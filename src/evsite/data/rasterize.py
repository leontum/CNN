"""Turns the extracted OSM/OpenChargeMap layers into the 6-channel raster stack
used as CNN input, one value per pixel per tile.

Two different rules for two different kinds of channel:

- road_density, poi_density, building_density, landuse_class only depend on what's
  inside a tile (plus a small smoothing radius), so they're rasterized per BLOCK of
  tiles directly, no padding needed.
- dist_to_main_road, dist_to_charger are distance transforms, so a pixel's value can
  depend on a feature outside its own tile (even outside its own block) up to
  DIST_CLIP_M away. Those are rasterized into a block padded by a HALO_M margin,
  distance-transformed, then cropped back down to the block's own extent. Padding
  every tile individually by 2km would blow up the pixel count ~80x; padding whole
  16km blocks by 2km instead keeps the overhead small.

Output is written to a zarr array (RASTER_STORE_PATH), shape
(n_tiles, N_CHANNELS, TILE_PIXELS, TILE_PIXELS), chunked one tile at a time and
zstd-compressed -- the state-wide stack is far too large to hold as a single dense
array (400k+ tiles), but any individual tile is cheap to read back for training.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import zarr
from numcodecs import Blosc
from rasterio.features import MergeAlg, rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm import tqdm

from evsite.config import (
    BLOCK_SIZE_TILES,
    BUILDING_SMOOTH_SIGMA_M,
    CHANNELS,
    DIST_CLIP_M,
    HALO_M,
    LANDUSE_CLASSES,
    N_CHANNELS,
    PIXEL_SIZE_M,
    POI_DENSITY_CLIP,
    POI_SMOOTH_SIGMA_M,
    RASTER_STORE_PATH,
    ROAD_DENSITY_CLIP,
    ROAD_SMOOTH_SIGMA_M,
    TILE_PIXELS,
)
from evsite.data.osm_extract import MAIN_ROAD_HIGHWAY_CLASSES


def _block_ranges(n_rows: int, n_cols: int, block_size: int):
    for row_start in range(0, n_rows, block_size):
        row_end = min(row_start + block_size, n_rows)
        for col_start in range(0, n_cols, block_size):
            col_end = min(col_start + block_size, n_cols)
            yield row_start, row_end, col_start, col_end


def _bbox_for(origin_x, origin_y, row_start, row_end, col_start, col_end, tile_size_m):
    x0 = origin_x + col_start * tile_size_m
    x1 = origin_x + col_end * tile_size_m
    y0 = origin_y + row_start * tile_size_m
    y1 = origin_y + row_end * tile_size_m
    return x0, y0, x1, y1


def _subset(gdf: gpd.GeoDataFrame | None, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame | None:
    if gdf is None or len(gdf) == 0:
        return None
    x0, y0, x1, y1 = bbox
    sub = gdf.cx[x0:x1, y0:y1]
    return sub if len(sub) > 0 else None


def _rasterize_lines_weighted(gdf, value_col, out_shape, transform) -> np.ndarray:
    if gdf is None:
        return np.zeros(out_shape, dtype=np.float32)
    shapes = list(zip(gdf.geometry, gdf[value_col].astype(float)))
    return rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0.0,
        merge_alg=MergeAlg.add, dtype="float32",
    )


def _rasterize_point_count(gdf, out_shape, transform) -> np.ndarray:
    if gdf is None:
        return np.zeros(out_shape, dtype=np.float32)
    shapes = [(geom, 1.0) for geom in gdf.geometry]
    return rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0.0,
        merge_alg=MergeAlg.add, dtype="float32",
    )


def _rasterize_binary(gdf, out_shape, transform, all_touched=True) -> np.ndarray:
    if gdf is None:
        return np.zeros(out_shape, dtype=bool)
    shapes = [(geom, 1) for geom in gdf.geometry]
    return rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0,
        all_touched=all_touched, dtype="uint8",
    ).astype(bool)


def _rasterize_landuse(landuse_gdf, natural_gdf, out_shape, transform) -> np.ndarray:
    out = np.zeros(out_shape, dtype=np.float32)  # 0 == "unknown"
    for gdf in (natural_gdf, landuse_gdf):  # natural first (background), landuse on top
        if gdf is None or len(gdf) == 0:
            continue
        class_ids = gdf["landuse_class"].map(LANDUSE_CLASSES).fillna(LANDUSE_CLASSES["other"])
        shapes = list(zip(gdf.geometry, class_ids.astype(float)))
        rasterize(
            shapes, out_shape=out_shape, transform=transform,
            all_touched=False, dtype="float32", out=out,
        )
    return out


def _sigma_px(sigma_m: float) -> float:
    return sigma_m / PIXEL_SIZE_M


def _distance_channel(feature_gdf, block_bbox, block_shape, transform_padded, halo_px) -> np.ndarray:
    padded_shape = (block_shape[0] + 2 * halo_px, block_shape[1] + 2 * halo_px)
    x0, y0, x1, y1 = block_bbox
    padded_bbox = (x0 - HALO_M, y0 - HALO_M, x1 + HALO_M, y1 + HALO_M)
    sub = _subset(feature_gdf, padded_bbox)
    mask = _rasterize_binary(sub, padded_shape, transform_padded, all_touched=True)

    if not mask.any():
        dist_m = np.full(block_shape, DIST_CLIP_M, dtype=np.float32)
    else:
        dist_px = distance_transform_edt(~mask)
        dist_m = (dist_px * PIXEL_SIZE_M).astype(np.float32)
        dist_m = dist_m[halo_px:halo_px + block_shape[0], halo_px:halo_px + block_shape[1]]

    return np.clip(dist_m / DIST_CLIP_M, 0.0, 1.0)


def rasterize_block(
    row_start: int, row_end: int, col_start: int, col_end: int,
    origin_x: float, origin_y: float, tile_size_m: float,
    layers: dict[str, gpd.GeoDataFrame | None],
    ocm_points: gpd.GeoDataFrame | None,
) -> np.ndarray:
    """Returns a (N_CHANNELS, block_h_px, block_w_px) float32 array for one block of
    tiles, in CHANNELS order, pixel row 0 = the block's north edge (standard raster
    convention -- note this is the OPPOSITE of the tile grid's row axis, which
    increases northward; callers must flip when slicing tiles back out)."""
    x0, y0, x1, y1 = _bbox_for(origin_x, origin_y, row_start, row_end, col_start, col_end, tile_size_m)
    block_h_px = (row_end - row_start) * TILE_PIXELS
    block_w_px = (col_end - col_start) * TILE_PIXELS
    out_shape = (block_h_px, block_w_px)
    transform = from_origin(x0, y1, PIXEL_SIZE_M, PIXEL_SIZE_M)

    bbox = (x0, y0, x1, y1)
    roads = _subset(layers["roads"], bbox)
    pois = _subset(layers["pois"], bbox)
    buildings = _subset(layers["buildings"], bbox)
    landuse = _subset(layers["landuse"], bbox)
    natural = _subset(layers["natural"], bbox)

    road_density = _rasterize_lines_weighted(roads, "road_weight", out_shape, transform)
    road_density = gaussian_filter(road_density, sigma=_sigma_px(ROAD_SMOOTH_SIGMA_M))
    road_density = np.clip(road_density / ROAD_DENSITY_CLIP, 0.0, 1.0)

    poi_density = _rasterize_point_count(pois, out_shape, transform)
    poi_density = gaussian_filter(poi_density, sigma=_sigma_px(POI_SMOOTH_SIGMA_M))
    poi_density = np.clip(poi_density / POI_DENSITY_CLIP, 0.0, 1.0)

    building_mask = _rasterize_binary(buildings, out_shape, transform, all_touched=True).astype(np.float32)
    building_density = gaussian_filter(building_mask, sigma=_sigma_px(BUILDING_SMOOTH_SIGMA_M))
    building_density = np.clip(building_density, 0.0, 1.0)

    landuse_class = _rasterize_landuse(landuse, natural, out_shape, transform)

    halo_px = round(HALO_M / PIXEL_SIZE_M)
    padded_transform = from_origin(x0 - HALO_M, y1 + HALO_M, PIXEL_SIZE_M, PIXEL_SIZE_M)

    main_roads = None
    roads_full = layers["roads"]
    if roads_full is not None:
        main_roads = roads_full[roads_full["is_main_road"]]
    dist_to_main_road = _distance_channel(main_roads, bbox, out_shape, padded_transform, halo_px)
    dist_to_charger = _distance_channel(ocm_points, bbox, out_shape, padded_transform, halo_px)

    stack = np.stack(
        [road_density, poi_density, building_density,
         dist_to_main_road, dist_to_charger, landuse_class],
        axis=0,
    ).astype(np.float32)
    assert stack.shape[0] == N_CHANNELS
    return stack


def _tile_window(block_stack: np.ndarray, row_start, row_end, col_start, col_end, r: int, c: int) -> np.ndarray:
    """Slices tile (r, c)'s (C, TILE_PIXELS, TILE_PIXELS) window out of a block array.
    Tile-grid rows increase northward but raster pixel rows increase southward
    (row 0 = top/north), so this flips the row axis; columns need no flip."""
    row_from_top = (row_end - 1 - r) - row_start
    col_from_left = c - col_start
    pr0 = row_from_top * TILE_PIXELS
    pc0 = col_from_left * TILE_PIXELS
    return block_stack[:, pr0:pr0 + TILE_PIXELS, pc0:pc0 + TILE_PIXELS]


def rasterize_region(
    tile_index,  # pandas DataFrame from grid.load_tile_index(): tile_id, row, col, ...
    origin_x: float, origin_y: float, n_rows: int, n_cols: int, tile_size_m: float,
    layers: dict[str, gpd.GeoDataFrame | None],
    ocm_points: gpd.GeoDataFrame | None,
    store_path=RASTER_STORE_PATH,
    block_size_tiles: int = BLOCK_SIZE_TILES,
) -> zarr.Array:
    n_tiles = n_rows * n_cols
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    z = zarr.open(
        str(store_path), mode="a",
        shape=(n_tiles, N_CHANNELS, TILE_PIXELS, TILE_PIXELS),
        chunks=(1, N_CHANNELS, TILE_PIXELS, TILE_PIXELS),
        dtype="float32", compressor=compressor,
    )
    z.attrs["channels"] = CHANNELS
    z.attrs["n_rows"] = n_rows
    z.attrs["n_cols"] = n_cols

    row_by_rc = tile_index.set_index(["row", "col"])["tile_id"]

    blocks = list(_block_ranges(n_rows, n_cols, block_size_tiles))
    for row_start, row_end, col_start, col_end in tqdm(blocks, desc="Rasterizing blocks"):
        block_stack = rasterize_block(
            row_start, row_end, col_start, col_end,
            origin_x, origin_y, tile_size_m, layers, ocm_points,
        )
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if (r, c) not in row_by_rc.index:
                    continue  # tile outside the requested grid (partial edge block)
                idx = r * n_cols + c
                z[idx] = _tile_window(block_stack, row_start, row_end, col_start, col_end, r, c)
    return z


if __name__ == "__main__":
    from evsite.config import FEATURES_DIR, OCM_CACHE_DIR, combined_bbox_wgs84
    from evsite.data import grid as grid_mod
    from evsite.data import openchargemap

    meta = grid_mod.load_grid_meta()
    tile_index = grid_mod.load_tile_index()

    import geopandas as gpd

    layers = {}
    for name in ["roads", "buildings", "landuse", "natural", "pois"]:
        path = FEATURES_DIR / f"{name}.geoparquet"
        layers[name] = gpd.read_parquet(path) if path.exists() else None

    ocm_path = OCM_CACHE_DIR / "openchargemap_points.geoparquet"
    ocm_points = gpd.read_parquet(ocm_path) if ocm_path.exists() else None

    z = rasterize_region(
        tile_index,
        meta["origin_x"], meta["origin_y"], meta["n_rows"], meta["n_cols"], meta["tile_size_m"],
        layers, ocm_points,
    )
    print(f"wrote {z.shape} to {RASTER_STORE_PATH}")
