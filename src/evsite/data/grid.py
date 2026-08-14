"""Deterministic equal-size tile grid over a region.

This is independent of any feature data: it just partitions a bounding box into
TILE_SIZE_M x TILE_SIZE_M cells in TARGET_CRS. rasterize.py fills each cell with
channel values later; labels get joined onto this grid by tile_id.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

from evsite.config import (
    GRID_DIR,
    TARGET_CRS,
    TILE_PIXELS,
    TILE_SIZE_M,
    WGS84,
    combined_bbox_wgs84,
)


def _projected_bbox(bbox_wgs84: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reprojects a WGS84 bbox to TARGET_CRS, covering it fully (not just corner-to-corner,
    since UTM warps a lon/lat rectangle into a non-axis-aligned shape -- we sample all
    four corners plus the midpoints of each edge and take the enclosing box)."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    lons = [lon_min, lon_max, (lon_min + lon_max) / 2]
    lats = [lat_min, lat_max, (lat_min + lat_max) / 2]
    sample_points = [(lon, lat) for lon in lons for lat in lats]

    transformer = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    xs, ys = [], []
    for lon, lat in sample_points:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def compute_grid_geometry(
    bbox_wgs84: tuple[float, float, float, float],
    tile_size_m: float = TILE_SIZE_M,
) -> tuple[float, float, int, int]:
    """Returns (origin_x, origin_y, n_rows, n_cols) in TARGET_CRS for bbox_wgs84.

    This is the single source of truth for how the tile lattice is placed --
    build_tile_grid() and rasterize.py both call it, so tile polygons and raster
    pixels are guaranteed to line up exactly.
    """
    minx, miny, maxx, maxy = _projected_bbox(bbox_wgs84)
    # Snap the origin down to a multiple of the tile size so the grid is stable
    # regardless of the exact bbox passed in (re-running with a slightly different
    # bbox still lines up on the same lattice).
    origin_x = math.floor(minx / tile_size_m) * tile_size_m
    origin_y = math.floor(miny / tile_size_m) * tile_size_m
    n_cols = math.ceil((maxx - origin_x) / tile_size_m)
    n_rows = math.ceil((maxy - origin_y) / tile_size_m)
    return origin_x, origin_y, n_rows, n_cols


def build_tile_grid(
    region_names: list[str] | None = None,
    tile_size_m: float = TILE_SIZE_M,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Builds the tile grid as a GeoDataFrame in TARGET_CRS.

    Columns: tile_id, row, col, minx, miny, maxx, maxy, centroid_lon, centroid_lat, geometry.
    tile_id is a stable string ("r{row:05d}_c{col:05d}") so it is reproducible across runs
    and can be used as a join key against labels or the raster pixel index.
    """
    bbox = bbox_wgs84 or combined_bbox_wgs84(region_names)
    origin_x, origin_y, n_rows, n_cols = compute_grid_geometry(bbox, tile_size_m)

    rows, cols, geoms = [], [], []
    for row in range(n_rows):
        y0 = origin_y + row * tile_size_m
        y1 = y0 + tile_size_m
        for col in range(n_cols):
            x0 = origin_x + col * tile_size_m
            x1 = x0 + tile_size_m
            rows.append(row)
            cols.append(col)
            geoms.append(box(x0, y0, x1, y1))

    gdf = gpd.GeoDataFrame({"row": rows, "col": cols, "geometry": geoms}, crs=TARGET_CRS)
    gdf["tile_id"] = [f"r{r:05d}_c{c:05d}" for r, c in zip(gdf["row"], gdf["col"])]
    bounds = gdf.geometry.bounds
    gdf["minx"], gdf["miny"], gdf["maxx"], gdf["maxy"] = (
        bounds["minx"],
        bounds["miny"],
        bounds["maxx"],
        bounds["maxy"],
    )

    centroids = gdf.geometry.centroid.to_crs(WGS84)
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y

    gdf = gdf[["tile_id", "row", "col", "minx", "miny", "maxx", "maxy",
               "centroid_lon", "centroid_lat", "geometry"]]
    return gdf


def save_grid(gdf: gpd.GeoDataFrame, out_dir: Path = GRID_DIR) -> tuple[Path, Path]:
    """Writes the grid twice: a geoparquet with geometry, and a plain parquet index
    (no geometry column) that's cheap to load with just pandas at training time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    geo_path = out_dir / "tile_grid.geoparquet"
    idx_path = out_dir / "tile_index.parquet"
    gdf.to_parquet(geo_path)
    pd.DataFrame(gdf.drop(columns="geometry")).to_parquet(idx_path)
    return geo_path, idx_path


def load_tile_index(path: Path = GRID_DIR / "tile_index.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def save_grid_meta(
    origin_x: float, origin_y: float, n_rows: int, n_cols: int,
    tile_size_m: float = TILE_SIZE_M, out_dir: Path = GRID_DIR,
) -> Path:
    """Persists the exact grid placement so rasterize.py and dataset.py never have
    to recompute it from a bbox (which risks float drift between runs/machines)."""
    meta = {
        "origin_x": origin_x, "origin_y": origin_y,
        "n_rows": n_rows, "n_cols": n_cols,
        "tile_size_m": tile_size_m, "tile_pixels": TILE_PIXELS,
        "pixel_size_m": tile_size_m / TILE_PIXELS,
        "crs": TARGET_CRS,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "grid_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def load_grid_meta(path: Path = GRID_DIR / "grid_meta.json") -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    bbox = combined_bbox_wgs84()
    origin_x, origin_y, n_rows, n_cols = compute_grid_geometry(bbox)
    grid = build_tile_grid(bbox_wgs84=bbox)
    geo_path, idx_path = save_grid(grid)
    meta_path = save_grid_meta(origin_x, origin_y, n_rows, n_cols)
    print(f"{len(grid)} tiles ({n_rows} rows x {n_cols} cols)")
    print(f"wrote {geo_path}")
    print(f"wrote {idx_path}")
    print(f"wrote {meta_path}")
