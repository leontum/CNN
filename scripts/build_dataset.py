#!/usr/bin/env python
"""Orchestrates the full pipeline: download OSM extracts -> extract features ->
fetch OpenChargeMap points -> build the tile grid -> rasterize the CNN input stack.

Examples:
    # Full Niedersachsen + Bremen (large: multi-GB download, hours of processing)
    python scripts/build_dataset.py --all

    # A small area for a quick end-to-end smoke test, e.g. central Hannover.
    # Writes into data_cache/*_bbox_test/ instead of the real cache dirs, so it
    # never clobbers a full-region run.
    python scripts/build_dataset.py --all --bbox 9.68 52.34 9.78 52.40
"""
from __future__ import annotations

import argparse

import geopandas as gpd

from evsite.config import DATA_DIR, REGIONS
from evsite.data import download_osm, grid, openchargemap, osm_extract, rasterize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download Geofabrik .pbf extracts")
    parser.add_argument("--extract", action="store_true", help="extract OSM feature layers")
    parser.add_argument("--ocm", action="store_true", help="fetch OpenChargeMap points")
    parser.add_argument("--grid", action="store_true", help="build the tile grid")
    parser.add_argument("--rasterize", action="store_true", help="build the raster stack")
    parser.add_argument("--all", action="store_true", help="run every step in order")
    parser.add_argument("--force", action="store_true", help="ignore caches, redo everything")
    parser.add_argument(
        "--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="override the region bbox (WGS84) -- use a small box for a fast smoke test. "
             "Writes to separate *_bbox_test cache dirs, never touches the full-region cache.",
    )
    args = parser.parse_args()
    if args.all:
        args.download = args.extract = args.ocm = args.grid = args.rasterize = True
    if not any([args.download, args.extract, args.ocm, args.grid, args.rasterize]):
        parser.error("choose at least one step, or pass --all")

    bbox = tuple(args.bbox) if args.bbox else None
    suffix = "_bbox_test" if bbox else ""
    features_dir = DATA_DIR / f"features{suffix}"
    ocm_cache_dir = DATA_DIR / f"openchargemap{suffix}"
    grid_dir = DATA_DIR / f"grid{suffix}"
    raster_store_path = DATA_DIR / f"rasters{suffix}" / "tiles.zarr"

    pbf_paths = None
    if args.download:
        print("== downloading OSM extracts ==")
        pbf_paths = download_osm.download_all(force=args.force)
        for name, path in pbf_paths.items():
            print(f"  {name}: {path}")

    if args.extract:
        print("== extracting OSM features ==")
        if pbf_paths is None:
            from evsite.config import RAW_OSM_DIR
            pbf_paths = {name: RAW_OSM_DIR / f"{name}-latest.osm.pbf" for name in REGIONS}
        layers = osm_extract.extract_all(
            pbf_paths, force=args.force, cache_dir=features_dir,
            bounding_box=list(bbox) if bbox else None,
        )
        for name, gdf in layers.items():
            print(f"  {name}: {0 if gdf is None else len(gdf)} features")

    if args.ocm:
        print("== fetching OpenChargeMap points ==")
        from evsite.config import combined_bbox_wgs84
        ocm_gdf = openchargemap.fetch_region(bbox or combined_bbox_wgs84())
        path = openchargemap.save(ocm_gdf, out_dir=ocm_cache_dir)
        print(f"  {len(ocm_gdf)} points -> {path}")

    if args.grid:
        print("== building tile grid ==")
        from evsite.config import combined_bbox_wgs84
        region_bbox = bbox or combined_bbox_wgs84()
        origin_x, origin_y, n_rows, n_cols = grid.compute_grid_geometry(region_bbox)
        tile_grid = grid.build_tile_grid(bbox_wgs84=region_bbox)
        geo_path, idx_path = grid.save_grid(tile_grid, out_dir=grid_dir)
        meta_path = grid.save_grid_meta(origin_x, origin_y, n_rows, n_cols, out_dir=grid_dir)
        print(f"  {len(tile_grid)} tiles ({n_rows} rows x {n_cols} cols)")
        print(f"  wrote {geo_path}, {idx_path}, {meta_path}")

    if args.rasterize:
        print("== rasterizing tile stack ==")
        meta = grid.load_grid_meta(grid_dir / "grid_meta.json")
        tile_index = grid.load_tile_index(grid_dir / "tile_index.parquet")

        layers = {}
        for name in ["roads", "buildings", "landuse", "natural", "pois"]:
            path = features_dir / f"{name}.geoparquet"
            layers[name] = gpd.read_parquet(path) if path.exists() else None

        ocm_path = ocm_cache_dir / "openchargemap_points.geoparquet"
        ocm_points = gpd.read_parquet(ocm_path) if ocm_path.exists() else None

        z = rasterize.rasterize_region(
            tile_index,
            meta["origin_x"], meta["origin_y"], meta["n_rows"], meta["n_cols"], meta["tile_size_m"],
            layers, ocm_points,
            store_path=raster_store_path,
        )
        print(f"  wrote {z.shape} to {raster_store_path}")


if __name__ == "__main__":
    main()
