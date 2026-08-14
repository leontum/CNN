"""Fast, network-free sanity checks using synthetic geometries -- not a substitute
for scripts/build_dataset.py --bbox ... against real data, but catches regressions
in the grid/rasterize math without any download."""
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from evsite.config import CHANNELS, N_CHANNELS, TARGET_CRS, TILE_PIXELS
from evsite.data.grid import build_tile_grid, compute_grid_geometry
from evsite.data.rasterize import rasterize_block

# A tiny bbox around a made-up point, just large enough for a 3x3 tile block.
TEST_BBOX = (9.70, 52.35, 9.715, 52.362)


def test_grid_tiles_are_500m_and_align_with_geometry():
    origin_x, origin_y, n_rows, n_cols = compute_grid_geometry(TEST_BBOX)
    grid = build_tile_grid(bbox_wgs84=TEST_BBOX)
    assert len(grid) == n_rows * n_cols
    widths = grid["maxx"] - grid["minx"]
    heights = grid["maxy"] - grid["miny"]
    assert np.allclose(widths, 500.0)
    assert np.allclose(heights, 500.0)
    assert grid["minx"].min() == origin_x
    assert grid["miny"].min() == origin_y


def _fake_layers(origin_x, origin_y):
    # One main road running east-west through the middle of the block, one charger
    # point on that road, one building and one POI a couple hundred meters away.
    mid_y = origin_y + 750  # middle of a 3-tile-tall (1500m) block
    road = gpd.GeoDataFrame(
        {"highway": ["primary"], "road_weight": [0.8], "is_main_road": [True]},
        geometry=[LineString([(origin_x, mid_y), (origin_x + 1500, mid_y)])],
        crs=TARGET_CRS,
    )
    charger = gpd.GeoDataFrame(
        geometry=[Point(origin_x + 750, mid_y)], crs=TARGET_CRS,
    )
    building = gpd.GeoDataFrame(
        geometry=[Polygon([
            (origin_x + 200, mid_y + 200), (origin_x + 230, mid_y + 200),
            (origin_x + 230, mid_y + 230), (origin_x + 200, mid_y + 230),
        ])],
        crs=TARGET_CRS,
    )
    poi = gpd.GeoDataFrame(
        {"shop": ["supermarket"]}, geometry=[Point(origin_x + 200, mid_y + 200)], crs=TARGET_CRS,
    )
    landuse = gpd.GeoDataFrame(
        {"landuse_class": ["residential"]},
        geometry=[Polygon([
            (origin_x, origin_y), (origin_x + 1500, origin_y),
            (origin_x + 1500, origin_y + 1500), (origin_x, origin_y + 1500),
        ])],
        crs=TARGET_CRS,
    )
    layers = {"roads": road, "buildings": building, "landuse": landuse,
              "natural": None, "pois": poi}
    return layers, charger


def test_rasterize_block_shapes_and_ranges():
    origin_x, origin_y, n_rows, n_cols = compute_grid_geometry(TEST_BBOX)
    layers, ocm_points = _fake_layers(origin_x, origin_y)

    row_end = min(3, n_rows)
    col_end = min(3, n_cols)
    stack = rasterize_block(0, row_end, 0, col_end, origin_x, origin_y, 500.0, layers, ocm_points)

    assert stack.shape == (N_CHANNELS, row_end * TILE_PIXELS, col_end * TILE_PIXELS)
    assert np.isfinite(stack).all()

    idx = {name: i for i, name in enumerate(CHANNELS)}
    for name in ["road_density", "poi_density", "building_density",
                 "dist_to_main_road", "dist_to_charger"]:
        chan = stack[idx[name]]
        assert chan.min() >= 0.0 - 1e-6
        assert chan.max() <= 1.0 + 1e-6

    # landuse_class should be a valid class id everywhere it was painted
    assert stack[idx["landuse_class"]].max() <= 9

    # the pixel exactly at the charger location should have ~zero distance
    dist_to_charger = stack[idx["dist_to_charger"]]
    assert dist_to_charger.min() < 0.05  # well under the 2km clip, near zero
