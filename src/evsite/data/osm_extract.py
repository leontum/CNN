"""Extracts roads, buildings, landuse/natural polygons and retail/amenity POIs from the
downloaded .pbf files via pyrosm, reprojects to TARGET_CRS, and caches each layer as
geoparquet so re-running the rasterizer never re-parses the raw OSM data.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyrosm import OSM

from evsite.config import FEATURES_DIR, LANDUSE_CLASSES, REGIONS, TARGET_CRS

# Road classes treated as "main roads" for the dist_to_main_road channel -- the
# tiers a driver would actually route along, not footpaths/tracks/service roads.
MAIN_ROAD_HIGHWAY_CLASSES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
}

# Relative weight per road class for the road_density channel: a motorway contributes
# more "road presence" per meter than a footpath even though both are lines.
ROAD_WEIGHTS = {
    "motorway": 1.0, "motorway_link": 0.8,
    "trunk": 1.0, "trunk_link": 0.8,
    "primary": 0.8, "primary_link": 0.6,
    "secondary": 0.6, "secondary_link": 0.5,
    "tertiary": 0.5, "tertiary_link": 0.4,
    "residential": 0.3, "living_street": 0.3, "unclassified": 0.3,
    "service": 0.2, "track": 0.1,
}
DEFAULT_ROAD_WEIGHT = 0.2

LANDUSE_TAG_MAP = {
    "residential": "residential",
    "commercial": "commercial",
    "industrial": "industrial",
    "retail": "retail",
    "farmland": "farmland", "farmyard": "farmland", "orchard": "farmland",
    "vineyard": "farmland", "allotments": "farmland", "meadow": "farmland",
    "forest": "forest",
    "grass": "grass_meadow", "recreation_ground": "grass_meadow",
    "village_green": "grass_meadow",
    "reservoir": "water", "basin": "water",
}
NATURAL_TAG_MAP = {
    "water": "water",
    "wood": "forest",
    "wetland": "grass_meadow", "scrub": "grass_meadow",
    "grassland": "grass_meadow", "heath": "grass_meadow",
}

POI_FILTER = {
    "shop": True,
    "amenity": ["supermarket", "restaurant", "cafe", "fast_food", "pharmacy",
                "bank", "marketplace", "fuel", "charging_station"],
    "leisure": ["fitness_centre", "sports_centre"],
}


def _reproject(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    if gdf is None or len(gdf) == 0:
        return None
    return gdf.to_crs(TARGET_CRS)


def extract_region(
    region_name: str,
    pbf_path: Path,
    bounding_box: list[float] | None = None,
) -> dict[str, gpd.GeoDataFrame | None]:
    """Extracts roads/buildings/landuse/pois from one .pbf file, in TARGET_CRS.

    bounding_box (WGS84 [lon_min, lat_min, lon_max, lat_max]) restricts parsing to a
    sub-area of the file -- handy for smoke-testing against a real extract without
    processing an entire state.
    """
    osm = OSM(str(pbf_path), bounding_box=bounding_box)

    roads = osm.get_network(network_type="driving")
    if roads is not None and len(roads) > 0:
        roads["road_weight"] = roads["highway"].map(ROAD_WEIGHTS).fillna(DEFAULT_ROAD_WEIGHT)
        roads["is_main_road"] = roads["highway"].isin(MAIN_ROAD_HIGHWAY_CLASSES)
    roads = _reproject(roads)

    buildings = _reproject(osm.get_buildings())

    landuse = osm.get_landuse()
    if landuse is not None and len(landuse) > 0:
        landuse["landuse_class"] = landuse["landuse"].map(LANDUSE_TAG_MAP).fillna("other")
    landuse = _reproject(landuse)

    natural = osm.get_natural()
    if natural is not None and len(natural) > 0:
        natural["landuse_class"] = natural["natural"].map(NATURAL_TAG_MAP).fillna("other")
    natural = _reproject(natural)

    pois = osm.get_pois(custom_filter=POI_FILTER)
    pois = _reproject(pois)

    return {"roads": roads, "buildings": buildings, "landuse": landuse,
            "natural": natural, "pois": pois}


def _concat(parts: list[gpd.GeoDataFrame | None]) -> gpd.GeoDataFrame | None:
    parts = [p for p in parts if p is not None and len(p) > 0]
    if not parts:
        return None
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=TARGET_CRS)


def extract_all(
    pbf_paths: dict[str, Path],
    force: bool = False,
    cache_dir: Path = FEATURES_DIR,
    bounding_box: list[float] | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Extracts every region and merges layers of the same kind. Cached as geoparquet
    per layer so repeated pipeline runs skip the (slow) pyrosm parse entirely."""
    layer_names = ["roads", "buildings", "landuse", "natural", "pois"]
    cache_paths = {name: cache_dir / f"{name}.geoparquet" for name in layer_names}

    if not force and all(p.exists() for p in cache_paths.values()):
        return {name: gpd.read_parquet(path) for name, path in cache_paths.items()}

    per_region = [extract_region(name, path, bounding_box=bounding_box) for name, path in pbf_paths.items()]
    merged = {
        name: _concat([region[name] for region in per_region])
        for name in layer_names
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, gdf in merged.items():
        if gdf is not None:
            gdf.to_parquet(cache_paths[name])

    return merged


if __name__ == "__main__":
    from evsite.config import RAW_OSM_DIR

    paths = {name: RAW_OSM_DIR / f"{name}-latest.osm.pbf" for name in REGIONS}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing .pbf files: {missing}. Run `python -m evsite.data.download_osm` first."
        )
    layers = extract_all(paths)
    for name, gdf in layers.items():
        n = 0 if gdf is None else len(gdf)
        print(f"{name}: {n} features")
