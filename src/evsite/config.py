"""Constants shared by every stage of the pipeline: CRS, tile geometry, regions, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Coordinate reference system -------------------------------------------------
# ETRS89 / UTM zone 32N: the standard metric CRS for German federal geodata, and
# the zone both Niedersachsen and Bremen fall into. All tiling/rasterization math
# happens in this CRS so "meters" means meters.
TARGET_CRS = "EPSG:25832"
WGS84 = "EPSG:4326"

# --- Tile geometry -----------------------------------------------------------------
TILE_SIZE_M = 500.0
TILE_PIXELS = 256
PIXEL_SIZE_M = TILE_SIZE_M / TILE_PIXELS  # ~1.953 m/px

# --- Feature channels (rasterized stack, in this fixed order) ----------------------
CHANNELS = [
    "road_density",       # normalized line-length of drivable roads per pixel
    "poi_density",        # normalized point density of retail/amenity POIs
    "building_density",   # fraction of pixel area covered by building footprints
    "dist_to_main_road",  # distance transform to nearest primary/trunk/secondary road, meters, clipped+normalized
    "dist_to_charger",    # distance transform to nearest existing OpenChargeMap point, meters, clipped+normalized
    "landuse_class",      # categorical id, see LANDUSE_CLASSES below
]
N_CHANNELS = len(CHANNELS)

# Categorical landuse classes rasterized into the "landuse_class" channel.
# 0 is reserved for "unknown/unmapped" so it survives as a real class, not a null.
LANDUSE_CLASSES = {
    "unknown": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
    "retail": 4,
    "farmland": 5,
    "forest": 6,
    "grass_meadow": 7,
    "water": 8,
    "other": 9,
}

# Distances beyond this are clipped before normalizing dist_to_* channels to [0, 1].
DIST_CLIP_M = 2000.0

# --- Rasterization tuning -------------------------------------------------------
# Tiles are processed in square blocks (not one by one, not the whole region at
# once): big enough to amortize per-call overhead, small enough that the halo
# needed for correct distance transforms (>= DIST_CLIP_M) doesn't dominate the
# block's own area. See rasterize.py for why distance channels need a halo and
# density/landuse channels don't.
BLOCK_SIZE_TILES = 32  # 32 * 500m = 16 km per block side
HALO_M = DIST_CLIP_M

# Gaussian smoothing sigma (meters) turning sparse point/line rasterizations into
# continuous density fields. Chosen as "a plausible walkable/local scale", not
# fitted -- revisit once you've looked at the actual value distributions.
ROAD_SMOOTH_SIGMA_M = 50.0
POI_SMOOTH_SIGMA_M = 100.0
BUILDING_SMOOTH_SIGMA_M = 25.0

# Clip values used to normalize the two density channels into [0, 1] after
# smoothing. Starting points, not calibrated against real data -- see README.
ROAD_DENSITY_CLIP = 3.0
POI_DENSITY_CLIP = 0.05

# --- Regions -------------------------------------------------------------------
# Geofabrik ships Bremen (a city-state, enclosed within Niedersachsen) as a
# separate extract from Niedersachsen itself, so both are needed for full coverage.
# Bboxes are WGS84 (lon_min, lat_min, lon_max, lat_max), deliberately generous
# rectangles around each state -- used only to seed the tile grid, not as an
# authoritative administrative boundary.
GEOFABRIK_BASE = "https://download.geofabrik.de/europe/germany"

@dataclass(frozen=True)
class Region:
    name: str
    pbf_url: str
    bbox_wgs84: tuple[float, float, float, float]


REGIONS: dict[str, Region] = {
    "niedersachsen": Region(
        name="niedersachsen",
        pbf_url=f"{GEOFABRIK_BASE}/niedersachsen-latest.osm.pbf",
        bbox_wgs84=(6.65, 51.29, 11.60, 53.90),
    ),
    "bremen": Region(
        name="bremen",
        pbf_url=f"{GEOFABRIK_BASE}/bremen-latest.osm.pbf",
        bbox_wgs84=(8.48, 53.01, 8.99, 53.61),
    ),
}


def combined_bbox_wgs84(region_names: list[str] | None = None) -> tuple[float, float, float, float]:
    """Union bbox (lon_min, lat_min, lon_max, lat_max) over the given regions."""
    regions = [REGIONS[n] for n in (region_names or list(REGIONS))]
    lon_min = min(r.bbox_wgs84[0] for r in regions)
    lat_min = min(r.bbox_wgs84[1] for r in regions)
    lon_max = max(r.bbox_wgs84[2] for r in regions)
    lat_max = max(r.bbox_wgs84[3] for r in regions)
    return (lon_min, lat_min, lon_max, lat_max)


# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EVSITE_DATA_DIR", PROJECT_ROOT / "data_cache"))
RAW_OSM_DIR = DATA_DIR / "raw_osm"          # downloaded .pbf files
FEATURES_DIR = DATA_DIR / "features"        # extracted OSM GeoDataFrames, geoparquet
OCM_CACHE_DIR = DATA_DIR / "openchargemap"  # cached OCM API responses
RASTER_DIR = DATA_DIR / "rasters"           # rasterized tile stack (zarr store)
RASTER_STORE_PATH = RASTER_DIR / "tiles.zarr"
GRID_DIR = DATA_DIR / "grid"                # tile index (geoparquet + parquet)

for _d in (RAW_OSM_DIR, FEATURES_DIR, OCM_CACHE_DIR, RASTER_DIR, GRID_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- OpenChargeMap -------------------------------------------------------------
OCM_BASE_URL = "https://api.openchargemap.io/v3/poi/"
# Optional: set OCM_API_KEY to raise the (otherwise tight) public rate limit.
# The pipeline works without one, just slower and more cautiously paced.
OCM_API_KEY = os.environ.get("OCM_API_KEY")
