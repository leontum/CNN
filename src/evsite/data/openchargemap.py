"""OpenChargeMap connector: fetches existing charging stations for a region.

The public API works without a key but is tightly rate-limited, so this chunks a
bbox into cells small enough that no single request hits OCM's result cap, caches
every cell's raw JSON response to disk, and paces requests. Set OCM_API_KEY (see
config.py) to raise the rate limit -- the code path is identical either way.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

from evsite.config import OCM_API_KEY, OCM_BASE_URL, OCM_CACHE_DIR, TARGET_CRS, WGS84

# Kept well under OCM's per-request cap so a dense metro area cell doesn't silently
# truncate results.
CELL_SIZE_DEG = 0.5
MAX_RESULTS_PER_CELL = 2000
REQUEST_DELAY_S = 1.1  # polite pacing for the unauthenticated public tier


def _cell_bounds(bbox_wgs84: tuple[float, float, float, float], cell_size_deg: float):
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    n_cols = math.ceil((lon_max - lon_min) / cell_size_deg)
    n_rows = math.ceil((lat_max - lat_min) / cell_size_deg)
    for row in range(n_rows):
        for col in range(n_cols):
            yield (
                lon_min + col * cell_size_deg,
                lat_min + row * cell_size_deg,
                min(lon_min + (col + 1) * cell_size_deg, lon_max),
                min(lat_min + (row + 1) * cell_size_deg, lat_max),
            )


def _cache_path(cell: tuple[float, float, float, float]) -> Path:
    key = hashlib.sha1(json.dumps(cell).encode()).hexdigest()[:16]
    return OCM_CACHE_DIR / f"cell_{key}.json"


def _fetch_cell(cell: tuple[float, float, float, float]) -> list[dict]:
    cache_path = _cache_path(cell)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    lon_min, lat_min, lon_max, lat_max = cell
    params = {
        "output": "json",
        "countrycode": "DE",
        "boundingbox": f"({lat_min},{lon_min}),({lat_max},{lon_max})",
        "maxresults": MAX_RESULTS_PER_CELL,
        "compact": "true",
        "verbose": "false",
    }
    if OCM_API_KEY:
        params["key"] = OCM_API_KEY

    resp = requests.get(OCM_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    time.sleep(REQUEST_DELAY_S)
    return data


def _flatten(poi: dict) -> dict | None:
    addr = poi.get("AddressInfo") or {}
    lon, lat = addr.get("Longitude"), addr.get("Latitude")
    if lon is None or lat is None:
        return None

    connections = poi.get("Connections") or []
    powers = [c["PowerKW"] for c in connections if c.get("PowerKW")]
    operator = poi.get("OperatorInfo") or {}

    return {
        "ocm_id": poi.get("ID"),
        "lon": lon,
        "lat": lat,
        "title": addr.get("Title"),
        "num_points": poi.get("NumberOfPoints"),
        "connection_count": len(connections),
        "max_power_kw": max(powers) if powers else None,
        "usage_cost": poi.get("UsageCost"),
        "operator": operator.get("Title"),
        "date_created": poi.get("DateCreated"),
        "date_last_verified": poi.get("DateLastVerified"),
    }


def fetch_region(
    bbox_wgs84: tuple[float, float, float, float],
    cell_size_deg: float = CELL_SIZE_DEG,
) -> gpd.GeoDataFrame:
    """Fetches all OCM points covering bbox_wgs84, de-duplicated by OCM id, in TARGET_CRS."""
    cells = list(_cell_bounds(bbox_wgs84, cell_size_deg))
    rows: dict[int, dict] = {}
    for cell in tqdm(cells, desc="OpenChargeMap cells"):
        for poi in _fetch_cell(cell):
            flat = _flatten(poi)
            if flat is not None:
                rows[flat["ocm_id"]] = flat

    df = pd.DataFrame(rows.values())
    if df.empty:
        return gpd.GeoDataFrame(columns=["ocm_id", "geometry"], geometry="geometry", crs=TARGET_CRS)

    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    return gdf.to_crs(TARGET_CRS)


def save(gdf: gpd.GeoDataFrame, out_dir: Path = OCM_CACHE_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "openchargemap_points.geoparquet"
    gdf.to_parquet(path)
    return path


def load(path: Path = OCM_CACHE_DIR / "openchargemap_points.geoparquet") -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)


if __name__ == "__main__":
    from evsite.config import combined_bbox_wgs84

    gdf = fetch_region(combined_bbox_wgs84())
    path = save(gdf)
    print(f"{len(gdf)} charging points -> {path}")
