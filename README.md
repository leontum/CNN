# evsite

Data pipeline that turns OpenStreetMap + OpenChargeMap data for **Niedersachsen +
Bremen** into a fixed-size, multi-channel tile stack usable as CNN input for
classifying EV charging site suitability (e.g. "schlecht" / "durchschnittlich" /
"gut"). This repo builds the **input tiles only** -- labels come from your own
labeling pipeline and get joined onto the tiles afterward by `tile_id`.

## What it produces

- A **tile grid**: `TILE_SIZE_M x TILE_SIZE_M` cells (default 500m x 500m) covering
  the region, each with a stable `tile_id`, pixel bounds, and a WGS84 centroid.
- A **raster stack**: one `(6, 256, 256)` float32 array per tile, stored in a
  chunked, compressed [zarr](https://zarr.readthedocs.io/) array so the whole
  region (400k+ tiles) never has to fit in memory at once. Channels, in order:

  | # | Channel | What it is |
  |---|---|---|
  | 0 | `road_density` | smoothed, class-weighted road presence, normalized to [0,1] |
  | 1 | `poi_density` | smoothed density of retail/amenity POIs, normalized to [0,1] |
  | 2 | `building_density` | smoothed building-footprint presence, [0,1] |
  | 3 | `dist_to_main_road` | distance to nearest primary/trunk/secondary road, clipped at 2km, normalized |
  | 4 | `dist_to_charger` | distance to nearest existing OpenChargeMap point, clipped at 2km, normalized |
  | 5 | `landuse_class` | categorical id, see `LANDUSE_CLASSES` in `src/evsite/config.py` |

  These are **rasterized vector features, not a rendered map screenshot** -- see
  "Why not just render map tiles?" below.

## Why not just render map tiles?

A screenshot of a rendered map (Mapnik-style) makes the CNN re-derive what a road
or a POI icon *means* from pixel patterns, and ties the model to one cartographic
style. Rasterizing the underlying OSM tags directly into interpretable channels
(road density, POI density, distances, land use) is more information-dense and
style-independent -- closer to what's actually causally relevant to charging-site
attractiveness. Note this still doesn't capture everything: traffic volume, grid
connection cost, land ownership, and zoning approval are invisible to any bird's-eye
raster and bound what accuracy is achievable from tiles alone.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Fast smoke test on a small area (few minutes, ~20MB) before committing to the
# full region -- writes to data_cache/*_bbox_test/, never touches the real cache.
python scripts/build_dataset.py --all --bbox 9.68 52.34 9.78 52.40

# Full Niedersachsen + Bremen. Downloads ~525MB of .pbf, then processes ~406k
# tiles -- see "Scale" below before running this unattended.
python scripts/build_dataset.py --download --extract --ocm
python scripts/build_dataset.py --grid --rasterize
```

Each step is independently re-runnable and cached (`--force` to bypass the cache):

```bash
python scripts/build_dataset.py --download    # Geofabrik .pbf -> data_cache/raw_osm/
python scripts/build_dataset.py --extract     # pyrosm -> data_cache/features/*.geoparquet
python scripts/build_dataset.py --ocm         # OpenChargeMap -> data_cache/openchargemap/
python scripts/build_dataset.py --grid        # tile grid -> data_cache/grid/
python scripts/build_dataset.py --rasterize   # raster stack -> data_cache/rasters/tiles.zarr
```

## Reading tiles back out

```python
from evsite.data.dataset import TileStore

store = TileStore()  # data_cache/rasters/tiles.zarr + data_cache/grid/tile_index.parquet
arr = store.get("r00312_c00450")   # (6, 256, 256) float32
meta = store.meta("r00312_c00450") # row, col, bounds, centroid_lon/lat
```

Once you have a labels table (`tile_id`, `label`) from your own annotation process:

```python
from evsite.data.dataset import TileStore, load_labels, make_torch_dataset

store = TileStore()
labels = load_labels("my_labels.parquet")
dataset = make_torch_dataset(store, labels=labels)  # torch.utils.data.Dataset
```

## Scale, and why the pipeline is block-based

At 500m tiles, Niedersachsen + Bremen is **~406,000 tiles** (588 rows x 691 cols).
Rasterizing tile-by-tile would mean 406k separate small function calls; rasterizing
the whole region as one dense array is ~27 billion pixels per channel and won't fit
in memory or reasonably on disk. `rasterize.py` instead processes **blocks** of
`BLOCK_SIZE_TILES` tiles (default 32 -> 16km) at a time:

- `road_density`, `poi_density`, `building_density`, `landuse_class` only depend on
  what's near a pixel, so they're rasterized directly at block resolution.
- `dist_to_main_road`, `dist_to_charger` are distance transforms, so a pixel can
  depend on a feature up to `DIST_CLIP_M` (2km) outside its own tile. Those channels
  pad each block by a `HALO_M` margin, run `scipy.ndimage.distance_transform_edt`,
  and crop back down -- padding whole 16km blocks by 2km is cheap; padding every
  500m tile individually by 2km would have meant ~80x more pixels per tile.

Output goes to a zarr array chunked one tile at a time (`chunks=(1, 6, 256, 256)`),
zstd-compressed, so training reads pull individual tiles cheaply without ever
materializing the full stack in memory.

**Before running the full-region build**, time a small `--bbox` run first and
extrapolate -- both wall-clock time and disk usage scale with block count and are
very machine-dependent (CPU core count matters a lot for the distance transforms).
Compressed disk size is hard to predict up front: dense urban blocks compress worse
than mostly-empty rural/forest blocks.

## Normalization constants are starting points, not calibrated

`ROAD_DENSITY_CLIP`, `POI_DENSITY_CLIP`, and the smoothing sigmas in
`src/evsite/config.py` are plausible defaults, not fitted to the real value
distribution. Once you've rasterized a real region, look at the actual channel
histograms (`TileStore().get(tile_id)`) and adjust the clip constants so the [0,1]
range is actually used -- there was no labeled data available yet to calibrate
against when this pipeline was built.

## OpenChargeMap access

**As of this writing, OCM's `/v3/poi/` endpoint returns `403 Forbidden` without a
key** ("You must specify an API key using the key query parameter or x-api-key
header") -- confirmed against the live API during development, so this isn't
theoretical. Register a free key at https://openchargemap.org (account settings ->
API Keys) and set it before running `--ocm`:

```bash
export OCM_API_KEY=your-key-here
python scripts/build_dataset.py --ocm
```

Without a key, `--ocm` fails and `dist_to_charger` falls back to the clip value
(2km, i.e. "no charger nearby") everywhere -- silently wrong input for a model, not
just a missing feature, so don't skip this step. The connector code itself doesn't
care whether a key is set (`_fetch_cell` in `openchargemap.py` adds it to the
request params only when present) -- it's purely an OCM-side requirement.

## Layout

```
src/evsite/
  config.py              CRS, tile size, channel definitions, regions, paths
  data/
    download_osm.py       Geofabrik .pbf download (resumable, cached)
    osm_extract.py        pyrosm feature extraction -> geoparquet cache
    openchargemap.py       OpenChargeMap connector (chunked, cached, rate-limited)
    grid.py                 tile grid + shared grid geometry (single source of truth)
    rasterize.py            block-wise multi-channel rasterization -> zarr
    dataset.py              tile_id -> (6,256,256) array / torch Dataset
scripts/
  build_dataset.py         CLI orchestrating every step above
```

## Not included here (yet)

- The CNN architecture and training loop -- this repo is the data pipeline only.
- A labeling pipeline -- `evsite.data.dataset.load_labels()` is the join point for
  whatever labels you produce; it just expects a `(tile_id, label)` table.
