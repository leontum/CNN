"""Reads individual tiles back out of the rasterized zarr stack.

Kept independent of torch at import time (only the Dataset class needs it, and that
import is local) so the rest of the data-prep pipeline works in an environment
without torch installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from evsite.config import GRID_DIR, RASTER_STORE_PATH


class TileStore:
    """Thin wrapper joining the tile index (tile_id -> row/col/bounds/centroid) with
    the zarr raster stack (row/col -> (C, H, W) array), independent of any ML framework."""

    def __init__(
        self,
        store_path: Path = RASTER_STORE_PATH,
        tile_index_path: Path = GRID_DIR / "tile_index.parquet",
    ):
        self.z = zarr.open(str(store_path), mode="r")
        self.tile_index = pd.read_parquet(tile_index_path).set_index("tile_id")
        self.n_cols = int(self.z.attrs["n_cols"])
        self.channels = list(self.z.attrs["channels"])

    def __len__(self) -> int:
        return len(self.tile_index)

    def tile_ids(self) -> list[str]:
        return list(self.tile_index.index)

    def _flat_index(self, tile_id: str) -> int:
        row = self.tile_index.at[tile_id, "row"]
        col = self.tile_index.at[tile_id, "col"]
        return int(row) * self.n_cols + int(col)

    def get(self, tile_id: str) -> np.ndarray:
        """Returns the (C, TILE_PIXELS, TILE_PIXELS) float32 array for one tile."""
        return self.z[self._flat_index(tile_id)]

    def meta(self, tile_id: str) -> pd.Series:
        return self.tile_index.loc[tile_id]


def load_labels(path: Path) -> pd.Series:
    """Loads a labels table with at least columns [tile_id, label] into a
    tile_id -> label Series. Point of integration for your own labeling pipeline --
    this repo only produces the tiles, not the labels."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return df.set_index("tile_id")["label"]


def make_torch_dataset(
    tile_store: TileStore,
    labels: pd.Series | None = None,
    tile_ids: list[str] | None = None,
):
    """Builds a torch.utils.data.Dataset over the given tile ids (default: every
    tile that has a label, or every tile in the store if labels is None). Import of
    torch is local so evsite.data stays usable without torch installed."""
    import torch
    from torch.utils.data import Dataset

    ids = tile_ids
    if ids is None:
        ids = list(labels.index) if labels is not None else tile_store.tile_ids()

    class _TileDataset(Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            tile_id = ids[i]
            arr = tile_store.get(tile_id)
            tensor = torch.from_numpy(np.ascontiguousarray(arr))
            if labels is not None:
                return tensor, labels.loc[tile_id], tile_id
            return tensor, tile_id

    return _TileDataset()
