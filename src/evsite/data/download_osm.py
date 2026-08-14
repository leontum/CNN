"""Downloads the Geofabrik .osm.pbf extracts needed to cover Niedersachsen + Bremen.

Geofabrik ships Bremen (a city-state enclosed within Niedersachsen) as a separate
extract, so both files are required for full coverage of the two states.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

from evsite.config import REGIONS, RAW_OSM_DIR


def _download_with_resume(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Streams url to dest, skipping the download if dest already matches the
    server's reported size (cheap resumability across re-runs, not partial-byte
    resume -- Geofabrik extracts are republished wholesale, not appended to)."""
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"[skip] {dest.name} already downloaded ({remote_size / 1e6:.0f} MB)")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", remote_size))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)


def download_region(region_name: str, force: bool = False) -> Path:
    """Downloads one region's .pbf extract into RAW_OSM_DIR, returns the local path."""
    region = REGIONS[region_name]
    dest = RAW_OSM_DIR / f"{region.name}-latest.osm.pbf"
    if force and dest.exists():
        dest.unlink()
    _download_with_resume(region.pbf_url, dest)
    return dest


def download_all(region_names: list[str] | None = None, force: bool = False) -> dict[str, Path]:
    """Downloads every requested region (default: all of REGIONS). Returns name -> local path."""
    names = region_names or list(REGIONS)
    return {name: download_region(name, force=force) for name in names}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    paths = download_all()
    for name, path in paths.items():
        size_mb = path.stat().st_size / 1e6
        print(f"{name}: {path} ({size_mb:.0f} MB)")
