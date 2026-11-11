"""High-level workflows that tie the building blocks together.

    run_tiling   scenes  -> normalised .npy tiles + manifest.csv  (multiprocess)
    make_pairs   tiles   -> matched LR/HR pairs + pairs.csv
    stitch_scene tiles   -> one reconstructed scene

Output layout of run_tiling (out_dir):
    manifest.csv                one row per tile, kept or skipped
    stats/<scene>.json          normalisation stats used for each scene
    tiles/<scene>/<scene>_r000_c000.npy
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tileprep.degrade import augment_pair, degrade
from tileprep.filters import FilterConfig, evaluate_tile
from tileprep.io import load_image
from tileprep.normalize import METHODS, NormStats, fit_stats
from tileprep.stitch import stitch_tiles
from tileprep.tiling import iter_tiles, make_grid, pad_to_grid

Logger = Callable[[str], None]

MANIFEST_COLUMNS = [
    "scene", "source", "tile_path", "row", "col", "y", "x", "tile_size", "overlap",
    "height", "width", "bands", "dtype", "norm_method", "stats_path",
    "kept", "reason", "nodata_frac", "cloud_frac", "std", "mean",
]  # fmt: skip


@dataclass(frozen=True)
class TileSettings:
    """Options for run_tiling. filters=None disables all skip rules."""

    tile_size: int = 256
    overlap: int = 32
    method: str = "percentile"
    low: float = 2.0
    high: float = 98.0
    nodata_value: float | None = 0
    filters: FilterConfig | None = field(default_factory=FilterConfig)
    save_skipped: bool = False

    def __post_init__(self) -> None:
        # Fail fast in the parent process instead of once per scene in workers.
        make_grid(1, 1, self.tile_size, self.overlap)
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")


def process_scene(
    path: str | Path, out_dir: str | Path, settings: TileSettings, stats: NormStats | None = None
) -> list[dict]:
    """Tile one scene and return its manifest rows.

    This runs inside worker processes, so it only takes picklable arguments
    and writes its own files; the parent process just collects the rows.
    """
    path, out_dir = Path(path), Path(out_dir)
    image = load_image(path)
    scene = path.stem

    if stats is None:
        s = settings
        stats = fit_stats(image, s.method, s.low, s.high, s.nodata_value)
    stats_rel = Path("stats") / f"{scene}.json"
    stats.save(out_dir / stats_rel)
    norm = stats.apply(image)

    height, width, bands = image.shape
    grid = make_grid(height, width, settings.tile_size, settings.overlap)
    raw_tiles = iter_tiles(pad_to_grid(image, grid), grid)
    norm_tiles = iter_tiles(pad_to_grid(norm, grid), grid)
    (out_dir / "tiles" / scene).mkdir(parents=True, exist_ok=True)

    rows = []
    for raw, tile in zip(raw_tiles, norm_tiles, strict=True):
        decision = evaluate_tile(raw.data, tile.data, settings.filters)
        tile_rel = ""
        if decision.keep or settings.save_skipped:
            tile_rel = f"tiles/{scene}/{scene}_r{tile.row:03d}_c{tile.col:03d}.npy"
            np.save(out_dir / tile_rel, tile.data)
        rows.append({
            "scene": scene, "source": str(path), "tile_path": tile_rel,
            "row": tile.row, "col": tile.col, "y": tile.y, "x": tile.x,
            "tile_size": grid.tile_size, "overlap": grid.overlap,
            "height": height, "width": width, "bands": bands, "dtype": str(image.dtype),
            "norm_method": stats.method, "stats_path": stats_rel.as_posix(),
            "kept": decision.keep, "reason": decision.reason,
            "nodata_frac": round(decision.nodata_frac, 4),
            "cloud_frac": round(decision.cloud_frac, 4),
            "std": round(decision.std, 4), "mean": round(decision.mean, 4),
        })  # fmt: skip
    return rows


def _summary(index: int, total: int, path: Path, rows: list[dict]) -> str:
    kept = sum(r["kept"] for r in rows)
    skipped = len(rows) - kept
    return f"[{index}/{total}] {path.name}: {len(rows)} tiles, {kept} kept, {skipped} skipped"


def run_tiling(
    paths: Sequence[str | Path],
    out_dir: str | Path,
    settings: TileSettings,
    *,
    stats: NormStats | None = None,
    workers: int = 1,
    log: Logger = print,
) -> pd.DataFrame:
    """Tile many scenes (in parallel when workers > 1) and write manifest.csv.

    Each scene is an independent job, so scenes are spread across processes.
    Processes (not threads) let the Python-level per-tile loop run truly in
    parallel despite the GIL. A scene that fails is logged and skipped so one
    corrupt file does not abort a long batch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in paths]
    rows: list[dict] = []

    if workers <= 1 or len(paths) == 1:
        for i, path in enumerate(paths, 1):
            try:
                scene_rows = process_scene(path, out_dir, settings, stats)
            except Exception as exc:  # keep going; report at the end
                log(f"[{i}/{len(paths)}] {path.name}: FAILED ({exc})")
                continue
            rows.extend(scene_rows)
            log(_summary(i, len(paths), path, scene_rows))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_scene, p, out_dir, settings, stats): p for p in paths}
            for i, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    scene_rows = future.result()
                except Exception as exc:
                    log(f"[{i}/{len(paths)}] {path.name}: FAILED ({exc})")
                    continue
                rows.extend(scene_rows)
                log(_summary(i, len(paths), path, scene_rows))

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(["scene", "row", "col"], ignore_index=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def load_manifest(tile_dir: str | Path) -> pd.DataFrame:
    """Read manifest.csv from a run_tiling output folder."""
    manifest = pd.read_csv(Path(tile_dir) / "manifest.csv", keep_default_na=False)
    manifest["kept"] = manifest["kept"].astype(str) == "True"
    return manifest


def make_pairs(
    tile_dir: str | Path,
    out_dir: str | Path,
    scale: int = 4,
    blur_sigma: float = 1.0,
    noise_std: float = 0.0,
    augment: int = 0,
    seed: int = 0,
    log: Logger = print,
) -> pd.DataFrame:
    """Create LR/HR pairs from the kept tiles of a run_tiling output.

    Writes out_dir/hr/*.npy, out_dir/lr/*.npy and out_dir/pairs.csv. With
    augment=N, N extra randomly flipped/rotated copies are written per tile.
    Each tile gets its own seeded RNG, so results do not depend on order.
    """
    tile_dir, out_dir = Path(tile_dir), Path(out_dir)
    manifest = load_manifest(tile_dir)
    kept = manifest[manifest["kept"] & (manifest["tile_path"] != "")]
    (out_dir / "hr").mkdir(parents=True, exist_ok=True)
    (out_dir / "lr").mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in enumerate(kept.itertuples(index=False)):
        hr = np.load(tile_dir / row.tile_path)
        rng = np.random.default_rng([seed, i])
        clip = row.norm_method != "zscore"
        lr = degrade(hr, scale, blur_sigma, noise_std, rng, clip=clip)

        variants = [(lr, hr, 0)] + [augment_pair(lr, hr, rng) for _ in range(augment)]
        stem = Path(row.tile_path).stem
        for j, (lr_v, hr_v, k) in enumerate(variants):
            name = stem if j == 0 else f"{stem}_aug{j}"
            np.save(out_dir / "hr" / f"{name}.npy", hr_v)
            np.save(out_dir / "lr" / f"{name}.npy", lr_v)
            records.append({
                "pair_id": name, "scene": row.scene, "tile_path": row.tile_path,
                "hr_path": f"hr/{name}.npy", "lr_path": f"lr/{name}.npy",
                "scale": scale, "blur_sigma": blur_sigma, "noise_std": noise_std,
                "dihedral_k": k,
            })  # fmt: skip

    pairs = pd.DataFrame(records)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    log(f"wrote {len(pairs)} pairs from {len(kept)} kept tiles (scale x{scale}) to {out_dir}")
    return pairs


def stitch_scene(
    tile_dir: str | Path,
    scene: str,
    tiles_from: str | Path | None = None,
    mode: str = "feather",
    denormalize: bool = False,
) -> np.ndarray:
    """Rebuild one scene from its tiles using the offsets in manifest.csv.

    Args:
        tile_dir: a run_tiling output folder.
        scene: scene name (file stem) to rebuild.
        tiles_from: optional folder of processed tiles with the same file
            names (e.g. model outputs). Tiles may be upscaled by an integer
            factor; offsets are scaled to match.
        mode: blending mode, "feather" or "mean".
        denormalize: map back to the scene's original units and dtype.

    Areas whose tiles were skipped (not saved) come back as 0.
    """
    tile_dir = Path(tile_dir)
    manifest = load_manifest(tile_dir)
    rows = manifest[(manifest["scene"] == scene) & (manifest["tile_path"] != "")]
    if rows.empty:
        raise ValueError(f"no saved tiles for scene {scene!r} in {tile_dir / 'manifest.csv'}")

    tiles = []
    for row in rows.itertuples(index=False):
        if tiles_from is None:
            path = tile_dir / row.tile_path
        else:
            path = Path(tiles_from) / Path(row.tile_path).name
        tiles.append((row.y, row.x, np.load(path)))

    first = rows.iloc[0]
    tile_size = int(first["tile_size"])
    scale, remainder = divmod(tiles[0][2].shape[0], tile_size)
    if remainder or scale < 1:
        raise ValueError(f"tile side {tiles[0][2].shape[0]} is not a multiple of {tile_size}")

    image, covered = stitch_tiles(
        [(y * scale, x * scale, t) for y, x, t in tiles],
        int(first["height"]) * scale,
        int(first["width"]) * scale,
        overlap=int(first["overlap"]) * scale,
        mode=mode,
        return_coverage=True,
    )
    if denormalize:
        image = NormStats.load(tile_dir / first["stats_path"]).invert(image)
        image[~covered] = 0  # holes left by skipped tiles become nodata, not p_low
    return image
