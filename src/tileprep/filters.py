"""Rules for skipping tiles that carry little useful training signal.

Three checks, applied in order:
    1. nodata   - too many pixels equal the nodata value in every band
                  (e.g. the black wedge outside a satellite swath)
    2. cloud    - too many pixels are bright in every band (white, saturated)
    3. variance - the tile is almost flat (open water, uniform haze)

The nodata check uses the raw tile. The other two use the normalised tile,
so their thresholds are in normalised units (0-1 for percentile/min-max).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the skip rules. Set nodata_value to None to disable nodata checks."""

    nodata_value: float | None = 0
    max_nodata_frac: float = 0.2
    cloud_threshold: float = 0.9
    max_cloud_frac: float = 0.3
    min_std: float = 0.02


@dataclass(frozen=True)
class TileDecision:
    """Outcome of evaluating one tile, plus the statistics behind it."""

    keep: bool
    reason: str
    nodata_frac: float
    cloud_frac: float
    std: float
    mean: float


def nodata_mask(raw: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """Boolean (H, W) mask of pixels equal to nodata_value in every band."""
    if nodata_value is None:
        return np.zeros(raw.shape[:2], dtype=bool)
    return np.all(raw == nodata_value, axis=-1)


def evaluate_tile(
    raw: np.ndarray, norm: np.ndarray, config: FilterConfig | None = None
) -> TileDecision:
    """Compute tile statistics and decide whether to keep the tile.

    Args:
        raw: tile in original units, shape (H, W, C).
        norm: the same tile after normalisation, shape (H, W, C).
        config: thresholds; None means "keep everything" (stats still computed).
    """
    nodata_value = config.nodata_value if config else 0
    invalid = nodata_mask(raw, nodata_value)
    nodata_frac = float(invalid.mean())

    valid = norm[~invalid]  # (N, C) pixels that carry data
    if valid.size == 0:
        keep = config is None
        return TileDecision(keep, "kept" if keep else "nodata", 1.0, 0.0, 0.0, 0.0)

    threshold = config.cloud_threshold if config else 0.9
    cloud_frac = float(np.all(valid >= threshold, axis=-1).mean())
    # Mean of per-band standard deviations: robust to one dull band.
    std = float(valid.std(axis=0).mean())
    mean = float(valid.mean())

    reason = "kept"
    if config is not None:
        if nodata_frac > config.max_nodata_frac:
            reason = "nodata"
        elif cloud_frac > config.max_cloud_frac:
            reason = "cloud"
        elif std < config.min_std:
            reason = "low_variance"
    return TileDecision(reason == "kept", reason, nodata_frac, cloud_frac, std, mean)
