"""Per-band normalisation with statistics that can be saved and reused.

Every method is expressed as the same affine transform per band:

    normalised = (raw - offset) / scale        (optionally clipped to [0, 1])

    percentile  offset = p_low,  scale = p_high - p_low,  clipped
    minmax      offset = min,    scale = max - min,       clipped
    zscore      offset = mean,   scale = std,             not clipped

Storing just (offset, scale) per band keeps the JSON small, makes the inverse
trivial, and guarantees that inference uses exactly the training transform.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tileprep.filters import nodata_mask

METHODS = ("percentile", "minmax", "zscore")


@dataclass
class NormStats:
    """Fitted per-band normalisation parameters."""

    method: str
    offset: list[float]
    scale: list[float]
    clip: bool
    source_dtype: str
    params: dict = field(default_factory=dict)

    @property
    def bands(self) -> int:
        return len(self.offset)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Normalise an (H, W, C) image to float32."""
        self._check_bands(image)
        out = (image.astype(np.float32) - np.float32(self.offset)) / np.float32(self.scale)
        if self.clip:
            np.clip(out, 0.0, 1.0, out=out)
        return out

    def invert(self, image: np.ndarray, to_source_dtype: bool = True) -> np.ndarray:
        """Map normalised values back to original units.

        Values clipped during apply() cannot be recovered: anything outside the
        percentile range comes back as p_low or p_high.
        """
        self._check_bands(image)
        out = image.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)
        if not to_source_dtype:
            return out
        dtype = np.dtype(self.source_dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        return out.astype(dtype)

    def save(self, path: str | Path) -> None:
        """Write the stats to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        """Read stats previously written by save()."""
        return cls(**json.loads(Path(path).read_text()))

    def _check_bands(self, image: np.ndarray) -> None:
        if image.shape[-1] != self.bands:
            raise ValueError(f"image has {image.shape[-1]} bands, stats expect {self.bands}")


def sample_pixels(
    image: np.ndarray,
    nodata_value: float | None = 0,
    max_samples: int = 500_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return up to max_samples valid pixels from an image as an (N, C) array.

    Percentiles over a random sample are much cheaper than over a full
    10k x 10k scene and are accurate enough for choosing clip limits.
    """
    rng = rng or np.random.default_rng(0)
    valid = image[~nodata_mask(image, nodata_value)]
    if len(valid) > max_samples:
        valid = valid[rng.choice(len(valid), size=max_samples, replace=False)]
    return valid


def fit_stats(
    images: np.ndarray | Iterable[np.ndarray],
    method: str = "percentile",
    low: float = 2.0,
    high: float = 98.0,
    nodata_value: float | None = 0,
    max_samples_per_image: int = 500_000,
) -> NormStats:
    """Fit normalisation stats on one image or on an iterable of images.

    Passing several scenes gives dataset-wide stats, so the same reflectance
    maps to the same normalised value in every scene.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if isinstance(images, np.ndarray):
        images = [images]

    rng = np.random.default_rng(0)
    samples, dtype = [], None
    for image in images:
        dtype = dtype or image.dtype
        samples.append(sample_pixels(image, nodata_value, max_samples_per_image, rng))
    if not samples:
        raise ValueError("no images given")
    if len({s.shape[1] for s in samples}) > 1:
        raise ValueError("all images must have the same number of bands")
    pixels = np.concatenate(samples).astype(np.float64)
    if len(pixels) == 0:
        raise ValueError("no valid (non-nodata) pixels to fit stats on")

    params: dict = {}
    if method == "percentile":
        offset, upper = np.percentile(pixels, [low, high], axis=0)
        scale = upper - offset
        params = {"low": low, "high": high}
    elif method == "minmax":
        offset = pixels.min(axis=0)
        scale = pixels.max(axis=0) - offset
    else:
        offset = pixels.mean(axis=0)
        scale = pixels.std(axis=0)

    # A constant band would divide by zero; leave it unscaled instead.
    scale = np.where(scale > 0, scale, 1.0)
    return NormStats(
        method=method,
        offset=[float(v) for v in offset],
        scale=[float(v) for v in scale],
        clip=method != "zscore",
        source_dtype=str(dtype),
        params=params,
    )
