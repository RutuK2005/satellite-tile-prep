"""Create low-resolution (LR) inputs from high-resolution (HR) tiles.

Super-resolution models learn LR -> HR. Real paired LR/HR satellite imagery is
rare, so the usual approach is to synthesise LR from HR with a known
degradation: blur (approximating the sensor's point spread function), bicubic
downsampling, and optional sensor noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def _restore_band_axis(image: np.ndarray) -> np.ndarray:
    # OpenCV drops the channel axis of single-band images; put it back.
    return image[:, :, np.newaxis] if image.ndim == 2 else image


def degrade(
    hr: np.ndarray,
    scale: int,
    blur_sigma: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Blur, bicubic-downscale and optionally add Gaussian noise to an HR tile.

    Args:
        hr: float32 tile of shape (H, W, C); H and W must be divisible by scale.
        scale: downscale factor, typically 2 or 4.
        blur_sigma: Gaussian blur sigma in HR pixels (0 disables blurring).
        noise_std: standard deviation of additive noise, in normalised units.
        rng: random generator for reproducible noise.
        clip: clip the result to [0, 1] (disable for z-score data).
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    height, width = hr.shape[:2]
    if height % scale or width % scale:
        raise ValueError(f"tile size {height}x{width} is not divisible by scale {scale}")

    image = hr.astype(np.float32)
    if blur_sigma > 0:
        # Blurring first acts as an anti-aliasing filter: plain bicubic
        # downscaling would alias fine detail such as road edges.
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma, borderType=cv2.BORDER_REFLECT)
    lr = cv2.resize(image, (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)
    lr = _restore_band_axis(lr)

    if noise_std > 0:
        rng = rng or np.random.default_rng()
        lr = lr + rng.normal(0.0, noise_std, size=lr.shape).astype(np.float32)
    if clip:
        lr = np.clip(lr, 0.0, 1.0)
    return lr


def bicubic_upsample(lr: np.ndarray, scale: int) -> np.ndarray:
    """Upsample an LR tile with bicubic interpolation (the usual SR baseline)."""
    height, width = lr.shape[:2]
    up = cv2.resize(lr, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    return _restore_band_axis(up)


def dihedral(image: np.ndarray, k: int) -> np.ndarray:
    """Apply one of the 8 square symmetries: k % 4 quarter-turns, then a flip if k >= 4.

    Satellite images have no canonical "up", so all 8 orientations are
    equally realistic training examples.
    """
    out = np.rot90(image, k % 4, axes=(0, 1))
    if k % 8 >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def augment_pair(
    lr: np.ndarray, hr: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the same random dihedral transform to an LR/HR pair.

    Using one transform for both keeps pixels aligned; transforming them
    independently would teach the model a nonsense mapping.
    """
    k = int(rng.integers(1, 8))  # skip k=0, the identity
    return dihedral(lr, k), dihedral(hr, k), k
