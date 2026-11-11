"""Reassemble a full image from overlapping tiles.

Where tiles overlap, each output pixel is a weighted average of all tiles that
cover it:

    output = sum(weight_i * tile_i) / sum(weight_i)

With "mean" blending every pixel of a tile has weight 1. With "feather"
blending the weight ramps down linearly across the overlap band towards the
tile edges. Models are least reliable near tile borders (less context), so
feathering hides seams that a plain average would leave visible.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

BLEND_MODES = ("feather", "mean")


def blend_weights(size: int, overlap: int, mode: str = "feather") -> np.ndarray:
    """Return a (size, size) weight window for one tile."""
    if mode not in BLEND_MODES:
        raise ValueError(f"mode must be one of {BLEND_MODES}, got {mode!r}")
    if mode == "mean" or overlap == 0:
        return np.ones((size, size), dtype=np.float32)

    idx = np.arange(size)
    distance_to_edge = np.minimum(idx, size - 1 - idx)
    # Linear ramp over the overlap band, flat (1.0) in the middle. The +1
    # keeps edge weights above zero so every pixel stays covered.
    ramp = np.minimum(1.0, (distance_to_edge + 1) / (overlap + 1)).astype(np.float32)
    return np.outer(ramp, ramp)


def stitch_tiles(
    tiles: Iterable[tuple[int, int, np.ndarray]],
    out_height: int,
    out_width: int,
    overlap: int,
    mode: str = "feather",
    return_coverage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Blend tiles back into one (out_height, out_width, C) float32 image.

    Args:
        tiles: (y, x, tile) triples; offsets are in output pixels and tiles
            are square, shape (S, S, C).
        out_height, out_width: size to crop the result to (removes padding).
        overlap: overlap between neighbouring tiles, in output pixels.
        mode: "feather" or "mean".
        return_coverage: also return a boolean (H, W) mask of covered pixels.

    Pixels not covered by any tile (e.g. skipped tiles) are returned as 0.
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles to stitch")
    size, bands = tiles[0][2].shape[0], tiles[0][2].shape[2]
    canvas_h = max(out_height, max(y for y, _, _ in tiles) + size)
    canvas_w = max(out_width, max(x for _, x, _ in tiles) + size)

    accum = np.zeros((canvas_h, canvas_w, bands), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
    window = blend_weights(size, overlap, mode)[:, :, np.newaxis]

    for y, x, tile in tiles:
        if tile.shape != (size, size, bands):
            raise ValueError(f"tile at ({y}, {x}) has shape {tile.shape}, expected {size}x{size}")
        accum[y : y + size, x : x + size] += tile * window
        weight_sum[y : y + size, x : x + size] += window

    covered = weight_sum > 0
    out = np.divide(accum, weight_sum, out=np.zeros_like(accum), where=covered)
    out = out[:out_height, :out_width]
    if return_coverage:
        return out, covered[:out_height, :out_width, 0]
    return out
