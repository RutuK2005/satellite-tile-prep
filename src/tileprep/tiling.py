"""Split a large scene into fixed-size, overlapping tiles.

Terminology:
    tile_size  side length of each square tile, in pixels
    overlap    pixels shared by neighbouring tiles
    stride     tile_size - overlap, i.e. how far the window moves each step

If the scene size is not an exact multiple of the stride, the bottom/right
edges are reflect-padded so that the last row/column of tiles is complete.
Padding only ever happens on the bottom/right, so tile offsets in the padded
image are also valid offsets in the original image.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileGrid:
    """Layout of tiles over an image of size (height, width)."""

    height: int
    width: int
    tile_size: int
    overlap: int
    n_rows: int
    n_cols: int

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    @property
    def padded_shape(self) -> tuple[int, int]:
        """(height, width) after padding so every tile fits completely."""
        return (
            (self.n_rows - 1) * self.stride + self.tile_size,
            (self.n_cols - 1) * self.stride + self.tile_size,
        )

    def positions(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield (row, col, y, x) for every tile, row by row."""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield row, col, row * self.stride, col * self.stride

    def __len__(self) -> int:
        return self.n_rows * self.n_cols


@dataclass
class Tile:
    """One tile cut from a scene, with its grid position and pixel offset."""

    row: int
    col: int
    y: int
    x: int
    data: np.ndarray


def _n_steps(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return 1
    return math.ceil((length - tile_size) / stride) + 1


def make_grid(height: int, width: int, tile_size: int, overlap: int = 0) -> TileGrid:
    """Compute the tile layout for an image.

    Raises ValueError if tile_size is not positive or overlap is not in
    [0, tile_size), because the stride would then be zero or negative.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap
    return TileGrid(
        height=height,
        width=width,
        tile_size=tile_size,
        overlap=overlap,
        n_rows=_n_steps(height, tile_size, stride),
        n_cols=_n_steps(width, tile_size, stride),
    )


def pad_to_grid(image: np.ndarray, grid: TileGrid) -> np.ndarray:
    """Reflect-pad an (H, W, C) image on the bottom/right to grid.padded_shape.

    Reflection is used instead of zeros so that edge tiles do not contain an
    artificial black border, which would look like nodata and confuse a model.
    """
    pad_h = grid.padded_shape[0] - image.shape[0]
    pad_w = grid.padded_shape[1] - image.shape[1]
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def iter_tiles(image: np.ndarray, grid: TileGrid) -> Iterator[Tile]:
    """Yield tiles from an image that has already been padded to the grid.

    Tiles are views into the padded array (no copy), so callers that keep
    them around should copy them first.
    """
    expected = grid.padded_shape
    if image.shape[:2] != expected:
        raise ValueError(f"image shape {image.shape[:2]} != padded grid shape {expected}")
    size = grid.tile_size
    for row, col, y, x in grid.positions():
        yield Tile(row, col, y, x, image[y : y + size, x : x + size])
