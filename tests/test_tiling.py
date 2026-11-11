import numpy as np
import pytest

from tileprep.tiling import iter_tiles, make_grid, pad_to_grid


def test_grid_exact_fit_needs_no_padding():
    grid = make_grid(256, 384, tile_size=128, overlap=0)
    assert (grid.n_rows, grid.n_cols) == (2, 3)
    assert grid.padded_shape == (256, 384)
    assert len(grid) == 6


def test_grid_with_overlap_and_remainder():
    # stride 48: tiles start at 0, 48, 96 -> last tile ends at 160 >= 150
    grid = make_grid(150, 64, tile_size=64, overlap=16)
    assert grid.stride == 48
    assert grid.n_rows == 3
    assert grid.n_cols == 1
    assert grid.padded_shape == (160, 64)


def test_image_smaller_than_tile_gives_single_tile():
    grid = make_grid(40, 50, tile_size=64, overlap=8)
    assert (grid.n_rows, grid.n_cols) == (1, 1)
    assert grid.padded_shape == (64, 64)


@pytest.mark.parametrize("overlap", [-1, 64, 100])
def test_invalid_overlap_raises(overlap):
    with pytest.raises(ValueError):
        make_grid(100, 100, tile_size=64, overlap=overlap)


def test_padding_is_reflective_and_keeps_original_pixels():
    image = np.arange(11 * 8 * 2, dtype=np.uint16).reshape(11, 8, 2)
    grid = make_grid(11, 8, tile_size=4, overlap=1)
    padded = pad_to_grid(image, grid)

    assert padded.shape[:2] == grid.padded_shape == (13, 10)
    np.testing.assert_array_equal(padded[:11, :8], image)
    # reflect mode mirrors around the last row without repeating it
    np.testing.assert_array_equal(padded[11, :8], image[9])


def test_tiles_match_source_windows():
    rng = np.random.default_rng(1)
    image = rng.random((70, 90, 3)).astype(np.float32)
    grid = make_grid(70, 90, tile_size=32, overlap=8)
    padded = pad_to_grid(image, grid)

    tiles = list(iter_tiles(padded, grid))
    assert len(tiles) == len(grid)
    for tile in tiles:
        assert tile.data.shape == (32, 32, 3)
        np.testing.assert_array_equal(tile.data, padded[tile.y : tile.y + 32, tile.x : tile.x + 32])
    # neighbouring tiles share exactly `overlap` columns
    first, second = tiles[0], tiles[1]
    np.testing.assert_array_equal(first.data[:, -8:], second.data[:, :8])


def test_iter_tiles_rejects_unpadded_image():
    grid = make_grid(70, 90, tile_size=32, overlap=8)
    with pytest.raises(ValueError):
        list(iter_tiles(np.zeros((70, 90, 1)), grid))
