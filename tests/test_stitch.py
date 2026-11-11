import numpy as np
import pytest

from tileprep.stitch import blend_weights, stitch_tiles
from tileprep.tiling import iter_tiles, make_grid, pad_to_grid


def tile_image(image: np.ndarray, tile_size: int, overlap: int):
    grid = make_grid(image.shape[0], image.shape[1], tile_size, overlap)
    padded = pad_to_grid(image, grid)
    return [(t.y, t.x, t.data.copy()) for t in iter_tiles(padded, grid)]


@pytest.mark.parametrize("mode", ["feather", "mean"])
@pytest.mark.parametrize(("height", "width", "overlap"), [(100, 130, 16), (64, 64, 0), (37, 90, 8)])
def test_tile_then_stitch_round_trips(mode, height, width, overlap):
    image = np.random.default_rng(0).random((height, width, 3)).astype(np.float32)
    tiles = tile_image(image, tile_size=32, overlap=overlap)

    stitched = stitch_tiles(tiles, height, width, overlap, mode)
    assert stitched.shape == image.shape
    np.testing.assert_allclose(stitched, image, atol=1e-6)


def test_stitch_upscaled_tiles():
    """Tiles that a 2x SR model upscaled can be stitched by scaling offsets."""
    image = np.random.default_rng(1).random((50, 70, 2)).astype(np.float32)
    tiles = tile_image(image, tile_size=32, overlap=8)
    upscaled = [(2 * y, 2 * x, t.repeat(2, axis=0).repeat(2, axis=1)) for y, x, t in tiles]

    stitched = stitch_tiles(upscaled, 100, 140, overlap=16)
    expected = image.repeat(2, axis=0).repeat(2, axis=1)
    np.testing.assert_allclose(stitched, expected, atol=1e-6)


def test_feather_blending_smooths_seams():
    # Two constant tiles with different values: feathering should create a
    # gradual transition across the overlap instead of a hard step.
    left = np.zeros((16, 16, 1), np.float32)
    right = np.ones((16, 16, 1), np.float32)
    tiles = [(0, 0, left), (0, 8, right)]

    feather = stitch_tiles(tiles, 16, 24, overlap=8, mode="feather")[0, :, 0]
    mean = stitch_tiles(tiles, 16, 24, overlap=8, mode="mean")[0, :, 0]
    assert np.all(np.diff(feather) >= 0)  # monotonic ramp from 0 to 1
    assert len(np.unique(np.round(feather[8:16], 4))) > 2
    np.testing.assert_allclose(mean[8:16], 0.5)


def test_missing_tiles_leave_zeros():
    tile = np.ones((8, 8, 1), np.float32)
    stitched, covered = stitch_tiles([(0, 0, tile)], 8, 16, overlap=0, return_coverage=True)
    assert stitched[:, :8].min() == 1.0
    assert stitched[:, 8:].max() == 0.0
    assert covered[:, :8].all()
    assert not covered[:, 8:].any()


def test_blend_weights_shape_and_bad_mode():
    weights = blend_weights(32, 8)
    assert weights.shape == (32, 32)
    assert weights.min() > 0
    assert weights[16, 16] == 1.0
    with pytest.raises(ValueError):
        blend_weights(32, 8, mode="median")
