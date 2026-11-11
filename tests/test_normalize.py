import numpy as np
import pytest

from tileprep.normalize import NormStats, fit_stats


@pytest.fixture
def scene() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.normal(3000, 800, size=(64, 64, 3)).clip(1, 10_000).astype(np.uint16)
    image[:5] = 0  # nodata strip
    return image


def test_percentile_output_is_unit_range(scene):
    stats = fit_stats(scene, method="percentile", low=2, high=98)
    norm = stats.apply(scene)
    assert norm.dtype == np.float32
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    # ~2% of valid pixels per band sit at each clip limit
    valid = norm[5:]
    assert 0.01 < (valid == 1.0).mean() < 0.04


def test_nodata_pixels_are_excluded_from_stats(scene):
    stats = fit_stats(scene, method="minmax", nodata_value=0)
    assert min(stats.offset) > 0
    stats_with_zeros = fit_stats(scene, method="minmax", nodata_value=None)
    assert stats_with_zeros.offset == [0.0, 0.0, 0.0]


def test_minmax_round_trip_is_exact(scene):
    stats = fit_stats(scene, method="minmax", nodata_value=None)
    restored = stats.invert(stats.apply(scene))
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, scene)


def test_zscore_is_centred_and_not_clipped(scene):
    valid = scene[5:]
    stats = fit_stats(valid, method="zscore")
    norm = stats.apply(valid)
    assert not stats.clip
    np.testing.assert_allclose(norm.mean(axis=(0, 1)), 0, atol=1e-3)
    np.testing.assert_allclose(norm.std(axis=(0, 1)), 1, atol=1e-3)
    assert norm.min() < 0


def test_stats_json_round_trip(tmp_path, scene):
    stats = fit_stats(scene)
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = NormStats.load(path)
    assert loaded == stats
    np.testing.assert_array_equal(loaded.apply(scene), stats.apply(scene))


def test_fit_on_multiple_images_and_constant_band():
    a = np.full((8, 8, 2), 100, np.uint16)
    b = np.full((8, 8, 2), 300, np.uint16)
    b[..., 1] = 100  # band 1 is constant across both images
    stats = fit_stats([a, b], method="minmax")
    assert stats.offset == [100.0, 100.0]
    assert stats.scale == [200.0, 1.0]


def test_band_mismatch_and_bad_method_raise(scene):
    stats = fit_stats(scene)
    with pytest.raises(ValueError):
        stats.apply(scene[..., :2])
    with pytest.raises(ValueError):
        fit_stats(scene, method="histogram")
