import numpy as np

from tileprep.filters import FilterConfig, evaluate_tile

CONFIG = FilterConfig(nodata_value=0, max_nodata_frac=0.2, max_cloud_frac=0.3, min_std=0.02)


def textured_tile(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    norm = rng.uniform(0.1, 0.7, size=(32, 32, 3)).astype(np.float32)
    raw = (norm * 10_000).astype(np.uint16)
    return raw, norm


def test_textured_tile_is_kept():
    raw, norm = textured_tile()
    decision = evaluate_tile(raw, norm, CONFIG)
    assert decision.keep
    assert decision.reason == "kept"
    assert decision.nodata_frac == 0.0


def test_mostly_nodata_tile_is_skipped():
    raw, norm = textured_tile()
    raw[:16] = 0  # half the tile is outside the swath
    decision = evaluate_tile(raw, norm, CONFIG)
    assert not decision.keep
    assert decision.reason == "nodata"
    assert decision.nodata_frac == 0.5


def test_fully_nodata_tile_is_skipped():
    raw = np.zeros((16, 16, 2), np.uint16)
    decision = evaluate_tile(raw, raw.astype(np.float32), CONFIG)
    assert decision.reason == "nodata"
    assert decision.nodata_frac == 1.0


def test_bright_tile_is_flagged_as_cloud():
    raw, norm = textured_tile()
    norm[:, :20] = 0.97  # bright in every band
    decision = evaluate_tile(raw, norm, CONFIG)
    assert decision.reason == "cloud"
    assert decision.cloud_frac > 0.3


def test_flat_tile_is_low_variance():
    raw = np.full((32, 32, 3), 1500, np.uint16)
    norm = np.full((32, 32, 3), 0.4, np.float32)
    decision = evaluate_tile(raw, norm, CONFIG)
    assert decision.reason == "low_variance"


def test_no_config_keeps_everything_but_reports_stats():
    raw = np.zeros((16, 16, 1), np.uint16)
    decision = evaluate_tile(raw, raw.astype(np.float32), None)
    assert decision.keep
    assert decision.nodata_frac == 1.0
