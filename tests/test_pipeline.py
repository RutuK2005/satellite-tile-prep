import numpy as np
import pytest

from tileprep.filters import FilterConfig
from tileprep.io import load_image, save_image
from tileprep.pipeline import (
    MANIFEST_COLUMNS,
    TileSettings,
    load_manifest,
    make_pairs,
    run_tiling,
    stitch_scene,
)


def make_scene(seed: int = 0, height: int = 90, width: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = rng.integers(500, 6000, size=(height, width, 3), dtype=np.uint16)
    scene[:, :30] = 0  # nodata strip on the left
    return scene


@pytest.fixture
def scene_dir(tmp_path):
    folder = tmp_path / "scenes"
    save_image(folder / "alpha.tif", make_scene(0))
    save_image(folder / "beta.tif", make_scene(1))
    return folder


def test_run_tiling_writes_tiles_and_manifest(scene_dir, tmp_path):
    out = tmp_path / "out"
    settings = TileSettings(tile_size=32, overlap=8)
    manifest = run_tiling(sorted(scene_dir.iterdir()), out, settings, log=lambda _: None)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert set(manifest["scene"]) == {"alpha", "beta"}
    assert (out / "manifest.csv").exists()
    assert (out / "stats" / "alpha.json").exists()
    # tiles fully inside the nodata strip are skipped, the rest are kept
    assert set(manifest["reason"]) == {"kept", "nodata"}
    kept = manifest[manifest["kept"]]
    for rel in kept["tile_path"]:
        tile = np.load(out / rel)
        assert tile.shape == (32, 32, 3)
        assert tile.dtype == np.float32
    assert (manifest.loc[~manifest["kept"], "tile_path"] == "").all()


def test_parallel_matches_sequential(scene_dir, tmp_path):
    settings = TileSettings(tile_size=32, overlap=8)
    paths = sorted(scene_dir.iterdir())
    seq = run_tiling(paths, tmp_path / "seq", settings, log=lambda _: None)
    par = run_tiling(paths, tmp_path / "par", settings, workers=2, log=lambda _: None)
    cols = ["scene", "row", "col", "kept", "reason", "std"]
    assert seq[cols].equals(par[cols])


def test_failed_scene_is_logged_not_fatal(scene_dir, tmp_path):
    (scene_dir / "broken.tif").write_bytes(b"not a tiff")
    messages = []
    settings = TileSettings(tile_size=32, overlap=8)
    manifest = run_tiling(
        sorted(scene_dir.iterdir()), tmp_path / "out", settings, log=messages.append
    )
    assert any("broken.tif: FAILED" in m for m in messages)
    assert set(manifest["scene"]) == {"alpha", "beta"}


def test_tile_then_stitch_restores_original(tmp_path):
    scene = make_scene(2, height=75, width=101)
    save_image(tmp_path / "gamma.tif", scene)
    settings = TileSettings(
        tile_size=32, overlap=8, method="minmax", nodata_value=None, filters=None
    )
    run_tiling([tmp_path / "gamma.tif"], tmp_path / "out", settings, log=lambda _: None)

    restored = stitch_scene(tmp_path / "out", "gamma", denormalize=True)
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, load_image(tmp_path / "gamma.tif"))


def test_skipped_tiles_become_nodata_after_denormalize(scene_dir, tmp_path):
    settings = TileSettings(tile_size=30, overlap=0)  # 90x120 -> 3x4 tiles, left column nodata
    run_tiling([scene_dir / "alpha.tif"], tmp_path / "out", settings, log=lambda _: None)
    restored = stitch_scene(tmp_path / "out", "alpha", denormalize=True)
    assert restored[:, :30].max() == 0
    assert restored[:, 30:].min() > 0


def test_stitch_processed_upscaled_tiles(tmp_path):
    save_image(tmp_path / "delta.tif", make_scene(3, height=64, width=80))
    settings = TileSettings(tile_size=32, overlap=8, filters=None)
    run_tiling([tmp_path / "delta.tif"], tmp_path / "out", settings, log=lambda _: None)

    # Pretend a 2x SR model processed every tile (nearest-neighbour upscale).
    processed = tmp_path / "processed"
    processed.mkdir()
    for rel in load_manifest(tmp_path / "out")["tile_path"]:
        tile = np.load(tmp_path / "out" / rel)
        np.save(processed / rel.split("/")[-1], tile.repeat(2, 0).repeat(2, 1))

    stitched = stitch_scene(tmp_path / "out", "delta", tiles_from=processed)
    direct = stitch_scene(tmp_path / "out", "delta")
    assert stitched.shape == (128, 160, 3)
    np.testing.assert_allclose(stitched, direct.repeat(2, 0).repeat(2, 1), atol=1e-6)


def test_make_pairs_with_augmentation(scene_dir, tmp_path):
    out = tmp_path / "out"
    filters = FilterConfig(nodata_value=0)
    settings = TileSettings(tile_size=32, overlap=8, filters=filters)
    run_tiling([scene_dir / "alpha.tif"], out, settings, log=lambda _: None)
    pairs = make_pairs(out, tmp_path / "pairs", scale=4, augment=2, seed=1, log=lambda _: None)

    n_kept = int(load_manifest(out)["kept"].sum())
    assert len(pairs) == 3 * n_kept
    row = pairs.iloc[1]
    assert np.load(tmp_path / "pairs" / row["hr_path"]).shape == (32, 32, 3)
    assert np.load(tmp_path / "pairs" / row["lr_path"]).shape == (8, 8, 3)
    assert (tmp_path / "pairs" / "pairs.csv").exists()


def test_stitch_unknown_scene_raises(scene_dir, tmp_path):
    settings = TileSettings(tile_size=32, overlap=8)
    run_tiling([scene_dir / "alpha.tif"], tmp_path / "out", settings, log=lambda _: None)
    with pytest.raises(ValueError, match="no saved tiles"):
        stitch_scene(tmp_path / "out", "nope")


def test_invalid_settings_fail_fast():
    with pytest.raises(ValueError):
        TileSettings(tile_size=32, overlap=32)
    with pytest.raises(ValueError):
        TileSettings(method="histogram")
