import numpy as np
import pytest

from tileprep.io import list_images, load_image, save_image


@pytest.mark.parametrize("bands", [1, 3, 4])
def test_uint16_tiff_round_trip(tmp_path, bands):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 10_000, size=(20, 30, bands), dtype=np.uint16)
    path = tmp_path / "scene.tif"
    save_image(path, image)

    loaded = load_image(path)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (20, 30, bands)
    np.testing.assert_array_equal(loaded, image)


def test_list_images_filters_extensions(tmp_path):
    save_image(tmp_path / "b.png", np.zeros((4, 4, 3), np.uint8))
    save_image(tmp_path / "a.tif", np.zeros((4, 4, 1), np.uint16))
    (tmp_path / "notes.txt").write_text("not an image")

    assert [p.name for p in list_images(tmp_path)] == ["a.tif", "b.png"]
    assert list_images(tmp_path / "a.tif") == [tmp_path / "a.tif"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.tif")
