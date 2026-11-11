import numpy as np
import pytest

from tileprep.degrade import augment_pair, bicubic_upsample, degrade, dihedral


@pytest.fixture
def hr() -> np.ndarray:
    return np.random.default_rng(0).random((64, 64, 4)).astype(np.float32)


@pytest.mark.parametrize("scale", [2, 4])
def test_degrade_shape_and_range(hr, scale):
    lr = degrade(hr, scale)
    assert lr.shape == (64 // scale, 64 // scale, 4)
    assert lr.dtype == np.float32
    assert lr.min() >= 0 and lr.max() <= 1


def test_degrade_keeps_single_band_axis():
    lr = degrade(np.zeros((32, 32, 1), np.float32), 2)
    assert lr.shape == (16, 16, 1)
    assert bicubic_upsample(lr, 2).shape == (32, 32, 1)


def test_degrade_rejects_indivisible_size(hr):
    with pytest.raises(ValueError):
        degrade(hr[:30], 4)


def test_blur_removes_high_frequency_detail(hr):
    sharp = degrade(hr, 2, blur_sigma=0)
    blurred = degrade(hr, 2, blur_sigma=1.5)
    assert blurred.std() < sharp.std()


def test_noise_is_reproducible_with_seed(hr):
    a = degrade(hr, 2, noise_std=0.01, rng=np.random.default_rng(7))
    b = degrade(hr, 2, noise_std=0.01, rng=np.random.default_rng(7))
    clean = degrade(hr, 2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, clean)


def test_bicubic_upsample_restores_shape(hr):
    lr = degrade(hr, 4)
    assert bicubic_upsample(lr, 4).shape == hr.shape


def test_dihedral_group_has_eight_distinct_members():
    image = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    variants = {dihedral(image, k).tobytes() for k in range(8)}
    assert len(variants) == 8
    np.testing.assert_array_equal(dihedral(image, 0), image)


def test_augment_pair_applies_same_transform(hr):
    lr = degrade(hr, 2, blur_sigma=0)
    lr_aug, hr_aug, k = augment_pair(lr, hr, np.random.default_rng(3))
    assert 1 <= k < 8
    # Degrading the augmented HR must equal augmenting the LR: pixels still align.
    np.testing.assert_allclose(degrade(hr_aug, 2, blur_sigma=0), lr_aug, atol=1e-5)
