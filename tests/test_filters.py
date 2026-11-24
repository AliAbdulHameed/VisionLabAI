import numpy as np
import pytest

from visionlab.filters import pencil_sketch


def _gradient_image(h=120, w=160):
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(row, (h, 1))
    return np.dstack([gray, gray, gray])


def test_returns_same_shape_three_channel_uint8():
    image = _gradient_image()
    out = pencil_sketch(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_output_is_binary():
    out = pencil_sketch(np.random.default_rng(0).integers(0, 256, (90, 90, 3), dtype=np.uint8))
    assert set(np.unique(out)) <= {0, 255}


def test_invert_flips_the_image():
    image = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    normal = pencil_sketch(image, invert=False)
    inverted = pencil_sketch(image, invert=True)
    assert np.array_equal(normal, 255 - inverted)


@pytest.mark.parametrize("detail", [0, 2, 4, 10])
def test_small_or_even_detail_is_corrected(detail):
    pencil_sketch(_gradient_image(), detail=detail)  # must not raise


def test_rejects_grayscale_input():
    with pytest.raises(ValueError):
        pencil_sketch(np.zeros((10, 10), dtype=np.uint8))
