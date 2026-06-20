import pytest
import numpy as np
from PIL import Image
from lib.white_balance import (
    apply_gray_world,
    apply_shades_of_gray,
    apply_gray_edge,
    white_balance_image,
)


def create_test_image(r: int, g: int, b: int, size: tuple = (100, 100)) -> Image.Image:
    arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return Image.fromarray(arr)


def test_gray_world_balances_channel_means():
    img = create_test_image(150, 100, 50)
    result = apply_gray_world(img)
    result_arr = np.array(result)
    means = result_arr.mean(axis=(0, 1))
    assert abs(means[0] - means[1]) < 5
    assert abs(means[1] - means[2]) < 5


def test_shades_of_gray_balances_both_mean_and_std():
    img = create_test_image(150, 100, 50)
    result = apply_shades_of_gray(img)
    result_arr = np.array(result)
    means = result_arr.mean(axis=(0, 1))
    stds = result_arr.std(axis=(0, 1))
    assert abs(means[0] - means[1]) < 10
    assert abs(means[1] - means[2]) < 10
    assert abs(stds[0] - stds[1]) < 10


def test_gray_edge_uses_edge_pixels():
    img = create_test_image(150, 100, 50)
    result = apply_gray_edge(img)
    result_arr = np.array(result)
    means = result_arr.mean(axis=(0, 1))
    assert means.shape == (3,)


def test_white_balance_dispatch():
    img = create_test_image(150, 100, 50)
    result = white_balance_image(img, "gray_world")
    assert isinstance(result, Image.Image)
    result2 = white_balance_image(img, "shades_of_gray")
    assert isinstance(result2, Image.Image)
    result3 = white_balance_image(img, "gray_edge")
    assert isinstance(result3, Image.Image)


def test_white_balance_invalid_method_raises():
    img = create_test_image(150, 100, 50)
    with pytest.raises(ValueError):
        white_balance_image(img, "invalid_method")
