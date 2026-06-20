"""
Test for the color matching module.

Run with:
    cd backend
    python -m pytest tests/test_color_matching.py -v -s
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from lib.color_matching import (
    color_match_image,
    apply_histogram_matching_lab,
    apply_wavelet_matching,
    apply_pca_matching,
    compute_lab_histogram,
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def test_import_color_matching():
    """Test that the color_matching module can be imported."""
    from lib import color_matching

    assert color_matching is not None


def test_histogram_matching():
    """Test LAB histogram matching method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = apply_histogram_matching_lab(source, ref)

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == source.size


def test_wavelet_matching():
    """Test wavelet matching method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = apply_wavelet_matching(source, ref)

    assert result is not None
    assert result.mode == "RGB"
    assert result.size[0] <= ref.size[0] and result.size[1] <= ref.size[1]


def test_pca_matching():
    """Test PCA matching method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = apply_pca_matching(source, ref)

    assert result is not None
    assert result.mode == "RGB"
    assert result.size[0] <= ref.size[0] and result.size[1] <= ref.size[1]


def test_color_match_image_histogram():
    """Test color_match_image dispatcher with histogram method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = color_match_image(source, ref, method="histogram")

    assert result is not None
    assert result.mode == "RGB"


def test_color_match_image_wavelet():
    """Test color_match_image dispatcher with wavelet method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = color_match_image(source, ref, method="wavelet")

    assert result is not None
    assert result.mode == "RGB"


def test_color_match_image_pca():
    """Test color_match_image dispatcher with pca method."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    result = color_match_image(source, ref, method="pca")

    assert result is not None
    assert result.mode == "RGB"


def test_color_match_invalid_method():
    """Test that invalid method raises ValueError."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")
    ref_path = os.path.join(ASSETS_DIR, "Snow_Leopard_on_Snowy_Rocks.jpg")

    source = Image.open(source_path)
    ref = Image.open(ref_path)

    try:
        color_match_image(source, ref, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown color matching method" in str(e)


def test_compute_lab_histogram():
    """Test LAB histogram computation."""
    source_path = os.path.join(ASSETS_DIR, "1200px-Snow_leopard_portrait.jpg")

    source = Image.open(source_path)
    histogram = compute_lab_histogram(source)

    assert histogram is not None
    assert "l" in histogram
    assert "a" in histogram
    assert "b" in histogram
    assert len(histogram["l"]) == 64
    assert len(histogram["a"]) == 64
    assert len(histogram["b"]) == 64


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
