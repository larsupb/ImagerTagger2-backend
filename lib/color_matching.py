"""Color matching utilities using histogram, wavelet, and PCA methods."""

import numpy as np
from PIL import Image
from typing import Tuple


def apply_histogram_matching_lab(
    source: Image.Image, reference: Image.Image
) -> Image.Image:
    """Apply histogram matching in LAB color space."""
    try:
        import skimage
    except ImportError:
        raise ImportError("scikit-image required for histogram matching")

    if source.mode != "RGB":
        source = source.convert("RGB")
    if reference.mode != "RGB":
        reference = reference.convert("RGB")

    source_arr = np.array(source)
    ref_arr = np.array(reference)

    source_lab = skimage.color.rgb2lab(source_arr)
    ref_lab = skimage.color.rgb2lab(ref_arr)

    matched_l = skimage.exposure.match_histograms(source_lab[:, :, 0], ref_lab[:, :, 0])
    matched_a = skimage.exposure.match_histograms(source_lab[:, :, 1], ref_lab[:, :, 1])
    matched_b = skimage.exposure.match_histograms(source_lab[:, :, 2], ref_lab[:, :, 2])

    matched_lab = np.stack([matched_l, matched_a, matched_b], axis=-1)
    matched_rgb = skimage.color.lab2rgb(matched_lab)

    matched_rgb = np.clip(matched_rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(matched_rgb)


def apply_wavelet_matching(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Apply wavelet-based color transfer."""
    try:
        import pywt
    except ImportError:
        raise ImportError("pywavelets required for wavelet matching")

    if source.mode != "RGB":
        source = source.convert("RGB")
    if reference.mode != "RGB":
        reference = reference.convert("RGB")

    source_arr = np.array(source, dtype=np.float32)
    ref_arr = np.array(reference, dtype=np.float32)

    source_resized = source.resize(reference.size, Image.LANCZOS)
    source_arr = np.array(source_resized, dtype=np.float32)

    result = np.zeros_like(source_arr)

    for channel in range(3):
        coeffs_src = pywt.wavedec2(source_arr[:, :, channel], "db1", level=2)
        coeffs_ref = pywt.wavedec2(ref_arr[:, :, channel], "db1", level=2)

        for level in range(len(coeffs_src)):
            if level == 0:
                coeffs_src[level] = (coeffs_src[level] - coeffs_src[level].mean()) / (
                    coeffs_src[level].std() + 1e-8
                ) * coeffs_ref[level].std() + coeffs_ref[level].mean()
            else:
                cH_s, cV_s, cD_s = coeffs_src[level]
                cH_r, cV_r, cD_r = coeffs_ref[level]

                coeffs_src[level] = (
                    (cH_s - cH_s.mean()) / (cH_s.std() + 1e-8) * cH_r.std()
                    + cH_r.mean(),
                    (cV_s - cV_s.mean()) / (cV_s.std() + 1e-8) * cV_r.std()
                    + cV_r.mean(),
                    (cD_s - cD_s.mean()) / (cD_s.std() + 1e-8) * cD_r.std()
                    + cD_r.mean(),
                )

        reconstructed = pywt.waverec2(coeffs_src, "db1")
        result[:, :, channel] = reconstructed[
            : source_arr.shape[0], : source_arr.shape[1]
        ]

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def apply_pca_matching(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Apply PCA-based color transfer."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scikit-learn required for PCA matching")

    if source.mode != "RGB":
        source = source.convert("RGB")
    if reference.mode != "RGB":
        reference = reference.convert("RGB")

    source_resized = source.resize(reference.size, Image.LANCZOS)
    source_arr = np.array(source_resized, dtype=np.float32)
    ref_arr = np.array(reference, dtype=np.float32)

    h, w = source_arr.shape[:2]
    source_flat = source_arr.reshape(-1, 3)
    ref_flat = ref_arr.reshape(-1, 3)

    pca_source = PCA(n_components=3)
    pca_ref = PCA(n_components=3)

    source_pca = pca_source.fit_transform(source_flat)
    ref_pca = pca_ref.fit_transform(ref_flat)

    source_pca_norm = (source_pca - source_pca.mean(axis=0)) / (
        source_pca.std(axis=0) + 1e-8
    )
    ref_pca_norm = (ref_pca - ref_pca.mean(axis=0)) / (ref_pca.std(axis=0) + 1e-8)

    matched_pca = source_pca_norm * ref_pca.std(axis=0) + ref_pca.mean(axis=0)

    matched_flat = pca_ref.inverse_transform(matched_pca)
    matched_arr = matched_flat.reshape(h, w, 3)

    result = np.clip(matched_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def color_match_image(
    source: Image.Image, reference: Image.Image, method: str = "histogram"
) -> Image.Image:
    """Apply color matching to source image using specified method."""
    if method == "histogram":
        return apply_histogram_matching_lab(source, reference)
    elif method == "wavelet":
        return apply_wavelet_matching(source, reference)
    elif method == "pca":
        return apply_pca_matching(source, reference)
    else:
        raise ValueError(f"Unknown color matching method: {method}")


def compute_lab_histogram(image: Image.Image) -> dict:
    """Compute LAB histogram for visualization."""
    import skimage

    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)
    lab = skimage.color.rgb2lab(arr)

    hist_l, _ = np.histogram(lab[:, :, 0].flatten(), bins=64, range=(0, 100))
    hist_a, _ = np.histogram(lab[:, :, 1].flatten(), bins=64, range=(-128, 128))
    hist_b, _ = np.histogram(lab[:, :, 2].flatten(), bins=64, range=(-128, 128))

    return {
        "l": hist_l.tolist(),
        "a": hist_a.tolist(),
        "b": hist_b.tolist(),
    }
