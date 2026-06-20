"""White balance correction algorithms."""

import numpy as np
from PIL import Image
from typing import Optional


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _scale_channel(
    channel: np.ndarray, target_mean: float, target_std: Optional[float] = None
) -> np.ndarray:
    """Scale a single channel to target mean (and optionally std)."""
    current_mean = channel.mean()
    target_mean = float(target_mean)
    scaled = channel.astype(np.float32) * (target_mean / (current_mean + 1e-8))
    if target_std is not None:
        current_std = scaled.std()
        scaled = (scaled - scaled.mean()) / (current_std + 1e-8) * target_std
        scaled = scaled + target_mean
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_gray_world(img: Image.Image) -> Image.Image:
    """Apply Gray World white balance - equalizes channel means."""
    img = _to_rgb(img)
    arr = np.array(img, dtype=np.float32)

    target_mean = arr.mean()

    for c in range(3):
        arr[:, :, c] = _scale_channel(arr[:, :, c], target_mean)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_shades_of_gray(img: Image.Image) -> Image.Image:
    """Apply Shades of Gray - equalizes both means and standard deviations."""
    img = _to_rgb(img)
    arr = np.array(img, dtype=np.float32)

    target_mean = arr.mean()
    target_std = arr.std()

    for c in range(3):
        arr[:, :, c] = _scale_channel(arr[:, :, c], target_mean, target_std)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_gray_edge(img: Image.Image, threshold: float = 10.0) -> Image.Image:
    """Apply Gray Edge white balance - uses edge pixel for correction."""
    img = _to_rgb(img)
    arr = np.array(img, dtype=np.float32)

    grad_x = np.diff(arr, axis=1)
    grad_y = np.diff(arr, axis=0)

    grad_mag = np.zeros((arr.shape[0], arr.shape[1]))
    grad_mag[:, :-1] += np.linalg.norm(grad_x, axis=2)
    grad_mag[:-1, :] += np.linalg.norm(grad_y, axis=2)

    edge_mask = grad_mag > threshold

    edge_pixels = arr[edge_mask]
    if len(edge_pixels) == 0:
        edge_pixels = arr

    target_mean = edge_pixels.mean(axis=0)

    for c in range(3):
        arr[:, :, c] = _scale_channel(arr[:, :, c], float(target_mean[c]))

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def white_balance_image(img: Image.Image, method: str = "gray_world") -> Image.Image:
    """Apply white balance to image using specified method."""
    if method == "gray_world":
        return apply_gray_world(img)
    elif method == "shades_of_gray":
        return apply_shades_of_gray(img)
    elif method == "gray_edge":
        return apply_gray_edge(img)
    else:
        raise ValueError(f"Unknown white balance method: {method}")
