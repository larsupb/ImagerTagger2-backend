import math

from PIL.Image import Image, Resampling
import cv2
import numpy as np
import torch


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def scale_to_megapixels(image, megapixels=0.5):
    width, height = image.size
    aspect_ratio = width / height

    target_pixels = 1_000_000 * megapixels
    target_height = int(math.sqrt(target_pixels / aspect_ratio))
    target_width = int(target_height * aspect_ratio)

    return image.resize((target_width, target_height), resample=Resampling.NEAREST)


def calculate_resolution_for_megapixels(
    width: int, height: int, megapixels: float
) -> int:
    aspect_ratio = width / height
    target_pixels = 1_000_000 * megapixels

    if width >= height:
        resolution = int(math.sqrt(target_pixels / aspect_ratio))
    else:
        resolution = int(math.sqrt(target_pixels * aspect_ratio))

    return resolution
