"""
Media caching utilities for thumbnail and GIF generation.

This module handles caching of thumbnails for images and animated GIF previews
for videos to improve gallery loading performance.
"""

import hashlib
import math
import os
from io import BytesIO
from typing import Optional

import PIL
from PIL import Image
import imageio

from lib.media_item import is_image, is_video


def get_cache_dir() -> str:
    """Get the cache directory path, creating it if necessary."""
    base_dir = os.getcwd()
    cache_dir = os.path.join(base_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def compute_cache_path(path: str, max_frames: int = 32, step: int = 2, duration: int = 100) -> str:
    """
    Build a unique cache file path for given media file and parameters.

    Args:
        path: Path to the media file
        max_frames: Maximum number of frames for video GIFs
        step: Frame step for video GIFs
        duration: Frame duration for video GIFs

    Returns:
        Full path to the cache file
    """
    cache_dir = get_cache_dir()
    key = f"{path}-{max_frames}-{step}-{duration}"
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key_hash}.gif")


def resize_to_target_megapixels(image: PIL.Image.Image, megapixels: float = 0.5) -> PIL.Image.Image:
    """
    Resize the image to a target number of megapixels while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        megapixels: Target size in megapixels

    Returns:
        Resized PIL Image
    """
    width, height = image.size
    aspect_ratio = width / height

    target_pixels = 1_000_000 * megapixels
    target_height = int(math.sqrt(target_pixels / aspect_ratio))
    target_width = int(target_height * aspect_ratio)

    return image.resize((target_width, target_height))


def resize_and_crop_square(img: PIL.Image.Image, size: int = 128) -> PIL.Image.Image:
    """
    Resize image keeping aspect ratio, then center crop to a square.

    Args:
        img: PIL Image to resize and crop
        size: Target square size in pixels

    Returns:
        Resized and cropped PIL Image
    """
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size

    return img.crop((left, top, right, bottom))


def generate_video_thumbnail(
    path: str,
    cache_path: str,
    max_frames: int = 32,
    step: int = 2,
    duration: int = 100,
    size: int = 128
) -> Optional[str]:
    """
    Generate an animated GIF thumbnail for a video file.

    Args:
        path: Path to the video file
        cache_path: Path where the cached GIF should be stored
        max_frames: Maximum number of frames to extract
        step: Take every nth frame
        duration: Duration of each frame in milliseconds
        size: Square size for the thumbnail

    Returns:
        Path to the cached GIF, or None on error
    """
    try:
        video = imageio.get_reader(path)
        frames = []

        for i, frame in enumerate(video):
            if i % step == 0:
                img = Image.fromarray(frame)
                img = resize_and_crop_square(img, size=size)
                frames.append(img)
            if len(frames) >= max_frames:
                break

        if not frames:
            raise ValueError("No frames extracted")

        # Convert frames to GIF
        frames[0].info['duration'] = duration
        frames[0].info['loop'] = 0
        gif_bytes = BytesIO()
        frames[0].save(
            gif_bytes,
            save_all=True,
            format="GIF",
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        gif_bytes.seek(0)

        # Persist to cache
        with open(cache_path, "wb") as f:
            f.write(gif_bytes.getbuffer())

        return cache_path
    except Exception as e:
        print(f"Error generating video thumbnail for {path}: {e}")
        return None


def generate_image_thumbnail(path: str, cache_path: str, megapixels: float = 0.2) -> Optional[str]:
    """
    Generate a thumbnail for an image file.

    Args:
        path: Path to the image file
        cache_path: Path where the cached thumbnail should be stored
        megapixels: Target size in megapixels

    Returns:
        Path to the cached thumbnail, or None on error
    """
    try:
        img = Image.open(path)
        img = resize_to_target_megapixels(img, megapixels)
        img.save(cache_path)
        return cache_path
    except Exception as e:
        print(f"Error generating image thumbnail for {path}: {e}")
        return None


def generate_thumbnail(
    path: str,
    max_frames: int = 32,
    step: int = 2,
    duration: int = 100,
    size: int = 128
) -> Optional[str]:
    """
    Load media and return path to a cached thumbnail (image) or animated GIF (video).

    This function checks the cache first and only generates new thumbnails
    if necessary.

    Args:
        path: Path to the media file
        max_frames: Maximum number of frames for video GIFs
        step: Take every nth frame for video GIFs
        duration: Duration of each frame in ms for video GIFs
        size: Square size for video thumbnails

    Returns:
        Path to the cached thumbnail/GIF, or None on error
    """
    cache_path = compute_cache_path(path, max_frames, step, duration)

    # Return cached version if available
    if os.path.exists(cache_path):
        return cache_path

    if is_video(path):
        return generate_video_thumbnail(path, cache_path, max_frames, step, duration, size)
    elif is_image(path):
        return generate_image_thumbnail(path, cache_path)

    return None


def clear_cache() -> int:
    """
    Clear all cached thumbnails.

    Returns:
        Number of files deleted
    """
    cache_dir = get_cache_dir()
    count = 0
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            count += 1
    return count
