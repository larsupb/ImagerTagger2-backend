"""
Bucket-based image resizing for LoRA training dataset preparation.

Generates a small set of resolution buckets around a target pixel count,
assigns each image to the nearest bucket by aspect ratio, and resizes+crops
to the exact bucket dimensions.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

from PIL import Image


@dataclass
class Bucket:
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    def __eq__(self, other) -> bool:
        return self.width == other.width and self.height == other.height


def generate_buckets(base_resolution: int = 1024, step: int = 128, max_steps: int = 2) -> List[Bucket]:
    """
    Generate resolution buckets around a base resolution.

    Creates buckets by trading width for height in increments of `step`,
    keeping the total pixel count close to base_resolution^2.

    Args:
        base_resolution: The square base resolution (e.g. 512, 768, 1024)
        step: Pixel increment for width/height adjustments
        max_steps: How many steps away from square to allow

    Returns:
        List of Bucket objects, sorted by aspect ratio (portrait first)
    """
    buckets = [Bucket(base_resolution, base_resolution)]

    for i in range(1, max_steps + 1):
        w = base_resolution + i * step
        h = base_resolution - i * step
        if h < step:
            break
        # Round to nearest multiple of step for clean numbers
        w = round(w / step) * step
        h = round(h / step) * step
        buckets.append(Bucket(w, h))  # landscape
        buckets.append(Bucket(h, w))  # portrait

    buckets.sort(key=lambda b: b.aspect_ratio)
    return buckets


def find_nearest_bucket(width: int, height: int, buckets: List[Bucket]) -> Bucket:
    """
    Find the bucket whose aspect ratio is closest to the image's aspect ratio.
    """
    ar = width / height
    return min(buckets, key=lambda b: abs(b.aspect_ratio - ar))


def analyze_buckets(dataset, buckets: List[Bucket]) -> Dict[Bucket, List[int]]:
    """
    Analyze a dataset and assign each image to its nearest bucket.

    Args:
        dataset: ImageDataSet instance
        buckets: List of available buckets

    Returns:
        Dict mapping each bucket to a list of image indices assigned to it
    """
    assignment: Dict[Bucket, List[int]] = {b: [] for b in buckets}

    for i in range(len(dataset)):
        item = dataset.get_item(i)
        if item is None:
            continue
        try:
            with Image.open(item.media_path) as img:
                bucket = find_nearest_bucket(img.width, img.height, buckets)
                assignment[bucket].append(i)
        except Exception:
            continue

    return assignment


def format_bucket_analysis(assignment: Dict[Bucket, List[int]], total: int) -> str:
    """Format bucket analysis as a human-readable string."""
    lines = ["Bucket Distribution:"]
    lines.append("-" * 45)

    for bucket in sorted(assignment.keys(), key=lambda b: b.aspect_ratio):
        count = len(assignment[bucket])
        if count == 0:
            continue
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * max(1, int(pct / 2))
        lines.append(f"  {bucket}  {count:>4} images ({pct:>5.1f}%)  {bar}")

    # Show empty buckets
    empty = [str(b) for b in assignment if len(assignment[b]) == 0]
    if empty:
        lines.append(f"\n  Empty buckets (will be unused): {', '.join(empty)}")

    lines.append("-" * 45)
    lines.append(f"  Total: {total} images across {sum(1 for v in assignment.values() if v)} buckets")
    return "\n".join(lines)


def resize_and_crop_to_bucket(image: Image.Image, bucket: Bucket) -> Image.Image:
    """
    Resize and center-crop an image to exactly match a bucket's dimensions.

    The image is first resized (preserving aspect ratio) so that it fully
    covers the bucket dimensions, then center-cropped to the exact size.

    Args:
        image: Input PIL Image
        bucket: Target bucket dimensions

    Returns:
        Resized and cropped PIL Image
    """
    img_ar = image.width / image.height
    bucket_ar = bucket.aspect_ratio

    # Scale so the image fully covers the bucket
    if img_ar > bucket_ar:
        # Image is wider than bucket: match height, crop width
        scale = bucket.height / image.height
    else:
        # Image is taller than bucket: match width, crop height
        scale = bucket.width / image.width

    new_w = round(image.width * scale)
    new_h = round(image.height * scale)

    resized = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # Center crop
    left = (new_w - bucket.width) // 2
    top = (new_h - bucket.height) // 2
    cropped = resized.crop((left, top, left + bucket.width, top + bucket.height))

    return cropped