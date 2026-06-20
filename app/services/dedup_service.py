import logging
from collections import defaultdict
from typing import Optional

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from lib.media_item import MediaItem

logger = logging.getLogger(__name__)


def compute_phash(media_path: str) -> Optional[imagehash.ImageHash]:
    try:
        img = Image.open(media_path).convert("RGB")
        return imagehash.phash(img)
    except Exception as e:
        logger.warning("compute_phash: failed for %s: %s", media_path, e)
        return None


def find_candidate_pairs(
    hashes: list[Optional[imagehash.ImageHash]],
    phash_threshold: int,
) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(hashes)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j] is None:
                continue
            if hashes[i] - hashes[j] <= phash_threshold:
                pairs.append((i, j))
    return pairs


def confirm_with_ssim(path_a: str, path_b: str, ssim_threshold: float) -> bool:
    try:
        size = (256, 256)
        img_a = np.array(Image.open(path_a).convert("L").resize(size))
        img_b = np.array(Image.open(path_b).convert("L").resize(size))
        score = ssim(img_a, img_b, data_range=255)
        return float(score) >= ssim_threshold
    except Exception as e:
        logger.warning("confirm_with_ssim: failed for %s / %s: %s", path_a, path_b, e)
        return False


def group_duplicates(
    items: list[MediaItem],
    phash_threshold: int = 10,
    ssim_threshold: float = 0.85,
) -> list[list[int]]:
    hashes = [compute_phash(item.media_path) for item in items]
    candidates = find_candidate_pairs(hashes, phash_threshold)

    confirmed = [
        (i, j)
        for i, j in candidates
        if confirm_with_ssim(items[i].media_path, items[j].media_path, ssim_threshold)
    ]

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, j in confirmed:
        union(i, j)

    groups_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(items)):
        groups_map[find(idx)].append(idx)

    return [sorted(g) for g in groups_map.values() if len(g) > 1]
