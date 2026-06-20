import math
import os
from typing import Tuple

import pandas as pd
from PIL import Image

from lib.image_aspects import ImageAspects
from lib.image_dataset import ImageDataSet


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


def validate_dataset(state_dict: dict, aspect_ratios=None):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized or dataset.is_empty:
        return None

    if aspect_ratios is not None and len(aspect_ratios) > 0:
        aspects = [ImageAspects.from_code(a) for a in aspect_ratios]
    else:
        aspects = [ImageAspects.SQUARE,
                   ImageAspects.POSTCARD,
                   ImageAspects.POSTCARD_INVERSE]

    def meets_format(d: tuple, epsilon=0.01):
        ratios_allowed = [a.value for a in aspects]
        ratio = d[0]/d[1]
        for da in ratios_allowed:
            if da - da * epsilon < ratio < da + da * epsilon:
                return True
        print(f"Missing format {d}: {ratio}")
        return False

    def calculate_bucket(image, megapixels=1, bucket_step_size=64) -> Tuple[float, float]:
        width, height = image.size
        aspect_ratio = width / height

        #calculate image pixels
        img_pixels = width * height
        # Calculate the new dimensions to get close to 1 megapixel
        target_pixels = min(1_000_000 * megapixels, img_pixels)
        target_height = int(math.sqrt(target_pixels / aspect_ratio))
        target_width = int(target_height * aspect_ratio)

        bucket_width = int(target_width / bucket_step_size) * bucket_step_size
        bucket_height = int(target_height/bucket_step_size) * bucket_step_size

        return bucket_width, bucket_height

    bucket_count = dict()
    bucket_assignments = dict()

    def validate(index, img_path, mask_path, caption):
        # img_validation = {'Image': '', 'Mask': '', 'Caption': ''}
        # if img_path is None or not os.path.exists(img_path):
        #     img_validation['Image'] = f'No image found'
        # elif mask_path is None or not os.path.exists(mask_path):
        #     img_validation['Mask'] = f'No mask found'
        # else:
        #     img = Image.open(img_path)
        #     mask = Image.open(mask_path)
        #     if img.size != mask.size:
        #         img_validation['Image'] = f'Image dims do not fit to mask dims!'
        #

        if img_path is not None and os.path.exists(img_path):
            img = Image.open(img_path)
            bucket_size = calculate_bucket(img, 1)
            if bucket_size in bucket_count:
                bucket_count[bucket_size] += 1
            else:
                bucket_count[bucket_size] = 1

            filename = img_path.split('/')[-1]
            if bucket_size in bucket_assignments:
                bucket_assignments[bucket_size].append(filename)
            else:
                bucket_assignments[bucket_size] = [filename]

        #     if not meets_format(img.size):
        #         img_validation['Image'] = f'Not in the allowed aspect ratios!'
        # if caption is None or len(caption) == 0:
        #     img_validation['Caption'] = f'Caption is empty!'

    validation_results = []
    # Scan the dataset and validate each image
    dataset.scan(validate)

    for k, v in bucket_count.items():
        validation_results.append({'Bucket': k, 'Count': v, 'Files': bucket_assignments[k]})

    return pd.DataFrame(data=validation_results, columns=['Bucket', 'Count', 'Files'])
