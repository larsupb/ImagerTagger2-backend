import json
import os
import threading
import requests
from PIL import Image
from typing import Union

import config
from lib.upscaling.util import scale_to_megapixels

_download_lock = threading.Lock()

UPSCALERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'upscalers.json')


def load_upscalers() -> list[dict]:
    with open(UPSCALERS_FILE, 'r') as f:
        return json.load(f)


def get_upscaler_names() -> list[str]:
    return [u['name'] for u in load_upscalers()]


def _find_upscaler(name: str) -> dict | None:
    return next((u for u in load_upscalers() if u['name'] == name), None)


def upscale_image(img: Union[Image.Image, str], upscaler_name: str, state_dict: dict, progress=None,
                  max_current_megapixels=2., target_megapixels=2.) -> Image.Image:
    from .spandrel_upscaler import upscale_image_spandrel

    # Load image from path if a string is provided
    if isinstance(img, str):
        img = Image.open(img)

    # skip if image is too large
    if img.width * img.height > max_current_megapixels * 1_000_000:
        print(f"Image's total megapixels count {img.width * img.height / 1_000_000} exceeds maximum megapixels "
              f"count {max_current_megapixels}, skipping upscale.")
        return img

    upscaler = _find_upscaler(upscaler_name)
    if upscaler is None:
        return None
    # read directory from current file
    upscaler_model = os.path.join(os.getcwd(), 'models', 'upscalers', upscaler['filename'])

    # if upscaler_model file does not exist, download it (lock to prevent concurrent downloads)
    if not os.path.exists(upscaler_model):
        with _download_lock:
            if not os.path.exists(upscaler_model):
                url = upscaler.get('url')
                if not url:
                    raise FileNotFoundError(
                        f"Model file {upscaler_model} not found and no download URL configured.")
                print(f"Downloading upscaler model {upscaler_model} from {url}")
                os.makedirs(os.path.dirname(upscaler_model), exist_ok=True)
                tmp_path = upscaler_model + '.tmp'
                r = requests.get(url, allow_redirects=True)
                with open(tmp_path, 'wb') as f:
                    f.write(r.content)
                os.replace(tmp_path, upscaler_model)

    # check that the image has 3 channels (RGB)
    if img.mode != 'RGB':
        print(f"Image mode {img.mode} is not RGB, converting to RGB.")
        img = img.convert('RGB')

    upscaled = upscale_image_spandrel(img, upscaler_model, upscale_factor=upscaler['scale_factor'], progress=progress)

    # if the upscaled image is larger than the target megapixels, resize it
    upscaled = scale_to_megapixels(upscaled, target_megapixels)

    return upscaled