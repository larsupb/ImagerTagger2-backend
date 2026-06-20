import logging
import re
import time
from typing import Mapping, Tuple, Dict

import PIL
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


# noinspection PyUnresolvedReferences
def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass

    return img


class WaifuDiffusionInterrogator:
    def __init__(
            self,
            repo='SmilingWolf/wd-v1-4-vit-tagger',
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            mode: str = "auto"
    ) -> None:
        self.__repo = repo
        self.__model_path = model_path
        self.__tags_path = tags_path
        self._provider_mode = mode

        self.__initialized = False
        self._model, self._tags = None, None

    def _init(self) -> None:
        if self.__initialized:
            return

        logger.info(f"Initializing WD14 model: {self.__repo}")
        start_time = time.time()

        model_path = hf_hub_download(self.__repo, filename=self.__model_path)
        tags_path = hf_hub_download(self.__repo, filename=self.__tags_path)

        self._model = InferenceSession(str(model_path))
        self._tags = pd.read_csv(tags_path)

        self.__initialized = True
        elapsed = time.time() - start_time
        logger.info(f"WD14 model initialized in {elapsed:.2f}s, loaded {len(self._tags)} tags")

    def _calculation(self, image: Image.Image) -> pd.DataFrame:
        self._init()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        logger.debug(f"Processing image for WD14: size={image.size}, mode={image.mode}")
        start_time = time.time()

        # convert an image to fit the model
        _, height, _, _ = self._model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        infer_start = time.time()
        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        confidence = self._model.run([label_name], {input_name: image})[0]
        infer_time = time.time() - infer_start
        logger.debug(f"WD14 inference completed in {infer_time:.3f}s")

        full_tags = self._tags[['name', 'category']].copy()
        full_tags['confidence'] = confidence[0]

        total_time = time.time() - start_time
        logger.debug(f"WD14 calculation completed in {total_time:.3f}s")

        return full_tags

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        full_tags = self._calculation(image)

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(full_tags[full_tags['category'] == 9][['name', 'confidence']].values)

        # rest are regular tags
        tags = dict(full_tags[full_tags['category'] != 9][['name', 'confidence']].values)

        return ratings, tags


WAIFU_MODELS: Mapping[str, WaifuDiffusionInterrogator] = {
    'wd14-vit': WaifuDiffusionInterrogator(),
    'wd14-convnext': WaifuDiffusionInterrogator(
        repo='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
}
RE_SPECIAL = re.compile(r'([\\()])')


def image_to_wd14_tags(image: Image.Image, model_name: str, threshold: float,
                       use_spaces: bool, use_escape: bool, include_ranks: bool, score_descend: bool) \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    logger.info(f"Converting image to WD14 tags using model: {model_name}, threshold: {threshold}")
    start_time = time.time()

    model = WAIFU_MODELS[model_name]
    ratings, tags = model.interrogate(image)

    filtered_tags = {
        tag: score for tag, score in tags.items()
        if score >= threshold
    }

    logger.info(f"Filtered {len(filtered_tags)} tags from {len(tags)} total (threshold: {threshold})")

    text_items = []
    tags_pairs = filtered_tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)
    output_text = ', '.join(text_items)

    elapsed = time.time() - start_time
    logger.info(f"WD14 tagging completed in {elapsed:.2f}s")

    return ratings, output_text, filtered_tags


def generate_wd14_caption(image_path: str):
    logger.info(f"Generating WD14 caption for: {image_path}")
    try:
        image = PIL.Image.open(image_path)
        caption = image_to_wd14_tags(image, 'wd14-vit', 0.5, True,
                                     False, False, True)
        logger.debug(f"Generated caption length: {len(caption[1])}")
        return caption[1]
    except Exception as e:
        logger.error(f"Error generating WD14 caption for {image_path}: {str(e)}", exc_info=True)
        raise
