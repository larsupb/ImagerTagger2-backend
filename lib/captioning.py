import logging
import os
import time

import config
from lib.image_dataset import ImageDataSet

logger = logging.getLogger(__name__)

TAGGERS = ["joytag", "wd14", "florence", "vlm-tagger", "ideogram", "combo"]


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get("dataset")


def generate_caption(current_index, option, state_dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        logger.warning("Cannot generate caption: dataset is None or not initialized")
        return

    item = dataset.get_item(current_index)
    if item is None:
        logger.warning(
            f"Cannot generate caption: item at index {current_index} is None"
        )
        return ""

    path = item.media_path
    logger.info(f"Generating caption for '{path}' using tagger: {option}")

    caption = ""
    if os.path.exists(path):
        start_time = time.time()
        try:
            if option == "joytag":
                from lib.tagging.joytag.joytag import generate_joytag_caption

                caption = generate_joytag_caption(path)
            elif option == "wd14":
                from lib.tagging.wd14_tagger import generate_wd14_caption

                caption = generate_wd14_caption(path)
            elif option == "florence":
                from lib.tagging.florence_tagger import generate_florence_caption

                caption = generate_florence_caption(
                    path, prompt=config.florence_settings(state_dict)["prompt"]
                )
            elif option == "vlm-tagger":
                from lib.tagging.openai_tagger import generate_vlm_caption

                caption = generate_vlm_caption(
                    path, config.vlm_endpoint_settings(state_dict)
                )
            elif option == "ideogram":
                from lib.tagging.ideogram_tagger import generate_ideogram_caption

                caption = generate_ideogram_caption(path)
            elif option == "combo":
                caption = generate_combo_caption(current_index, state_dict)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Caption generated successfully in {elapsed_time:.2f}s: '{caption[:100]}...' (length: {len(caption)})"
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Error generating caption with {option} after {elapsed_time:.2f}s: {str(e)}",
                exc_info=True,
            )
            raise
    else:
        logger.error(f"Image path does not exist: {path}")

    return caption


def generate_combo_caption(current_index, state_dict):
    """
    Generate a caption using a combination of taggers
    """
    caption = []
    taggers = config.combo_taggers(state_dict)
    logger.info(f"Generating combo caption using taggers: {taggers}")
    for t in taggers:
        caption.append(generate_caption(current_index, t, state_dict))
    combined_caption = ", ".join(caption)
    logger.info(
        f"Combo caption completed: {len(taggers)} taggers, total length: {len(combined_caption)}"
    )
    return combined_caption


def save_caption(index, caption_text, file_name=None, dataset: ImageDataSet = None):
    """
    Save caption text for an image.

    :param index: Image index, or -1 if using file_name
    :param caption_text: Caption text to save
    :param file_name: Optional file name to look up index
    :param dataset: ImageDataSet instance (required)
    """
    if caption_text is None or dataset is None or not dataset.is_initialized:
        return
    if file_name is not None:
        index = dataset.find_index(file_name)
    dataset.save_caption(index, caption_text)
