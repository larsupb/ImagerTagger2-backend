"""
ComfyUI workflow definition and helpers.
"""

import copy
import json
import os

from app.config import comfyui_settings
from lib.upscaling.util import calculate_resolution_for_megapixels


DEFAULT_WORKFLOW_FILE = os.path.join(
    os.path.dirname(__file__), "workflows", "seedvr2.json"
)


def get_workflow() -> dict:
    """
    Load workflow from settings or use default.

    Returns:
        Workflow dictionary (API format)
    """
    settings = comfyui_settings()
    workflow_file = settings.get("workflow_file")

    if workflow_file and os.path.exists(workflow_file):
        with open(workflow_file, "r") as f:
            return json.load(f)

    if os.path.exists(DEFAULT_WORKFLOW_FILE):
        with open(DEFAULT_WORKFLOW_FILE, "r") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No workflow file found. Expected at {DEFAULT_WORKFLOW_FILE} or custom path."
    )


import uuid


def inject_image(
    workflow: dict,
    image_filename: str,
    target_megapixels: float = None,
    input_width: int = None,
    input_height: int = None,
) -> dict:
    """
    Inject the image filename into the workflow's LoadImage node
    and set random seed for unique results.

    Args:
        workflow: The workflow dictionary (API format)
        image_filename: The filename of the image to process
        target_megapixels: Target output megapixels (calculates resolution)
        input_width: Input image width (required if target_megapixels set)
        input_height: Input image height (required if target_megapixels set)

    Returns:
        Modified workflow dict with image filename set and random seed
    """
    result = copy.deepcopy(workflow)

    if "10" in result and "inputs" in result["10"]:
        result["10"]["inputs"]["image"] = image_filename

    if target_megapixels and input_width and input_height:
        resolution = calculate_resolution_for_megapixels(
            input_width, input_height, target_megapixels
        )
        if "1" in result and "inputs" in result["1"]:
            result["1"]["inputs"]["resolution"] = resolution

    return result
