"""
ComfyUI workflow definition and helpers.
"""

import copy
import json
import logging
import os

from app.config import comfyui_settings
from lib.upscaling.util import calculate_resolution_for_megapixels


DEFAULT_WORKFLOW_FILE = os.path.join(
    os.path.dirname(__file__), "workflows", "seedvr2.json"
)


def get_workflow(workflow_file: str = None) -> dict:
    """
    Load workflow from path or settings or use default.

    Args:
        workflow_file: Path to workflow JSON file. Can be absolute or relative
            to the workflows directory.

    Returns:
        Workflow dictionary (API format)
    """
    path = None

    if workflow_file:
        if os.path.isabs(workflow_file):
            path = workflow_file
        else:
            path = os.path.join(os.path.dirname(__file__), "workflows", workflow_file)

    if not path or not os.path.exists(path):
        settings = comfyui_settings()
        configured = settings.get("workflow_file")
        if configured and os.path.exists(configured):
            path = configured

    if not path or not os.path.exists(path):
        path = DEFAULT_WORKFLOW_FILE

    with open(path, "r") as f:
        return json.load(f)


import uuid


def _find_node_by_class(workflow: dict, class_type: str) -> tuple[str, dict] | None:
    """Find the first node with the given class_type. Returns (node_id, node)."""
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node_id, node
    return None


def _find_nodes_by_class(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
    """Find all nodes with the given class_type."""
    return [
        (node_id, node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]


def inject_params(workflow: dict, injections: list) -> dict:
    """
    Inject workflow-specific parameters into their declared nodes.

    Args:
        workflow: Workflow dict already processed by inject_image()
        injections: List of resolved specs, each a dict with keys
            "node_id", "input_name", and "value". Targeting is explicit so
            that several nodes sharing an input name (e.g. multiple
            PrimitiveStringMultiline "value" inputs) can be set independently.

    Returns:
        Modified workflow dict with params injected
    """
    result = copy.deepcopy(workflow)

    for spec in injections:
        value = spec.get("value")
        if value is None:
            continue

        node_id = spec.get("node_id")
        input_name = spec.get("input_name")
        node = result.get(node_id)
        if isinstance(node, dict) and input_name in node.get("inputs", {}):
            node["inputs"][input_name] = value
        else:
            logging.warning(
                "inject_params: could not resolve node_id=%r input_name=%r",
                node_id,
                input_name,
            )

    return result


def inject_image(
    workflow: dict,
    image_filename: str,
    target_megapixels: float = None,
    input_width: int = None,
    input_height: int = None,
    params: dict = None,
) -> dict:
    """
    Inject the image filename into the workflow's LoadImage node
    and set resolution/megapixels inputs appropriately.

    Args:
        workflow: The workflow dictionary (API format)
        image_filename: The filename of the image to process
        target_megapixels: Target output megapixels
        input_width: Input image width
        input_height: Input image height
        params: List of resolved injection specs (see inject_params) to apply

    Returns:
        Modified workflow dict with image filename and resolution injected
    """
    result = copy.deepcopy(workflow)

    load_image = _find_node_by_class(result, "LoadImage")
    if load_image:
        node_id, node = load_image
        node["inputs"]["image"] = image_filename

    if target_megapixels and input_width and input_height:
        resolution = calculate_resolution_for_megapixels(
            input_width, input_height, target_megapixels
        )

        img_scale_nodes = _find_nodes_by_class(result, "ImageScaleToTotalPixels")
        if img_scale_nodes:
            for node_id, node in img_scale_nodes:
                if "megapixels" in node["inputs"]:
                    node["inputs"]["megapixels"] = target_megapixels

        resolution_calc_nodes = _find_nodes_by_class(result, "PixelResolutionCalculator")
        if resolution_calc_nodes:
            for node_id, node in resolution_calc_nodes:
                if "megapixels" in node["inputs"]:
                    node["inputs"]["megapixels"] = target_megapixels

        if "1" in result and "inputs" in result["1"]:
            if "resolution" in result["1"]["inputs"]:
                result["1"]["inputs"]["resolution"] = resolution

    seed = str(uuid.uuid4().int)[:18]
    sampler_nodes = _find_nodes_by_class(result, "KSampler")
    for node_id, node in sampler_nodes:
        if "seed" in node["inputs"]:
            node["inputs"]["seed"] = int(seed)

    if params:
        result = inject_params(result, params)

    return result
