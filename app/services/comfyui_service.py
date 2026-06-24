import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PIL import Image
from app.sessions import Session
from app.config import comfyui_settings
from lib.comfyui.client import ComfyClient

from lib.upscaling.util import scale_to_megapixels


def upscale_with_comfyui(
    session: Session,
    index: int,
    target_megapixels: float = 2.0,
    workflow_file: str = None,
    upscaler_name: str = None,
    cancel_check=None,
) -> str:
    """Upscale image using ComfyUI workflow."""
    settings = comfyui_settings()
    url = settings.get("url", "http://localhost:8188")
    api_token = settings.get("api_token", "")

    if not url:
        raise ValueError("ComfyUI URL not configured")

    if workflow_file:
        workflow_file = os.path.join(
            os.path.dirname(__file__), "..", "..", workflow_file
        )

    fparams = settings.get("comfyui_fparams", {})
    workflow_params = fparams.get(upscaler_name, {}) if upscaler_name else {}

    client = ComfyClient(url, api_token)

    ds = session.dataset
    item = ds.get_item(index)

    img = Image.open(item.media_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    filename = client.upload_image(img)
    img_width, img_height = img.size

    prompt_id = client.run_workflow(
        filename,
        target_megapixels=target_megapixels,
        input_width=img_width,
        input_height=img_height,
        workflow_file=workflow_file,
        params=workflow_params,
    )

    result = client.wait_for_completion(prompt_id)

    import logging

    logging.warning(f"ComfyUI result keys: {result.keys()}")
    logging.warning(f"ComfyUI result: {result}")

    upscaled = None
    outputs = result.get("outputs", {})
    if not outputs:
        outputs = result
    logging.warning(f"Outputs: {outputs}")
    for node_id, node_output in outputs.items():
        logging.warning(f"Node {node_id}: {node_output}")
        if isinstance(node_output, dict):
            if "images" in node_output:
                for img_data in node_output["images"]:
                    output_filename = img_data.get("filename")
                    if output_filename:
                        upscaled = client.download_output(output_filename)
                        break
                break

    if upscaled is None:
        raise ValueError("Could not find output image in ComfyUI result")

    upscaled = scale_to_megapixels(upscaled, target_megapixels)

    session.upscaled_image = upscaled
    session.upscaled_index = index

    return "ok"
