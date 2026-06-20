import logging
import os
import time

import PIL
import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from huggingface_hub import snapshot_download

from PIL import Image
from lib.tagging.joytag.joytag_model import VisionModel

logger = logging.getLogger(__name__)

THRESHOLD = 0.4
MODEL_PATH = 'models/taggers/joytag'


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

    return image_tensor


@torch.no_grad()
def predict(image: Image.Image):
    logger.info("Starting JoyTag prediction")
    start_time = time.time()

    # Download model if needed
    download_start = time.time()
    snapshot_download('fancyfeast/joytag', local_dir=MODEL_PATH)
    download_time = time.time() - download_start
    if download_time > 1.0:
        logger.info(f"Model download/check completed in {download_time:.2f}s")

    # Load and prepare model
    load_start = time.time()
    model = VisionModel.load_model(MODEL_PATH)
    model.eval()
    model = model.to('cuda')
    load_time = time.time() - load_start
    logger.info(f"Model loaded and moved to CUDA in {load_time:.2f}s")

    # Prepare image
    prep_start = time.time()
    image_tensor = prepare_image(image, model.image_size)
    batch = {
        'image': image_tensor.unsqueeze(0).to('cuda'),
    }
    prep_time = time.time() - prep_start
    logger.debug(f"Image prepared in {prep_time:.3f}s")

    # Run inference
    infer_start = time.time()
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        preds = model(batch)
        tag_preds = preds['tags'].sigmoid().cpu()
    infer_time = time.time() - infer_start
    logger.info(f"Inference completed in {infer_time:.3f}s")

    # Load top tags and filter
    with open(os.path.join(MODEL_PATH, 'top_tags.txt'), 'r') as f:
        top_tags = [line.strip() for line in f.readlines() if line.strip()]

    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    tag_string = ', '.join(predicted_tags)

    total_time = time.time() - start_time
    logger.info(f"JoyTag prediction complete: {len(predicted_tags)} tags generated in {total_time:.2f}s total")

    return tag_string, scores


def generate_joytag_caption(image_path: str):
    logger.info(f"Generating JoyTag caption for: {image_path}")
    try:
        image = PIL.Image.open(image_path)
        logger.debug(f"Image loaded: size={image.size}, mode={image.mode}")
        tag_string, scores = predict(image)
        logger.debug(f"Top 5 scores: {sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return tag_string
    except Exception as e:
        logger.error(f"Error generating JoyTag caption for {image_path}: {str(e)}", exc_info=True)
        raise
