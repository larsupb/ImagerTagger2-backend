import base64
import io
import json
import logging
import re

import PIL
from PIL import Image
from openai import OpenAI, APITimeoutError, APIConnectionError

from lib.tagging.ideogram_schema import normalize_caption, serialize_caption


def _scale_to_megapixels(image: Image.Image, megapixels: float = 0.5) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height
    target_pixels = 1_000_000 * megapixels
    target_height = int((target_pixels / aspect_ratio) ** 0.5)
    target_width = int(target_height * aspect_ratio)
    return image.resize((target_width, target_height), resample=Image.Resampling.NEAREST)

logger = logging.getLogger(__name__)

JSON_SCHEMA_INSTRUCTIONS = """
Return exactly one compact valid JSON object. No markdown. No commentary.

Schema:
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {"type": "obj", "desc": "..."},
      {"type": "text", "text": "...", "desc": "..."}
    ]
  }
}

Field guidance:
- high_level_description: one or two sentences summarizing the whole image.
- aesthetics: concise visual style keywords, e.g. "moody, cinematic, desaturated" or "warm, playful, vibrant".
- lighting: concrete light quality, source, and shadow behavior, e.g. "golden hour, rim light, dramatic shadows" or "bright afternoon sunlight, long soft shadows".
- photo: camera, lens, viewpoint, focus, and photographic traits for photos, e.g. "35mm, f/1.4, bokeh", "shallow depth of field, eye-level, 85mm lens", or "wide angle, f/8, long exposure".
- medium: use a compact medium label such as "photograph", "illustration", "3d_render", "painting", or "graphic_design".
- art_style: style and medium traits for non-photo captions, e.g. "flat vector illustration, bold outlines" or "flat vector design, generous whitespace, sans-serif typography".
- background: describe the environment, setting, distant scenery, surfaces, and atmosphere.
- elements desc: describe each subject/object with its visible appearance, clothing/materials, pose/action, and important props.

Rules:
- Include high_level_description, style_description, and compositional_deconstruction.
- compositional_deconstruction must contain background first, then elements.
- Use "photo" for photographic images, or replace it with "art_style" for non-photo artwork.
- Use exactly one of "photo" or "art_style".
- Do not include bbox values. Bboxes are added in a separate pass.
- Do not include color_palette fields.
- type is "obj" for normal subjects/objects and "text" only for literal visible text.
- Text elements must preserve the literal visible text exactly.
- A coherent subject is one element; do not split people, vehicles, plants, buildings, or products into parts.
- Put ground, sky, walls, distant scenery, and ambient environment into background.
- Put people, animals, vehicles, products, furniture, props, signs, and visible text into elements.
- Keep trigger tokens, names, identifiers, and stylized spelling exactly.
""".strip()

IMAGE_TO_JSON_SYSTEM = """
You inspect an image and produce an Ideogram 4 structured JSON caption.
The image is authoritative. Describe only what is visible.
""".strip()

IMAGE_TO_JSON_USER = """
Create an Ideogram 4 structured JSON caption for this image.
Do not reference any existing sidecar caption.
""".strip()

SYSTEM_PROMPT = f"""{IMAGE_TO_JSON_SYSTEM}

{JSON_SCHEMA_INSTRUCTIONS}
"""


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def extract_json(text: str) -> dict:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in model output: {raw[:200]!r}")
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse model JSON: {exc}") from exc
    return parsed


def _predict_ideogram_json(
    image_path: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> dict:
    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=base_url.rstrip("/") + "/",
        timeout=timeout,
        max_retries=1,
    )

    image = Image.open(image_path).convert("RGB")
    image = _scale_to_megapixels(image, 0.5)
    base64_image = _encode_image(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": IMAGE_TO_JSON_USER},
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("VLM returned empty response")

    raw = content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    parsed = extract_json(raw)
    return normalize_caption(parsed)


def generate_ideogram_caption(image_path: str) -> str:
    logger.info(f"Generating Ideogram JSON caption for: {image_path}")
    try:
        from app.config import vlm_endpoint_settings

        settings = vlm_endpoint_settings()
        api_key = settings.get("api_key", "")
        base_url = settings.get("base_url", "http://localhost:11434/v1")
        model = settings.get("model", "qwen2.5-vl-7b")
        timeout = settings.get("timeout", 120.0)

        caption_dict = _predict_ideogram_json(
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        return serialize_caption(caption_dict)

    except APITimeoutError as e:
        logger.error(f"Ideogram API timeout for {image_path}: {e}")
        return ""
    except APIConnectionError as e:
        logger.error(f"Ideogram API connection error for {image_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error generating Ideogram caption for {image_path}: {e}", exc_info=True)
        return ""