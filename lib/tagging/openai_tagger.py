"""
OpenAI-compatible VLM Tagger

Supports any OpenAI-compatible API endpoint including:
- OpenAI (api.openai.com)
- Ollama (localhost:11434)
- OpenRouter, Together.ai, etc.
"""

import base64
import io
import logging
import time

from PIL import Image
from openai import OpenAI, APITimeoutError, APIConnectionError

from lib.upscaling.util import scale_to_megapixels

logger = logging.getLogger(__name__)


class VLMTagger:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3-vl:32b",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        """
        Initialize the OpenAI-compatible tagger.

        Args:
            api_key: API key for the service (can be empty for local Ollama)
            base_url: Base URL for the API endpoint
            model: Model name to use for captioning
            timeout: Timeout in seconds for API requests (default: 120s)
            max_retries: Maximum number of retries for failed requests (default: 2)
        """
        logger.info(
            f"Initializing OpenAI-compatible tagger: model={model}, base_url={base_url}, timeout={timeout}s, max_retries={max_retries}"
        )
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key else "ollama",  # Ollama doesn't need a real key
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 JPEG."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        return base64_image

    def predict(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        target_megapixels: float = 1.0,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a caption for the given image.

        Args:
            image_path: Path to the image file
            prompt: The prompt/instruction for the VLM
            target_megapixels: Target size in megapixels for image resizing (default 1.0)
            temperature: Sampling temperature; lower is more grounded (default 0.3)
            top_p: Nucleus sampling probability mass (default 0.9)
            max_tokens: Maximum tokens to generate (default 512)

        Returns:
            Generated caption string
        """
        logger.info(f"Generating OpenAI caption for: {image_path}")
        start_time = time.time()

        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = scale_to_megapixels(image, target_megapixels)
        logger.debug(f"Image resized from {original_size} to {image.size}")

        encode_start = time.time()
        base64_image = self._encode_image(image)
        encode_time = time.time() - encode_start
        logger.debug(f"Image encoded to base64 in {encode_time:.3f}s")

        api_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            api_time = time.time() - api_start
            logger.info(f"API request completed in {api_time:.2f}s")
        except APITimeoutError as e:
            api_time = time.time() - api_start
            logger.error(
                f"API request timed out after {api_time:.2f}s for {image_path}: {str(e)}"
            )
            logger.warning(
                f"Consider increasing timeout (current: {self.client.timeout}s) or reducing image resolution"
            )
            return ""
        except APIConnectionError as e:
            api_time = time.time() - api_start
            logger.error(
                f"API connection error after {api_time:.2f}s for {image_path}: {str(e)}"
            )
            logger.warning(
                f"Check if the API endpoint is accessible: {self.client.base_url}"
            )
            return ""
        except Exception as e:
            api_time = time.time() - api_start
            logger.error(
                f"Unexpected API error after {api_time:.2f}s for {image_path}: {str(e)}",
                exc_info=True,
            )
            return ""

        # Check response validity
        if not response.choices:
            logger.warning(f"OpenAI API returned empty choices list for {image_path}")
            logger.debug(f"Full response object: {response}")
            return ""

        result = response.choices[0].message.content

        # Check for empty or None result
        if result is None or (isinstance(result, str) and not result.strip()):
            logger.warning(f"VLM API returned empty caption for {image_path}")
            logger.debug(
                f"Response details - model: {response.model}, finish_reason: {response.choices[0].finish_reason}"
            )
            logger.debug(f"Full response object: {response}")

            # Check if there's an error or refusal
            if (
                hasattr(response.choices[0].message, "refusal")
                and response.choices[0].message.refusal
            ):
                logger.error(
                    f"API refused request: {response.choices[0].message.refusal}"
                )

            return ""

        total_time = time.time() - start_time
        logger.info(
            f"VLM prediction complete in {total_time:.2f}s: '{result[:100]}...'"
        )

        return result


def generate_vlm_caption(image_path: str, settings: dict) -> str:
    """
    Generate a caption using an OpenAI-compatible VLM API.

    Args:
        image_path: Path to the image file
        settings: Dictionary containing vlm_endpoint with keys:
                  - api_key: API key for the service
                  - base_url: Base URL for the API
                  - model: Model name
                  - prompt: Prompt for caption generation
                  - timeout: Request timeout in seconds (optional, default: 120)
                  - max_retries: Maximum retry attempts (optional, default: 2)
                  - target_megapixels: Resize target in MP (optional, default: 1.0)
                  - temperature: Sampling temperature (optional, default: 0.3)
                  - top_p: Nucleus sampling probability mass (optional, default: 0.9)
                  - max_tokens: Maximum tokens to generate (optional, default: 512)

    Returns:
        Generated caption string
    """
    logger.info(f"Generating OpenAI-compatible caption for: {image_path}")
    logger.debug(
        f"Settings: base_url={settings.get('base_url', 'default')}, model={settings.get('model', 'default')}"
    )

    try:
        api_key = settings.get("api_key", "")
        base_url = settings.get("base_url", "http://localhost:11434/v1")
        model = settings.get("model", "qwen3-vl:32b")
        prompt = settings.get("prompt", "Describe this image in detail.")
        timeout = settings.get("timeout", 120.0)
        max_retries = settings.get("max_retries", 2)
        target_megapixels = settings.get("target_megapixels", 1.0)
        temperature = settings.get("temperature", 0.3)
        top_p = settings.get("top_p", 0.9)
        max_tokens = settings.get("max_tokens", 512)

        tagger = VLMTagger(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
        caption = tagger.predict(
            image_path,
            prompt=prompt,
            target_megapixels=target_megapixels,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return caption
    except Exception as e:
        logger.error(
            f"Error generating OpenAI caption for {image_path}: {str(e)}", exc_info=True
        )
        raise
