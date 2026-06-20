import logging
import random
import re
import httpx
from openai import OpenAI

from app.sessions import Session
from app.db.repository import CaptionRepository
from app.config import llm_endpoint_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at creating detailed image generation prompts. 
Based on the following example descriptions, create a new prompt that captures the 
same style, mood, and visual element patterns. The prompt should be suitable for 
AI image generation models like Midjourney, Stable Diffusion, etc.

Examples:
{examples}

Create a single, detailed prompt that follows the same pattern:"""


def generate_prompt(
    session: Session, caption_type: str, example_count: int, user_prompt: str
) -> str:
    """Generate an image prompt based on caption examples.

    Args:
        session: Current session
        caption_type: The caption type to fetch examples from
        example_count: Number of examples to use
        user_prompt: Additional user instructions

    Returns:
        Generated prompt string
    """
    logger.info(
        f"generate_prompt called: caption_type={caption_type}, example_count={example_count}"
    )
    logger.info(f"session.dataset: {session.dataset}")

    if session.dataset is None:
        raise ValueError("No dataset loaded in session")

    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    logger.info(f"Found {len(rows)} rows for caption_type={caption_type}")

    non_empty = [r for r in rows if r.get("content")]
    logger.info(f"Found {len(non_empty)} non-empty captions")

    if not non_empty:
        raise ValueError(f"No captions found for type: {caption_type}")

    examples = random.sample(non_empty, min(example_count, len(non_empty)))
    formatted_example = "\n".join(f"- {r['content']}" for r in examples)

    system_msg = SYSTEM_PROMPT.format(examples=formatted_example)
    user_msg = (
        user_prompt.strip()
        if user_prompt.strip()
        else "Create a prompt in the same style."
    )

    settings = llm_endpoint_settings()
    logger.info(
        f"LLM endpoint: base_url={settings.get('base_url')}, model={settings.get('model')}"
    )

    timeout = max(settings.get("timeout", 180.0), 180.0)
    logger.info(f"Using timeout: {timeout}")

    client = OpenAI(
        api_key=settings.get("api_key", "") or "ollama",
        base_url=settings.get("base_url", "http://localhost:11434/v1"),
        timeout=timeout,
        max_retries=settings.get("max_retries", 2),
        http_client=httpx.Client(http2=False),
    )

    response = client.chat.completions.create(
        model=settings.get("model", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2048,
    )

    if not response.choices:
        raise ValueError("Empty response from OpenAI")

    content = response.choices[0].message.content or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content
