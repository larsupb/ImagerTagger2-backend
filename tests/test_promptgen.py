"""
Simple integration test for the promptgen service against the configured OpenAI-compatible API.

Run with:
    cd backend
    source .venv/bin/activate
    python -m pytest tests/test_promptgen.py -v -s
or directly:
    python tests/test_promptgen.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

import httpx
from openai import OpenAI

SETTINGS = {
    "api_key": os.environ.get("TEST_LLM_API_KEY", ""),
    "base_url": os.environ.get("TEST_LLM_BASE_URL", "http://localhost:11434/v1"),
    "model": os.environ.get("TEST_LLM_MODEL", "qwen3-vl:32b"),
}

EXAMPLE_CAPTIONS = [
    "Sarah stands confidently in front of a blurred city background, wearing a dark blazer and smiling warmly at the camera.",
    "A close-up shot of Sarah in profile, her hair pulled back, wearing a light summer dress, looking thoughtfully to the side.",
    "Sarah sits casually on a park bench, dressed in jeans and a striped shirt, laughing naturally in soft afternoon light.",
    "Medium shot of Sarah facing the camera directly, arms crossed, wearing a red jacket, against a plain white studio backdrop.",
    "Sarah walks towards the camera on a cobblestone street, wearing a long coat, captured from a low angle with shallow depth of field.",
]

SYSTEM_PROMPT = """You are an expert at creating detailed image generation prompts.
Based on the following example descriptions, create a new prompt that captures the
same style, mood, and visual element patterns. The prompt should be suitable for
AI image generation models like Midjourney, Stable Diffusion, etc.

Examples:
{examples}

Create a single, detailed prompt that follows the same pattern:"""


def test_openai_connection():
    """Test that we can connect to the API and get a response."""
    client = OpenAI(
        api_key=SETTINGS["api_key"] or "ollama",
        base_url=SETTINGS["base_url"],
        timeout=300.0,
        max_retries=0,
        http_client=httpx.Client(http2=False),
    )

    examples_text = "\n".join(f"- {c}" for c in EXAMPLE_CAPTIONS)
    system_msg = SYSTEM_PROMPT.format(examples=examples_text)
    user_msg = "Create a prompt in the same style."

    print(f"\nConnecting to: {SETTINGS['base_url']}")
    print(f"Model: {SETTINGS['model']}")
    print(f"Sending {len(EXAMPLE_CAPTIONS)} example captions...")

    response = client.chat.completions.create(
        model=SETTINGS["model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2048,
    )

    assert response.choices, "Response has no choices"
    import re

    result = re.sub(
        r"<think>.*?</think>",
        "",
        response.choices[0].message.content or "",
        flags=re.DOTALL,
    ).strip()
    assert result, "Response content is empty after stripping think tags"

    print(f"\nGenerated prompt:\n{result}")
    return result


def test_promptgen_service_logic():
    """Test the promptgen service generate_prompt function with a mock session."""
    import random

    class MockDB:
        def execute(self, query, params=()):
            class MockCursor:
                def fetchall(self):
                    return [
                        {"filename": f"image_{i:03d}.jpg", "content": caption}
                        for i, caption in enumerate(EXAMPLE_CAPTIONS)
                    ]

            return MockCursor()

    class MockDataset:
        pass

    class MockSession:
        dataset = MockDataset()
        db = MockDB()

    from unittest.mock import patch
    from app.services.promptgen_service import generate_prompt

    print("\nTesting promptgen service with mock session...")
    with patch(
        "app.services.promptgen_service.llm_endpoint_settings", return_value=SETTINGS
    ):
        result = generate_prompt(
            session=MockSession(),
            caption_type="tags",
            example_count=3,
            user_prompt="",
        )

    assert result and result.strip(), "Generated prompt is empty"
    print(f"\nGenerated prompt:\n{result}")
    return result


if __name__ == "__main__":
    print("=== Test 1: Direct OpenAI API connection ===")
    try:
        result = test_openai_connection()
        print("\n[PASS] Direct API connection works")
    except Exception as e:
        print(f"\n[FAIL] Direct API connection: {e}")

    print("\n=== Test 2: Promptgen service logic ===")
    try:
        result = test_promptgen_service_logic()
        print("\n[PASS] Promptgen service logic works")
    except Exception as e:
        print(f"\n[FAIL] Promptgen service logic: {e}")
