"""
Integration test for the OpenAITagger service against the configured OpenAI-compatible API.

Run with:
    cd backend
    source .venv/bin/activate
    python -m pytest tests/test_openai_tagger.py -v -s
or directly:
    python tests/test_openai_tagger.py
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

from lib.tagging.openai_tagger import VLMTagger

SETTINGS = {
    "api_key": os.environ.get("TEST_OPENAI_API_KEY", ""),
    "base_url": os.environ.get("TEST_OPENAI_BASE_URL", "http://localhost:11434/v1"),
    "model": os.environ.get("TEST_OPENAI_MODEL", "qwen3-vl:32b"),
}

TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "Snow_Leopard_on_Snowy_Rocks.jpg"
)


def test_vlm_tagger():
    """Test that VLMTagger can generate a caption for the test image."""
    tagger = VLMTagger(
        api_key=SETTINGS["api_key"],
        base_url=SETTINGS["base_url"],
        model=SETTINGS["model"],
        timeout=300.0,
    )

    print(f"\nConnecting to: {SETTINGS['base_url']}")
    print(f"Model: {SETTINGS['model']}")
    print(f"Image: {TEST_IMAGE_PATH}")

    caption = tagger.predict(TEST_IMAGE_PATH)

    assert caption, "Caption is empty"
    assert isinstance(caption, str), f"Caption is not a string: {type(caption)}"
    assert len(caption.strip()) > 0, "Caption is only whitespace"

    print(f"\nGenerated caption:\n{caption}")
    return caption


if __name__ == "__main__":
    print("=== Test: VLMTagger generates caption for snow leopard image ===")
    try:
        result = test_vlm_tagger()
        print("\n[PASS] VLMTagger successfully generated caption")
    except Exception as e:
        print(f"\n[FAIL] VLMTagger: {e}")
        import traceback

        traceback.print_exc()
