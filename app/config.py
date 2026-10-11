import json
import os
from typing import Optional

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "settings.json")

DEFAULTS = {
    "ignore_list": ["masklabel"],
    "upscaler": "NMKD_Siax_200k_4x",
    "upscale_target_megapixels": 2.0,
    "default_tagger": "florence",
    "tagger_instruction": "A descriptive caption for this image:\n",
    "combo_taggers": ["florence", "wd14"],
    "florence_settings": {"prompt": "<DETAILED_CAPTION>"},
    "rembg": {"model": "u2net_human_seg"},
    "vlm_endpoint": {
        "api_key": "",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2-vl:7b",
        "prompt": "Describe the image in continuous text.",
        "target_megapixels": 1.0,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 512,
    },
    "llm_endpoint": {
        "api_key": "",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:32b",
    },
    "embedding_endpoint": {
        "api_key": "",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3-embedding-8b",
        "timeout": 120,
    },
    "comfyui": {
        "url": "http://localhost:8188",
        "api_token": "",
        "workflow_file": "",
    },
    "comfyui_fparams": {},
    "ideogram_settings": {
        "prompt": None,
    },
}


def read_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w") as f:
            json.dump({}, f)
        return dict(DEFAULTS)
    with open(SETTINGS_FILE, "r") as f:
        data = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def save_settings(settings: dict):
    serializable = {
        k: v
        for k, v in settings.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


def update_setting(key: str, value) -> dict:
    settings = read_settings()
    settings[key] = value
    save_settings(settings)
    return settings


def get_setting(key: str, default=None):
    settings = read_settings()
    return settings.get(key, default)


def merge_settings(project_overrides: Optional[dict] = None) -> dict:
    """Merge global settings with optional project-specific overrides."""
    global_settings = read_settings()
    if not project_overrides:
        return global_settings
    merged = dict(global_settings)
    _deep_merge(merged, project_overrides)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override dict into base dict in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _settings_source(state_dict: dict = None) -> dict:
    if state_dict:
        config = state_dict.get("config")
        if isinstance(config, dict) and config:
            return config
    return read_settings()


def florence_settings(state_dict: dict = None) -> dict:
    return _settings_source(state_dict).get("florence_settings", DEFAULTS["florence_settings"])


def vlm_endpoint_settings(state_dict: dict = None) -> dict:
    return _settings_source(state_dict).get("vlm_endpoint", DEFAULTS["vlm_endpoint"])


def ideogram_settings(state_dict=None) -> dict:
    return _settings_source(state_dict).get("ideogram_settings", {"prompt": None})


OVERRIDABLE_KEYS = [
    "florence_settings.prompt",
    "vlm_endpoint.prompt",
    "upscaler",
    "upscale_target_megapixels",
    "rembg.model",
]


def flatten_overridable(settings: dict) -> dict:
    flat = {}
    for path in OVERRIDABLE_KEYS:
        parent, _, leaf = path.partition(".")
        if leaf:
            section = settings.get(parent)
            flat[path] = section.get(leaf) if isinstance(section, dict) else None
        else:
            flat[path] = settings.get(path)
    return flat


def llm_endpoint_settings() -> dict:
    """Get LLM endpoint settings for PromptGen."""
    return read_settings().get("llm_endpoint", DEFAULTS["llm_endpoint"])


def embedding_endpoint_settings() -> dict:
    """Get OpenAI-compatible embedding endpoint settings."""
    return read_settings().get("embedding_endpoint", DEFAULTS["embedding_endpoint"])


def combo_taggers(state_dict: dict = None) -> list:
    """Get list of taggers for combo mode."""
    return read_settings().get("combo_taggers", DEFAULTS["combo_taggers"])


def comfyui_settings(state_dict: dict = None) -> dict:
    """Get ComfyUI settings."""
    return read_settings().get(
        "comfyui", {"url": "http://localhost:8188", "api_token": ""}
    )
