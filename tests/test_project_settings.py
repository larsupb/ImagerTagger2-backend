from app.config import florence_settings, vlm_endpoint_settings


def test_helpers_prefer_state_dict_config():
    cfg = {
        "florence_settings": {"prompt": "proj florence"},
        "vlm_endpoint": {"prompt": "proj vlm", "base_url": "http://proj:1/v1"},
    }
    assert florence_settings({"config": cfg})["prompt"] == "proj florence"
    vlm = vlm_endpoint_settings({"config": cfg})
    assert vlm["prompt"] == "proj vlm"
    assert vlm["base_url"] == "http://proj:1/v1"


def test_helpers_fall_back_to_global_without_config():
    assert vlm_endpoint_settings()["base_url"]
    assert vlm_endpoint_settings({"dataset": None})["base_url"]
    assert florence_settings({})["prompt"]
