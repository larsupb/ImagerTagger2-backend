"""
Test for the ComfyUI client module.

Run with:
    cd backend
    python -m pytest tests/test_comfyui.py -v -s
or directly:
    python tests/test_comfyui.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import_comfyui_module():
    """Test that the comfyui module can be imported."""
    from lib.comfyui import workflow, client

    assert workflow is not None
    assert client is not None


def test_workflow_loads_from_file():
    """Test that get_workflow() loads the default workflow JSON file."""
    from lib.comfyui.workflow import get_workflow

    workflow = get_workflow()

    assert isinstance(workflow, dict)
    assert len(workflow) > 0


def test_inject_image_function():
    """Test inject_image helper function."""
    from lib.comfyui.workflow import inject_image, get_workflow

    result = inject_image(get_workflow(), "test_image.png")

    assert isinstance(result, dict)


def test_comfy_client_init():
    """Test ComfyClient can be initialized."""
    from lib.comfyui.client import ComfyClient

    client = ComfyClient(url="http://localhost:8188", api_token="test_token")

    assert client.url == "http://localhost:8188"
    assert client.api_token == "test_token"


def test_comfy_client_upload_image_method():
    """Test ComfyClient has upload_image method."""
    from lib.comfyui.client import ComfyClient

    client = ComfyClient(url="http://localhost:8188", api_token="test_token")

    assert hasattr(client, "upload_image")
    assert callable(client.upload_image)


def test_comfy_client_run_workflow_method():
    """Test ComfyClient has run_workflow method."""
    from lib.comfyui.client import ComfyClient

    client = ComfyClient(url="http://localhost:8188", api_token="test_token")

    assert hasattr(client, "run_workflow")
    assert callable(client.run_workflow)


def test_comfy_client_wait_for_completion_method():
    """Test ComfyClient has wait_for_completion method."""
    from lib.comfyui.client import ComfyClient

    client = ComfyClient(url="http://localhost:8188", api_token="test_token")

    assert hasattr(client, "wait_for_completion")
    assert callable(client.wait_for_completion)


def test_comfy_client_download_output_method():
    """Test ComfyClient has download_output method."""
    from lib.comfyui.client import ComfyClient

    client = ComfyClient(url="http://localhost:8188", api_token="test_token")

    assert hasattr(client, "download_output")
    assert callable(client.download_output)


def test_inject_params():
    """Test that params are injected into the correct workflow nodes."""
    from lib.comfyui.workflow import inject_image, inject_params, get_workflow

    workflow = get_workflow("image_enhance.json")
    injections = [
        {"node_id": "158:77", "input_name": "denoise", "value": 0.5},
        {"node_id": "158:77", "input_name": "steps", "value": 10},
        {"node_id": "158:77", "input_name": "seed", "value": 42},
    ]
    result = inject_image(workflow, "test.jpg", target_megapixels=1.0, input_width=512, input_height=512)

    params_result = inject_params(result, injections)

    sampler = params_result["158:77"]
    assert sampler.get("class_type") == "KSampler"
    assert sampler["inputs"]["denoise"] == 0.5, f"denoise={sampler['inputs']['denoise']}, expected 0.5"
    assert sampler["inputs"]["steps"] == 10, f"steps={sampler['inputs']['steps']}, expected 10"
    assert sampler["inputs"]["seed"] == 42, f"seed={sampler['inputs']['seed']}, expected 42"


def test_inject_params_text():
    """Text params target individual PrimitiveStringMultiline nodes that share
    the 'value' input name, so each must be set independently."""
    from lib.comfyui.workflow import inject_image, inject_params, get_workflow

    workflow = get_workflow("image_enhance.json")
    injections = [
        {"node_id": "150", "input_name": "value", "value": "enhance details"},
        {"node_id": "151", "input_name": "value", "value": "sharpen features"},
    ]
    result = inject_image(workflow, "test.jpg", target_megapixels=1.0, input_width=512, input_height=512)

    params_result = inject_params(result, injections)

    assert params_result["150"]["inputs"]["value"] == "enhance details"
    assert params_result["151"]["inputs"]["value"] == "sharpen features"
    # 152 was not targeted, so it keeps its workflow-file default
    assert params_result["152"]["inputs"]["value"] != "enhance details"


if __name__ == "__main__":
    import traceback

    tests = [
        test_import_comfyui_module,
        test_workflow_loads_from_file,
        test_inject_image_function,
        test_comfy_client_init,
        test_comfy_client_upload_image_method,
        test_comfy_client_run_workflow_method,
        test_comfy_client_wait_for_completion_method,
        test_comfy_client_download_output_method,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n=== {test.__name__} ===")
        try:
            test()
            print("[PASS]")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
