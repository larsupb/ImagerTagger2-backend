"""
Test for the ComfyUI service module.

Run with:
    cd backend
    python -m pytest tests/test_comfyui_service.py -v -s
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import_comfyui_service():
    """Test that the comfyui_service module can be imported."""
    from app.services import comfyui_service

    assert comfyui_service is not None


def test_upscale_with_comfyui_function_exists():
    """Test that upscale_with_comfyui function exists."""
    from app.services.comfyui_service import upscale_with_comfyui

    assert callable(upscale_with_comfyui)


if __name__ == "__main__":
    import traceback

    tests = [
        test_import_comfyui_service,
        test_upscale_with_comfyui_function_exists,
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
