import json
import os

import pytest

UPSCALERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upscalers.json")


@pytest.fixture(scope="module")
def upscalers() -> list[dict]:
    with open(UPSCALERS_PATH) as f:
        return json.load(f)


def _select_params(upscalers: list[dict]):
    for upscaler in upscalers:
        for param in upscaler.get("params") or []:
            if param.get("options"):
                yield upscaler["name"], param


def test_select_param_options_are_unique(upscalers):
    for name, param in _select_params(upscalers):
        options = param["options"]
        duplicates = sorted({o for o in options if options.count(o) > 1})
        assert not duplicates, f"{name}.{param['key']} has duplicate options: {duplicates}"


def test_select_param_options_have_no_stray_whitespace(upscalers):
    for name, param in _select_params(upscalers):
        for option in param["options"]:
            assert option == option.strip() and option, (
                f"{name}.{param['key']} option {option!r} has stray whitespace"
            )


def test_select_param_default_is_one_of_its_options(upscalers):
    for name, param in _select_params(upscalers):
        assert param["default"] in param["options"], (
            f"{name}.{param['key']} default {param['default']!r} is not in options"
        )
