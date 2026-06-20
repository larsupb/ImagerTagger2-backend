import pytest
import json
from lib.tagging.ideogram_schema import (
    default_caption,
    normalize_caption,
    serialize_caption,
    parse_caption_text,
)


def test_default_caption_has_required_keys():
    d = default_caption()
    assert "high_level_description" in d
    assert "style_description" in d
    assert "compositional_deconstruction" in d


def test_normalize_caption_returns_dict():
    result = normalize_caption({"high_level_description": "test"})
    assert isinstance(result, dict)


def test_normalize_caption_defaults_on_non_dict():
    result = normalize_caption("not a dict")
    assert isinstance(result, dict)
    assert result["high_level_description"] == ""


def test_serialize_caption_returns_valid_json():
    d = default_caption()
    d["high_level_description"] = "a test image"
    result = serialize_caption(d)
    parsed = json.loads(result)
    assert parsed["high_level_description"] == "a test image"


def test_parse_caption_text_parses_json_string():
    d = default_caption()
    d["high_level_description"] = "parsed caption"
    text = json.dumps(d)
    result = parse_caption_text(text)
    assert result["high_level_description"] == "parsed caption"


def test_normalize_caption_extracts_style_fields():
    d = {
        "high_level_description": "sunset over mountains",
        "style_description": {
            "aesthetics": "warm, cinematic",
            "lighting": "golden hour",
            "medium": "photograph",
            "photo": "35mm, f/1.8",
            "color_palette": ["#FF6B00", "#FFD700"],
        },
    }
    result = normalize_caption(d)
    assert result["style_description"]["aesthetics"] == "warm, cinematic"
    assert result["style_description"]["lighting"] == "golden hour"
    assert result["style_description"]["photo"] == "35mm, f/1.8"
    assert "color_palette" not in result["style_description"]


def test_normalize_caption_extracts_elements():
    d = {
        "high_level_description": "a dog in a park",
        "compositional_deconstruction": {
            "background": "green park",
            "elements": [
                {"type": "obj", "desc": "brown dog"},
                {"type": "text", "text": "hello", "desc": "sign"},
            ]
        }
    }
    result = normalize_caption(d)
    elements = result["compositional_deconstruction"]["elements"]
    assert len(elements) == 2
    assert elements[0]["desc"] == "brown dog"
    assert elements[1]["text"] == "hello"


def test_normalize_caption_normalizes_bbox():
    d = {
        "high_level_description": "image with bboxes",
        "compositional_deconstruction": {
            "background": "",
            "elements": [
                {"type": "obj", "bbox": [100, 200, 500, 600]},
            ]
        }
    }
    result = normalize_caption(d)
    element = result["compositional_deconstruction"]["elements"][0]
    assert element.get("bbox") == [100, 200, 500, 600]


from lib.tagging.ideogram_tagger import (
    extract_json,
    generate_ideogram_caption,
)


def test_extract_json_strips_markdown_fence():
    raw = '```json\n{"high_level_description": "test"}\n```'
    result = extract_json(raw)
    assert result["high_level_description"] == "test"


def test_extract_json_finds_json_without_fence():
    raw = 'Here is some text before {"high_level_description": "test"} and after'
    result = extract_json(raw)
    assert result["high_level_description"] == "test"


def test_extract_json_finds_nested_braces():
    raw = '{"high_level_description": "test", "nested": {"a": 1}}'
    result = extract_json(raw)
    assert result["nested"]["a"] == 1





def test_extract_json_raises_on_no_json():
    raw = 'no json here at all'
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json(raw)


def test_generate_ideogram_caption_returns_json_string(tmp_path, monkeypatch):
    from PIL import Image
    from lib.tagging.ideogram_schema import default_caption

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    mock_caption = default_caption()
    mock_caption["high_level_description"] = "a test image"

    class FakePredict:
        call_count = 0
        def __call__(self, **kwargs):
            FakePredict.call_count += 1
            return mock_caption

    fake = FakePredict()
    monkeypatch.setattr("lib.tagging.ideogram_tagger._predict_ideogram_json", fake)

    result = generate_ideogram_caption(str(img_path))

    assert result.startswith("{")
    assert "a test image" in result
    assert FakePredict.call_count == 1


def test_generate_ideogram_caption_returns_empty_on_error(tmp_path, monkeypatch):
    from PIL import Image

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    def fake_error(**kwargs):
        raise Exception("API error")

    monkeypatch.setattr("lib.tagging.ideogram_tagger._predict_ideogram_json", fake_error)

    result = generate_ideogram_caption(str(img_path))

    assert result == ""