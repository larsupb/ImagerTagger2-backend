from app.attributes import (
    ATTRIBUTES,
    coerce_value,
    parse_attributes_response,
    validate_attributes,
)


def test_vocabularies_match_spec():
    assert ATTRIBUTES["shot_type"]["applies_to"] == "all"
    assert "selfie" in ATTRIBUTES["shot_type"]["values"]
    assert ATTRIBUTES["camera_angle"]["applies_to"] == "person"
    assert ATTRIBUTES["expression"]["values"] == [
        "neutral", "smile", "laugh", "serious", "surprised", "sad", "angry",
    ]
    assert len(ATTRIBUTES) == 10


def test_coerce_value_normalizes():
    assert coerce_value("shot_type", "Selfie") == "selfie"
    assert coerce_value("framing", "Medium Wide") == "medium_wide"
    assert coerce_value("pose", " Running  Jumping ") == "running_jumping"
    assert coerce_value("shot_type", "cats_and_dogs") is None


def test_coerce_value_aliases():
    assert coerce_value("framing", "closeup") == "close_up"
    assert coerce_value("camera_angle", "ots") == "over_the_shoulder"
    assert coerce_value("shot_type", "candid") == "candid_street"
    assert coerce_value("medium", "closeup") is None


def test_validate_attributes():
    valid, errors = validate_attributes(
        {"shot_type": "selfie", "framing": "blurry", "unknown_attr": "x"}
    )
    assert valid == {"shot_type": "selfie"}
    assert errors["framing"] == "blurry"
    assert "unknown_attr" in errors


def test_parse_plain_json():
    parsed, has_person = parse_attributes_response(
        '{"has_person": true, "shot_type": "headshot", "pose": "standing",'
        ' "camera_angle": "low angle"}'
    )
    assert parsed == {
        "shot_type": "headshot", "pose": "standing", "camera_angle": "low_angle",
    }
    assert has_person is True


def test_parse_prose_wrapped_and_junk():
    parsed, has_person = parse_attributes_response(
        'Sure! {"has_person": false, "shot_type": "landscape",'
        ' "pose": "dancing", "medium": "photo"} hope that helps'
    )
    # pose ignored: has_person false -> person-only attributes dropped
    assert parsed == {"shot_type": "landscape", "medium": "photo"}
    assert has_person is False


def test_parse_trailing_brace_block():
    parsed, has_person = parse_attributes_response(
        'answer {"has_person": false, "medium": "photo"} more {x} here'
    )
    assert parsed == {"medium": "photo"}
    assert has_person is False


def test_parse_out_of_vocab_and_no_json():
    parsed, _ = parse_attributes_response(
        '{"has_person": true, "shot_type": "vibes", "framing": "close-up"}'
    )
    assert parsed == {"framing": "close_up"}
    parsed, _ = parse_attributes_response("the image shows a cat")
    assert parsed == {}
