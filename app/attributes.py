import json
import re

ATTRIBUTES: dict[str, dict] = {
    "shot_type": {
        "applies_to": "all",
        "values": [
            "selfie", "headshot", "upper_body_portrait", "full_body_portrait",
            "environmental_portrait", "lifestyle", "candid_street", "group_shot",
            "couple_shot", "landscape", "aerial", "macro", "architecture",
            "interior", "product", "still_life", "food", "animal",
            "sports_action", "event",
        ],
    },
    "framing": {
        "applies_to": "all",
        "values": [
            "extreme_close_up", "close_up", "medium_close_up", "medium_shot",
            "cowboy_shot", "medium_wide", "full_shot", "wide_shot", "extreme_wide",
        ],
    },
    "camera_angle": {
        "applies_to": "person",
        "values": [
            "eye_level", "low_angle", "high_angle", "birds_eye", "worms_eye",
            "dutch_angle", "pov", "over_the_shoulder",
        ],
    },
    "subject_orientation": {
        "applies_to": "person",
        "values": [
            "frontal_looking_at_camera", "frontal_looking_away", "three_quarter",
            "profile", "back_view",
        ],
    },
    "pose": {
        "applies_to": "person",
        "values": [
            "standing", "walking", "running_jumping", "sitting", "sitting_on_floor",
            "kneeling", "crouching", "lying_reclining", "leaning", "dancing",
        ],
    },
    "medium": {
        "applies_to": "all",
        "values": ["photo", "illustration_anime", "3d_render", "screenshot"],
    },
    "setting": {
        "applies_to": "all",
        "values": [
            "studio_plain", "indoor_home", "indoor_public", "urban_outdoor",
            "nature_outdoor", "beach_water", "snow",
        ],
    },
    "lighting": {
        "applies_to": "all",
        "values": [
            "natural_soft", "direct_sunlight", "golden_hour", "blue_hour",
            "overcast", "window_light", "studio_strobe", "flash", "backlit_rim",
            "night_low_light", "neon",
        ],
    },
    "expression": {
        "applies_to": "person",
        "values": ["neutral", "smile", "laugh", "serious", "surprised", "sad", "angry"],
    },
    "quality_flags": {
        "applies_to": "all",
        "values": ["sharp", "blurry", "watermarked", "text_overlay", "very_low_res"],
    },
}

_ALIASES = {
    ("framing", "close_up"): ["closeup", "close up"],
    ("framing", "extreme_close_up"): ["ecu", "extreme close up"],
    ("framing", "medium_close_up"): ["mcu", "medium close up"],
    ("framing", "medium_shot"): ["ms", "medium shot"],
    ("framing", "medium_wide"): ["medium wide shot", "three quarter shot", "knees up"],
    ("framing", "wide_shot"): ["long shot", "ws"],
    ("framing", "extreme_wide"): ["extreme long shot", "ews"],
    ("camera_angle", "pov"): ["point of view"],
    ("camera_angle", "over_the_shoulder"): ["ots"],
    ("shot_type", "candid_street"): ["candid", "street", "street photography"],
}


def coerce_value(attribute: str, raw: str) -> str | None:
    spec = ATTRIBUTES.get(attribute)
    if spec is None or not isinstance(raw, str):
        return None
    candidate = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    if candidate in spec["values"]:
        return candidate
    for (attr, value), aliases in _ALIASES.items():
        if attr == attribute and candidate in [a.replace(" ", "_") for a in aliases]:
            return value
    return None


def validate_attributes(attributes: dict) -> tuple[dict, dict[str, str]]:
    valid: dict[str, str] = {}
    errors: dict[str, str] = {}
    for key, value in attributes.items():
        if key not in ATTRIBUTES:
            errors[key] = "unknown attribute"
        elif value is None:
            continue
        else:
            coerced = coerce_value(key, value)
            if coerced is None:
                errors[key] = str(value)
            else:
                valid[key] = coerced
    return valid, errors


def build_classify_prompt() -> str:
    all_attrs = [k for k, s in ATTRIBUTES.items() if s["applies_to"] == "all"]
    person_attrs = [k for k, s in ATTRIBUTES.items() if s["applies_to"] == "person"]

    def lines(keys: list[str]) -> str:
        return "\n".join(f"- {k}: one of {', '.join(ATTRIBUTES[k]['values'])}" for k in keys)

    return (
        "You are an image classifier for a photography dataset. Respond with ONLY a"
        " flat JSON object, no markdown, no prose. Keys and allowed values:\n"
        + lines(all_attrs)
        + "\nAlso answer \"has_person\": true if one or more people are visible,"
        " else false. When has_person is false, omit these keys:"
        " " + ", ".join(person_attrs)
        + "\n"
        + lines(person_attrs)
        + "\nOmit any key you cannot determine confidently. Example:"
        ' {"has_person": true, "shot_type": "selfie", "framing": "close_up",'
        ' "camera_angle": "eye_level", "pose": "standing"}'
    )


def parse_attributes_response(raw: str) -> tuple[dict, bool]:
    if not raw:
        return {}, False
    for pattern in (r"\{.*\}", r"\{.*?\}"):
        match = re.search(pattern, raw, re.DOTALL)
        if not match:
            return {}, False
        try:
            data = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue
    else:
        return {}, False
    if not isinstance(data, dict):
        return {}, False
    has_person = data.get("has_person") is True
    parsed, _ = validate_attributes(
        {
            k: v
            for k, v in data.items()
            if k != "has_person" and (has_person or ATTRIBUTES.get(k, {}).get("applies_to") != "person")
        }
    )
    return parsed, has_person
