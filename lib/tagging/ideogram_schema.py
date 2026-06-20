from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def default_caption() -> dict[str, Any]:
    return {
        "high_level_description": "",
        "style_description": {
            "aesthetics": "",
            "lighting": "",
            "photo": "",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "",
            "elements": [],
        },
    }


def normalize_palette(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        values: list[str] = []
        for raw in re.split(r"[,\s]+", value.strip()):
            if not raw:
                continue
            item = raw.upper()
            if HEX_COLOR_RE.match(item) and len(values) < limit:
                values.append(item)
        return values
    if not isinstance(value, list):
        return []
    colors: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        color = item.strip().upper()
        if HEX_COLOR_RE.match(color):
            colors.append(color)
        if len(colors) >= limit:
            break
    return colors


def normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        y1, x1, y2, x2 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    y1 = max(0, min(1000, y1))
    x1 = max(0, min(1000, x1))
    y2 = max(0, min(1000, y2))
    x2 = max(0, min(1000, x2))
    top, bottom = sorted((y1, y2))
    left, right = sorted((x1, x2))
    if top == bottom or left == right:
        return None
    return [top, left, bottom, right]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_caption(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return default_caption()

    caption: dict[str, Any] = {}
    caption["high_level_description"] = _as_str(data.get("high_level_description", ""))

    style_in = data.get("style_description")
    if isinstance(style_in, dict):
        has_art = "art_style" in style_in and "photo" not in style_in
        style: dict[str, Any] = {
            "aesthetics": _as_str(style_in.get("aesthetics", "")),
            "lighting": _as_str(style_in.get("lighting", "")),
        }
        if has_art:
            style["medium"] = _as_str(style_in.get("medium", "illustration")) or "illustration"
            style["art_style"] = _as_str(style_in.get("art_style", ""))
        else:
            style["photo"] = _as_str(style_in.get("photo", ""))
            style["medium"] = _as_str(style_in.get("medium", "photograph")) or "photograph"
        caption["style_description"] = style
    else:
        caption["style_description"] = deepcopy(default_caption()["style_description"])

    comp_in = data.get("compositional_deconstruction", {})
    elements_in = comp_in.get("elements", []) if isinstance(comp_in, dict) else []
    elements: list[dict[str, Any]] = []
    if isinstance(elements_in, list):
        for item in elements_in:
            if not isinstance(item, dict):
                continue
            element_type = item.get("type")
            if element_type not in ("obj", "text"):
                element_type = "text" if "text" in item else "obj"
            element: dict[str, Any] = {"type": element_type}
            bbox = normalize_bbox(item.get("bbox"))
            if bbox:
                element["bbox"] = bbox
            if element_type == "text":
                element["text"] = _as_str(item.get("text", ""))
            element["desc"] = _as_str(item.get("desc", item.get("description", "")))
            elements.append(element)

    caption["compositional_deconstruction"] = {
        "background": _as_str(comp_in.get("background", "")) if isinstance(comp_in, dict) else "",
        "elements": elements,
    }
    return caption


def serialize_caption(data: dict[str, Any]) -> str:
    return json.dumps(normalize_caption(data), separators=(",", ":"), ensure_ascii=False)


def parse_caption_text(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Caption JSON must be an object.")
    return normalize_caption(parsed)