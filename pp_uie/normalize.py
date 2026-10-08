"""Convert PP-UIE's relation tree to a stable business JSON shape."""

from typing import Any


def normalize_scene_output(raw: dict[str, Any], requested_schema: Any) -> dict[str, Any]:
    del requested_schema
    scenes = []
    for item in raw.get("拍照公式场景", []):
        parameters: dict[str, list[str]] = {}
        for field, values in item.get("relations", {}).items():
            texts = [value["text"] for value in values if isinstance(value, dict) and "text" in value]
            if texts:
                parameters[field] = texts
        scenes.append({
            "scene": item.get("text", ""),
            "parameters": parameters,
            "unlabeled_parameters": [],
        })
    return {"scenes": scenes}

