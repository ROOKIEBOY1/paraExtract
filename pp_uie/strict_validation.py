"""Evidence-backed validation for the supported Pocket camera parameters."""

from __future__ import annotations

import re
from typing import Any, Sequence


ALLOWED_FIELDS = (
    "白平衡", "APP美颜", "锐度", "去噪", "纹理", "分辨率", "帧率",
    "胶片影调", "滤镜浓度", "色彩模式", "感光度", "曝光补偿", "快门速度",
)

EN_TO_CN = {
    "White Balance": "白平衡", "WhiteBalance": "白平衡",
    "APP Beauty": "APP美颜", "APPBeauty": "APP美颜", "Beauty": "APP美颜",
    "Sharpness": "锐度", "Noise Reduction": "去噪", "NoiseReduction": "去噪",
    "Denoise": "去噪", "Texture": "纹理", "Resolution": "分辨率",
    "Frame Rate": "帧率", "FrameRate": "帧率", "Film Look": "胶片影调",
    "FilmLook": "胶片影调", "Filter Intensity": "滤镜浓度",
    "FilterIntensity": "滤镜浓度", "Color Mode": "色彩模式",
    "ColorMode": "色彩模式", "ISO Sensitivity": "感光度",
    "ISOSensitivity": "感光度", "ISO": "感光度",
    "Exposure Compensation": "曝光补偿", "ExposureCompensation": "曝光补偿",
    "EV": "曝光补偿", "Shutter Speed": "快门速度", "ShutterSpeed": "快门速度",
}

_CN_ALIASES = {
    "曝光": "曝光补偿", "机内曝光": "曝光补偿", "感光度": "感光度",
    "色彩": "色彩模式", "美颜": "APP美颜", "锐化": "锐度", "快门": "快门速度",
}

_LABELS = {
    "白平衡": ("白平衡", "White Balance", "WhiteBalance"),
    "APP美颜": ("APP美颜", "美颜", "APP Beauty", "APPBeauty", "Beauty"),
    "锐度": ("锐度", "锐化", "Sharpness"),
    "去噪": ("去噪", "Noise Reduction", "NoiseReduction", "Denoise"),
    "纹理": ("纹理", "Texture"),
    "分辨率": ("分辨率", "Resolution"),
    "帧率": ("帧率", "Frame Rate", "FrameRate"),
    "胶片影调": ("胶片影调", "Film Look", "FilmLook"),
    "滤镜浓度": ("滤镜浓度", "滤镜", "Filter Intensity", "FilterIntensity"),
    "色彩模式": ("色彩模式", "色彩", "Color Mode", "ColorMode"),
    "感光度": ("感光度", "ISO", "ISO Sensitivity", "ISOSensitivity"),
    "曝光补偿": ("曝光补偿", "曝光", "EV", "Exposure Compensation", "ExposureCompensation"),
    "快门速度": ("快门速度", "快门", "Shutter Speed", "ShutterSpeed"),
}

_ISO_POINTS = {50, 100, 200, 400, 800, 1600, 3200, 6400, 12800}
_EXPOSURES = {
    "-2.0", "-1.7", "-1.3", "-1.0", "-0.7", "-0.3", "0.0",
    "+0.3", "+0.7", "+1.0", "+1.3", "+1.7", "+2.0",
}
_SHUTTERS = {"1/8000", "1/4000", "1/2000", "1/1000", "1/500", "1/250", "1/120", "1/60", "1/30"}


def _canonical_field(field: str) -> str:
    value = field.strip()
    if value in ALLOWED_FIELDS:
        return value
    if value in EN_TO_CN:
        return EN_TO_CN[value]
    if value in _CN_ALIASES:
        return _CN_ALIASES[value]
    raise ValueError(f"unsupported field: {field}")


def canonicalize_fields(fields: Sequence[str]) -> list[str]:
    result = []
    for field in fields:
        canonical = _canonical_field(str(field))
        if canonical not in result:
            result.append(canonical)
    return result


def _normalize(field: str, raw: str) -> str | None:
    value = raw.strip().replace("～", "-").replace("~", "-").replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", value)
    if field == "白平衡":
        if compact.upper() == "AWB" or compact in {"自动", "AWB自动"}:
            return "AWB"
        match = re.search(r"(?i)(\d{4,5})k?", compact)
        if match and 2000 <= int(match.group(1)) <= 10000 and int(match.group(1)) % 100 == 0:
            return f"{int(match.group(1))}K"
    elif field == "APP美颜":
        if compact in {"开", "开启", "打开", "是"}:
            return "开启"
        if compact in {"关", "关闭", "关掉", "否"}:
            return "关闭"
    elif field in {"锐度", "去噪", "纹理"}:
        match = re.fullmatch(r"\+?(-?\d)", compact)
        allowed = {"锐度": {-2, -1, 0, 1, 2}, "去噪": {-2, -1, 0, 1}, "纹理": {-2, -1, 0, 1, 2}}[field]
        if match and int(match.group(1)) in allowed:
            return str(int(match.group(1)))
    elif field == "分辨率":
        match = re.search(r"(?i)(1080p|2\.7k|4k)", compact)
        if match:
            return {"1080p": "1080P", "2.7k": "2.7K", "4k": "4K"}[match.group(1).lower()]
    elif field == "帧率":
        matches = re.findall(r"(?i)(24|25|30|48|50|60)(?:fps|p|帧)", compact)
        if not matches:
            matches = re.findall(r"(?i)(?:1080p|2\.7k|4k)(24|25|30|48|50|60)$", compact)
        if matches:
            return f"{matches[-1]}fps"
    elif field == "胶片影调":
        upper = compact.upper()
        if "NC" in upper:
            return "NC胶片"
        if "CC" in upper:
            return "CC胶片"
        for option in ("清新", "暖调", "电影", "复古"):
            if option in compact:
                return option
    elif field == "滤镜浓度":
        match = re.search(r"(30|50|70|100)%", compact)
        if match:
            return f"{match.group(1)}%"
    elif field == "色彩模式":
        if re.search(r"(?i)d-?log\s*m", value):
            return "D-Log M"
        if re.search(r"(?i)HLG", value):
            return "HLG"
        if "普通" in compact:
            return "普通"
    elif field == "感光度":
        if compact.lower() in {"自动", "auto", "auto自动"}:
            return "自动"
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", compact)
        if match:
            points = [int(item) for item in match.groups() if item is not None]
            if all(point in _ISO_POINTS for point in points):
                return "-".join(str(point) for point in points)
    elif field == "曝光补偿":
        cleaned = re.sub(r"(?i)EV$", "", compact)
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
            number = float(cleaned)
            normalized = "0.0" if number == 0 else (f"{number:+.1f}" if cleaned.startswith("+") else f"{number:.1f}")
            if normalized in _EXPOSURES:
                return normalized
    elif field == "快门速度":
        if compact.lower() in {"自动", "auto"}:
            return "自动"
        if compact in _SHUTTERS:
            return compact
    return None


def _evidence_line(text: str, field: str, value: str) -> str | None:
    candidates = [line.strip() for line in text.splitlines() if value in line]
    if not candidates:
        return None
    useful_labels = [label for label in _LABELS[field] if label.upper() != "EV"]
    for line in candidates:
        if any(label.lower() in line.lower() for label in useful_labels):
            return line
    return candidates[0]


def _has_field_evidence(field: str, line: str, raw: str) -> bool:
    lower = line.lower()
    if any(label.lower() in lower for label in _LABELS[field]):
        return True
    compact = re.sub(r"\s+", "", raw)
    if field == "分辨率":
        return bool(re.search(r"(?i)(1080p|2\.7k|4k)", compact))
    if field == "帧率":
        return bool(re.search(r"(?i)(24|25|30|48|50|60)(fps|p|帧)", compact) or re.fullmatch(r"(?i)(1080p|2\.7k|4k)(24|25|30|48|50|60)", compact))
    if field == "白平衡":
        return compact.upper() == "AWB" or bool(re.fullmatch(r"(?i)\d{4,5}k", compact))
    if field == "曝光补偿":
        return compact.upper().endswith("EV")
    if field == "快门速度":
        return compact in _SHUTTERS
    if field == "胶片影调":
        return any(token in compact.upper() for token in ("NC", "CC")) or any(token in compact for token in ("清新", "暖调", "电影", "复古"))
    return False


def _has_sign_mismatch(text: str, raw: str) -> bool:
    if not re.fullmatch(r"\+?\d+(?:\.\d+)?(?:EV)?", raw.strip(), re.IGNORECASE):
        return False
    matches = list(re.finditer(re.escape(raw.strip()), text, re.IGNORECASE))
    return bool(matches) and all(match.start() > 0 and text[match.start() - 1] == "-" for match in matches)


def _label_value_mismatch(field: str, line: str, raw: str) -> bool:
    """Check that a labeled value is the value immediately following its label."""
    labels = [label for label in _LABELS[field] if label.upper() != "EV"]
    found_label = False
    lower = line.lower()
    for label in sorted(labels, key=len, reverse=True):
        start = 0
        while True:
            index = lower.find(label.lower(), start)
            if index < 0:
                break
            found_label = True
            tail = line[index + len(label):].lstrip(" \t:：")
            if tail.lower().startswith(raw.strip().lower()):
                return False
            start = index + len(label)
    return found_label


def validate_extraction(text: str, raw: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    verified: dict[str, list[dict[str, str]]] = {}
    normalized: dict[str, list[str]] = {}
    rejected: list[dict[str, Any]] = []
    status: dict[str, str] = {}

    for field in fields:
        candidates = raw.get(field, [])
        if not candidates:
            status[field] = "not_found"
            continue
        for item in candidates:
            candidate = str(item.get("text", "")) if isinstance(item, dict) else str(item)
            reasons = []
            line = _evidence_line(text, field, candidate)
            value = _normalize(field, candidate)
            if line is None:
                reasons.append("value_not_in_source")
            elif _has_sign_mismatch(text, candidate):
                reasons.append("numeric_sign_mismatch")
            if value is None:
                reasons.append("value_not_allowed")
            if line is not None and not _has_field_evidence(field, line, candidate):
                reasons.append("field_evidence_missing")
            elif line is not None and _label_value_mismatch(field, line, candidate):
                reasons.append("field_value_mismatch")
            if reasons:
                rejection = {"field": field, "value": candidate, "reasons": reasons}
                if line is not None:
                    rejection["evidence"] = line
                rejected.append(rejection)
                continue
            entry = {"value": value, "raw_value": candidate, "evidence": line}
            if value not in normalized.setdefault(field, []):
                normalized[field].append(value)
                verified.setdefault(field, []).append(entry)
        status[field] = "verified" if field in verified else "rejected"

    return {
        "normalized_output": normalized,
        "verified_output": verified,
        "rejected_candidates": rejected,
        "field_status": status,
    }
