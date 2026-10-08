import pytest

from pp_uie.strict_validation import (
    ALLOWED_FIELDS,
    canonicalize_fields,
    validate_extraction,
)


def test_allowed_fields_are_exactly_the_user_whitelist():
    assert ALLOWED_FIELDS == (
        "白平衡", "APP美颜", "锐度", "去噪", "纹理", "分辨率", "帧率",
        "胶片影调", "滤镜浓度", "色彩模式", "感光度", "曝光补偿", "快门速度",
    )


def test_english_names_are_canonicalized_and_duplicates_removed():
    assert canonicalize_fields(["White Balance", "WhiteBalance", "ISO", "EV"]) == [
        "白平衡", "感光度", "曝光补偿"
    ]


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="unsupported field"):
        canonicalize_fields(["收音模式"])


def test_strict_validation_accepts_source_evidence_and_normalizes_values():
    text = "机内参数：\n4k60帧\n曝光：-0.3EV\nISO：50-800\n白平衡：AWB\n色彩：普通10bit"
    raw = {
        "分辨率": [{"text": "4k60帧"}],
        "帧率": [{"text": "4k60帧"}],
        "曝光补偿": [{"text": "-0.3EV"}],
        "感光度": [{"text": "50-800"}],
        "白平衡": [{"text": "AWB"}],
        "色彩模式": [{"text": "普通"}],
    }
    result = validate_extraction(text, raw, list(raw))
    assert result["normalized_output"] == {
        "分辨率": ["4K"],
        "帧率": ["60fps"],
        "曝光补偿": ["-0.3"],
        "感光度": ["50-800"],
        "白平衡": ["AWB"],
        "色彩模式": ["普通"],
    }
    assert result["rejected_candidates"] == []
    assert result["field_status"]["曝光补偿"] == "verified"
    assert result["verified_output"]["曝光补偿"][0]["evidence"] == "曝光：-0.3EV"


def test_strict_validation_rejects_wrong_field_fill_even_when_value_is_in_source():
    text = "曝光:+0.7EV\n感光度:50-3200"
    raw = {
        "曝光补偿": [{"text": "+0.7EV"}],
        "感光度": [{"text": "50-3200"}],
        "分辨率": [{"text": "50-3200"}],
    }
    result = validate_extraction(text, raw, list(raw))
    assert result["normalized_output"] == {
        "曝光补偿": ["+0.7"],
        "感光度": ["50-3200"],
    }
    rejected = result["rejected_candidates"]
    assert rejected[0]["field"] == "分辨率"
    assert "value_not_allowed" in rejected[0]["reasons"]
    assert "field_evidence_missing" in rejected[0]["reasons"]
    assert result["field_status"]["分辨率"] == "rejected"


def test_strict_validation_rejects_value_not_copied_from_source():
    result = validate_extraction(
        "白平衡：AWB", {"白平衡": [{"text": "4800K"}]}, ["白平衡"]
    )
    assert result["normalized_output"] == {}
    assert result["rejected_candidates"][0]["reasons"] == ["value_not_in_source"]


def test_strict_validation_detects_dropped_numeric_sign():
    result = validate_extraction(
        "曝光：-0.3EV", {"曝光补偿": [{"text": "0.3EV"}]}, ["曝光补偿"]
    )
    assert result["normalized_output"] == {}
    assert "numeric_sign_mismatch" in result["rejected_candidates"][0]["reasons"]


def test_empty_model_answer_is_not_found_not_rejected():
    result = validate_extraction("ISO：800", {"白平衡": []}, ["白平衡"])
    assert result["field_status"] == {"白平衡": "not_found"}
    assert result["rejected_candidates"] == []


def test_zero_values_use_their_own_field_label_as_evidence():
    result = validate_extraction(
        "曝光：-0.3EV\n图像调节：纹理0去噪0",
        {"纹理": [{"text": "0"}], "去噪": [{"text": "0"}]},
        ["纹理", "去噪"],
    )
    assert result["normalized_output"] == {"纹理": ["0"], "去噪": ["0"]}
    assert result["verified_output"]["纹理"][0]["evidence"] == "图像调节：纹理0去噪0"


def test_compact_video_spec_and_positive_zero_are_normalized():
    result = validate_extraction(
        "分辨率：4k50\n曝光：+0.0ev",
        {"帧率": [{"text": "4k50"}], "曝光补偿": [{"text": "+0.0ev"}]},
        ["帧率", "曝光补偿"],
    )
    assert result["normalized_output"] == {"帧率": ["50fps"], "曝光补偿": ["0.0"]}


def test_later_suggested_value_is_not_accepted_as_labeled_exposure():
    result = validate_extraction(
        "曝光：-0.3（室内阴暗建议调整到0）",
        {"曝光补偿": [{"text": "0"}]},
        ["曝光补偿"],
    )
    assert result["normalized_output"] == {}
    assert "field_value_mismatch" in result["rejected_candidates"][0]["reasons"]
