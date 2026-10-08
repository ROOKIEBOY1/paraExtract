import pytest

from pp_uie.schema import iter_prompts, parse_schema


def test_nested_scene_schema_generates_parent_prefixed_prompts():
    nodes = parse_schema({"拍照公式场景": ["曝光", "白平衡"]})
    prompts = list(iter_prompts(nodes, parents=("夜景赛博朋克",)))
    assert [p.prompt for p in prompts] == ["夜景赛博朋克的曝光", "夜景赛博朋克的白平衡"]


def test_invalid_schema_reports_json_path():
    with pytest.raises(TypeError, match=r"\$\.拍照公式场景 must be string or list"):
        parse_schema({"拍照公式场景": {"曝光": 3}})

