import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
INPUT_PATH = ROOT / "testdata/pocket_edge_cases.jsonl"
EXPECTED_PATH = ROOT / "testdata/pocket_edge_cases_expected.jsonl"
MATRIX_PATH = ROOT / "testdata/pocket_edge_cases_matrix.md"
ALLOWED_FIELDS = {
    "白平衡", "APP美颜", "锐度", "去噪", "纹理", "分辨率", "帧率",
    "胶片影调", "滤镜浓度", "色彩模式", "感光度", "曝光补偿", "快门速度",
}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_edge_case_dataset_has_the_approved_40_case_distribution():
    rows = _read_jsonl(INPUT_PATH)
    assert len(rows) == 40
    assert len({row["id"] for row in rows}) == 40
    assert Counter(row["priority"] for row in rows) == {"P0": 30, "P1": 6, "MIXED": 4}
    counts = Counter(row["primary_rule"] for row in rows)
    assert counts == {
        **{f"P0-{index:02d}": 3 for index in range(1, 11)},
        "P1-01": 3,
        "P1-02": 3,
        "MIXED": 4,
    }


def test_inputs_are_traceable_controlled_mutations_with_explicit_schema():
    rows = _read_jsonl(INPUT_PATH)
    valid_sources = {f"pocket_user_{index:02d}" for index in range(1, 7)}
    for row in rows:
        assert row["source_type"] == "synthetic_controlled_mutation"
        assert row["source_ids"]
        assert set(row["source_ids"]) <= valid_sources
        assert row["text"].strip()
        assert row["schema"]
        assert len(row["schema"]) == len(set(row["schema"]))
        assert set(row["schema"]) <= ALLOWED_FIELDS
        assert row["rules"]
        assert row["primary_rule"] in row["rules"] or row["primary_rule"] == "MIXED"


def test_inputs_read_like_rich_social_posts_without_losing_challenge_anchors():
    rows = _read_jsonl(INPUT_PATH)
    social_markers = ("#", "📸", "✨", "🎬", "🌊", "☀️", "@", "后期", "调色", "拍摄")
    meta_test_terms = ("模型", "Schema", "抽取", "测试", "标准结果", "标准化", "值域", "数据清洗", "严格模式")
    for row in rows:
        text = row["text"]
        assert 180 <= len(text) <= 900, row["id"]
        assert text.count("\n") >= 4, row["id"]
        assert any(marker in text for marker in social_markers), row["id"]
        assert row["challenge_anchors"], row["id"]
        assert all(anchor in text for anchor in row["challenge_anchors"]), row["id"]
        assert not any(term in text for term in meta_test_terms), row["id"]


def test_each_post_recommends_a_full_parameter_set_beside_the_target_challenge():
    inputs = _read_jsonl(INPUT_PATH)
    expected = {row["id"]: row for row in _read_jsonl(EXPECTED_PATH)}
    for row in inputs:
        assert len(row["schema"]) >= 11, row["id"]
        assert row["challenge_fields"], row["id"]
        assert row["baseline_fields"], row["id"]
        assert set(row["schema"]) == set(row["challenge_fields"]) | set(row["baseline_fields"])
        assert set(row["challenge_fields"]).isdisjoint(row["baseline_fields"])
        assert all(field in row["text"] for field in row["baseline_fields"]), row["id"]

        gold = expected[row["id"]]
        baseline_group = next(
            group for group in gold["expected_output"]["groups"]
            if group["name"] == "完整机内参数"
        )
        assert set(baseline_group["params"]) == set(row["baseline_fields"])
        valid_fields = {
            field
            for group in gold["expected_output"]["groups"]
            for field in group["params"]
        }
        assert len(valid_fields) >= 7, row["id"]


def test_gold_records_align_with_inputs_and_follow_compact_output_contract():
    inputs = _read_jsonl(INPUT_PATH)
    expected = _read_jsonl(EXPECTED_PATH)
    assert [row["id"] for row in expected] == [row["id"] for row in inputs]

    inputs_by_id = {row["id"]: row for row in inputs}
    for gold in expected:
        source = inputs_by_id[gold["id"]]
        assert set(gold) == {
            "id", "expected_output", "omitted_fields", "ambiguous_candidates",
            "rejected_candidates", "conflicts", "notes",
        }
        assert set(gold["omitted_fields"]) <= set(source["schema"])
        assert isinstance(gold["notes"], str) and gold["notes"].strip()
        groups = gold["expected_output"]["groups"]
        assert isinstance(groups, list)
        for group in groups:
            assert set(group) == {"name", "params"}
            assert group["name"].strip()
            assert set(group["params"]) <= set(source["schema"])
            for value in group["params"].values():
                assert isinstance(value, str) and value.strip()
                assert " | " not in value


def test_gold_keeps_uncertain_or_invalid_values_out_of_normalized_params():
    expected = _read_jsonl(EXPECTED_PATH)
    for gold in expected:
        normalized_pairs = {
            (field, value)
            for group in gold["expected_output"]["groups"]
            for field, value in group["params"].items()
        }
        for item in gold["rejected_candidates"]:
            assert item["reason"]
            assert (item["field"], item["raw"]) not in normalized_pairs
        for item in gold["ambiguous_candidates"]:
            assert item["reason"]
            assert item["candidates"]


def test_coverage_matrix_documents_rules_and_output_policy():
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    for rule in [*(f"P0-{index:02d}" for index in range(1, 11)), "P1-01", "P1-02", "MIXED"]:
        assert rule in matrix
    assert "value|value" in matrix
    assert "缺失字段直接省略" in matrix
    assert "机内参数" in matrix and "后期参数" in matrix
