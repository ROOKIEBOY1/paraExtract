from pathlib import Path

from scripts.evaluate_uie_mini import (
    build_uie_definition,
    evaluate_rows,
    load_evaluation_rows,
    render_report,
    resolve_output_paths,
    summarize_results,
)
from pp_uie.uie_backend import UIETaskflowBackend, UIETaskflowConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeService:
    def __init__(self):
        self.calls = []

    def extract(self, text, fields, strict=True):
        self.calls.append((text, fields, strict))
        index = len(self.calls)
        raw = {fields[0]: [{"text": f"value-{index}", "probability": 0.75}]}
        return {
            "raw_output": raw,
            "normalized_output": {fields[0]: [f"normalized-{index}"]},
            "verified_output": {
                fields[0]: [{"value": f"normalized-{index}", "raw_value": f"value-{index}"}]
            },
            "rejected_candidates": ([{"field": "bad", "value": "x"}] if index % 2 == 0 else []),
            "field_status": {fields[0]: "verified"},
            "timing_ms": {"schema": 0.1, "inference": float(index), "total": index + 0.1},
        }


def test_load_evaluation_rows_retains_all_46_samples_and_assigns_schema():
    rows = load_evaluation_rows(
        PROJECT_ROOT / "testdata/pocket_user_six_samples.jsonl",
        PROJECT_ROOT / "testdata/pocket_edge_cases.jsonl",
    )

    assert len(rows) == 46
    assert rows[0]["id"] == "pocket_user_01"
    assert rows[-1]["id"] == "pocket_edge_mixed_04"
    assert rows[0]["schema"]
    assert rows[6]["schema"] == [
        "白平衡", "曝光补偿", "快门速度", "APP美颜", "锐度", "去噪", "纹理",
        "分辨率", "帧率", "胶片影调", "滤镜浓度",
    ]


def test_evaluate_rows_preserves_order_schema_and_all_output_sections():
    rows = [
        {"id": f"sample-{index:02d}", "text": f"text-{index}", "schema": [f"field-{index}"]}
        for index in range(1, 47)
    ]
    service = FakeService()

    output = evaluate_rows(service, rows)

    assert [item["sample_id"] for item in output] == [item["id"] for item in rows]
    assert service.calls == [
        (row["text"], row["schema"], True) for row in rows
    ]
    assert output[1] == {
        "sample_id": "sample-02",
        "schema": ["field-2"],
        "raw_output": {"field-2": [{"text": "value-2", "probability": 0.75}]},
        "normalized_output": {"field-2": ["normalized-2"]},
        "verified_output": {
            "field-2": [{"value": "normalized-2", "raw_value": "value-2"}]
        },
        "rejected_candidates": [{"field": "bad", "value": "x"}],
        "field_status": {"field-2": "verified"},
        "timing_ms": {"schema": 0.1, "inference": 2.0, "total": 2.1},
    }


def test_summary_counts_candidates_and_uses_deterministic_nearest_rank_percentiles():
    service = FakeService()
    rows = [
        {"id": f"sample-{index}", "text": "x", "schema": ["感光度"]}
        for index in range(1, 11)
    ]
    results = evaluate_rows(service, rows)

    summary = summarize_results(
        results,
        model_size_bytes=123,
        load_ms=456.5,
        offline_success=True,
    )

    assert summary["sample_count"] == 10
    assert summary["model_size_bytes"] == 123
    assert summary["load_ms"] == 456.5
    assert summary["offline_success"] is True
    assert summary["accepted_candidate_count"] == 10
    assert summary["rejected_candidate_count"] == 5
    assert summary["latency_ms"] == {
        "min": 1.0,
        "mean": 5.5,
        "p50": 5.0,
        "p95": 10.0,
        "max": 10.0,
    }


def test_build_uie_base_evaluation_definition_uses_local_taskflow_model(tmp_path):
    definition = build_uie_definition("uie-base", tmp_path, 384)

    assert definition.model_id == "uie-base"
    assert definition.label == "UIE-base"
    assert definition.backend_factory is UIETaskflowBackend
    assert definition.config == UIETaskflowConfig(
        tmp_path,
        max_seq_len=384,
        model="uie-base",
    )


def test_uie_base_report_names_its_own_raw_output_file():
    summary = {
        "sample_count": 0,
        "model_size_bytes": 1,
        "load_ms": 1.0,
        "offline_success": True,
        "accepted_candidate_count": 0,
        "rejected_candidate_count": 0,
        "latency_ms": {"min": 0, "mean": 0, "p50": 0, "p95": 0, "max": 0},
    }

    report = render_report(
        summary,
        [],
        model_label="UIE-base",
        output_filename="uie_base_outputs.jsonl",
    )

    assert "uie_base_outputs.jsonl" in report
    assert "uie_mini_outputs.jsonl" not in report


def test_uie_base_default_outputs_do_not_overwrite_uie_mini_results():
    output, report = resolve_output_paths("uie-base", None, None)

    assert output == PROJECT_ROOT / "results/uie_base_outputs.jsonl"
    assert report == PROJECT_ROOT / "results/uie_base_assessment.md"
