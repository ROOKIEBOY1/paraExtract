#!/usr/bin/env python3
"""Evaluate one resident local UIE-mini model on the 46 Pocket posts."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text_extractor.backends.uie import (
    UIETaskflowBackend,
    UIETaskflowConfig,
    UIE_MODEL_SPECS,
    validate_uie_model_dir,
)
from text_extractor.core.registry import ModelDefinition
from text_extractor.validation.strict import ALLOWED_FIELDS
from text_extractor.web.application import ExtractionService


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_evaluation_rows(source_path: Path, edge_path: Path) -> list[dict[str, Any]]:
    source_rows = _read_jsonl(source_path)
    for row in source_rows:
        row["schema"] = list(ALLOWED_FIELDS)
    return source_rows + _read_jsonl(edge_path)


def build_uie_definition(
    model_id: str, model_path: Path, max_seq_len: int
) -> ModelDefinition:
    try:
        spec = UIE_MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"unsupported UIE model: {model_id}") from exc
    return ModelDefinition(
        model_id,
        spec.directory,
        UIETaskflowConfig(model_path, max_seq_len=max_seq_len, model=model_id),
        UIETaskflowBackend,
    )


def resolve_output_paths(
    model_id: str,
    output: Optional[Path],
    report: Optional[Path],
) -> tuple[Path, Path]:
    stem = model_id.replace("-", "_")
    return (
        output or ROOT / f"results/{stem}_outputs.jsonl",
        report or ROOT / f"results/{stem}_assessment.md",
    )


def evaluate_rows(
    service: ExtractionService, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        result = service.extract(row["text"], row["schema"], strict=True)
        output.append({
            "sample_id": row["id"],
            "schema": row["schema"],
            "raw_output": result["raw_output"],
            "normalized_output": result["normalized_output"],
            "verified_output": result["verified_output"],
            "rejected_candidates": result["rejected_candidates"],
            "field_status": result["field_status"],
            "timing_ms": result["timing_ms"],
        })
    return output


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_results(
    results: Sequence[dict[str, Any]],
    model_size_bytes: int,
    load_ms: float,
    offline_success: bool,
) -> dict[str, Any]:
    latencies = [float(row["timing_ms"]["inference"]) for row in results]
    accepted = sum(
        len(values)
        for row in results
        for values in row["verified_output"].values()
    )
    rejected = sum(len(row["rejected_candidates"]) for row in results)
    return {
        "sample_count": len(results),
        "model_size_bytes": model_size_bytes,
        "load_ms": load_ms,
        "offline_success": offline_success,
        "accepted_candidate_count": accepted,
        "rejected_candidate_count": rejected,
        "latency_ms": {
            "min": min(latencies, default=0.0),
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
    }


def render_report(
    summary: dict[str, Any],
    results: Sequence[dict[str, Any]],
    model_label: str = "UIE-mini",
    output_filename: str = "uie_mini_outputs.jsonl",
) -> str:
    latency = summary["latency_ms"]
    examples = []
    for row in results:
        for rejection in row["rejected_candidates"]:
            examples.append({"sample_id": row["sample_id"], **rejection})
            if len(examples) == 10:
                break
        if len(examples) == 10:
            break
    offline = "成功" if summary["offline_success"] else "未在离线环境变量下验证"
    return f"""# {model_label} Pocket 参数评估

- 样本数：{summary['sample_count']}
- 模型文件大小：{summary['model_size_bytes']} bytes
- 模型加载时间：{summary['load_ms']:.3f} ms
- 离线环境运行：{offline}
- 通过严格校验的候选：{summary['accepted_candidate_count']}
- 被拒绝候选：{summary['rejected_candidate_count']}

## 推理时延（不含加载）

- 平均：{latency['mean']:.3f} ms
- P50：{latency['p50']:.3f} ms
- P95：{latency['p95']:.3f} ms
- 最小 / 最大：{latency['min']:.3f} / {latency['max']:.3f} ms

## 代表性拒绝案例

```json
{json.dumps(examples, ensure_ascii=False, indent=2)}
```

原始跨度、概率和偏移量完整保存在 `{output_filename}` 的 `raw_output` 中；本报告不改写模型预测。
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(UIE_MODEL_SPECS), default="uie-mini")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-samples", type=Path, default=ROOT / "testdata/pocket_user_six_samples.jsonl")
    parser.add_argument("--edge-samples", type=Path, default=ROOT / "testdata/pocket_edge_cases.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args(argv)
    args.output, args.report = resolve_output_paths(args.model, args.output, args.report)

    validation = validate_uie_model_dir(args.model_path, args.model)
    rows = load_evaluation_rows(args.source_samples, args.edge_samples)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        rows = rows[:args.limit]

    definition = build_uie_definition(args.model, args.model_path, args.max_seq_len)
    service = ExtractionService.create(definition)
    try:
        results = evaluate_rows(service, rows)
    finally:
        service.close()

    offline_success = all(
        os.environ.get(name) == "1"
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "AISTUDIO_OFFLINE")
    )
    summary = summarize_results(
        results,
        model_size_bytes=validation.total_bytes,
        load_ms=service.load_ms,
        offline_success=offline_success,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    args.report.write_text(
        render_report(summary, results, definition.label, args.output.name), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
