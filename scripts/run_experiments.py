#!/usr/bin/env python3
"""Run the common Pocket corpus, dynamic schemas, nesting, metrics, and RSS capture."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text_extractor.backends.pp_uie import (
    MODEL_SPECS,
    PPUIEBackend,
    RuntimeConfig,
    validate_model_dir,
)
from text_extractor.core.normalize import normalize_scene_output
from text_extractor.evaluation.experiments import run_schema_switches
from text_extractor.evaluation.metrics import evaluate_records
from text_extractor.evaluation.monitoring import PeakRSSMonitor
from text_extractor.runtime.offline import block_network


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.5b", "1.5b"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--samples", type=Path, default=ROOT / "testdata/samples.jsonl")
    parser.add_argument("--expected", type=Path, default=ROOT / "testdata/expected_results.jsonl")
    parser.add_argument("--schema", type=Path, default=ROOT / "configs/pocket_schema.json")
    parser.add_argument("--nested-schema", type=Path, default=ROOT / "configs/nested_pocket_schema.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    if args.offline:
        block_network()
    model_path = (args.model_path or ROOT / "models" / MODEL_SPECS[args.model].directory).resolve()
    model_summary = validate_model_dir(model_path)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    nested_schema = json.loads(args.nested_schema.read_text(encoding="utf-8"))
    samples = read_jsonl(args.samples)
    gold_rows = {row["id"]: row for row in read_jsonl(args.expected)}
    backend = PPUIEBackend(RuntimeConfig(
        model_path, "cpu", "float32", args.batch_size, args.max_length, args.max_new_tokens
    ))

    outputs = []
    metrics = {}
    load_started = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        backend.load()
        load_ms = (time.perf_counter() - load_started) * 1000
        backend.set_schema(schema)
        for sample in samples:
            started = time.perf_counter()
            raw = backend.extract([sample["text"]])[0]
            inference_ms = (time.perf_counter() - started) * 1000
            normalized = normalize_scene_output(raw, schema)
            gold = gold_rows[sample["id"]]
            if gold.get("reference_id"):
                gold = gold_rows[gold["reference_id"]]
            report = evaluate_records(gold.get("scenes", []), normalized["scenes"])
            metrics[sample["id"]] = asdict(report)
            outputs.append({
                "id": sample["id"],
                "source_type": sample.get("source_type", "manual_transcription"),
                "text": sample["text"],
                "business_schema": schema,
                "pp_uie_schema": schema,
                "model": args.model,
                "model_path": str(model_path),
                "runtime": {"device": "cpu", "precision": "float32", "batch_size": args.batch_size,
                            "max_length": args.max_length, "max_new_tokens": args.max_new_tokens},
                "raw_output": raw,
                "normalized_output": normalized,
                "timing_ms": {"load": load_ms, "inference": inference_ms},
                "error": None,
            })

        single = next(row for row in samples if row["id"] == "single_night_cyberpunk")
        dynamic_schemas = [
            ["拍照公式场景"],
            {"拍照公式场景": ["曝光", "白平衡"]},
            {"拍照公式场景": ["感光度", "分辨率", "去噪"]},
        ]
        dynamic = run_schema_switches(backend, single["text"], dynamic_schemas)
        clean = next(row for row in samples if row["id"] == "all_scenes_clean")
        backend.set_schema(nested_schema)
        started = time.perf_counter()
        nested_raw = backend.extract([clean["text"]])[0]
        nested_ms = (time.perf_counter() - started) * 1000
        nested_normalized = normalize_scene_output(nested_raw, nested_schema)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model
    output_path = args.output_dir / f"outputs_{slug}.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8"
    )
    experiment = {
        "model": args.model,
        "model_path": str(model_path),
        "model_files_bytes": model_summary.total_bytes,
        "load_ms": load_ms,
        "peak_rss_bytes": monitor.peak_bytes,
        "metrics_by_sample": metrics,
        "dynamic_schema": {"model_reload_count": 0, "cases": dynamic},
        "nested_schema": {
            "business_schema": nested_schema,
            "pp_uie_schema": nested_schema,
            "inference_ms": nested_ms,
            "raw_output": nested_raw,
            "business_output": nested_normalized,
        },
    }
    (args.output_dir / f"experiment_{slug}.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_path), "experiment": str(args.output_dir / f"experiment_{slug}.json"),
        "load_ms": load_ms, "peak_rss_bytes": monitor.peak_bytes,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
