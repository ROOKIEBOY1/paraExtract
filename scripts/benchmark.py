#!/usr/bin/env python3
"""Benchmark a locally loaded PP-UIE model with warmups and raw samples."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text_extractor.backends.pp_uie import PPUIEBackend, RuntimeConfig
from text_extractor.evaluation.benchmarking import run_latency_matrix
from text_extractor.evaluation.monitoring import PeakRSSMonitor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Tokenizer input-token limit; --lengths are character-count cases")
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = max(records, key=lambda row: len(row["text"]))["text"]
    backend = PPUIEBackend(RuntimeConfig(
        args.model_path, "cpu", "float32", max(args.batch_sizes), args.max_length, args.max_new_tokens
    ))
    load_started = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        backend.load()
        load_ms = (time.perf_counter() - load_started) * 1000
        backend.set_schema(schema)
        results = run_latency_matrix(
            backend, source, args.lengths, args.batch_sizes, args.warmups, args.runs
        )
    for row in results:
        row["peak_rss_bytes"] = monitor.peak_bytes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "warmups": args.warmups,
        "runs": args.runs,
        "load_ms": load_ms,
        "model_path": str(args.model_path.resolve()),
        "max_length_tokens": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
