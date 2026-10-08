#!/usr/bin/env python3
"""Unified PP-UIE local inference entry point."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pp_uie.cli import batched, parse_args, resolve_runtime
from pp_uie.offline import block_network


def _inputs(args):
    if args.text is not None:
        return [{"id": "direct_text", "text": args.text}]
    return [json.loads(line) for line in args.text_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.offline:
        block_network()
    from pp_uie.backend import PPUIEBackend, RuntimeConfig
    from pp_uie.models import MODEL_SPECS, validate_model_dir
    from pp_uie.normalize import normalize_scene_output

    device, precision = resolve_runtime(args.device, args.precision)
    model_path = (args.model_path or ROOT / "models" / MODEL_SPECS[args.model].directory).resolve()
    validate_model_dir(model_path)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    config = RuntimeConfig(model_path, device, precision, args.batch_size, args.max_length, args.max_new_tokens)
    backend = PPUIEBackend(config)
    load_start = time.perf_counter()
    backend.load()
    load_ms = (time.perf_counter() - load_start) * 1000
    backend.set_schema(schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for rows in batched(_inputs(args), args.batch_size):
            started = time.perf_counter()
            errors = [None] * len(rows)
            try:
                raw_outputs = backend.extract([row["text"] for row in rows])
            except Exception as exc:
                raw_outputs = [None] * len(rows)
                errors = [{"type": type(exc).__name__, "message": str(exc)}] * len(rows)
            elapsed = (time.perf_counter() - started) * 1000
            for row, raw, error in zip(rows, raw_outputs, errors):
                normalized = normalize_scene_output(raw, schema) if raw is not None else None
                output = {
                    "id": row.get("id", "direct_text"),
                    "text": row["text"],
                    "business_schema": schema,
                    "pp_uie_schema": schema,
                    "model": args.model,
                    "model_path": str(model_path),
                    "runtime": {"device": device, "precision": precision, "batch_size": args.batch_size,
                                "max_length": args.max_length, "max_new_tokens": args.max_new_tokens},
                    "raw_output": raw,
                    "normalized_output": normalized,
                    "truncation": {"enabled": True, "max_length": args.max_length, "occurred": None},
                    "timing_ms": {"load": load_ms, "inference_batch_total": elapsed,
                                  "inference_per_item": elapsed / len(rows)},
                    "error": error,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
