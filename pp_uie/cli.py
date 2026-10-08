"""Argument and runtime validation for the unified inference command."""

import argparse
from pathlib import Path
from typing import Iterable


def batched(values, size: int):
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run PP-UIE from an explicit local model directory")
    parser.add_argument("--model", choices=["0.5b", "1.5b"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "npu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args(argv)


def resolve_runtime(device: str, precision: str, available: Iterable[str] = ("cpu",)) -> tuple[str, str]:
    available = set(available)
    device = "cpu" if device == "auto" else device
    if device not in available:
        label = {"gpu": "GPU", "npu": "NPU"}.get(device, device.upper())
        raise ValueError(f"{label} is not available")
    precision = "float32" if precision == "auto" else precision
    if device == "cpu" and precision == "float16":
        raise ValueError("float16 is not supported for this CPU experiment")
    if device == "cpu" and precision == "bfloat16":
        raise ValueError("bfloat16 is not enabled for this CPU experiment")
    return device, precision
