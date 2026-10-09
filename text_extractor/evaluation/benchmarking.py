"""Reusable benchmark matrix logic that never reloads the model between cases."""

from dataclasses import asdict
import time
from typing import Callable, Sequence

from .monitoring import summarize_latencies


def build_length_text(source: str, target_characters: int) -> str:
    if target_characters <= 0:
        raise ValueError("target_characters must be positive")
    if not source:
        raise ValueError("source text must not be empty")
    repeated = (source + "\n") * (target_characters // (len(source) + 1) + 1)
    return repeated[:target_characters]


def run_latency_matrix(
    backend,
    source: str,
    lengths: Sequence[int],
    batch_sizes: Sequence[int],
    warmups: int,
    runs: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, object]]:
    if warmups < 0 or runs <= 0:
        raise ValueError("warmups must be non-negative and runs must be positive")
    rows: list[dict[str, object]] = []
    for length in lengths:
        text = build_length_text(source, length)
        for batch_size in batch_sizes:
            batch = [text] * batch_size
            for _ in range(warmups):
                backend.extract(batch)
            samples = []
            for _ in range(runs):
                started = clock()
                backend.extract(batch)
                samples.append((clock() - started) * 1000)
            mean_ms = sum(samples) / len(samples)
            rows.append({
                "length_chars": len(text),
                "batch_size": batch_size,
                "latencies_ms": samples,
                "summary": asdict(summarize_latencies(samples)),
                "throughput_items_per_second": batch_size * 1000 / mean_ms,
            })
    return rows
