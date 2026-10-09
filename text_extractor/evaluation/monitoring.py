"""Latency statistics and peak resident-memory sampling."""

from dataclasses import dataclass
import math
import threading
import time

import psutil


def process_tree_rss(process) -> int:
    try:
        processes = [process] + process.children(recursive=True)
    except (OSError, psutil.Error):
        processes = [process]
    total = 0
    for item in processes:
        try:
            if item.is_running():
                total += item.memory_info().rss
        except (OSError, psutil.Error):
            continue
    return total


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float


def summarize_latencies(values_ms):
    values = sorted(values_ms)
    if not values:
        raise ValueError("latency list must not be empty")
    nearest = lambda percentile: values[max(0, math.ceil(percentile * len(values)) - 1)]
    return LatencySummary(sum(values) / len(values), nearest(0.50), nearest(0.95))


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, process_tree_rss(process))
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
