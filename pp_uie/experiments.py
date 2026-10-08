"""Experiment helpers shared by the operational runner and tests."""

import time
from typing import Any, Callable, Sequence


def run_schema_switches(
    backend,
    text: str,
    schemas: Sequence[Any],
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    rows = []
    for schema in schemas:
        started = clock()
        backend.set_schema(schema)
        schema_switch_ms = (clock() - started) * 1000
        started = clock()
        raw = backend.extract([text])[0]
        inference_ms = (clock() - started) * 1000
        rows.append({
            "business_schema": schema,
            "pp_uie_schema": schema,
            "schema_switch_ms": schema_switch_ms,
            "inference_ms": inference_ms,
            "raw_output": raw,
        })
    return rows
