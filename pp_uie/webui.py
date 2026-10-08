"""Persistent-model extraction service and local web UI configuration."""

import gc
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Sequence

from .backend import PPUIEBackend, RuntimeConfig
from .strict_validation import ALLOWED_FIELDS, canonicalize_fields, validate_extraction


FIELD_GROUPS = [
    {
        "name": "Pocket 标准参数",
        "fields": list(ALLOWED_FIELDS),
    },
]

EDGE_RULE_LABELS = {
    "P0-01": "模糊语义",
    "P0-02": "数值格式",
    "P0-03": "曝光歧义",
    "P0-04": "参数互斥",
    "P0-05": "参数名缺省",
    "P0-06": "参数值缺省",
    "P0-07": "名称和值缺省",
    "P0-08": "范围与多选",
    "P0-09": "非法参数",
    "P0-10": "输出格式",
    "P1-01": "OCR噪声",
    "P1-02": "别名与错别字",
    "MIXED": "综合场景",
}

EDGE_GROUP_LABELS = {
    "P0": "P0 边界场景（30条）",
    "P1": "P1 OCR/别名场景（6条）",
    "MIXED": "综合压力场景（4条）",
}


def load_text_samples(path: Path) -> list[dict[str, str]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"id": row["id"], "label": f"测试文本 {index}", "text": row["text"]})
    return rows


def load_web_samples(original_path: Path, edge_path: Path) -> list[dict[str, str]]:
    samples = []
    for index, row in enumerate(load_text_samples(original_path), start=1):
        samples.append({
            **row,
            "label": f"原始文本 {index}",
            "group": "原有测试文本（6条）",
        })

    rule_counts: dict[str, int] = {}
    for line in edge_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rule = row["primary_rule"]
        priority = row["priority"]
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
        samples.append({
            "id": row["id"],
            "label": f"{rule} · {EDGE_RULE_LABELS[rule]} · {rule_counts[rule]:02d}",
            "text": row["text"],
            "group": EDGE_GROUP_LABELS[priority],
        })
    return samples


def normalize_flat_output(raw: dict[str, Any], fields: Sequence[str]) -> dict[str, list[str]]:
    normalized = {}
    for field in fields:
        values = [
            str(item["text"])
            for item in raw.get(field, [])
            if isinstance(item, dict) and "text" in item
        ]
        if values:
            normalized[field] = values
    return normalized


class ExtractionService:
    def __init__(self, backend, model_name: str, load_ms: float, clock: Callable[[], float]):
        self.backend = backend
        self.model_name = model_name
        self.load_ms = load_ms
        self.clock = clock
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        config: RuntimeConfig,
        model_name: str,
        backend_factory=PPUIEBackend,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "ExtractionService":
        started = clock()
        backend = backend_factory(config)
        backend.load()
        load_ms = (clock() - started) * 1000
        return cls(backend, model_name, load_ms, clock)

    def extract(self, text: str, fields: Sequence[str], strict: bool = True) -> dict[str, Any]:
        text = text.strip()
        requested_fields = [str(field).strip() for field in fields if str(field).strip()]
        if not text:
            raise ValueError("text must not be empty")
        if not requested_fields:
            raise ValueError("select at least one field")
        if len(requested_fields) != len(set(requested_fields)):
            raise ValueError("fields must be unique")
        fields = canonicalize_fields(requested_fields)
        with self._lock:
            started = self.clock()
            self.backend.set_schema(fields)
            schema_ms = (self.clock() - started) * 1000
            started = self.clock()
            raw = self.backend.extract([text])[0]
            inference_ms = (self.clock() - started) * 1000
        validation = validate_extraction(text, raw, fields) if strict else {
            "normalized_output": normalize_flat_output(raw, fields),
            "verified_output": {},
            "rejected_candidates": [],
            "field_status": {},
        }
        return {
            "model": self.model_name,
            "fields": fields,
            "raw_output": raw,
            "strict_mode": strict,
            **validation,
            "timing_ms": {
                "schema": schema_ms,
                "inference": inference_ms,
                "total": schema_ms + inference_ms,
                "model_load_excluded": True,
            },
        }

    def close(self) -> None:
        backend, self.backend = self.backend, None
        close = getattr(backend, "close", None)
        if close is not None:
            close()


class SwitchingExtractionService:
    """Own exactly one loaded model and replace it on explicit selection."""

    def __init__(
        self,
        configs: dict[str, RuntimeConfig],
        service: ExtractionService,
        backend_factory,
        clock: Callable[[], float],
        collect_memory: Callable[[], Any],
    ):
        self.configs = dict(configs)
        self._service = service
        self.backend_factory = backend_factory
        self.clock = clock
        self.collect_memory = collect_memory
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        configs: dict[str, RuntimeConfig],
        initial_model: str,
        backend_factory=PPUIEBackend,
        clock: Callable[[], float] = time.perf_counter,
        collect_memory: Callable[[], Any] = gc.collect,
    ) -> "SwitchingExtractionService":
        if initial_model not in configs:
            raise ValueError(f"unsupported model: {initial_model}")
        service = ExtractionService.create(
            configs[initial_model], initial_model, backend_factory=backend_factory, clock=clock
        )
        return cls(configs, service, backend_factory, clock, collect_memory)

    @property
    def available_models(self) -> list[str]:
        return list(self.configs)

    @property
    def model_name(self) -> str:
        return self._service.model_name

    @property
    def load_ms(self) -> float:
        return self._service.load_ms

    def switch_model(self, model_name: str) -> dict[str, Any]:
        if model_name not in self.configs:
            raise ValueError(f"unsupported model: {model_name}")
        with self._lock:
            if model_name == self._service.model_name:
                return {
                    "model": model_name,
                    "changed": False,
                    "load_ms": self._service.load_ms,
                    "switch_ms": 0.0,
                }
            started = self.clock()
            old_service, self._service = self._service, None
            old_service.close()
            del old_service
            self.collect_memory()
            self._service = ExtractionService.create(
                self.configs[model_name],
                model_name,
                backend_factory=self.backend_factory,
                clock=self.clock,
            )
            return {
                "model": model_name,
                "changed": True,
                "load_ms": self._service.load_ms,
                "switch_ms": (self.clock() - started) * 1000,
            }

    def extract(self, text: str, fields: Sequence[str], strict: bool = True) -> dict[str, Any]:
        with self._lock:
            return self._service.extract(text, fields, strict=strict)


class WebApplication:
    def __init__(self, service: ExtractionService, samples: list[dict[str, str]]):
        self.service = service
        self.samples = samples

    def config(self) -> dict[str, Any]:
        return {
            "model": self.service.model_name,
            "load_ms": self.service.load_ms,
            "models": self.service.available_models,
            "samples": self.samples,
            "field_groups": FIELD_GROUPS,
        }

    def extract(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not isinstance(payload.get("fields"), list):
            raise ValueError("request must be a JSON object with text and fields")
        strict = payload.get("strict", True)
        if not isinstance(strict, bool):
            raise ValueError("strict must be boolean")
        return self.service.extract(payload["text"], payload["fields"], strict=strict)

    def switch_model(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise ValueError("request must be a JSON object with model")
        return self.service.switch_model(payload["model"])
