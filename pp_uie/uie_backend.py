"""Adapter for the official PaddleNLP UIE-mini Taskflow pipeline."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Optional, Sequence

from .strict_validation import EN_TO_CN


@dataclass(frozen=True)
class UIETaskflowConfig:
    model_path: Path
    device: str = "cpu"
    batch_size: int = 1
    max_seq_len: int = 512
    position_prob: float = 0.5
    model: str = "uie-mini"


def _default_taskflow_factory(**kwargs: Any) -> Any:
    from paddlenlp import Taskflow

    return Taskflow(**kwargs)


def _repair_static_cache_marker(model_path: Path) -> None:
    """Work around the Taskflow/Paddle 3 cache-suffix mismatch.

    PaddleNLP 3.0.0b4 writes ``md5 + 'taskflow'`` but Paddle 3.3 uses a
    ``.json`` static-model suffix and trims only that suffix length when it
    checks the marker.  Repair only a marker that already belongs to the
    current weights and only when complete static artifacts exist.
    """
    weights = model_path / "model_state.pdparams"
    marker = model_path / ".cache_info"
    static_prefix = model_path / "static" / "inference"
    suffix = next(
        (candidate for candidate in (".json", ".pdmodel") if static_prefix.with_suffix(candidate).is_file()),
        None,
    )
    if (
        suffix is None
        or not static_prefix.with_suffix(".pdiparams").is_file()
        or not weights.is_file()
        or not marker.is_file()
    ):
        return
    digest_builder = hashlib.md5()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    current = marker.read_text(encoding="utf-8")
    expected = digest + suffix
    if current.startswith(digest) and current != expected:
        marker.write_text(expected, encoding="utf-8")


class UIETaskflowBackend:
    def __init__(
        self,
        config: UIETaskflowConfig,
        taskflow_factory: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.taskflow_factory = taskflow_factory or _default_taskflow_factory
        self.taskflow: Any | None = None
        self._canonical_schema: Any = []
        self._schema_key_map: dict[str, str] = {}

    def load(self) -> None:
        if self.taskflow is not None:
            return
        _repair_static_cache_marker(self.config.model_path)
        device_id = -1 if self.config.device == "cpu" else 0
        self.taskflow = self.taskflow_factory(
            task="information_extraction",
            model=self.config.model,
            task_path=str(self.config.model_path.resolve()),
            schema=[],
            device_id=device_id,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            position_prob=self.config.position_prob,
        )

    def close(self) -> None:
        self.taskflow = None

    def set_schema(self, schema: Any) -> None:
        if self.taskflow is None:
            raise RuntimeError("backend is not loaded")
        self._canonical_schema = schema
        self._schema_key_map = {
            field: field for field in schema if isinstance(field, str)
        } if isinstance(schema, (list, tuple)) else {}
        self.taskflow.set_schema(schema)

    def prepare_schema(self, texts: Sequence[str]) -> None:
        """Add only source-present English label queries to a flat schema.

        Taskflow returns offsets into the untouched source text.  We therefore
        expand the query schema instead of translating the input, then map the
        added query keys back to their canonical Chinese field names.
        """
        if self.taskflow is None:
            raise RuntimeError("backend is not loaded")
        if not isinstance(self._canonical_schema, (list, tuple)) or not all(
            isinstance(field, str) for field in self._canonical_schema
        ):
            return

        source = "\n".join(str(text) for text in texts)
        aliases_by_field: dict[str, list[str]] = {}
        for alias, canonical in EN_TO_CN.items():
            aliases_by_field.setdefault(canonical, []).append(alias)

        expanded: list[str] = []
        key_map: dict[str, str] = {}
        for field in self._canonical_schema:
            expanded.append(field)
            key_map[field] = field
            for alias in aliases_by_field.get(field, []):
                pattern = rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])"
                if re.search(pattern, source, flags=re.IGNORECASE) is None:
                    continue
                query = f"{alias}值"
                if query not in key_map:
                    expanded.append(query)
                    key_map[query] = field

        self._schema_key_map = key_map
        self.taskflow.set_schema(expanded)

    def extract(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if self.taskflow is None:
            raise RuntimeError("backend is not loaded")
        inputs = list(texts)
        results = self.taskflow(inputs)
        if not isinstance(results, list):
            raise TypeError("Taskflow result must be a list")
        if len(results) != len(inputs):
            raise ValueError(
                f"Taskflow returned {len(results)} results for {len(inputs)} inputs"
            )
        for index, record in enumerate(results):
            if not isinstance(record, dict):
                raise TypeError(
                    f"Taskflow result at index {index} must be a dictionary"
                )
        if all(query == canonical for query, canonical in self._schema_key_map.items()):
            return results

        merged_results: list[dict[str, Any]] = []
        for record in results:
            merged: dict[str, Any] = {}
            seen: dict[str, set[tuple[Any, Any, Any]]] = {}
            for query, candidates in record.items():
                canonical = self._schema_key_map.get(query, query)
                if not isinstance(candidates, list):
                    merged[canonical] = candidates
                    continue
                target = merged.setdefault(canonical, [])
                field_seen = seen.setdefault(canonical, set())
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        signature = (
                            candidate.get("text"),
                            candidate.get("start"),
                            candidate.get("end"),
                        )
                    else:
                        signature = (candidate, None, None)
                    if signature in field_seen:
                        continue
                    field_seen.add(signature)
                    target.append(candidate)
            merged_results.append(merged)
        return merged_results
