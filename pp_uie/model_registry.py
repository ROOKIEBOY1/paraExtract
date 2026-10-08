"""Typed definitions for heterogeneous extraction backends."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    label: str
    config: Any
    backend_factory: Callable[[Any], Any]


def model_options(definitions: Sequence[ModelDefinition]) -> list[dict[str, str]]:
    return [
        {"id": definition.model_id, "label": definition.label}
        for definition in definitions
    ]


def index_definitions(
    definitions: Sequence[ModelDefinition],
) -> dict[str, ModelDefinition]:
    indexed = {}
    for definition in definitions:
        if definition.model_id in indexed:
            raise ValueError(f"duplicate model id: {definition.model_id}")
        indexed[definition.model_id] = definition
    return indexed
