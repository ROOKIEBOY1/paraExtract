"""Validation and traversal for PP-UIE dynamic schemas."""

from dataclasses import dataclass
from typing import Iterator, Union


SchemaValue = Union[str, list["SchemaValue"], dict[str, "SchemaValue"]]


@dataclass(frozen=True)
class SchemaNode:
    name: str
    children: tuple["SchemaNode", ...] = ()


@dataclass(frozen=True)
class PromptSpec:
    field: str
    prompt: str


def parse_schema(value: SchemaValue, path: str = "$") -> tuple[SchemaNode, ...]:
    if isinstance(value, str):
        return (SchemaNode(value),)
    if isinstance(value, list):
        return tuple(node for index, item in enumerate(value) for node in parse_schema(item, f"{path}[{index}]"))
    if isinstance(value, dict):
        nodes = []
        for name, children in value.items():
            if not isinstance(children, (str, list)):
                raise TypeError(f"{path}.{name} must be string or list")
            nodes.append(SchemaNode(name, parse_schema(children, f"{path}.{name}")))
        return tuple(nodes)
    raise TypeError(f"{path} must be string, list, or object")


def iter_prompts(nodes: tuple[SchemaNode, ...], parents: tuple[str, ...] = ()) -> Iterator[PromptSpec]:
    for node in nodes:
        if node.children and parents:
            for parent in parents:
                yield from iter_prompts(node.children, (parent,))
        else:
            for parent in parents or ("",):
                yield PromptSpec(node.name, f"{parent}的{node.name}" if parent else node.name)

