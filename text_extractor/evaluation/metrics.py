"""Field-level extraction metrics for scene-associated parameter values."""

from dataclasses import dataclass
import unicodedata
from typing import Any, Callable


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    strict: Counts
    normalized: Counts
    field_strict: Counts
    field_normalized: Counts
    association_errors: int
    extra_fields: int
    missing_fields: int
    schema_violations: int
    json_valid: bool
    nested_structure_correct: bool


def normalize_value(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    keep = {"+", "-", ".", "/", "~"}
    return "".join(char for char in value if not char.isspace() and (not unicodedata.category(char).startswith("P") or char in keep))


def _tuples(records: list[dict[str, Any]], transform: Callable[[str], str]) -> set[tuple[str, str, str]]:
    return {
        (row.get("scene", ""), field, transform(str(value)))
        for row in records
        for field, values in row.get("parameters", {}).items()
        for value in values
    }


def _counts(expected, predicted) -> Counts:
    return Counts(len(expected & predicted), len(predicted - expected), len(expected - predicted))


def _field_tuples(records: list[dict[str, Any]], transform: Callable[[str], str]) -> set[tuple[str, str]]:
    return {
        (field, transform(str(value)))
        for row in records
        for field, values in row.get("parameters", {}).items()
        for value in values
    }


def evaluate_records(expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> EvaluationReport:
    strict_expected = _tuples(expected, lambda value: value)
    strict_predicted = _tuples(predicted, lambda value: value)
    normalized_expected = _tuples(expected, normalize_value)
    normalized_predicted = _tuples(predicted, normalize_value)
    field_strict_expected = _field_tuples(expected, lambda value: value)
    field_strict_predicted = _field_tuples(predicted, lambda value: value)
    field_normalized_expected = _field_tuples(expected, normalize_value)
    field_normalized_predicted = _field_tuples(predicted, normalize_value)
    expected_by_field_value = {(field, value): scene for scene, field, value in normalized_expected}
    association_errors = sum(
        1 for scene, field, value in normalized_predicted
        if (field, value) in expected_by_field_value and expected_by_field_value[(field, value)] != scene
    )
    expected_fields = {(row.get("scene", ""), field) for row in expected for field in row.get("parameters", {})}
    predicted_fields = {(row.get("scene", ""), field) for row in predicted for field in row.get("parameters", {})}
    return EvaluationReport(
        _counts(strict_expected, strict_predicted),
        _counts(normalized_expected, normalized_predicted),
        _counts(field_strict_expected, field_strict_predicted),
        _counts(field_normalized_expected, field_normalized_predicted),
        association_errors,
        len(predicted_fields - expected_fields),
        len(expected_fields - predicted_fields),
        len(predicted_fields - expected_fields),
        True,
        all(isinstance(row.get("parameters", {}), dict) for row in predicted),
    )
