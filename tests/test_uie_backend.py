from pathlib import Path
import hashlib

import pytest

from pp_uie.uie_backend import UIETaskflowBackend, UIETaskflowConfig


class FakeTaskflow:
    def __init__(self, result=None):
        self.result = result if result is not None else [{}]
        self.schemas = []
        self.calls = []

    def set_schema(self, schema):
        self.schemas.append(schema)

    def __call__(self, texts):
        self.calls.append(texts)
        return self.result


def test_uie_backend_loads_explicit_local_path_once(tmp_path):
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return FakeTaskflow()

    config = UIETaskflowConfig(
        tmp_path / "model",
        device="cpu",
        batch_size=3,
        max_seq_len=384,
        position_prob=0.5,
    )
    backend = UIETaskflowBackend(config, taskflow_factory=factory)

    backend.load()
    backend.load()

    assert created == [{
        "task": "information_extraction",
        "model": "uie-mini",
        "task_path": str((tmp_path / "model").resolve()),
        "schema": [],
        "device_id": -1,
        "batch_size": 3,
        "max_seq_len": 384,
        "position_prob": 0.5,
    }]


def test_uie_backend_changes_schema_without_reloading(tmp_path):
    taskflow = FakeTaskflow()
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return taskflow

    backend = UIETaskflowBackend(UIETaskflowConfig(tmp_path), factory)
    backend.load()
    backend.set_schema(["感光度"])
    backend.set_schema(["白平衡", "帧率"])

    assert len(factory_calls) == 1
    assert taskflow.schemas == [["感光度"], ["白平衡", "帧率"]]


def test_uie_backend_preserves_span_metadata(tmp_path):
    span = {"text": "-0.3EV", "start": 4, "end": 10, "probability": 0.987}
    expected = [{"曝光补偿": [span]}]
    taskflow = FakeTaskflow(expected)
    backend = UIETaskflowBackend(UIETaskflowConfig(tmp_path), lambda **kwargs: taskflow)
    backend.load()

    actual = backend.extract(["曝光：-0.3EV"])

    assert actual is expected
    assert actual[0]["曝光补偿"][0] is span


def test_uie_backend_close_drops_taskflow_reference(tmp_path):
    backend = UIETaskflowBackend(UIETaskflowConfig(tmp_path), lambda **kwargs: FakeTaskflow())
    backend.load()
    assert backend.taskflow is not None

    backend.close()

    assert backend.taskflow is None


def test_uie_backend_rejects_wrong_result_count(tmp_path):
    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: FakeTaskflow([{}])
    )
    backend.load()

    with pytest.raises(ValueError, match="Taskflow returned 1 results for 2 inputs"):
        backend.extract(["一", "二"])


def test_uie_backend_rejects_non_mapping_records(tmp_path):
    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: FakeTaskflow([[{"text": "错误层级"}]])
    )
    backend.load()

    with pytest.raises(TypeError, match="Taskflow result at index 0 must be a dictionary"):
        backend.extract(["文本"])


def test_uie_backend_requires_load_before_extract(tmp_path):
    backend = UIETaskflowBackend(UIETaskflowConfig(tmp_path), lambda **kwargs: FakeTaskflow())

    with pytest.raises(RuntimeError, match="backend is not loaded"):
        backend.extract(["文本"])


def test_uie_taskflow_config_is_immutable(tmp_path):
    config = UIETaskflowConfig(Path(tmp_path))

    with pytest.raises(Exception):
        config.batch_size = 2


def test_uie_backend_repairs_taskflow_static_cache_marker(tmp_path):
    model_bytes = b"model"
    (tmp_path / "model_state.pdparams").write_bytes(model_bytes)
    static = tmp_path / "static"
    static.mkdir()
    (static / "inference.json").write_text("{}", encoding="utf-8")
    (static / "inference.pdiparams").write_bytes(b"params")
    digest = hashlib.md5(model_bytes).hexdigest()
    marker = tmp_path / ".cache_info"
    marker.write_text(digest + "taskflow", encoding="utf-8")

    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: FakeTaskflow()
    )
    backend.load()

    assert marker.read_text(encoding="utf-8") == digest + ".json"


def test_uie_backend_passes_configured_uie_base_model(tmp_path):
    created = []

    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path, model="uie-base"),
        lambda **kwargs: created.append(kwargs) or FakeTaskflow(),
    )
    backend.load()

    assert created[0]["model"] == "uie-base"
    assert created[0]["task_path"] == str(tmp_path.resolve())


def test_uie_backend_expands_only_english_aliases_found_in_source(tmp_path):
    taskflow = FakeTaskflow()
    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: taskflow
    )
    backend.load()
    backend.set_schema(["感光度", "白平衡", "锐度"])

    backend.prepare_schema(["ISO: 50-800，White Balance：AWB。没有锐度设置。"])

    assert taskflow.schemas[-1] == [
        "感光度", "ISO值", "白平衡", "White Balance值", "锐度"
    ]


def test_uie_backend_merges_alias_results_without_rebuilding_spans(tmp_path):
    iso_span = {"text": "50-800", "start": 5, "end": 11, "probability": 0.9}
    taskflow = FakeTaskflow([{"ISO值": [iso_span]}])
    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: taskflow
    )
    backend.load()
    backend.set_schema(["感光度"])
    backend.prepare_schema(["ISO: 50-800"])

    result = backend.extract(["ISO: 50-800"])

    assert result == [{"感光度": [iso_span]}]
    assert result[0]["感光度"][0] is iso_span


def test_uie_backend_deduplicates_same_span_from_canonical_and_alias_queries(tmp_path):
    canonical = {"text": "800", "start": 4, "end": 7, "probability": 0.8}
    alias = {"text": "800", "start": 4, "end": 7, "probability": 0.9}
    taskflow = FakeTaskflow([{"感光度": [canonical], "ISO值": [alias]}])
    backend = UIETaskflowBackend(
        UIETaskflowConfig(tmp_path), lambda **kwargs: taskflow
    )
    backend.load()
    backend.set_schema(["感光度"])
    backend.prepare_schema(["ISO:800"])

    result = backend.extract(["ISO:800"])

    assert result == [{"感光度": [canonical]}]
