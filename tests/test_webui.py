import json
from pathlib import Path

import pytest

from pp_uie.backend import RuntimeConfig
from pp_uie.webui import (
    ExtractionService,
    FIELD_GROUPS,
    SwitchingExtractionService,
    WebApplication,
    load_text_samples,
)


def test_load_text_samples_returns_the_six_user_records():
    path = Path(__file__).parents[1] / "testdata/pocket_user_six_samples.jsonl"
    rows = load_text_samples(path)
    assert [row["id"] for row in rows] == [f"pocket_user_{index:02d}" for index in range(1, 7)]
    assert rows[0]["label"] == "测试文本 1"
    assert "焦段1.0x～1.3x" in rows[0]["text"]


def test_service_loads_model_once_and_reuses_it_for_dynamic_fields(tmp_path):
    events = []

    class Backend:
        def __init__(self, config):
            events.append(("create", config.model_path))

        def load(self):
            events.append(("load",))

        def set_schema(self, schema):
            events.append(("schema", schema))

        def extract(self, texts):
            events.append(("extract", texts))
            return [{field: [{"text": f"value-{field}"}] for field in events[-2][1]}]

    ticks = iter(index / 1000 for index in range(30))
    service = ExtractionService.create(
        RuntimeConfig(tmp_path),
        model_name="0.5b",
        backend_factory=Backend,
        clock=lambda: next(ticks),
    )
    first = service.extract("sample one", ["曝光", "ISO"], strict=False)
    second = service.extract("sample two", ["白平衡"], strict=False)

    assert [event[0] for event in events].count("load") == 1
    assert [event for event in events if event[0] == "schema"] == [
        ("schema", ["曝光补偿", "感光度"]),
        ("schema", ["白平衡"]),
    ]
    assert first["normalized_output"] == {
        "曝光补偿": ["value-曝光补偿"], "感光度": ["value-感光度"]
    }
    assert second["model"] == "0.5b"
    assert first["timing_ms"]["schema"] == 1.0
    assert first["timing_ms"]["inference"] == 1.0
    assert service.load_ms == 1.0


@pytest.mark.parametrize(("text", "fields", "message"), [
    ("", ["曝光"], "text must not be empty"),
    ("sample", [], "select at least one field"),
    ("sample", ["曝光", "曝光"], "fields must be unique"),
])
def test_service_rejects_invalid_requests(tmp_path, text, fields, message):
    class Backend:
        def __init__(self, config):
            pass

        def load(self):
            pass

    service = ExtractionService.create(RuntimeConfig(tmp_path), "0.5b", backend_factory=Backend)
    with pytest.raises(ValueError, match=message):
        service.extract(text, fields)


def test_service_strict_mode_returns_verified_and_rejected_candidates(tmp_path):
    class Backend:
        def __init__(self, config):
            pass

        def load(self):
            pass

        def set_schema(self, schema):
            self.schema = schema

        def extract(self, texts):
            return [{
                "感光度": [{"text": "50-800"}],
                "分辨率": [{"text": "50-800"}],
            }]

    service = ExtractionService.create(RuntimeConfig(tmp_path), "0.5b", backend_factory=Backend)
    result = service.extract("ISO：50-800", ["ISO", "Resolution"], strict=True)
    assert result["strict_mode"] is True
    assert result["fields"] == ["感光度", "分辨率"]
    assert result["normalized_output"] == {"感光度": ["50-800"]}
    assert result["rejected_candidates"][0]["field"] == "分辨率"


def test_switcher_releases_current_model_before_loading_selected_model(tmp_path):
    events = []

    class Backend:
        def __init__(self, config):
            self.name = config.model_path.name
            events.append(("create", self.name))

        def load(self):
            events.append(("load", self.name))

        def close(self):
            events.append(("close", self.name))

    configs = {
        "0.5b": RuntimeConfig(tmp_path / "0.5b"),
        "1.5b": RuntimeConfig(tmp_path / "1.5b"),
    }
    switcher = SwitchingExtractionService.create(
        configs,
        initial_model="0.5b",
        backend_factory=Backend,
        collect_memory=lambda: events.append(("collect",)),
    )

    result = switcher.switch_model("1.5b")

    assert result["model"] == "1.5b"
    assert result["changed"] is True
    assert events == [
        ("create", "0.5b"), ("load", "0.5b"),
        ("close", "0.5b"), ("collect",),
        ("create", "1.5b"), ("load", "1.5b"),
    ]
    assert switcher.model_name == "1.5b"


def test_switcher_does_not_reload_the_already_active_model(tmp_path):
    loads = []

    class Backend:
        def __init__(self, config):
            self.name = config.model_path.name

        def load(self):
            loads.append(self.name)

    switcher = SwitchingExtractionService.create(
        {"0.5b": RuntimeConfig(tmp_path / "0.5b")},
        initial_model="0.5b",
        backend_factory=Backend,
    )

    assert switcher.switch_model("0.5b")["changed"] is False
    assert loads == ["0.5b"]


def test_switcher_rejects_unknown_model(tmp_path):
    class Backend:
        def __init__(self, config):
            pass

        def load(self):
            pass

    switcher = SwitchingExtractionService.create(
        {"0.5b": RuntimeConfig(tmp_path)}, "0.5b", backend_factory=Backend
    )
    with pytest.raises(ValueError, match="unsupported model"):
        switcher.switch_model("3b")


def test_web_page_contains_required_controls():
    page = (Path(__file__).parents[1] / "web/index.html").read_text(encoding="utf-8")
    assert 'id="sample-select"' in page
    assert 'id="input-text"' in page
    assert 'id="field-list"' in page
    assert 'id="extract-button"' in page
    assert 'id="elapsed"' in page
    assert "/api/extract" in page
    assert 'id="strict-mode"' in page
    assert 'id="rejected-output"' in page
    assert 'id="model-select"' in page
    assert "/api/model" in page
    assert "function showError" in page
    assert "errorBox.innerHTML" not in page


def test_web_application_exposes_ready_config_and_delegates_extraction():
    class Service:
        model_name = "0.5b"
        load_ms = 1234.5
        available_models = ["0.5b", "1.5b"]

        def extract(self, text, fields, strict=True):
            return {"text": text, "fields": fields, "strict": strict}

        def switch_model(self, model):
            return {"model": model, "changed": True, "load_ms": 4567.0}

    samples = [{"id": "one", "label": "测试文本 1", "text": "ISO：800"}]
    app = WebApplication(Service(), samples)

    assert app.config() == {
        "model": "0.5b",
        "load_ms": 1234.5,
        "models": ["0.5b", "1.5b"],
        "samples": samples,
        "field_groups": FIELD_GROUPS,
    }
    assert app.extract({"text": "ISO：800", "fields": ["ISO"], "strict": False}) == {
        "text": "ISO：800", "fields": ["ISO"], "strict": False
    }
    assert app.switch_model({"model": "1.5b"}) == {
        "model": "1.5b", "changed": True, "load_ms": 4567.0
    }


@pytest.mark.parametrize("payload", [None, {}, {"text": "x", "fields": "ISO"}])
def test_web_application_rejects_malformed_payload(payload):
    app = WebApplication(object(), [])
    with pytest.raises(ValueError, match="JSON object with text and fields"):
        app.extract(payload)


def test_web_application_rejects_non_boolean_strict_mode():
    class Service:
        def extract(self, text, fields, strict=True):
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="strict must be boolean"):
        WebApplication(Service(), []).extract({"text": "x", "fields": ["ISO"], "strict": "yes"})


@pytest.mark.parametrize("payload", [None, {}, {"model": 1.5}])
def test_web_application_rejects_malformed_model_switch(payload):
    with pytest.raises(ValueError, match="JSON object with model"):
        WebApplication(object(), []).switch_model(payload)
