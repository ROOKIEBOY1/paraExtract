import json
from pathlib import Path

import pytest
import pp_uie.webui as webui
import scripts.web_ui as web_ui_script

from pp_uie.backend import PPUIEBackend, RuntimeConfig
from pp_uie.model_registry import ModelDefinition, model_options
from pp_uie.uie_backend import UIETaskflowBackend, UIETaskflowConfig
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


def test_load_web_samples_groups_the_original_six_and_all_40_edge_posts():
    root = Path(__file__).parents[1]
    assert hasattr(webui, "load_web_samples")
    rows = webui.load_web_samples(
        root / "testdata/pocket_user_six_samples.jsonl",
        root / "testdata/pocket_edge_cases.jsonl",
    )

    assert len(rows) == 46
    assert [row["id"] for row in rows[:6]] == [f"pocket_user_{index:02d}" for index in range(1, 7)]
    assert rows[0]["label"] == "原始文本 1"
    assert rows[0]["group"] == "原有测试文本（6条）"
    assert rows[6]["label"] == "P0-01 · 模糊语义 · 01"
    assert rows[6]["group"] == "P0 边界场景（30条）"
    assert rows[36]["group"] == "P1 OCR/别名场景（6条）"
    assert rows[42]["label"] == "MIXED · 综合场景 · 01"
    assert rows[42]["group"] == "综合压力场景（4条）"


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
        ModelDefinition("0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path), Backend),
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


def test_service_prepares_text_aware_schema_when_backend_supports_it(tmp_path):
    events = []

    class Backend:
        def __init__(self, config):
            pass

        def load(self):
            pass

        def set_schema(self, schema):
            events.append(("schema", schema))

        def prepare_schema(self, texts):
            events.append(("prepare", texts))

        def extract(self, texts):
            events.append(("extract", texts))
            return [{}]

    service = ExtractionService.create(
        ModelDefinition("uie-mini", "UIE-mini", object(), Backend)
    )

    service.extract("ISO: 800", ["感光度"], strict=False)

    assert events == [
        ("schema", ["感光度"]),
        ("prepare", ["ISO: 800"]),
        ("extract", ["ISO: 800"]),
    ]


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

    service = ExtractionService.create(
        ModelDefinition("0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path), Backend)
    )
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

    service = ExtractionService.create(
        ModelDefinition("0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path), Backend)
    )
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

    definitions = [
        ModelDefinition("0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path / "0.5b"), Backend),
        ModelDefinition("1.5b", "PP-UIE-1.5B", RuntimeConfig(tmp_path / "1.5b"), Backend),
    ]
    switcher = SwitchingExtractionService.create(
        definitions,
        initial_model="0.5b",
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
        [ModelDefinition(
            "0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path / "0.5b"), Backend
        )],
        initial_model="0.5b",
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
        [ModelDefinition("0.5b", "PP-UIE-0.5B", RuntimeConfig(tmp_path), Backend)],
        "0.5b",
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
    assert "function renderSampleOptions" in page
    assert "document.createElement('optgroup')" in page
    assert "option.textContent = row.label" in page
    assert "document.createElement('option')" in page
    assert "option.value = model.id" in page
    assert "option.textContent = model.label" in page
    assert "modelSelect.innerHTML" not in page
    assert "已驻留内存的 PP-UIE 模型" not in page


def test_build_model_definitions_exposes_four_local_backends(tmp_path):
    definitions = web_ui_script.build_model_definitions(tmp_path, 1024, 50)

    assert [(item.model_id, item.label) for item in definitions] == [
        ("0.5b", "PP-UIE-0.5B"),
        ("1.5b", "PP-UIE-1.5B"),
        ("uie-mini", "UIE-mini"),
        ("uie-base", "UIE-base"),
    ]
    assert [item.backend_factory for item in definitions] == [
        PPUIEBackend,
        PPUIEBackend,
        UIETaskflowBackend,
        UIETaskflowBackend,
    ]
    assert definitions[0].config == RuntimeConfig(
        (tmp_path / "models/PP-UIE-0.5B").resolve(),
        "cpu",
        "float32",
        1,
        1024,
        50,
    )
    assert definitions[1].config.model_path == (tmp_path / "models/PP-UIE-1.5B").resolve()
    assert definitions[2].config == UIETaskflowConfig(
        (tmp_path / "models/UIE-mini").resolve(),
        device="cpu",
        batch_size=1,
        max_seq_len=1024,
        position_prob=0.5,
    )
    assert definitions[3].config == UIETaskflowConfig(
        (tmp_path / "models/UIE-base").resolve(),
        device="cpu",
        batch_size=1,
        max_seq_len=1024,
        position_prob=0.5,
        model="uie-base",
    )


def test_web_application_exposes_ready_config_and_delegates_extraction():
    class Service:
        model_name = "0.5b"
        load_ms = 1234.5
        available_models = [
            {"id": "0.5b", "label": "PP-UIE-0.5B"},
            {"id": "1.5b", "label": "PP-UIE-1.5B"},
        ]

        def extract(self, text, fields, strict=True):
            return {"text": text, "fields": fields, "strict": strict}

        def switch_model(self, model):
            return {"model": model, "changed": True, "load_ms": 4567.0}

    samples = [{"id": "one", "label": "测试文本 1", "text": "ISO：800"}]
    app = WebApplication(Service(), samples)

    assert app.config() == {
        "model": "0.5b",
        "load_ms": 1234.5,
        "models": [
            {"id": "0.5b", "label": "PP-UIE-0.5B"},
            {"id": "1.5b", "label": "PP-UIE-1.5B"},
        ],
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


def test_switcher_uses_each_models_own_backend_factory(tmp_path):
    events = []

    def make_factory(kind):
        class Backend:
            def __init__(self, config):
                events.append(("create", kind, config))

            def load(self):
                events.append(("load", kind))

            def close(self):
                events.append(("close", kind))

        return Backend

    pp_config = RuntimeConfig(tmp_path / "pp")
    uie_config = object()
    switcher = SwitchingExtractionService.create(
        [
            ModelDefinition("0.5b", "PP-UIE-0.5B", pp_config, make_factory("pp")),
            ModelDefinition("uie-mini", "UIE-mini", uie_config, make_factory("uie")),
        ],
        "0.5b",
        collect_memory=lambda: events.append(("collect",)),
    )

    switcher.switch_model("uie-mini")

    assert events == [
        ("create", "pp", pp_config),
        ("load", "pp"),
        ("close", "pp"),
        ("collect",),
        ("create", "uie", uie_config),
        ("load", "uie"),
    ]


def test_failed_switch_has_no_active_model_and_can_retry(tmp_path):
    should_fail = {"value": True}

    class WorkingBackend:
        def __init__(self, config):
            pass

        def load(self):
            pass

        def close(self):
            pass

    class RetryBackend(WorkingBackend):
        def load(self):
            if should_fail["value"]:
                raise RuntimeError("load failed")

    switcher = SwitchingExtractionService.create(
        [
            ModelDefinition("0.5b", "PP-UIE-0.5B", object(), WorkingBackend),
            ModelDefinition("uie-mini", "UIE-mini", object(), RetryBackend),
        ],
        "0.5b",
    )

    with pytest.raises(RuntimeError, match="load failed"):
        switcher.switch_model("uie-mini")
    assert switcher.model_name is None
    assert switcher.load_ms is None
    with pytest.raises(RuntimeError, match="no active model"):
        switcher.extract("ISO：800", ["感光度"])

    should_fail["value"] = False
    assert switcher.switch_model("uie-mini")["changed"] is True
    assert switcher.model_name == "uie-mini"


def test_available_models_exposes_ids_and_labels(tmp_path):
    class Backend:
        def __init__(self, config):
            pass

        def load(self):
            pass

    definitions = [
        ModelDefinition("0.5b", "PP-UIE-0.5B", object(), Backend),
        ModelDefinition("uie-mini", "UIE-mini", object(), Backend),
    ]
    switcher = SwitchingExtractionService.create(definitions, "0.5b")

    assert model_options(definitions) == [
        {"id": "0.5b", "label": "PP-UIE-0.5B"},
        {"id": "uie-mini", "label": "UIE-mini"},
    ]
    assert switcher.available_models == model_options(definitions)


def test_switcher_rejects_duplicate_model_ids():
    definition = ModelDefinition("same", "First", object(), lambda config: None)

    with pytest.raises(ValueError, match="duplicate model id: same"):
        SwitchingExtractionService.create(
            [definition, ModelDefinition("same", "Second", object(), lambda config: None)],
            "same",
        )
