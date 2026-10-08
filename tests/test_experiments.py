from pp_uie.experiments import run_schema_switches


def test_schema_switches_measure_set_and_inference_without_loading():
    events = []

    class Backend:
        def set_schema(self, schema):
            events.append(("schema", schema))

        def extract(self, texts):
            events.append(("extract", texts))
            return [{"拍照公式场景": []}]

    ticks = iter(index / 1000 for index in range(20))
    rows = run_schema_switches(
        Backend(), "sample", [["场景"], {"场景": ["曝光"]}], clock=lambda: next(ticks)
    )

    assert [event[0] for event in events] == ["schema", "extract", "schema", "extract"]
    assert len(rows) == 2
    assert rows[0]["schema_switch_ms"] == 1.0
    assert rows[0]["inference_ms"] == 1.0
