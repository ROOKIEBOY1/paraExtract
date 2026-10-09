from text_extractor.evaluation.metrics import evaluate_records, normalize_value


def test_strict_and_normalized_metrics_are_separate():
    expected = [{"scene": "阳光微曝", "parameters": {"曝光": ["+1.0EV"]}}]
    predicted = [{"scene": "阳光微曝", "parameters": {"曝光": ["+1.0 EV"]}}]
    report = evaluate_records(expected, predicted)
    assert report.strict.tp == 0
    assert report.strict.fp == 1
    assert report.strict.fn == 1
    assert report.normalized.tp == 1
    assert normalize_value("+1.0 EV") == "+1.0ev"


def test_wrong_parent_is_counted_as_association_error():
    expected = [{"scene": "演唱会参数", "parameters": {"分辨率": ["4K30p/60p"]}}]
    predicted = [{"scene": "演唱会拍摄舞台", "parameters": {"分辨率": ["4K30p/60p"]}}]
    report = evaluate_records(expected, predicted)
    assert report.association_errors == 1
    assert report.strict.tp == 0
    assert report.field_strict.tp == 1
    assert report.field_normalized.tp == 1
