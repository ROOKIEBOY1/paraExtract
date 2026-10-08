import pytest

from pp_uie.cli import batched, parse_args, resolve_runtime


@pytest.mark.parametrize("argv", [
    ["--model", "0.5b", "--schema", "s.json"],
    ["--model", "0.5b", "--schema", "s.json", "--text", "x", "--text-file", "x.jsonl"],
])
def test_exactly_one_text_source_is_required(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize(("device", "precision", "message"), [
    ("gpu", "float32", "GPU is not available"),
    ("npu", "float32", "NPU is not available"),
    ("cpu", "float16", "float16 is not supported for this CPU experiment"),
])
def test_unavailable_runtime_fails_explicitly(device, precision, message):
    with pytest.raises(ValueError, match=message):
        resolve_runtime(device, precision, available={"cpu"})


def test_batched_preserves_rows_and_uses_requested_size():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_non_positive_size():
    with pytest.raises(ValueError, match="batch size must be positive"):
        list(batched([1], 0))
