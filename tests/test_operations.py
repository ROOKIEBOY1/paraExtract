import json

from text_extractor.core.operations import OperationLog


def test_operation_log_writes_reproducible_jsonl(tmp_path):
    path = tmp_path / "operations.jsonl"
    OperationLog(path).record(["python", "-V"], 0, "Python 3.9.6", "", "start", "end")
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {
        "command": ["python", "-V"],
        "exit_code": 0,
        "stdout": "Python 3.9.6",
        "stderr": "",
        "started_at": "start",
        "ended_at": "end",
    }
