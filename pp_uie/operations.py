"""Append-only command evidence log."""

import json
from pathlib import Path
from typing import Sequence


class OperationLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(
        self,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        started_at: str,
        ended_at: str,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "command": list(command),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "started_at": started_at,
            "ended_at": ended_at,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

