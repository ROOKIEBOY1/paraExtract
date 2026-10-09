#!/usr/bin/env python3
"""Capture host details without modifying the host."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_extractor.runtime.environment import (
    ProbeResult,
    collect_environment,
    render_environment_markdown,
)


def run(command: list[str]) -> ProbeResult:
    if shutil.which(command[0]) is None:
        return ProbeResult(127, "", f"{command[0]}: not found")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return ProbeResult(completed.returncode, completed.stdout, completed.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/environment.md"))
    args = parser.parse_args()
    data = collect_environment(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_environment_markdown(data), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
