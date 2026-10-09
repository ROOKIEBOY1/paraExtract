from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/web_ui.py",
        "scripts/download_models.py",
        "scripts/infer.py",
        "scripts/benchmark.py",
        "scripts/run_experiments.py",
    ],
)
def test_script_help_loads_with_the_refactored_package(script):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
