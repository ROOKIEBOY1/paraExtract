from text_extractor.runtime.environment import ProbeResult, collect_environment, render_environment_markdown


def test_environment_report_distinguishes_absent_accelerators():
    responses = {
        ("uname", "-m"): ProbeResult(0, "arm64\n", ""),
        ("python3", "--version"): ProbeResult(0, "Python 3.9.6\n", ""),
        ("conda", "--version"): ProbeResult(127, "", "not found"),
        ("nvidia-smi",): ProbeResult(127, "", "not found"),
        ("npu-smi", "info"): ProbeResult(127, "", "not found"),
    }
    data = collect_environment(lambda cmd: responses.get(tuple(cmd), ProbeResult(0, "fixture", "")))
    report = render_environment_markdown(data)
    assert data["architecture"] == "arm64"
    assert data["conda"]["status"] == "not_installed"
    assert data["cuda_gpu"]["status"] == "not_available"
    assert data["npu"]["status"] == "not_available"
    assert "Apple GPU 不等同于 Paddle CUDA GPU" in report


def test_environment_report_redacts_hardware_identifiers():
    data = {
        "architecture": "arm64",
        "conda": {"status": "not_installed"},
        "cuda_gpu": {"status": "not_available"},
        "npu": {"status": "not_available"},
        "hardware": {
            "status": "available",
            "stdout": "Chip: Apple M4\nSerial Number (system): SECRET\nHardware UUID: UUID-SECRET",
            "stderr": "",
        },
    }
    report = render_environment_markdown(data)
    assert "Apple M4" in report
    assert "SECRET" not in report
    assert "UUID-SECRET" not in report
