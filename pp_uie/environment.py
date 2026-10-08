"""Environment probes that never mutate the host."""

from dataclasses import dataclass
import json
from typing import Callable, Mapping


@dataclass(frozen=True)
class ProbeResult:
    returncode: int
    stdout: str
    stderr: str


def _status(result: ProbeResult) -> dict[str, str]:
    return {
        "status": "available" if result.returncode == 0 else "not_installed",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_environment(run: Callable[[list[str]], ProbeResult]) -> dict[str, object]:
    architecture = run(["uname", "-m"]).stdout.strip()
    cuda = _status(run(["nvidia-smi"]))
    cuda["status"] = "available" if cuda["status"] == "available" else "not_available"
    npu = _status(run(["npu-smi", "info"]))
    npu["status"] = "available" if npu["status"] == "available" else "not_available"
    probes = {
        "architecture": architecture,
        "python": _status(run(["python3", "--version"])),
        "pip": _status(run(["python3", "-m", "pip", "--version"])),
        "conda": _status(run(["conda", "--version"])),
        "cuda_gpu": cuda,
        "npu": npu,
        "paddle": _status(run(["python3", "-c", "import paddle; print(paddle.__version__)"])),
        "paddlenlp": _status(run(["python3", "-c", "import paddlenlp; print(paddlenlp.__version__)"])),
        "disk": _status(run(["df", "-h", "."])),
        "hardware": _status(run(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])),
        "memory": _status(run(["sysctl", "-n", "hw.memsize"])),
        "cpu": _status(run(["sysctl", "-n", "machdep.cpu.brand_string"])),
        "os": _status(run(["sw_vers"])),
        "cuda_compiler": _status(run(["nvcc", "--version"])),
        "cudnn": _status(run(["sh", "-c", "ls /usr/local/cuda/include/cudnn_version.h"])),
        "cann": _status(run(["sh", "-c", "ls /usr/local/Ascend/ascend-toolkit/latest"])),
    }
    return probes


def render_environment_markdown(data: Mapping[str, object]) -> str:
    safe_data = json.loads(json.dumps(data))
    hardware = safe_data.get("hardware")
    if isinstance(hardware, dict) and isinstance(hardware.get("stdout"), str):
        private_prefixes = ("Serial Number", "Hardware UUID", "Provisioning UDID")
        hardware["stdout"] = "\n".join(
            line for line in hardware["stdout"].splitlines()
            if not line.strip().startswith(private_prefixes)
        )
    return (
        "# Environment\n\n"
        f"- Architecture: `{data['architecture']}`\n"
        f"- Conda: `{data['conda']['status']}`\n"
        f"- CUDA GPU: `{data['cuda_gpu']['status']}`\n"
        f"- NPU: `{data['npu']['status']}`\n"
        "- Apple GPU 不等同于 Paddle CUDA GPU；macOS 本次使用 CPU/FP32。\n"
        "\n## Raw probes\n\n"
        f"```json\n{json.dumps(safe_data, ensure_ascii=False, indent=2)}\n```\n"
    )
