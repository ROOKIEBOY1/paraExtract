# Environment

- Architecture: `arm64`
- Conda: `not_installed`
- CUDA GPU: `not_available`
- NPU: `not_available`
- Apple GPU 不等同于 Paddle CUDA GPU；macOS 本次使用 CPU/FP32。

## Raw probes

```json
{
  "architecture": "arm64",
  "python": {
    "status": "available",
    "stdout": "Python 3.9.6",
    "stderr": ""
  },
  "pip": {
    "status": "available",
    "stdout": "pip 21.2.4 from /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages/pip (python 3.9)",
    "stderr": ""
  },
  "conda": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "conda: not found"
  },
  "cuda_gpu": {
    "status": "not_available",
    "stdout": "",
    "stderr": "nvidia-smi: not found"
  },
  "npu": {
    "status": "not_available",
    "stdout": "",
    "stderr": "npu-smi: not found"
  },
  "paddle": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nModuleNotFoundError: No module named 'paddle'"
  },
  "paddlenlp": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nModuleNotFoundError: No module named 'paddlenlp'"
  },
  "disk": {
    "status": "available",
    "stdout": "Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on\n/dev/disk3s5   228Gi    66Gi   138Gi    33%    901k  1.4G    0%   /System/Volumes/Data",
    "stderr": ""
  },
  "hardware": {
    "status": "available",
    "stdout": "Hardware:\n\n    Hardware Overview:\n\n      Model Name: Mac mini\n      Model Identifier: Mac16,10\n      Model Number: MU9D3CH/A\n      Chip: Apple M4\n      Total Number of Cores: 10 (4 performance and 6 efficiency)\n      Memory: 16 GB\n      System Firmware Version: 13822.41.1\n      OS Loader Version: 13822.41.1\n\nGraphics/Displays:\n\n    Apple M4:\n\n      Chipset Model: Apple M4\n      Type: GPU\n      Bus: Built-In\n      Total Number of Cores: 10\n      Vendor: Apple (0x106b)\n      Metal: Supported",
    "stderr": ""
  },
  "memory": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "sysctl: sysctl fmt -1 1024 1: Operation not permitted"
  },
  "cpu": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "sysctl: sysctl fmt -1 1024 1: Operation not permitted"
  },
  "os": {
    "status": "available",
    "stdout": "ProductName:\t\tmacOS\nProductVersion:\t\t26.1\nBuildVersion:\t\t25B78",
    "stderr": ""
  },
  "cuda_compiler": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "nvcc: not found"
  },
  "cudnn": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "ls: /usr/local/cuda/include/cudnn_version.h: No such file or directory"
  },
  "cann": {
    "status": "not_installed",
    "stdout": "",
    "stderr": "ls: /usr/local/Ascend/ascend-toolkit/latest: No such file or directory"
  }
}
```
