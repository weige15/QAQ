#!/usr/bin/env python3
"""Inspect the host and Python environment without changing the machine."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_PATH = Path(__file__).resolve().parent.parent

COMMANDS = [
    "source ~/.venv/bin/activate",
    "which python",
    "python --version",
    "uname -a",
    ". /etc/os-release; printf 'NAME=%s\\nVERSION=%s\\nPRETTY_NAME=%s\\n' \"$NAME\" \"$VERSION\" \"$PRETTY_NAME\"",
    "lscpu",
    "cat /proc/meminfo",
    f"df -P -k {PROJECT_PATH}",
    "command -v nvidia-smi",
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
    "nvidia-smi",
    "command -v nvcc",
    "nvcc --version",
    "printf '%s\\n' \"$CUDA_HOME\" \"$PATH\"",
    "command -v gcc; gcc --version",
    "command -v g++; g++ --version",
    "which python",
    "python --version",
    "python -m pip --version",
    "python -c 'import sys; print(sys.prefix)'",
    "python -c 'import torch; ...'",
    "python -c 'import transformers; ...'",
]


def run(command: list[str], *, timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": " ".join(command), "returncode": None, "stdout": "", "stderr": "not_found"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": " ".join(command), "returncode": None, "stdout": "", "stderr": str(exc)}


def first_line(value: str) -> str | None:
    return value.splitlines()[0] if value.splitlines() else None


def parse_version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)+)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def parse_lscpu() -> dict[str, Any]:
    result = run(["lscpu"])
    fields: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    logical = fields.get("CPU(s)")
    sockets = fields.get("Socket(s)")
    cores_per_socket = fields.get("Core(s) per socket")
    physical = None
    if sockets and cores_per_socket:
        try:
            physical = int(sockets) * int(cores_per_socket)
        except ValueError:
            pass
    return {
        "model": fields.get("Model name"),
        "logical_count": int(logical) if logical and logical.isdigit() else os.cpu_count(),
        "physical_count": physical,
        "sockets": int(sockets) if sockets and sockets.isdigit() else None,
        "cores_per_socket": int(cores_per_socket) if cores_per_socket and cores_per_socket.isdigit() else None,
        "lscpu_output": result["stdout"] or None,
    }


def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def parse_memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            match = re.search(r"(\d+)", value)
            if match:
                values[key] = int(match.group(1)) * 1024
    except (OSError, ValueError):
        pass
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "total_gib": round(values["MemTotal"] / 1024**3, 2) if "MemTotal" in values else None,
        "available_gib": round(values["MemAvailable"] / 1024**3, 2) if "MemAvailable" in values else None,
    }


def parse_disk() -> dict[str, Any]:
    result = run(["df", "-P", "-k", str(PROJECT_PATH)])
    line = result["stdout"].splitlines()[-1] if result["stdout"] else ""
    fields = line.split()
    if len(fields) >= 6:
        try:
            return {
                "path": str(PROJECT_PATH),
                "filesystem": fields[0],
                "size_bytes": int(fields[1]) * 1024,
                "used_bytes": int(fields[2]) * 1024,
                "available_bytes": int(fields[3]) * 1024,
                "available_gib": round(int(fields[3]) / 1024**2, 2),
                "use_percent": fields[4],
                "mount_point": fields[5],
            }
        except ValueError:
            pass
    return {"path": str(PROJECT_PATH), "filesystem": None, "available_bytes": None, "available_gib": None}


def inspect_nvidia() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], {"available": False, "executable": "not_found", "driver_version": None, "cuda_version_reported_by_driver": None}
    query = run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in query["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            try:
                memory = int(float(parts[1]))
            except ValueError:
                memory = None
            gpus.append({"index": len(gpus), "model": parts[0], "total_vram_mib": memory})
    full = run([smi])
    driver_match = re.search(r"Driver Version:\s*([^\s|]+)", full["stdout"])
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", full["stdout"])
    return gpus, {
        "available": True,
        "executable": smi,
        "driver_version": driver_match.group(1) if driver_match else None,
        "cuda_version_reported_by_driver": cuda_match.group(1) if cuda_match else None,
        "query_output": query["stdout"] or None,
    }


def inspect_nvcc() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return {"available": False, "executable": "not_found", "version": None, "raw_output": None}
    result = run([nvcc, "--version"])
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)+)", result["stdout"])
    return {
        "available": result["returncode"] == 0,
        "executable": nvcc,
        "version": match.group(1) if match else None,
        "raw_output": result["stdout"] or result["stderr"] or None,
    }


def inspect_compiler(name: str) -> dict[str, Any]:
    executable = shutil.which(name)
    if not executable:
        return {"available": False, "executable": "not_found", "version": None, "raw_output": None}
    result = run([executable, "--version"])
    version = parse_version(first_line(result["stdout"]))
    return {
        "available": result["returncode"] == 0,
        "executable": executable,
        "version": ".".join(map(str, version)) if version else None,
        "raw_output": result["stdout"] or result["stderr"] or None,
    }


def inspect_python() -> dict[str, Any]:
    pip = run([sys.executable, "-m", "pip", "--version"])
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "pip": pip["stdout"] or pip["stderr"] or None,
    }


def inspect_pytorch() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host
        return {
            "imports": False,
            "version": None,
            "built_with_cuda": None,
            "cuda_available": None,
            "device_count": None,
            "devices": [],
            "cuda_smoke_test": {"result": "UNKNOWN", "reason": "torch could not be imported", "error": repr(exc)},
        }

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            })
        except Exception as exc:  # pragma: no cover - depends on host
            devices.append({"index": index, "name": None, "compute_capability": None, "error": repr(exc)})

    smoke = {"result": "UNKNOWN", "reason": None}
    if not cuda_available:
        smoke = {"result": "FAIL", "reason": "torch.cuda.is_available() returned false"}
    else:
        try:
            left = torch.tensor([1.0, 2.0], device="cuda")
            right = torch.tensor([3.0, 4.0], device="cuda")
            result = left + right
            torch.cuda.synchronize()
            smoke = {"result": "PASS", "reason": "two small CUDA tensors were added successfully", "result_values": result.cpu().tolist()}
        except Exception as exc:  # pragma: no cover - depends on host
            smoke = {"result": "FAIL", "reason": "minimal CUDA tensor addition failed", "error": repr(exc)}

    return {
        "imports": True,
        "version": torch.__version__,
        "built_with_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "cuda_smoke_test": smoke,
    }


def inspect_transformers() -> dict[str, Any]:
    try:
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host
        return {"imports": False, "version": None, "status": "absent", "error": repr(exc)}
    return {"imports": True, "version": transformers.__version__, "status": "present"}


def prerequisite(status: str, observed: Any, reason: str) -> dict[str, Any]:
    return {"result": status, "observed": observed, "reason": reason}


def main() -> None:
    gpus, driver = inspect_nvidia()
    toolkit = inspect_nvcc()
    gcc = inspect_compiler("gcc")
    gxx = inspect_compiler("g++")
    python_info = inspect_python()
    pytorch = inspect_pytorch()
    transformers = inspect_transformers()

    python_ok = python_info["version_info"][:2] == [3, 11]
    toolkit_version = parse_version(toolkit.get("version"))
    gcc_version = parse_version(gcc.get("version"))
    toolkit_ok = toolkit_version is not None and toolkit_version >= (12, 0)
    gcc_ok = gcc_version is not None and gcc_version >= (9, 0)

    if python_ok:
        python_prereq = prerequisite("PASS", python_info["version"], "Python 3.11 is installed")
    else:
        python_prereq = prerequisite("FAIL", python_info["version"], "documented prerequisite is Python 3.11")
    if toolkit_ok:
        toolkit_prereq = prerequisite("PASS", toolkit["version"], "nvcc reports CUDA Toolkit 12 or newer")
    elif toolkit_version is None:
        toolkit_prereq = prerequisite("UNKNOWN", toolkit.get("version") or "not_found", "CUDA Toolkit version could not be determined")
    else:
        toolkit_prereq = prerequisite("FAIL", toolkit["version"], "nvcc reports a CUDA Toolkit older than 12")
    if gcc_ok:
        gcc_prereq = prerequisite("PASS", gcc["version"], "GCC is version 9 or newer")
    elif gcc_version is None:
        gcc_prereq = prerequisite("UNKNOWN", gcc.get("version") or "not_found", "GCC version could not be determined")
    else:
        gcc_prereq = prerequisite("FAIL", gcc["version"], "GCC is older than version 9")

    output = {
        "report_scope": "S00 environment capture only; no package installation, model download, compilation, quantization, or implementation",
        "commands_executed": COMMANDS,
        "host": {
            "operating_system": platform.system(),
            "os_release": parse_os_release(),
            "kernel": platform.release(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "cpu": parse_lscpu(),
        "memory": parse_memory(),
        "disk": parse_disk(),
        "gpu": {"count": len(gpus), "devices": gpus},
        "nvidia_driver": driver,
        "cuda_toolkit": toolkit,
        "compiler": {"gcc": gcc, "g++": gxx},
        "environment": {"CUDA_HOME": os.environ.get("CUDA_HOME"), "PATH": os.environ.get("PATH")},
        "python": python_info,
        "pytorch": pytorch,
        "transformers": transformers,
        "environment_check_results": {
            "python_3_11": python_prereq,
            "cuda_toolkit_12_or_newer": toolkit_prereq,
            "gcc_9_or_newer": gcc_prereq,
            "pytorch_cuda_smoke_test": pytorch["cuda_smoke_test"],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
