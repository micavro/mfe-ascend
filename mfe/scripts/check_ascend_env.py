#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 openEuler/Ascend/vLLM Ascend 运行环境。"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"ok": False, "error": f"{cmd[0]} not found"}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _package_version(dist_name: str, import_name: str | None = None) -> dict[str, Any]:
    import_name = import_name or dist_name.replace("-", "_")
    result: dict[str, Any] = {"installed": False}
    try:
        result["version"] = importlib.metadata.version(dist_name)
        result["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        result["version"] = None
    try:
        importlib.import_module(import_name)
        result["importable"] = True
    except Exception as exc:
        result["importable"] = False
        result["import_error"] = repr(exc)
    return result


def _os_release() -> dict[str, str]:
    path = "/etc/os-release"
    if not os.path.exists(path):
        return {}
    data: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            data[key] = value.strip('"')
    return data


def _torch_npu_summary() -> dict[str, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

    summary: dict[str, Any] = {"ok": True}
    try:
        summary["device_count"] = int(torch.npu.device_count())
        summary["is_available"] = bool(torch.npu.is_available())
    except Exception as exc:
        summary["device_error"] = repr(exc)
    return summary


def main() -> None:
    packages = {
        "torch": _package_version("torch"),
        "torch-npu": _package_version("torch-npu", "torch_npu"),
        "vllm": _package_version("vllm"),
        "vllm-ascend": _package_version("vllm-ascend", "vllm_ascend"),
        "transformers": _package_version("transformers"),
    }
    env = {
        key: os.environ.get(key)
        for key in (
            "MFE_ACCELERATOR",
            "ASCEND_HOME_PATH",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "VLLM_TARGET_DEVICE",
            "VLLM_USE_V1",
            "LD_LIBRARY_PATH",
        )
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": _os_release(),
        "env": env,
        "commands": {
            "npu-smi info": _run(["npu-smi", "info"]),
            "uname -a": _run(["uname", "-a"]),
            "gcc --version": _run(["gcc", "--version"]),
        },
        "packages": packages,
        "torch_npu": _torch_npu_summary(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
