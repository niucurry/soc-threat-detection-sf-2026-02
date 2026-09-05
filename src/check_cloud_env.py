from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import duckdb
import numpy
import pandas
import pyarrow
import torch


def main() -> None:
    try:
        drain3_version = version("drain3")
    except PackageNotFoundError:
        drain3_version = None
    torch_npu_version = None
    npu_available = False
    npu_count = 0
    npu_name = None
    try:
        import torch_npu

        torch_npu_version = getattr(torch_npu, "__version__", "unknown")
        npu_available = bool(hasattr(torch, "npu") and torch.npu.is_available())
        if npu_available:
            npu_count = int(torch.npu.device_count())
            try:
                npu_name = str(torch.npu.get_device_name(0))
            except Exception:
                npu_name = "available-name-unreported"
    except ImportError:
        pass

    result = {
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": torch_npu_version,
        "npu_available": npu_available,
        "npu_count": npu_count,
        "npu_name": npu_name,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "duckdb": duckdb.__version__,
        "drain3": drain3_version,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not npu_available:
        raise SystemExit(
            "NPU is not available. Confirm that the selected image contains "
            "matching torch and torch_npu packages and that an NPU resource is attached."
        )


if __name__ == "__main__":
    main()
