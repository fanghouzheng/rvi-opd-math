#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import yaml

REQUIRED_PATHS = [
    Path("docs/EXPERIMENT_PLAN.md"),
    Path("experiments/manifest.csv"),
    Path("schemas/state_record.schema.json"),
]


def main() -> int:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
    config = yaml.safe_load(config_path.read_text())
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": str(config_path),
        "protocol_version": config.get("protocol_version"),
        "git": shutil.which("git"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "missing_contract_files": missing,
        "gpu_training_ready": shutil.which("nvidia-smi") is not None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
