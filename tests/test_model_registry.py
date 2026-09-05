from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"^(?:b|v)\d+\.\d+$")


def test_registry_versions_and_runners_are_valid() -> None:
    registry = json.loads((PROJECT_ROOT / "model_registry.json").read_text("utf-8"))
    versions = registry["versions"]
    names = [item["version"] for item in versions]
    assert len(names) == len(set(names))
    assert registry["current_version"] in names
    for item in versions:
        assert VERSION_PATTERN.fullmatch(item["version"])
        for key in ("runner", "incremental_runner", "seed_runner"):
            value = item.get(key)
            if value is not None:
                assert (PROJECT_ROOT / value).is_file(), (item["version"], key, value)
