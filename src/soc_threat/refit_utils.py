"""Small, atomic artifact helpers for the full-label experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from importlib.metadata import PackageNotFoundError, version
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def bind_manifest(path: Path, identity: dict) -> str:
    if path.exists() and json.loads(path.read_text('utf-8')) != identity:
        raise ValueError(f'Input/code/settings changed; choose a NEW directory: {path.parent}')
    write_json(path, identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))}


def environment_info() -> dict:
    packages = {}
    for name in ['torch','torch-npu','numpy','pandas','pyarrow','duckdb']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def git_revision() -> str | None:
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_checksum(path: Path) -> None:
    target = Path(str(path) + '.sha256')
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(f'{sha256(path)}  {path.name}\n', encoding='utf-8')
    os.replace(temporary, target)


def verify_artifacts(directory: Path, names: list[str], report: dict) -> bool:
    return all((directory / name).is_file()
               and sha256(directory / name) == report.get('artifact_sha256', {}).get(name)
               for name in names)
