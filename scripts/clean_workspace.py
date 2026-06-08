#!/usr/bin/env python3
"""Remove generated local clutter without touching runtime trading data."""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "storage"}
DELETE_FILES = {".DS_Store"}
DELETE_DIRS = {"__pycache__", ".pytest_cache"}


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(PROJECT_ROOT).parts)


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if path == PROJECT_ROOT or _is_skipped(path):
            continue
        if path.is_dir() and path.name in DELETE_DIRS:
            targets.append(path)
        elif path.is_file() and (path.name in DELETE_FILES or path.suffix == ".pyc"):
            targets.append(path)
    return sorted(targets, key=lambda item: len(item.parts), reverse=True)


def main() -> None:
    targets = collect_targets()
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print(f"Removed {len(targets)} generated file(s)/folder(s).")


if __name__ == "__main__":
    main()
