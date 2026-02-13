from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    """Run densigrav CLI via `python -m densigrav ...`."""
    return subprocess.run(
        [sys.executable, "-m", "densigrav", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
