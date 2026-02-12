from __future__ import annotations

import subprocess
import sys


def run_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "densigrav", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_runs():
    p = run_cmd("--help")
    assert p.returncode == 0
    assert "densigrav" in p.stdout


def test_version_runs():
    p = run_cmd("--version")
    assert p.returncode == 0
    assert "densigrav" in p.stdout
