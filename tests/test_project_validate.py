from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "densigrav", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def test_project_validate_success(tmp_path: Path):
    # dummy inputs (existence check uses these)
    write_text(tmp_path / "data" / "gravity_points.csv", "E,N,elev_m,bouguer_mgal\n0,0,0,0\n")
    write_text(tmp_path / "data" / "dem.tif", "dummy")  # v0.1-02では中身は読まない

    project_yaml = """
project:
  name: "Test"
  crs: "EPSG:6677"
  gravity_unit: "mGal"
  length_unit: "m"
paths:
  data_dir: "data"
  cache_dir: "cache"
  results_dir: "results"
inputs:
  gravity_points:
    path: "data/gravity_points.csv"
    value_kind: "bouguer_anomaly"
    columns:
      x: "E"
      y: "N"
      z: "elev_m"
      value: "bouguer_mgal"
  dem:
    path: "data/dem.tif"
    z_unit: "m"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    p = run_cmd("project", "validate", str(tmp_path / "project.yaml"), "--check-exists")
    assert p.returncode == 0, p.stderr
    assert "OK: project validated" in p.stdout


def test_project_validate_failure_missing_key(tmp_path: Path):
    # Missing inputs.dem
    project_yaml = """
project:
  name: "Test"
  crs: "EPSG:6677"
inputs:
  gravity_points:
    path: "data/gravity_points.csv"
    value_kind: "bouguer_anomaly"
    columns:
      x: "E"
      y: "N"
      value: "bouguer_mgal"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    p = run_cmd("project", "validate", str(tmp_path / "project.yaml"))
    assert p.returncode != 0
    assert "validation failed" in p.stderr.lower()


def test_project_validate_failure_bad_yaml(tmp_path: Path):
    write_text(tmp_path / "project.yaml", "project: [this is not: valid")
    p = run_cmd("project", "validate", str(tmp_path / "project.yaml"))
    assert p.returncode != 0
    assert "invalid yaml" in p.stderr.lower()
