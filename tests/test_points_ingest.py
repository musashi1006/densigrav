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


def test_points_ingest_csv_to_gpkg(tmp_path: Path):
    # CSV (input unit: uGal)
    csv = """E,N,elev_m,g_uGal,s_uGal
0,0,100,1000,50
1000,0,120,2000,50
0,1000,90,500,50
"""
    write_text(tmp_path / "data" / "gravity_points.csv", csv)

    # project.yaml (uGal -> target mGal)
    project_yaml = """
project:
  name: "Test"
  crs: "EPSG:4326"
  gravity_unit: "uGal"
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
      value: "g_uGal"
      sigma: "s_uGal"
  dem:
    path: "data/dem.tif"
    z_unit: "m"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    out_gpkg = tmp_path / "cache" / "out.gpkg"
    p = run_cmd(
        "points",
        "ingest",
        str(tmp_path / "project.yaml"),
        "--out",
        str(out_gpkg),
        "--overwrite",
        "--target-unit",
        "mGal",
    )
    assert p.returncode == 0, p.stderr
    assert out_gpkg.exists()

    import geopandas as gpd

    gdf = gpd.read_file(out_gpkg, layer="gravity_points")
    # standardized columns exist
    assert "x" in gdf.columns
    assert "y" in gdf.columns
    assert "z" in gdf.columns
    assert "g_in" in gdf.columns
    assert "sigma" in gdf.columns

    # conversion check: 1000 uGal = 1 mGal
    assert abs(float(gdf.loc[0, "g_in"]) - 1.0) < 1e-9
    assert abs(float(gdf.loc[0, "sigma"]) - 0.05) < 1e-9
