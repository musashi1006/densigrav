from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


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


def test_preprocess_step1_outputs_cba(tmp_path: Path):
    # DEM
    import rasterio
    from rasterio.transform import from_origin

    dem = tmp_path / "data" / "dem.tif"
    dem.parent.mkdir(parents=True, exist_ok=True)

    width, height = 40, 40
    pixel = 50.0
    transform = from_origin(0.0, 2000.0, pixel, pixel)
    crs = "EPSG:6677"
    z = (100 + np.random.RandomState(0).normal(0, 1, (height, width))).astype(np.float32)

    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=None,
    ) as dst:
        dst.write(z, 1)

    # points CSV (Bouguer anomaly)
    csv = """E,N,elev_m,bouguer_mgal,sigma_mgal
200,1800,120,0.10,0.05
400,1600,130,0.20,0.05
600,1400,,  0.00,0.05
"""
    write_text(tmp_path / "data" / "gravity_points.csv", csv)

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
      sigma: "sigma_mgal"
  dem:
    path: "data/dem.tif"
    z_unit: "m"

step1_preprocess:
  terrain_correction:
    density_kgm3: 2670.0
    outer_radius_m: 1000.0
    station_epsilon_m: 0.1
  outputs:
    points_gpkg: "results/points_cba.gpkg"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    p = run_cmd("preprocess", str(tmp_path / "project.yaml"), "--overwrite")
    assert p.returncode == 0, p.stderr

    out = tmp_path / "results" / "points_cba.gpkg"
    assert out.exists()

    import geopandas as gpd

    gdf = gpd.read_file(out, layer="gravity_points")
    assert "tc_mgal" in gdf.columns
    assert "cba_mgal" in gdf.columns
    assert np.isfinite(gdf["tc_mgal"].to_numpy()).all()
    assert np.isfinite(gdf["cba_mgal"].to_numpy()).all()
