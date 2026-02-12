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


def test_dem_prepare_clips_and_fills_z(tmp_path: Path):
    # --- Create a tiny DEM GeoTIFF
    import rasterio
    from rasterio.transform import from_origin

    dem_path = tmp_path / "data" / "dem.tif"
    dem_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 50, 50
    pixel = 10.0
    transform = from_origin(0.0, 500.0, pixel, pixel)  # x0, y0, dx, dy
    crs = "EPSG:6677"

    data = np.arange(width * height, dtype=np.float32).reshape(height, width)

    with rasterio.open(
        dem_path,
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
        dst.write(data, 1)

    # --- Gravity points CSV (no z column in mapping -> should be filled)
    csv = """E,N,bouguer_mgal,sigma_mgal
50,450,0.1,0.05
100,400,0.2,0.05
150,350,0.0,0.05
"""
    write_text(tmp_path / "data" / "gravity_points.csv", csv)

    # --- project.yaml (omit z mapping)
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
      value: "bouguer_mgal"
      sigma: "sigma_mgal"
  dem:
    path: "data/dem.tif"
    z_unit: "m"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    out_dem = tmp_path / "cache" / "dem_clipped.tif"
    out_pts = tmp_path / "cache" / "points_with_z.gpkg"

    p = run_cmd(
        "dem",
        "prepare",
        str(tmp_path / "project.yaml"),
        "--buffer-m",
        "20",
        "--out-dem",
        str(out_dem),
        "--out-points",
        str(out_pts),
        "--overwrite",
    )
    assert p.returncode == 0, p.stderr
    assert out_dem.exists()
    assert out_pts.exists()

    import geopandas as gpd

    gdf = gpd.read_file(out_pts, layer="gravity_points")
    assert "z" in gdf.columns
    assert np.isfinite(gdf["z"].to_numpy()).all()
