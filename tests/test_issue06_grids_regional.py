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


def test_issue06_preprocess_outputs_grids_and_regional(tmp_path: Path):
    # ---- DEM (projected CRS)
    import rasterio
    from rasterio.transform import from_origin

    dem = tmp_path / "data" / "dem.tif"
    dem.parent.mkdir(parents=True, exist_ok=True)

    crs = "EPSG:6677"
    res = 50.0
    width, height = 80, 80
    transform = from_origin(0.0, 4000.0, res, res)

    rng = np.random.RandomState(0)
    topo = (100 + 10 * rng.normal(size=(height, width))).astype(np.float32)

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
        dst.write(topo, 1)

    # ---- points CSV (Bouguer anomaly in mGal)
    n = 120
    E = rng.uniform(200, 3800, n)
    N = rng.uniform(200, 3800, n)
    elev = 120 + 5 * rng.normal(size=n)
    bouguer = 0.2 * np.sin(E / 800) - 0.1 * np.cos(N / 900) + 0.02 * rng.normal(size=n)
    sigma = np.full(n, 0.05)

    rows = ["E,N,elev_m,bouguer_mgal,sigma_mgal"]
    for i in range(n):
        rows.append(f"{E[i]:.3f},{N[i]:.3f},{elev[i]:.3f},{bouguer[i]:.6f},{sigma[i]:.3f}")
    write_text(tmp_path / "data" / "gravity_points.csv", "\n".join(rows) + "\n")

    # ---- project.yaml (enable regional + grids)
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
  bouguer_density_kgm3: 2670.0
  terrain_correction:
    enabled: true
    method: "dem_prism"
    density_kgm3: 2670.0
    outer_radius_m: 1500.0
    station_epsilon_m: 0.1
  regional:
    enabled: true
    method: "polynomial"
    order: 2
    output: "residual"
  outputs:
    points_gpkg: "cache/preprocessed_points.gpkg"
    grids:
      enabled: true
      resolution_m: 200.0
      cba_grid: "cache/cba.tif"
      tc_grid: "cache/terrain_correction.tif"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    p = run_cmd("preprocess", str(tmp_path / "project.yaml"), "--overwrite")
    assert p.returncode == 0, p.stderr

    # points output
    out_pts = tmp_path / "cache" / "preprocessed_points.gpkg"
    assert out_pts.exists()

    import geopandas as gpd

    gdf = gpd.read_file(out_pts, layer="gravity_points")
    for col in ("tc_mgal", "cba_mgal", "regional_mgal", "residual_mgal"):
        assert col in gdf.columns

    # grids output
    tc_tif = tmp_path / "cache" / "terrain_correction.tif"
    cba_tif = tmp_path / "cache" / "cba.tif"
    assert tc_tif.exists()
    assert cba_tif.exists()

    import rasterio

    with rasterio.open(tc_tif) as ds:
        assert ds.crs.to_string() == "EPSG:6677"
        assert abs(ds.transform.a - 200.0) < 1e-9
        a = ds.read(1)
        assert np.isfinite(a[a != ds.nodata]).any()

    with rasterio.open(cba_tif) as ds:
        assert ds.crs.to_string() == "EPSG:6677"
        assert abs(ds.transform.a - 200.0) < 1e-9
        a = ds.read(1)
        assert np.isfinite(a[a != ds.nodata]).any()
