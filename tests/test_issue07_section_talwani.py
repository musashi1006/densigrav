from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import run_cmd, write_text


def test_issue07_section_extract(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    crs = "EPSG:6677"
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    # --- preprocessed points gpkg (minimal)
    rng = np.random.RandomState(0)
    n = 50
    E = rng.uniform(0, 5000, n)
    N = rng.uniform(0, 2000, n)
    cba = 0.2 * np.sin(E / 800) + 0.05 * rng.normal(size=n)
    tc = 0.05 * np.cos(N / 400) + 0.02 * rng.normal(size=n)

    pts = gpd.GeoDataFrame(
        {
            "cba_mgal": cba,
            "tc_mgal": tc,
            "residual_mgal": cba,  # for test
        },
        geometry=[Point(float(x), float(y)) for x, y in zip(E, N, strict=True)],
        crs=crs,
    )
    pp = tmp_path / "cache" / "preprocessed_points.gpkg"
    pts.to_file(pp, layer="gravity_points", driver="GPKG")

    # --- section line gpkg
    line = gpd.GeoDataFrame(
        {"name": ["A"]},
        geometry=[LineString([(0, 1000), (5000, 1000)])],
        crs=crs,
    )
    lf = tmp_path / "data" / "section_line.gpkg"
    line.to_file(lf, layer="line", driver="GPKG")

    # --- project.yaml (enable step2_section outputs)
    project_yaml = f"""
project:
  name: "Test"
  crs: "{crs}"
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
  dem:
    path: "data/dem.tif"
    z_unit: "m"

step1_preprocess:
  outputs:
    points_gpkg: "cache/preprocessed_points.gpkg"

step2_section:
  enabled: true
  buffer_width_m: 500.0
  sampling_interval_m: 250.0
  outputs:
    result_dir: "results/2d/section_A"
    section_points_gpkg: "section_points.gpkg"
"""
    write_text(tmp_path / "project.yaml", project_yaml)

    p = run_cmd(
        "section",
        "extract",
        str(tmp_path / "project.yaml"),
        "--line",
        str(lf),
        "--layer",
        "line",
        "--no-plot",
        "--overwrite",
    )
    assert p.returncode == 0, p.stderr

    out_dir = tmp_path / "results" / "2d" / "section_A"
    assert (out_dir / "section_points.gpkg").exists()
    assert (out_dir / "profile.csv").exists()


def test_issue07_talwani_forward_and_invert(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")

    from densigrav.section.talwani2d import talwani_gz_polygon
    from densigrav.section.talwani_io import TalwaniModel, save_talwani_model

    # synthetic profile
    x = np.linspace(0, 5000, 101)
    z = np.zeros_like(x)

    true_model = TalwaniModel(
        density_contrast_kgm3=300.0,
        vertices_xz_m=np.asarray([[2200, 300], [2800, 300], [3200, 1400], [1800, 1400]]),
    )
    y = talwani_gz_polygon(x, z, true_model.vertices_xz_m, true_model.density_contrast_kgm3)
    y = y + 0.01 * np.random.RandomState(1).normal(size=y.size)

    prof = tmp_path / "profile.csv"
    df = pd.DataFrame({"dist_m": x, "residual_mgal": y})
    df.to_csv(prof, index=False)

    model_yaml = tmp_path / "true_model.yaml"
    save_talwani_model(model_yaml, true_model)

    out_csv = tmp_path / "pred.csv"
    p = run_cmd(
        "section",
        "talwani-forward",
        "--profile",
        str(prof),
        "--model",
        str(model_yaml),
        "--out",
        str(out_csv),
        "--overwrite",
    )
    assert p.returncode == 0, p.stderr
    assert out_csv.exists()

    out_model = tmp_path / "fit_model.yaml"
    p2 = run_cmd(
        "section",
        "talwani-invert",
        "--profile",
        str(prof),
        "--out-model",
        str(out_model),
        "--overwrite",
    )
    assert p2.returncode == 0, p2.stderr
    assert out_model.exists()
