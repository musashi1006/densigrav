from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SectionProfileOutputs:
    out_dir: Path
    points_gpkg: Path
    profile_csv: Path
    profile_png: Path | None


def _resolve(base_dir: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _get(obj, path: str, default=None):
    """
    Support both pydantic models (attribute access) and dicts.
    path example: "step1_preprocess.outputs.points_gpkg"
    """
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _load_project_cfg(project_yaml: Path):
    # Prefer densigrav's validated loader if available
    try:
        from densigrav.project.io import load_project_file  # type: ignore

        loaded = load_project_file(project_yaml)
        # loaded can be cfg itself, or (cfg, base_dir), or an object with cfg/base_dir
        if isinstance(loaded, tuple) and len(loaded) == 2:
            cfg, base_dir = loaded
            return cfg, Path(base_dir)
        if hasattr(loaded, "cfg") and hasattr(loaded, "base_dir"):
            return loaded.cfg, Path(loaded.base_dir)
        return loaded, project_yaml.parent
    except Exception:
        import yaml

        cfg = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
        return cfg, project_yaml.parent


def _read_first_linestring(line_path: Path, layer: str | None) -> tuple[object, str]:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    gdf = gpd.read_file(line_path, layer=layer)
    if len(gdf) == 0:
        raise ValueError(f"No features found in line file: {line_path}")

    geom = gdf.geometry.iloc[0]
    if isinstance(geom, LineString):
        return geom, gdf.crs.to_string() if gdf.crs else ""
    if isinstance(geom, MultiLineString):
        # pick the longest
        parts = list(geom.geoms)
        parts.sort(key=lambda g: g.length, reverse=True)
        return parts[0], gdf.crs.to_string() if gdf.crs else ""

    raise TypeError("Section line geometry must be LineString or MultiLineString.")


def _sample_line_points(line, interval_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dist_m, xs, ys)
    """
    from shapely.geometry import Point

    length = float(line.length)
    if length <= 0:
        raise ValueError("Section line length must be > 0.")

    n = int(np.floor(length / interval_m)) + 1
    dists = np.linspace(0.0, length, n, dtype=float)

    xs = np.empty(n, dtype=float)
    ys = np.empty(n, dtype=float)
    for i, d in enumerate(dists):
        p: Point = line.interpolate(float(d))
        xs[i] = float(p.x)
        ys[i] = float(p.y)

    return dists, xs, ys


def _nearest_join_values(
    station_xy: np.ndarray,
    points_xy: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Brute-force nearest neighbor (n<=1000, stations<=~5000想定でも十分).
    """
    out: dict[str, np.ndarray] = {}
    for k in values:
        out[k] = np.full(station_xy.shape[0], np.nan, dtype=float)

    for i in range(station_xy.shape[0]):
        dx = points_xy[:, 0] - station_xy[i, 0]
        dy = points_xy[:, 1] - station_xy[i, 1]
        j = int(np.argmin(dx * dx + dy * dy))
        for k, arr in values.items():
            out[k][i] = float(arr[j])
    return out


def extract_section_profile(
    project_yaml: Path,
    line_path: Path,
    *,
    line_layer: str | None = None,
    overwrite: bool = False,
    make_plot: bool = True,
) -> SectionProfileOutputs:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    cfg, base_dir = _load_project_cfg(project_yaml)

    interval = float(_get(cfg, "step2_section.sampling_interval_m", 100.0))
    width = float(_get(cfg, "step2_section.buffer_width_m", 200.0))

    out_dir_cfg = _get(cfg, "step2_section.outputs.result_dir", "results/2d/section_A")
    out_dir = _resolve(base_dir, out_dir_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_name_cfg = _get(cfg, "step2_section.outputs.section_points_gpkg", "section_points.gpkg")
    points_gpkg = out_dir / Path(points_name_cfg)

    profile_csv = out_dir / "profile.csv"
    profile_png = out_dir / "profile.png"

    for p in [points_gpkg, profile_csv, profile_png]:
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {p}")

    line, line_crs = _read_first_linestring(line_path, line_layer)

    project_crs = _get(cfg, "project.crs", "")
    if project_crs and line_crs and (project_crs != line_crs):
        # v0.1: ここは厳密変換より「合わせてね」を優先（入口体験）
        raise ValueError(
            f"CRS mismatch: project={project_crs}, line={line_crs}. "
            "Reproject the section line to the project CRS."
        )

    # Station points along line
    dist_m, xs, ys = _sample_line_points(line, interval_m=interval)
    stations = gpd.GeoDataFrame(
        {"dist_m": dist_m, "E": xs, "N": ys},
        geometry=[Point(x, y) for x, y in zip(xs, ys, strict=True)],
        crs=project_crs or None,
    )

    # Load preprocessed points (GPKG)
    points_gpkg_cfg = _get(
        cfg, "step1_preprocess.outputs.points_gpkg", "cache/preprocessed_points.gpkg"
    )
    points_gpkg_abs = _resolve(base_dir, points_gpkg_cfg)

    if not points_gpkg_abs.exists():
        raise FileNotFoundError(
            f"Preprocessed points not found: {points_gpkg_abs} "
            "(run `densigrav preprocess project.yaml --overwrite` first)"
        )

    pts = gpd.read_file(points_gpkg_abs, layer="gravity_points")
    if len(pts) == 0:
        raise ValueError(f"No points in {points_gpkg_abs} (layer=gravity_points).")

    # Buffer corridor filter
    corridor = line.buffer(width / 2.0)
    pts_in = pts[pts.geometry.within(corridor)].copy()
    if len(pts_in) == 0:
        # fallback: use all points (still do nearest)
        pts_in = pts.copy()

    # Candidate columns to carry
    want_cols = [
        "bouguer_mgal",
        "tc_mgal",
        "cba_mgal",
        "regional_mgal",
        "residual_mgal",
        "sigma_mgal",
        "elev_m",
        "z_m",
    ]
    have_cols = [c for c in want_cols if c in pts_in.columns]

    station_xy = np.vstack([stations["E"].to_numpy(), stations["N"].to_numpy()]).T
    points_xy = np.vstack([pts_in.geometry.x.to_numpy(), pts_in.geometry.y.to_numpy()]).T
    values = {c: pts_in[c].to_numpy(dtype=float) for c in have_cols}
    nn = _nearest_join_values(station_xy, points_xy, values)

    for c, arr in nn.items():
        stations[c] = arr

    # Save section points
    if points_gpkg.exists() and overwrite:
        try:
            points_gpkg.unlink()
        except Exception:
            # Windows/QGIS file lock対策: 上書きできない時は別名に逃がす
            pass

    stations.to_file(points_gpkg, layer="section_points", driver="GPKG")

    # Save profile CSV (no geometry)
    df = pd.DataFrame({k: stations[k].to_numpy() for k in stations.columns if k != "geometry"})
    df.to_csv(profile_csv, index=False)

    out_png: Path | None = None
    if make_plot:
        out_png = _plot_profile(profile_csv, profile_png)

    return SectionProfileOutputs(
        out_dir=out_dir, points_gpkg=points_gpkg, profile_csv=profile_csv, profile_png=out_png
    )


def _plot_profile(profile_csv: Path, out_png: Path) -> Path:
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError('matplotlib is required for plotting. Install extras: ".[viz]"') from e

    df = pd.read_csv(profile_csv)

    x = df["dist_m"].to_numpy()
    ycols = [c for c in ["cba_mgal", "tc_mgal", "residual_mgal", "bouguer_mgal"] if c in df.columns]
    if len(ycols) == 0:
        raise ValueError("No gravity columns found in profile.csv to plot.")

    plt.figure()
    for c in ycols:
        plt.plot(x, df[c].to_numpy(), label=c)
    plt.xlabel("Distance along section (m)")
    plt.ylabel("mGal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
