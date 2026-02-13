from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class TalwaniModel:
    density_contrast_kgm3: float
    vertices_xz_m: np.ndarray  # shape (n,2)


def load_talwani_model(path: Path) -> TalwaniModel:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    rho = float(obj["density_contrast_kgm3"])
    verts = np.asarray(obj["vertices_xz_m"], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError("vertices_xz_m must be a list of [x,z].")
    return TalwaniModel(density_contrast_kgm3=rho, vertices_xz_m=verts)


def save_talwani_model(path: Path, model: TalwaniModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "density_contrast_kgm3": float(model.density_contrast_kgm3),
        "vertices_xz_m": [[float(x), float(z)] for x, z in model.vertices_xz_m],
    }
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def load_polygon_vertices_xy_from_vector(
    path: Path, *, layer: str | None = None, feature_index: int = 0
) -> np.ndarray:
    """
    Read a single polygon from a vector file and return its exterior vertices as (x, y).

    - If MultiPolygon -> use largest part
    - Ignore holes (v0.1)
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if len(gdf) == 0:
        raise ValueError(f"No features found in polygon file: {path}")
    if not (0 <= feature_index < len(gdf)):
        raise IndexError(f"feature_index out of range: {feature_index} (n={len(gdf)})")

    geom = gdf.geometry.iloc[int(feature_index)]
    if geom is None:
        raise ValueError(f"Polygon feature has null geometry: {path} (index={feature_index})")

    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
        parts.sort(key=lambda g: float(g.area), reverse=True)
        geom = parts[0]

    if not isinstance(geom, Polygon):
        raise TypeError(f"Polygon geometry required (got {type(geom).__name__})")

    coords = np.asarray(geom.exterior.coords, dtype=float)
    return coords[:, :2].copy()


def model_from_polygon_vector(
    path: Path,
    *,
    layer: str | None = None,
    feature_index: int = 0,
    density_contrast_kgm3: float,
    polygon_y_axis: str = "elev",
) -> TalwaniModel:
    """
    polygon_y_axis:
      - "elev": y is elevation (up). Convert to z-down by z = -y
      - "zdown": y is already z-down
    """
    xy = load_polygon_vertices_xy_from_vector(path, layer=layer, feature_index=feature_index)
    x = xy[:, 0]
    y = xy[:, 1]
    if polygon_y_axis not in ("elev", "zdown"):
        raise ValueError('polygon_y_axis must be "elev" or "zdown"')
    z = (-y) if polygon_y_axis == "elev" else y
    verts = np.vstack([x, z]).T
    return TalwaniModel(float(density_contrast_kgm3), verts)
