from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


class PointsIOError(Exception):
    pass


@dataclass(frozen=True)
class StandardizedPoints:
    """
    Standard columns:
      - x, y: coordinates
      - z: elevation (optional)
      - g_in: gravity value (converted to target unit)
      - sigma: uncertainty (optional)
    Geometry: Point(x, y) in given CRS
    """

    gdf: "pd.DataFrame"  # actually GeoDataFrame; typed loosely to avoid hard import at module load
    crs: str
    gravity_unit: str  # target unit


def _unit_factor_to_mgal(unit: str) -> float:
    """
    Return multiplier to convert given unit to mGal.
    Supported: mGal, uGal, µGal
    """
    u = unit.strip()
    if u == "mGal":
        return 1.0
    if u in ("uGal", "µGal"):
        return 1.0 / 1000.0  # 1 uGal = 0.001 mGal
    raise PointsIOError(f"Unsupported gravity unit: {unit!r} (use mGal, uGal, µGal)")


def convert_gravity_units(values: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    """
    Convert gravity values between units (mGal, uGal, µGal).
    """
    if from_unit == to_unit:
        return values.astype(float)

    # convert from -> mGal
    to_mgal = _unit_factor_to_mgal(from_unit)
    v_mgal = values.astype(float) * to_mgal

    # convert mGal -> target
    if to_unit == "mGal":
        return v_mgal
    if to_unit in ("uGal", "µGal"):
        return v_mgal * 1000.0

    raise PointsIOError(f"Unsupported target unit: {to_unit!r} (use mGal, uGal, µGal)")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise PointsIOError(f"CSV not found: {path}") from e
    except Exception as e:
        raise PointsIOError(f"Failed to read CSV: {path}\n{e}") from e


def _read_vector(path: Path, layer: Optional[str] = None) -> "pd.DataFrame":
    """
    Read GeoPackage/GeoJSON/Shapefile as GeoDataFrame.
    """
    try:
        import geopandas as gpd
    except Exception as e:
        raise PointsIOError(
            "geopandas is required for reading vector files. Install with: pip install -e '.[geo]'"
        ) from e

    try:
        if layer:
            return gpd.read_file(path, layer=layer)
        return gpd.read_file(path)
    except FileNotFoundError as e:
        raise PointsIOError(f"Vector file not found: {path}") from e
    except Exception as e:
        raise PointsIOError(f"Failed to read vector file: {path}\n{e}") from e


def read_points_any(path: Path, layer: Optional[str] = None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return _read_csv(path)
    # treat others as vector
    return _read_vector(path, layer=layer)


def standardize_points(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: Optional[str],
    value_col: str,
    sigma_col: Optional[str],
    crs: str,
    input_unit: str,
    target_unit: str = "mGal",
) -> StandardizedPoints:
    """
    Convert an input table into a GeoDataFrame with standardized columns.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        raise PointsIOError(
            "geopandas/shapely are required. Install with: pip install -e '.[geo]'"
        ) from e

    for c in (x_col, y_col, value_col):
        if c not in df.columns:
            raise PointsIOError(f"Missing required column: {c!r}. Available: {list(df.columns)}")

    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    z = None
    if z_col:
        if z_col not in df.columns:
            raise PointsIOError(f"z column {z_col!r} not found in input.")
        z = df[z_col].astype(float)

    g_in = convert_gravity_units(df[value_col], from_unit=input_unit, to_unit=target_unit)

    sigma = None
    if sigma_col:
        if sigma_col not in df.columns:
            raise PointsIOError(f"sigma column {sigma_col!r} not found in input.")
        sigma = convert_gravity_units(df[sigma_col], from_unit=input_unit, to_unit=target_unit)

    out = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "g_in": g_in,
        }
    )
    if z is not None:
        out["z"] = z
    if sigma is not None:
        out["sigma"] = sigma

    geom = [Point(xx, yy) for xx, yy in zip(out["x"].to_list(), out["y"].to_list())]
    gdf = gpd.GeoDataFrame(out, geometry=geom, crs=crs)

    return StandardizedPoints(gdf=gdf, crs=crs, gravity_unit=target_unit)


def write_gpkg(
    gdf: "pd.DataFrame",
    out_path: Path,
    *,
    layer: str = "gravity_points",
    overwrite: bool = False,
) -> None:
    """
    Write GeoDataFrame to GeoPackage using pyogrio (preferred) if available.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and out_path.exists():
        out_path.unlink()

    try:
        import geopandas as gpd  # noqa: F401
    except Exception as e:
        raise PointsIOError(
            "geopandas is required for writing GeoPackage. Install with: pip install -e '.[geo]'"
        ) from e

    # geopandas >=0.14 supports engine="pyogrio" in to_file
    try:
        gdf.to_file(out_path, layer=layer, driver="GPKG", engine="pyogrio")
    except TypeError:
        # fallback for older geopandas
        gdf.to_file(out_path, layer=layer, driver="GPKG")
    except Exception as e:
        raise PointsIOError(f"Failed to write GeoPackage: {out_path}\n{e}") from e
