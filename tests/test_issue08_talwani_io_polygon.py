from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")


def _write_gpkg(gdf, path: Path, layer: str) -> None:
    # pyogrio があれば速い / 無ければ fiona でもOK
    try:
        gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
    except TypeError:
        gdf.to_file(path, layer=layer, driver="GPKG")


def test_load_polygon_vertices_xy_from_vector_basic(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Polygon

    from densigrav.section.talwani_io import load_polygon_vertices_xy_from_vector

    poly = Polygon([(0, 0), (10, 0), (10, -5), (0, -5), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:6677")

    gpkg = tmp_path / "poly.gpkg"
    _write_gpkg(gdf, gpkg, "body")

    xy = load_polygon_vertices_xy_from_vector(gpkg, layer="body", feature_index=0)
    assert xy.shape[1] == 2
    assert xy.shape[0] >= 4
    # 1点目が期待通り（厳密一致でなくてもOKだが、ここは固定にしてよい）
    assert np.allclose(xy[0], [0, 0])


def test_model_from_polygon_vector_y_axis_convert(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Polygon

    from densigrav.section.talwani_io import model_from_polygon_vector

    # y は elevation（上向き正）として扱う想定
    # → zdown = -y になるので、y=-50 は z=+50 になる
    poly = Polygon([(0, 0), (10, 0), (10, -50), (0, -50), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:6677")

    gpkg = tmp_path / "poly.gpkg"
    _write_gpkg(gdf, gpkg, "body")

    m = model_from_polygon_vector(
        gpkg, layer="body", density_contrast_kgm3=300.0, polygon_y_axis="elev"
    )
    verts = m.vertices_xz_m
    # 最深点（y=-50）が z=+50 へ反転していること
    assert np.isclose(verts[:, 1].max(), 50.0)


def test_load_polygon_vertices_multipolygon_uses_largest(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    from densigrav.section.talwani_io import load_polygon_vertices_xy_from_vector

    small = Polygon([(0, 0), (1, 0), (1, -1), (0, -1), (0, 0)])
    large = Polygon([(0, 0), (10, 0), (10, -5), (0, -5), (0, 0)])
    mp = MultiPolygon([small, large])

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[mp], crs="EPSG:6677")
    gpkg = tmp_path / "mp.gpkg"
    _write_gpkg(gdf, gpkg, "body")

    xy = load_polygon_vertices_xy_from_vector(gpkg, layer="body", feature_index=0)
    # large の頂点が含まれているはず（例えば x=10 が存在）
    assert np.any(np.isclose(xy[:, 0], 10.0))
