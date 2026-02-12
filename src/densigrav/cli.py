from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .io.dem import (
    DemIOError,
    aoi_from_points_bbox,
    clip_dem_to_geom,
    sample_dem_to_points,
    write_dem_prepare_points_gpkg,
)
from .io.gravity_points import PointsIOError, read_points_any, standardize_points, write_gpkg
from .project.io import ProjectValidationError, load_project, validate_project


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="densigrav",
        description="densigrav: gravity anomaly preprocessing (CBA) and 2D section tools",
    )
    p.add_argument("--version", action="store_true", help="Show version and exit")

    sub = p.add_subparsers(dest="command")

    # project
    project_p = sub.add_parser("project", help="Project utilities")
    project_sub = project_p.add_subparsers(dest="project_cmd", required=True)

    val = project_sub.add_parser("validate", help="Validate project YAML")
    val.add_argument("path", type=str, help="Path to project.yaml")
    val.add_argument(
        "--no-resolve", action="store_true", help="Do not resolve relative paths to absolute paths"
    )
    val.add_argument(
        "--check-exists", action="store_true", help="Check whether referenced input files exist"
    )

    # points (Issue03)
    points_p = sub.add_parser("points", help="Gravity points I/O")
    points_sub = points_p.add_subparsers(dest="points_cmd", required=True)

    ing = points_sub.add_parser(
        "ingest", help="Read gravity points and write standardized GeoPackage"
    )
    ing.add_argument("project", type=str, help="Path to project.yaml")
    ing.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output GeoPackage path (default: cache/gravity_points.gpkg)",
    )
    ing.add_argument("--layer", type=str, default="gravity_points", help="GeoPackage layer name")
    ing.add_argument(
        "--target-unit",
        type=str,
        default="mGal",
        choices=["mGal", "uGal", "µGal"],
        help="Convert gravity values to this unit",
    )
    ing.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    ing.add_argument(
        "--no-resolve", action="store_true", help="Do not resolve relative paths to absolute paths"
    )
    ing.add_argument(
        "--in-layer", type=str, default=None, help="Input layer name (only for GeoPackage input)"
    )

    # dem (Issue04)
    dem_p = sub.add_parser("dem", help="DEM utilities")
    dem_sub = dem_p.add_subparsers(dest="dem_cmd", required=True)

    prep = dem_sub.add_parser(
        "prepare", help="Clip DEM by AOI and fill point elevations (z) from DEM when missing"
    )
    prep.add_argument("project", type=str, help="Path to project.yaml")
    prep.add_argument(
        "--buffer-m", type=float, default=5000.0, help="AOI buffer (CRS units, typically meters)"
    )
    prep.add_argument(
        "--out-dem",
        type=str,
        default=None,
        help="Output clipped DEM path (default: cache/dem_clipped.tif)",
    )
    prep.add_argument(
        "--out-points",
        type=str,
        default=None,
        help="Output points GPKG path (default: cache/points_with_z.gpkg)",
    )
    prep.add_argument("--layer", type=str, default="gravity_points", help="Output GPKG layer name")
    prep.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    prep.add_argument(
        "--no-resolve", action="store_true", help="Do not resolve relative paths to absolute paths"
    )
    prep.add_argument(
        "--in-layer", type=str, default=None, help="Input layer name (only for GeoPackage input)"
    )
    prep.add_argument(
        "--error-if-outside",
        action="store_true",
        help="Fail if any points are outside DEM bounds",
    )

    # stubs (later)
    sub.add_parser("preprocess", help="Compute CBA with DEM terrain correction [stub]")
    sub.add_parser("section", help="Extract section and run 2D tools [stub]")

    return p


def _resolve_out_path(base_dir: Path, out: Optional[str], default_rel: Path) -> Path:
    if out is None:
        p = base_dir / default_rel
    else:
        p = Path(out)
        if not p.is_absolute():
            p = base_dir / p
    return p.resolve()


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"densigrav {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    try:
        # ---- project validate (Issue02)
        if args.command == "project" and args.project_cmd == "validate":
            loaded = validate_project(
                Path(args.path),
                resolve_paths=not args.no_resolve,
                check_exists=bool(args.check_exists),
            )
            print("OK: project validated")
            print(f"base_dir: {loaded.base_dir}")
            print(f"gravity_points: {loaded.config.inputs.gravity_points.path}")
            print(f"dem: {loaded.config.inputs.dem.path}")
            return 0

        # ---- points ingest (Issue03)
        if args.command == "points" and args.points_cmd == "ingest":
            loaded = load_project(Path(args.project), resolve_paths=not args.no_resolve)
            cfg = loaded.config

            gp_path = cfg.inputs.gravity_points.path
            if not gp_path.exists():
                raise PointsIOError(f"Gravity points file not found: {gp_path}")

            df = read_points_any(gp_path, layer=args.in_layer)
            cols = cfg.inputs.gravity_points.columns
            std = standardize_points(
                df,
                x_col=cols.x,
                y_col=cols.y,
                z_col=cols.z,
                value_col=cols.value,
                sigma_col=cols.sigma,
                crs=cfg.project.crs,
                input_unit=cfg.project.gravity_unit.value,
                target_unit=args.target_unit,
            )

            default_rel = cfg.paths.cache_dir / "gravity_points.gpkg"
            out_path = _resolve_out_path(loaded.base_dir, args.out, default_rel)

            write_gpkg(std.gdf, out_path, layer=args.layer, overwrite=bool(args.overwrite))

            print("OK: points ingested")
            print(f"out: {out_path}")
            print(f"layer: {args.layer}")
            print(f"unit: {std.gravity_unit}")
            return 0

        # ---- dem prepare (Issue04)
        if args.command == "dem" and args.dem_cmd == "prepare":
            loaded = load_project(Path(args.project), resolve_paths=not args.no_resolve)
            cfg = loaded.config

            gp_path = cfg.inputs.gravity_points.path
            dem_path = cfg.inputs.dem.path

            if not gp_path.exists():
                raise PointsIOError(f"Gravity points file not found: {gp_path}")
            if not dem_path.exists():
                raise DemIOError(f"DEM file not found: {dem_path}")

            # Read + standardize points
            df = read_points_any(gp_path, layer=args.in_layer)
            cols = cfg.inputs.gravity_points.columns
            std = standardize_points(
                df,
                x_col=cols.x,
                y_col=cols.y,
                z_col=cols.z,
                value_col=cols.value,
                sigma_col=cols.sigma,
                crs=cfg.project.crs,
                input_unit=cfg.project.gravity_unit.value,
                target_unit=cfg.project.gravity_unit.value,  # keep unit here; Issue05で統一する
            )

            points_gdf = std.gdf

            # Build AOI in points CRS, then reproject to DEM CRS inside dem functions
            aoi = aoi_from_points_bbox(points_gdf, buffer_m=float(args.buffer_m))

            # Clip DEM (need AOI in DEM CRS)
            import rasterio

            with rasterio.open(dem_path) as src:
                dem_crs = src.crs.to_string() if src.crs else ""
            if not dem_crs:
                raise DemIOError(f"DEM has no CRS: {dem_path}")

            # Reproject AOI to DEM CRS if needed
            from .io.dem import (
                _reproject_shapely_geom,  # local import to avoid exposing as public API
            )

            aoi_dem = _reproject_shapely_geom(aoi, points_gdf.crs.to_string(), dem_crs)

            out_dem = _resolve_out_path(
                loaded.base_dir, args.out_dem, cfg.paths.cache_dir / "dem_clipped.tif"
            )
            out_points = _resolve_out_path(
                loaded.base_dir, args.out_points, cfg.paths.cache_dir / "points_with_z.gpkg"
            )

            clip_dem_to_geom(dem_path, aoi_dem, out_dem, overwrite=bool(args.overwrite))

            # Fill z if missing
            filled_points, stats = sample_dem_to_points(
                out_dem,
                points_gdf,
                z_col="z",
                fill_only_missing=True,
                error_if_outside=bool(args.error_if_outside),
            )

            write_dem_prepare_points_gpkg(
                filled_points, out_points, layer=args.layer, overwrite=bool(args.overwrite)
            )

            print("OK: DEM prepared")
            print(f"out_dem: {out_dem}")
            print(f"out_points: {out_points} (layer={args.layer})")
            # print(
            #    f"points: {stats.n_points}, filled_z: {stats.n_filled}, outside_dem: {stats.n_outside}"
            # )
            print(
                f"points: {stats.n_points}, "
                f"z_missing_before: {stats.n_missing_before}, "
                f"z_filled_from_dem: {stats.n_filled_from_dem}, "
                f"z_missing_after: {stats.n_missing_after}, "
                f"outside_dem: {stats.n_outside}"
            )
            return 0

        # stubs
        if args.command in ("preprocess", "section"):
            print(f"[stub] command='{args.command}' is not implemented yet.")
            return 0

        parser.print_help()
        return 0

    except (ProjectValidationError, PointsIOError, DemIOError) as e:
        print(str(e), file=sys.stderr)
        return 2
