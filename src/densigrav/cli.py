from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .project.io import ProjectValidationError, validate_project


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
        "--no-resolve",
        action="store_true",
        help="Do not resolve relative paths to absolute paths",
    )
    val.add_argument(
        "--check-exists",
        action="store_true",
        help="Check whether referenced input files exist",
    )

    # stubs (Issue03以降で実装)
    sub.add_parser("preprocess", help="Compute CBA with DEM terrain correction [stub]")
    sub.add_parser("section", help="Extract section and run 2D tools [stub]")

    return p


def main(argv: list[str] | None = None) -> int:
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

        # stubs
        if args.command in ("preprocess", "section"):
            print(f"[stub] command='{args.command}' is not implemented yet.")
            return 0

        parser.print_help()
        return 0

    except ProjectValidationError as e:
        print(str(e), file=sys.stderr)
        return 2
