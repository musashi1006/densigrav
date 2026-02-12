from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="densigrav",
        description="densigrav: gravity anomaly preprocessing (CBA) and 2D section tools",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    sub = p.add_subparsers(dest="command")

    # v0.1-01では“枠”だけ。中身はIssue02以降で実装。
    sub.add_parser("project", help="Project utilities (init/validate) [stub]")
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

    # v0.1-01: コマンド未実装はヘルプに誘導（終了コードは0）
    if args.command is None:
        parser.print_help()
        return 0

    print(f"[stub] command='{args.command}' is not implemented yet.")
    return 0
