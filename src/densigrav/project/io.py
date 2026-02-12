from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import ProjectFile


class ProjectValidationError(Exception):
    pass


@dataclass(frozen=True)
class LoadedProject:
    config: ProjectFile
    base_dir: Path  # project.yaml があるディレクトリ（相対パス解決の基準）


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ProjectValidationError(f"Project YAML not found: {path}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ProjectValidationError(f"Invalid YAML: {path}\n{e}") from e

    if not isinstance(data, dict):
        raise ProjectValidationError(f"YAML root must be a mapping (dict): {path}")

    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _resolve_path(base_dir: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_project_paths(cfg: ProjectFile, base_dir: Path) -> ProjectFile:
    """
    既知のパス項目を base_dir 基準で絶対パスに解決して返す。
    """
    gp = cfg.inputs.gravity_points.model_copy(
        update={"path": _resolve_path(base_dir, cfg.inputs.gravity_points.path)}
    )
    dem = cfg.inputs.dem.model_copy(update={"path": _resolve_path(base_dir, cfg.inputs.dem.path)})

    inputs = cfg.inputs.model_copy(update={"gravity_points": gp, "dem": dem})

    # outputs (step1)
    s1_out = cfg.step1_preprocess.outputs.model_copy(
        update={"points_gpkg": _resolve_path(base_dir, cfg.step1_preprocess.outputs.points_gpkg)}
    )
    s1_grids = cfg.step1_preprocess.outputs.grids.model_copy(
        update={
            "cba_grid": _resolve_path(base_dir, cfg.step1_preprocess.outputs.grids.cba_grid),
            "tc_grid": _resolve_path(base_dir, cfg.step1_preprocess.outputs.grids.tc_grid),
        }
    )
    s1_out = s1_out.model_copy(update={"grids": s1_grids})
    step1 = cfg.step1_preprocess.model_copy(update={"outputs": s1_out})

    # outputs (step2)
    s2_out = cfg.step2_section.outputs.model_copy(
        update={
            "result_dir": _resolve_path(base_dir, cfg.step2_section.outputs.result_dir),
            "section_points_gpkg": _resolve_path(
                base_dir, cfg.step2_section.outputs.section_points_gpkg
            ),
        }
    )
    step2 = cfg.step2_section.model_copy(update={"outputs": s2_out})

    return cfg.model_copy(
        update={"inputs": inputs, "step1_preprocess": step1, "step2_section": step2}
    )


def load_project(path: Path, resolve_paths: bool = True) -> LoadedProject:
    """
    project.yaml を読み込み、Pydanticで検証して返す。
    resolve_paths=True の場合、既知のパスを絶対パスに解決する。
    """
    path = Path(path)
    base_dir = path.parent.resolve()

    raw = _read_yaml(path)
    try:
        cfg = ProjectFile.model_validate(raw)
    except ValidationError as e:
        raise ProjectValidationError(f"Project schema validation failed: {path}\n{e}") from e

    if resolve_paths:
        cfg = resolve_project_paths(cfg, base_dir)

    return LoadedProject(config=cfg, base_dir=base_dir)


def validate_project(
    path: Path, resolve_paths: bool = True, check_exists: bool = False
) -> LoadedProject:
    """
    load + (option) existence check.
    """
    loaded = load_project(path, resolve_paths=resolve_paths)

    if check_exists:
        gp = loaded.config.inputs.gravity_points.path
        dem = loaded.config.inputs.dem.path
        missing = [p for p in [gp, dem] if not p.exists()]
        if missing:
            msg = "\n".join([f"- missing: {p}" for p in missing])
            raise ProjectValidationError(f"Input files do not exist:\n{msg}")

    return loaded


def save_project_yaml(path: Path, cfg: ProjectFile) -> None:
    """
    ProjectFile をYAMLに書き戻す（v0.1では主にデバッグ/将来用途）。
    パスは cfg に入っている値をそのまま出力する。
    """
    data = cfg.model_dump(mode="python")
    _write_yaml(Path(path), data)
