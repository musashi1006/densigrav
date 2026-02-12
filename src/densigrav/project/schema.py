from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GravityUnit(str, Enum):
    mGal = "mGal"
    uGal = "uGal"
    microGal = "µGal"


class LengthUnit(str, Enum):
    m = "m"


class ValueKind(str, Enum):
    bouguer_anomaly = "bouguer_anomaly"
    gravity_disturbance = "gravity_disturbance"
    observed_gravity = "observed_gravity"


class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    crs: str = Field(min_length=1)  # e.g. "EPSG:6677"
    gravity_unit: GravityUnit = GravityUnit.mGal
    length_unit: LengthUnit = LengthUnit.m


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results")


class GravityPointColumns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str
    y: str
    value: str
    z: Optional[str] = None
    sigma: Optional[str] = None


class GravityPointsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    value_kind: ValueKind = ValueKind.bouguer_anomaly
    columns: GravityPointColumns


class DemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    z_unit: str = "m"


class InputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gravity_points: GravityPointsInput
    dem: DemInput


class TerrainCorrectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    method: str = "dem_prism"
    density_kgm3: float = 2670.0
    outer_radius_m: float = 20000.0
    station_epsilon_m: float = 0.1


class RegionalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str = "polynomial"
    order: int = 2
    output: str = "residual"  # residual | regional | both (v0.1は文字列でOK)


class GridOutputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    resolution_m: float = 250.0
    cba_grid: Path = Path("cache/cba.tif")
    tc_grid: Path = Path("cache/terrain_correction.tif")


class PreprocessOutputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points_gpkg: Path = Path("cache/preprocessed_points.gpkg")
    grids: GridOutputsConfig = GridOutputsConfig()


class Step1PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bouguer_density_kgm3: float = 2670.0
    terrain_correction: TerrainCorrectionConfig = TerrainCorrectionConfig()
    regional: RegionalConfig = RegionalConfig()
    outputs: PreprocessOutputsConfig = PreprocessOutputsConfig()


class Step2SectionOutputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_dir: Path = Path("results/2d/section_A")
    section_points_gpkg: Path = Path("section_points.gpkg")
    figures: list[str] = Field(default_factory=list)


class Step2SectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    buffer_width_m: float = 2000.0
    sampling_interval_m: float = 50.0
    outputs: Step2SectionOutputsConfig = Step2SectionOutputsConfig()


class ProjectFile(BaseModel):
    """
    v0.1 (Step2まで) の最小プロジェクト定義。
    タイポ検出のため基本は extra=forbid にしている。
    """

    model_config = ConfigDict(extra="forbid")

    project: ProjectMeta
    paths: PathsConfig = PathsConfig()
    inputs: InputsConfig

    step1_preprocess: Step1PreprocessConfig = Field(default_factory=Step1PreprocessConfig)
    step2_section: Step2SectionConfig = Step2SectionConfig()

    # 将来拡張用（v0.1は何でも入れてよい“逃げ道”を用意）
    future: dict[str, Any] = Field(default_factory=dict)
