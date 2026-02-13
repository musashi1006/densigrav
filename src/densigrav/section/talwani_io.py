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
