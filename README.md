# densigrav (v0.1)

Gravity anomaly preprocessing focused on **DEM-based terrain correction (TC)** and a smooth **QGIS-first** workflow.

v0.1 (Issues **01–07**) delivers:

- `project.yaml` schema validation (typo detection; `extra=forbid`)
- Gravity points ingest (CSV/GPKG) → standardized GeoPackage
- DEM AOI clipping + point elevation (`z`) completion from DEM
- **Bouguer anomaly + DEM → terrain correction (TC) → complete Bouguer anomaly (CBA)** (points + optional grids)
- Optional **regional/residual separation** (polynomial) and **gridding** outputs
- **Section extraction**: preprocessed points + a section line → `profile.csv` + `section_points.gpkg`
- (Prototype) 2D Talwani utilities on the extracted profile (`talwani-forward`, `talwani-invert`)

---

## Requirements / Notes (Important)

### CRS (Coordinate Reference System)

`densigrav` assumes a **projected CRS in meters** (e.g., UTM).

- If `project.crs` is geographic (e.g., EPSG:4326), distance-based parameters in meters (e.g., `outer_radius_m`, `buffer_width_m`) become invalid, so densigrav will stop with an error.
- Reproject **both** the point data and the DEM to the **same projected CRS** using QGIS (or other GIS tools) before running densigrav.

**Recommended choices (manual):**

- Around Japan: JGD2011 / Japan Plane Rectangular CS, or UTM
- Global: UTM (choose the zone from the longitude of the point set’s centroid)

---

## Install

Development install (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Full v0.1 feature set (Step1 + Step2 + plots + quick inversion)
python -m pip install -e ".[dev,geo,raster,grav,grid,viz]"
```

Minimal (Step1 only, no plots / no quick inversion):

```bash
python -m pip install -e ".[geo,raster,grav]"
```

Verify core deps:

```bash
python -c "import geopandas, rasterio, harmonica; print('ok')"
```

---

## Quickstart

### 1) Prepare `project.yaml`

Minimal example (v0.1):

```yaml
project:
  name: "demo"
  crs: "EPSG:6677"        # projected CRS (meters)
  gravity_unit: "mGal"    # unit of input gravity values
  length_unit: "m"

paths:
  data_dir: "data"
  cache_dir: "cache"
  results_dir: "results"

inputs:
  gravity_points:
    path: "data/gravity_points.csv"
    value_kind: "bouguer_anomaly"   # v0.1 Step1 assumes Bouguer anomaly input
    columns:
      x: "E"
      y: "N"
      z: "elev_m"          # optional (filled from DEM if omitted or missing)
      value: "bouguer_mgal"
      sigma: "sigma_mgal"  # optional
  dem:
    path: "data/dem.tif"
    z_unit: "m"

step1_preprocess:
  bouguer_density_kgm3: 2670.0
  terrain_correction:
    enabled: true
    method: "dem_prism"
    density_kgm3: 2670.0
    outer_radius_m: 20000.0
    station_epsilon_m: 0.1

  # Optional (Issue06): regional/residual
  regional:
    enabled: false
    method: "polynomial"
    order: 2
    output: "residual"     # residual | regional | both

  # Optional (Issue06): grids
  outputs:
    points_gpkg: "cache/preprocessed_points.gpkg"
    grids:
      enabled: false
      resolution_m: 250.0
      cba_grid: "cache/cba.tif"
      tc_grid: "cache/terrain_correction.tif"

# Issue07: section extraction
step2_section:
  enabled: true
  buffer_width_m: 2000.0
  sampling_interval_m: 50.0
  outputs:
    result_dir: "results/2d/section_A"
    section_points_gpkg: "section_points.gpkg"
```

### 2) Validate the project

```bash
densigrav project validate project.yaml
# To also check whether input files exist:
densigrav project validate project.yaml --check-exists
```

### 3) Step 1 (Preprocess): BA + DEM → TC → CBA

```bash
densigrav preprocess project.yaml --overwrite
```

Outputs (default):

- `cache/preprocessed_points.gpkg` (layer: `gravity_points`)
- `cache/dem_clipped_tc.tif` (DEM clipped to AOI)

Main added columns (in `preprocessed_points.gpkg`):

- `slab_mgal` (Bouguer slab attraction)
- `terrain_effect_mgal` (terrain effect computed from DEM prisms)
- `tc_mgal` (terrain correction = slab - terrain_effect)
- `cba_mgal` (complete Bouguer anomaly = bouguer + tc)

If `step1_preprocess.regional.enabled: true`:

- `regional_mgal`, `residual_mgal` (depending on `regional.output`)

If `step1_preprocess.outputs.grids.enabled: true`:

- `cache/terrain_correction.tif`
- `cache/cba.tif`

---

## Step 2 (Issue07): Section extraction

### Prepare a section line

Create a **LineString** in QGIS (or any GIS tool), and export it as a vector file (GeoPackage recommended).

**Important:** the line CRS must match `project.crs`.

### Run section extraction

```bash
densigrav section extract project.yaml \
  --line data/section_line.gpkg \
  --layer line \
  --overwrite
```

Outputs (default): `results/2d/section_A/`

- `section_points.gpkg` (layer: `section_points`)
- `profile.csv` (no geometry)
- `profile.png` (unless `--no-plot`)

How it works (v0.1):

- Densigrav samples station points along the line at `sampling_interval_m`
- It filters preprocessed points inside a corridor of `buffer_width_m`
- For each station point, it attaches the nearest point values (CBA/TC/residual…)

---

## (Prototype) 2D Talwani utilities

> These commands exist in v0.1 to support early Step2 workflows.
> The interface will likely evolve as Issue08/09 progresses.

### Talwani model format (YAML)

`data/talwani/model.yaml` example:

```yaml
density_contrast_kgm3: 300.0
vertices_xz_m:
  - [2200, 300]
  - [2800, 300]
  - [3200, 1400]
  - [1800, 1400]
```

Notes:

- `x` is distance along the section (`dist_m`)
- `z` is **positive downward** (z-down)

### Forward calculation on `profile.csv`

```bash
densigrav section talwani-forward \
  --profile results/2d/section_A/profile.csv \
  --model data/talwani/model.yaml \
  --out results/2d/section_A/profile_talwani.csv \
  --overwrite
```

- Output CSV adds `talwani_pred_mgal` and `talwani_resid_mgal`.
- If your profile has `elev_m` and you want to use it as observation height:
  add `--use-elev` (uses `z_obs = -elev_m`).

### Quick inversion (single trapezoid)

Fits a single trapezoid body (x0, z_top, z_bottom, halfwidth_top, halfwidth_bottom).
Density contrast is fixed via `--drho`.

```bash
densigrav section talwani-invert \
  --profile results/2d/section_A/profile.csv \
  --drho 300 \
  --out-model results/2d/section_A/fit_model.yaml \
  --overwrite
```

Requires `scipy` (install `.[grid]`).

---

## Optional utilities

### points ingest (standardize points to GeoPackage)

Reads CSV/GPKG, applies column mapping and unit conversion, and writes a standardized GeoPackage.

```bash
densigrav points ingest project.yaml --overwrite
# Output (default): cache/gravity_points.gpkg
```

### dem prepare (clip DEM AOI + fill z)

Clips the DEM by point bbox + buffer and fills missing point elevations (`z`) from the DEM.

```bash
densigrav dem prepare project.yaml --overwrite
# Outputs (default):
# - cache/dem_clipped.tif
# - cache/points_with_z.gpkg
```

---

## Viewing in QGIS (v0.1)

1. Load `cache/preprocessed_points.gpkg` via **Add Vector Layer**
2. Style by `cba_mgal` / `tc_mgal` / `residual_mgal`
3. (Issue07) Load `results/2d/section_A/section_points.gpkg` and inspect attributes along the section
4. Optionally load `cache/cba.tif` / `cache/terrain_correction.tif` if grids are enabled

---

## Troubleshooting

### `Extra inputs are not permitted`

Your `project.yaml` contains **keys that are not defined in the schema** (densigrav uses `extra=forbid` to catch typos).

### `CRS must be projected`

You are using a geographic CRS (e.g., EPSG:4326). Reproject both points and DEM to a projected CRS (meters).

### `CRS mismatch: project=..., line=...`

Your section line CRS differs from `project.crs`. Reproject the line to the project CRS.

### `Permission denied` when overwriting `.gpkg`

GeoPackage is SQLite; QGIS can lock the file.

- Remove the layer from QGIS (or close QGIS), then retry
- Or change the output path to a new filename

---

## Development

Lint/Test:

```bash
ruff check .
pytest
```

---

## License

MIT License
