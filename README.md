# densigrav (v0.1)

Gravity anomaly preprocessing focused on **DEM-based terrain correction** and a smooth **QGIS-first** workflow.

v0.1 (Issues 01–05) delivers:
- `project.yaml` schema validation (typo detection; `extra=forbid`)
- Gravity points (CSV/GPKG) → standardized GeoPackage output
- DEM AOI clipping + elevation (`z`) completion for points
- **Bouguer anomaly + DEM → terrain correction (TC) → complete Bouguer anomaly (CBA)** exported as a point GeoPackage

---

## Requirements / Notes (Important)

### CRS (Coordinate Reference System)
v0.1 assumes a **projected CRS in meters** (e.g., UTM).
- If `project.crs` is geographic (e.g., EPSG:4326), distance-based parameters in meters (e.g., `outer_radius_m`, `resolution_m`) become invalid, so densigrav will stop with an error.
- Reproject both the point data and the DEM to the same projected CRS (meters) using QGIS (or other GIS tools) before running densigrav.

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

# Everything needed for v0.1 (lint/test + vector + raster + gravity)
python -m pip install -e ".[dev,geo,raster,grav]"
```

Verify dependencies:

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
    value_kind: "bouguer_anomaly"
    columns:
      x: "E"
      y: "N"
      z: "elev_m"          # optional (filled from DEM if omitted or missing)
      value: "bouguer_mgal"
      sigma: "sigma_mgal"
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
  regional:
    enabled: false
    method: "polynomial"
    order: 2
    output: "residual"
  outputs:
    points_gpkg: "cache/preprocessed_points.gpkg"
    grids:
      enabled: false
      resolution_m: 250.0
      cba_grid: "cache/cba.tif"
      tc_grid: "cache/terrain_correction.tif"
```

### 2) Validate the project

```bash
densigrav project validate project.yaml
# To also check whether input files exist:
densigrav project validate project.yaml --check-exists
```

### 3) Step 1 (Preprocess): BA + DEM → TC → CBA (point output)

```bash
densigrav preprocess project.yaml --overwrite
```

Outputs (default):

* `cache/preprocessed_points.gpkg` (layer: `gravity_points`)

Main added columns:

* `slab_mgal` (Bouguer slab attraction)
* `terrain_effect_mgal` (terrain effect computed from DEM prisms)
* `tc_mgal` (terrain correction = slab - terrain_effect)
* `cba_mgal` (complete Bouguer anomaly = bouguer + tc)

---

## Optional utilities (v0.1)

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
2. Style by `cba_mgal` / `tc_mgal`
3. Section tools / along-line extraction will be added after v0.1 (Step 2)

---

## Troubleshooting

### `Extra inputs are not permitted`

Your `project.yaml` contains **keys that are not defined in the schema** (densigrav uses `extra=forbid` to catch typos).
Example: remove `step1_preprocess.outputs.dem_clipped` because it does not exist in the v0.1 schema.

### `CRS must be projected`

You are using a geographic CRS (e.g., EPSG:4326). Reproject both points and DEM to a projected CRS (meters).

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
