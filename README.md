# densigrav (v0.1)
Gravity anomaly preprocessing focused on **DEM-based terrain correction** and a smooth **QGIS-first** workflow.

v0.1 (Issue01〜05) の到達点：
- `project.yaml` のスキーマ検証（typo検出を重視：`extra=forbid`）
- 点データ（CSV/GPKG）→標準化GPKG出力
- DEMのAOIクリップ＋点への標高（z）補完
- **Bouguer anomaly + DEM → terrain correction (TC) → complete Bouguer anomaly (CBA)** を点GPKGに出力

---

## Requirements / 注意（重要）
### CRS（座標系）
v0.1 は **投影座標系（メートル）** 前提です（UTMなど）。
- `project.crs` が地理座標（例：EPSG:4326）のままだと、距離（m）パラメータ（`outer_radius_m`, `resolution_m` など）が破綻するため、処理を停止します。
- まず QGIS 等で点データとDEMを同じ投影CRSへ再投影してから使ってください。

**おすすめの選び方（手動）：**
- 日本付近：JGD2011 / 平面直角座標系、または UTM
- 世界一般：UTM（点群の中心経度からゾーンを選ぶ）

---

## Install
開発インストール（推奨）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# v0.1で必要な一式（lint/test + vector + raster + gravity）
python -m pip install -e ".[dev,geo,raster,grav]"
```

依存の確認：

```bash
python -c "import geopandas, rasterio, harmonica; print('ok')"
```

---

## Quickstart

### 1) `project.yaml` を用意

最小例（v0.1）：

```yaml
project:
  name: "demo"
  crs: "EPSG:6677"        # 投影CRS（m）
  gravity_unit: "mGal"    # 入力の重力値の単位
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
      z: "elev_m"          # 無ければ省略可（DEMから補完）
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

### 2) project の検証

```bash
densigrav project validate project.yaml
# 入力ファイルの存在も確認する場合：
densigrav project validate project.yaml --check-exists
```

### 3) Step1（前処理）：BA + DEM → TC → CBA（点出力）

```bash
densigrav preprocess project.yaml --overwrite
```

生成物（デフォルト）：

* `cache/preprocessed_points.gpkg`（layer: `gravity_points`）

主な追加列：

* `slab_mgal`（ブーゲースラブ）
* `terrain_effect_mgal`（DEMプリズムの地形効果）
* `tc_mgal`（terrain correction = slab - terrain_effect）
* `cba_mgal`（complete Bouguer anomaly = bouguer + tc）

---

## Optional utilities (v0.1)

### points ingest（点データを標準化してGPKGへ）

CSV/GPKGを読み、列マッピングと単位変換を行ってGPKGを作ります。

```bash
densigrav points ingest project.yaml --overwrite
# 出力（デフォルト）：cache/gravity_points.gpkg
```

### dem prepare（DEMのAOIクリップ＋z補完）

点のbbox＋bufferでDEMをクリップし、zが欠損の点をDEMから補完します。

```bash
densigrav dem prepare project.yaml --overwrite
# 出力（デフォルト）：
# - cache/dem_clipped.tif
# - cache/points_with_z.gpkg
```

---

## QGISでの閲覧（v0.1）

1. `cache/preprocessed_points.gpkg` を Add Vector Layer で読み込む
2. `cba_mgal` / `tc_mgal` をスタイルで可視化
3. 断面や沿線抽出は v0.1 以降（Step2）で拡張予定

---

## Troubleshooting

### `Extra inputs are not permitted` が出る

`project.yaml` に **スキーマに存在しないキー**が入っています（typo検出のため `extra=forbid`）。
例：`step1_preprocess.outputs.dem_clipped` は v0.1 スキーマに無いので削除してください。

### `CRS must be projected` が出る

EPSG:4326 など地理座標のままです。点とDEMを投影CRS（メートル）へ再投影してください。

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
