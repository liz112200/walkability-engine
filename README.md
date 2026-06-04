# Predictive Walkability Engine

> A Geospatial AI system that predicts multi-dimensional pedestrian walkability for Chicago, IL - going beyond traditional Walk Score metrics through Graph Neural Networks, street-level imagery analysis, and an equity audit of walkability disparities across demographic groups.


---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Reproducing Results](#reproducing-results)
- [Roadmap (16-week plan)](#roadmap)
- [References](#references)
- [License](#license)

---

## Overview

Standard walkability metrics (e.g. Walk Score) reduce complex pedestrian environments to a single scalar. This project builds a **multi-dimensional predictive engine** that:

1. Assembles a 30–40 feature vector per H3 hexagonal cell from OpenStreetMap, GTFS transit feeds, SRTM elevation, US Census ACS, and Vision Zero crash data.
2. Trains an **ensemble of XGBoost + LightGBM + Graph Neural Network** models using spatial cross-validation to avoid leakage.
3. Adds a **street-level imagery dimension** via fine-tuned EfficientNet on Google Street View / Mapillary imagery.
4. Produces **SHAP-based explanations** for every predicted score — telling you *why* an area is scored the way it is.
5. Runs an **equity audit** testing whether low-walkability areas are disproportionately distributed across income and demographic groups using spatial autocorrelation (Moran's I).

**Target city:** Chicago, IL — chosen for its rich open data ecosystem, pronounced walkability gradients (Loop vs. far South/West sides), and dense pedestrian crash dataset.

---

## Architecture

```
Data Ingestion          Feature Engineering       ML Core             Output
──────────────          ───────────────────       ───────             ──────
OSMnx (OSM)      ─┐
GTFS (CTA)        ├──► H3 hex aggregation ──►  XGBoost  ─┐
Census ACS        │    (res-9, ~174m cells)    LightGBM   ├──► Ensemble ──► Score + SHAP
SRTM elevation   ─┤                            GNN (GAT)  │    + equity audit
Vision Zero crash │                            EfficientNet┘
Street View imgs ─┘

```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/walkability-engine.git
cd walkability-engine
```

### 2. Create and activate the conda environment

```bash
# This resolves GDAL/PROJ correctly via conda-forge
conda env create -f environment.yml
conda activate walkability

# Verify the geospatial stack is wired
python -c "import osmnx, geopandas, h3; print('Environment OK')"
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your Walk Score, Census, and Street View API keys
```

### 4. Run Week 1 ingestion

```bash
# Downloads Chicago pedestrian network (~180k nodes, ~10 min first run)
# Results are cached in .osmnx_cache/ for fast re-runs
python -m src.ingestion.fetch_osm_network

# Expected output:
#   data/raw/osm/chicago_walk_graph.graphml   (~200 MB)
#   data/raw/osm/chicago_walk_edges.gpkg      (~80 MB)
#   data/raw/osm/chicago_walk_nodes.gpkg      (~20 MB)
```

### 5. Open the sanity-check notebook

```bash
jupyter lab notebooks/01_sanity_check_network.ipynb
# Run all cells — all 8 checks in the final cell must pass
```

---

## Project Structure

```
walkability-engine/
├── configs/
│   └── city_config.yml          # City, CRS, H3 resolution, all data paths
├── data/
│   ├── raw/                     # Original downloaded files (git-ignored)
│   │   ├── osm/                 # GraphML + GeoPackages from OSMnx
│   │   ├── gtfs/                # CTA GTFS feed
│   │   ├── census/              # ACS 5-year estimates
│   │   ├── elevation/           # SRTM 30m raster tiles
│   │   ├── poi/                 # Point-of-interest data
│   │   └── crash/               # Vision Zero pedestrian crash data
│   └── processed/
│       ├── features/            # H3-indexed feature GeoDataFrame (Parquet)
│       ├── labels/              # Walk Score labels
│       └── splits/              # Spatial CV fold assignments
├── notebooks/
│   ├── 01_sanity_check_network.ipynb
│   ├── 02_h3_grid_features.ipynb       # Week 2
│   ├── 03_poi_transit_features.ipynb   # Week 3
│   └── ...
├── src/
│   ├── ingestion/               # Data download scripts
│   ├── features/                # Feature engineering
│   ├── models/                  # XGBoost, LightGBM, GNN
│   ├── evaluation/              # Spatial CV, metrics, SHAP
│   ├── visualization/           # Map and chart helpers
│   └── utils/                   # Config loader, logging, H3 helpers
├── outputs/
│   ├── figures/                 # PNG exports from notebooks
│   ├── maps/                    # GeoJSON for Kepler.gl / Streamlit
│   ├── models/                  # Serialised model artefacts
│   └── reports/                 # Final PDF report
├── tests/                       # pytest unit tests
├── .env.example                 # API key template
├── environment.yml              # Conda environment (conda-forge)
└── README.md
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| OpenStreetMap via OSMnx | Street network, POIs, sidewalks | Free |
| CTA GTFS | Transit stops, headways, routes | Free via transit.land |
| US Census ACS (5-yr) | Income, race, age by tract | Free API key |
| NASA SRTM | 30m elevation tiles | Free (Earthdata account) |
| Vision Zero Chicago | Pedestrian crash locations | Free, city open data portal |
| Walk Score API | Ground truth labels | Free tier, 5k calls/day |
| Google Street View Static | Imagery for EfficientNet | Pay-per-use ($7/1k images) |
| Mapillary | Open imagery alternative | Free API |

---

## Reproducing Results

Each week has a corresponding notebook that must be run in order:

```bash
# Week 1: Ingestion
python -m src.ingestion.fetch_osm_network
jupyter nbconvert --to notebook --execute notebooks/01_sanity_check_network.ipynb

# Week 2: H3 + network features  (coming Week 2)
# Week 3: POI + transit features  (coming Week 3)
# ...
```

All notebooks are designed to be **restart-and-run-all clean**.

---

## Roadmap

| Phase | Weeks | Focus |
|-------|-------|-------|
| Foundation | 1–3 | Environment, OSM ingestion, H3 grid, POI/transit/terrain features |
| Feature Engineering | 4–6 | Census integration, spatial CV setup, ground truth labels |
| ML Modeling | 7–10 | XGBoost baseline, LightGBM ensemble, GNN (PyTorch Geometric) |
| Explainability & Equity | 11–12 | SHAP attribution, Moran's I, demographic correlation analysis |
| Application Layer | 13–14 | FastAPI backend, PostGIS, Streamlit dashboard |
| Writeup & Polish | 15–16 | Report, presentation, reproducibility pass |

---

## References

- Frank, L. et al. (2010). *Stepping towards causation: Do built environments or neighborhood and travel preferences explain physical activity, driving, and obesity?* Social Science & Medicine.
- Sallis, J. et al. (2016). *Physical activity in relation to urban environments in 14 cities worldwide.* The Lancet.
- Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems.
- Brownson, R. et al. (2009). *Measuring the built environment for physical activity.* American Journal of Preventive Medicine.
- Hamilton, W. et al. (2017). *Inductive representation learning on large graphs (GraphSAGE).* NeurIPS.

---

## License

MIT License — see [LICENSE](LICENSE).