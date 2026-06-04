# Chicago Walkability Engine

Predicts and explains pedestrian walkability for every neighbourhood in Chicago
using machine learning, street network analysis, and SHAP attribution. Goes
beyond Walk Score by identifying *which features* drive each prediction and
*whether those features cluster along demographic lines*.

## Live Demo

| Service | URL |
|---|---|
| Dashboard | https://walkability-engine.streamlit.app |
| API | https://walkability-engine-production.up.railway.app |
| API Docs | https://walkability-engine-production.up.railway.app/docs |

## What It Does

Walk Score gives you a number. This system gives you a number plus an
explanation plus a demographic overlay. For any of Chicago's 5,023
H3 resolution-9 hex cells (~174m edge), it predicts a walkability score,
decomposes that prediction into per-feature SHAP attributions, and
quantifies whether low-scoring areas cluster in minority and low-income
neighbourhoods — and why.

## Results

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge baseline | 11.802 | 0.512 |
| XGBoost | 9.610 | 0.680 |
| LightGBM | 9.507 | 0.686 |
| Ensemble | 9.371 | 0.695 |
| **GraphSAGE GNN** | **7.780** | **0.790** |

The GNN beats the tabular ensemble by 1.591 RMSE points by incorporating
spatial adjacency — what surrounds a neighbourhood matters, not just
what is in it. Optuna consistently chose a single-layer architecture
(59→192→1), confirming that walkability is a 1-hop local signal (~174m radius).

## Key Equity Findings

- **Moran's I = 0.866** (p < 0.0001) — walkability is near-maximally
  spatially clustered. Low-scoring hexes form solid contiguous blocks
  on Chicago's South and West Sides.
- **Population density r = +0.650** — the strongest demographic predictor.
- **% Black r = −0.259**, **% minority r = −0.231** — significant negative
  correlations after 1,000-trial permutation testing.
- **% Hispanic r = +0.020** — not significant. Pilsen (high walkability)
  and Northwest Side (moderate) cancel out in the aggregate.
- **Restaurant density accounts for 34% of the 10-point walkability gap**
  between highest-minority and lowest-minority quartiles. Safety
  infrastructure is not the driver — amenity distribution is.
- **Safety crash density shows a positive SHAP effect** — a Walk Score label
  artefact. The model learned that crash density proxies urban activity,
  not danger, because Walk Score does not penalise safety.

## Architecture

```
OSM street network  ──┐
Vision Zero crashes   ├──▶ Feature pipeline ──▶ H3 hex grid (5,023 cells)
CTA GTFS transit      │    (60 features)        │
Census ACS 2022       │                         ▼
SRTM elevation      ──┘                    ML models
                                           (XGBoost / LightGBM /
Walk Score API ────────────────────────▶   GNN ensemble)
(weak labels)                              │
                                           ▼
                                      SHAP attribution
                                           │
                                           ▼
                              FastAPI ──▶ Streamlit dashboard
                              (Railway)   (Streamlit Cloud)
```

## Project Structure

```
walkability-engine/
├── src/
│   ├── ingestion/          OSM network download
│   ├── features/           Feature engineering modules
│   │   ├── edge_preprocessing.py
│   │   ├── network_features.py
│   │   ├── poi_transit_features.py
│   │   ├── terrain_safety_features.py
│   │   ├── census_features.py
│   │   └── labels.py           Walk Score API + spatial CV
│   ├── models/
│   │   ├── utils.py             Shared CV utilities
│   │   ├── tabular.py           XGBoost + Optuna
│   │   ├── lgbm.py              LightGBM + Optuna
│   │   ├── ensemble.py          Ridge meta-learner
│   │   └── gnn.py               GraphSAGE + Optuna
│   ├── evaluation/
│   │   ├── shap_analysis.py
│   │   └── equity_audit.py
│   ├── api/
│   │   └── main.py              FastAPI backend (6 endpoints)
│   └── dashboard/
│       ├── app.py               Streamlit entry point
│       ├── api_client.py        Backend API calls
│       ├── map_builder.py       Pydeck H3 hex map
│       ├── charts.py            Plotly SHAP charts
│       ├── styles.py            Injected CSS
│       └── config.py            API URL resolution
├── data/processed/
│   ├── shap_values.parquet      1.5 MB — SHAP matrix (5023 × 65)
│   └── master_features.parquet  4.8 MB — all features (11484 × 71)
├── Dockerfile
├── requirements.txt
└── configs/city_config.yml
```

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/your-username/walkability-engine
cd walkability-engine
conda env create -f environment-dev.yml
conda activate walkability

# Terminal 1 — start API
python -m uvicorn src.api.main:app --reload --port 8001

# Terminal 2 — start dashboard
API_URL=http://localhost:8001 streamlit run src/dashboard/app.py
```

## Reproducing the Full Pipeline

```bash
# Weeks 2-5: feature engineering
python -m src.ingestion.fetch_osm_network
python -m src.features.edge_preprocessing
python -m src.features.network_features
python -m src.features.poi_transit_features
python -m src.features.terrain_safety_features
python -m src.features.census_features
python -m src.features.feature_store

# Week 6: Walk Score labels + spatial CV
python -m src.features.labels --step scores
python -m src.features.labels --step splits

# Weeks 7-9: models
python -m src.models.tabular
python -m src.models.lgbm --fast
python -m src.models.ensemble
python -m src.models.gnn

# Weeks 10-11: analysis
python -m src.evaluation.shap_analysis
python -m src.evaluation.equity_audit
```

## Data Sources

| Source | What it provides | License |
|---|---|---|
| OpenStreetMap | Street network, POIs | ODbL |
| Chicago Vision Zero | Pedestrian crash data | Public domain |
| CTA GTFS | Transit stops, headways | Public domain |
| Census ACS 2022 | Demographics (Cook County) | Public domain |
| SRTM 30m | Elevation | Public domain |
| Walk Score API | Walkability labels | Commercial (queried once) |

## API Endpoints

```
GET  /health                    Liveness check
GET  /hex/{h3_index}            Full data for one hex cell
GET  /predict/{lat}/{lng}       Lat/lng → H3 → hex data
POST /compare                   Side-by-side neighbourhood comparison
GET  /city/summary              City-wide aggregate statistics
GET  /neighbourhood/{name}      Named neighbourhood shortcut
```

## Release Tags

| Tag | Week | What was delivered |
|---|---|---|
| v0.2.0-week2 | 2 | OSM network features (26 features) |
| v0.3.0-week3 | 3 | POI + transit features |
| v0.4.0-week4 | 4 | Terrain + safety features |
| v0.5.0-week5 | 5 | Census demographics |
| v0.6.0-week6 | 6 | Walk Score labels + KMeans spatial CV |
| v0.7.0-week7 | 7 | XGBoost model (RMSE 9.610) |
| v0.8.0-week8 | 8 | LightGBM + Ridge ensemble (RMSE 9.371) |
| v0.9.0-week9 | 9 | GraphSAGE GNN (RMSE 7.780) |
| v0.10.0-week10 | 10 | SHAP explainability |
| v0.11.0-week11 | 11 | Spatial equity audit |
| v0.12.0-week12 | 12 | FastAPI backend + deployment |