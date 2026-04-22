"""
src/features/labels.py
───────────────────────
Week 6: Walk Score ground truth labels and spatial cross-validation splits.

Writes:
    data/processed/labels/chicago_walk_scores.parquet
    data/processed/splits/chicago_spatial_cv.parquet

Walk Score API
--------------
Calls the Walk Score API for each H3 hex centroid and stores the raw score
(0–100) as the modeling target. Free tier: 5,000 calls/day.

Chicago has ~11,484 hex cells, of which ~5,313 are dense (have street network).
At 5,000 calls/day the full dataset takes 2 days. To run in one session, we
query only dense cells (~5,313 calls) and skip sparse cells (lake, airport).

API documentation: https://www.walkscore.com/professional/api.php

Spatial Cross-Validation
------------------------
Standard random train/test splits are INVALID for geospatial data because
neighbouring hex cells share features (spatial autocorrelation). A model
trained with random splits will appear to generalise but will fail on new
cities or held-out neighbourhoods.

We use 5 geographic blocks (N / S / E / W / Center) as CV folds.
Each fold is used as the test set once; the other 4 are training.

The Center fold (roughly the Loop ± 3km) is always held out as the
final test set — it has the most varied walkability and provides the
cleanest evaluation of model generalisation.

Usage
-----
    # Fetch Walk Score labels (requires WALK_SCORE_API_KEY in .env)
    python -m src.features.labels --step scores

    # Build spatial CV splits (no API key needed)
    python -m src.features.labels --step splits

    # Both
    python -m src.features.labels
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.utils.config import cfg

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
WALK_SCORE_URL = "https://api.walkscore.com/score"
RATE_LIMIT_DELAY = 0.25   # seconds between API calls (4 calls/sec max)

# Chicago city center (Daley Plaza) for spatial fold assignment
CHICAGO_CENTER_LAT = 41.8840
CHICAGO_CENTER_LNG = -87.6320

# Radius in metres for the "Center" spatial fold
CENTER_RADIUS_M = 5_500


# ── Walk Score API ─────────────────────────────────────────────────────────────

def _fetch_walk_score(lat: float, lng: float, api_key: str) -> dict | None:
    """
    Fetch Walk Score for a single lat/lng point.
    Returns dict with walk_score, transit_score, bike_score or None on failure.
    """
    try:
        r = requests.get(
            WALK_SCORE_URL,
            params={
                "format":   "json",
                "lat":      lat,
                "lon":      lng,
                "wsapikey": api_key,
                "transit":  1,
                "bike":     1,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != 1:
            return None
        return {
            "walk_score":    data.get("walkscore"),
            "transit_score": data.get("transit", {}).get("score"),
            "bike_score":    data.get("bike", {}).get("score"),
            "ws_description": data.get("description", ""),
        }
    except Exception:
        return None


def build_walk_score_labels(
    out_path: Path | None = None,
    dense_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch Walk Score labels for all (or dense-only) H3 hex centroids.
    """
    load_dotenv()
    api_key = os.getenv("WALK_SCORE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "WALK_SCORE_API_KEY not found in .env\n"
        )

    slug     = cfg.city.slug
    out_path = out_path or (cfg.paths.labels / f"{slug}_walk_scores.parquet")
    cfg.paths.labels.mkdir(parents=True, exist_ok=True)

    net_path = cfg.paths.processed / f"{slug}_network_features.parquet"
    assert net_path.exists(), "Run Week 2 first."
    net = gpd.read_parquet(str(net_path))[
        ["h3_index", "centroid_lat", "centroid_lng", "data_sparse"]
    ]

    # Resume support — load existing scores if partial run was interrupted
    if out_path.exists():
        existing = pd.read_parquet(str(out_path))
        already_fetched = set(existing["h3_index"].values)
        logger.info(f"Resuming — {len(already_fetched):,} scores already fetched")
    else:
        existing = pd.DataFrame()
        already_fetched = set()

    # Select hexes to query
    if dense_only:
        to_fetch = net[net["data_sparse"] == 0].copy()
    else:
        to_fetch = net.copy()

    to_fetch = to_fetch[~to_fetch["h3_index"].isin(already_fetched)]
    logger.info(
        f"Walk Score labels: {len(to_fetch):,} hexes to fetch  |  "
        f"API key: {'*' * 8}{api_key[-4:]}"
    )

    if len(to_fetch) == 0:
        logger.info("All scores already fetched.")
        return existing

    # Estimate time
    est_minutes = len(to_fetch) * RATE_LIMIT_DELAY / 60
    logger.info(f"Estimated fetch time: {est_minutes:.0f} min at {1/RATE_LIMIT_DELAY:.0f} calls/sec")

    # Fetch
    rows = []
    failed = 0
    t0 = time.perf_counter()

    for _, hex_row in tqdm(to_fetch.iterrows(), total=len(to_fetch), desc="Walk Score"):
        result = _fetch_walk_score(
            hex_row["centroid_lat"],
            hex_row["centroid_lng"],
            api_key,
        )
        if result:
            rows.append({"h3_index": hex_row["h3_index"], **result})
        else:
            rows.append({
                "h3_index":     hex_row["h3_index"],
                "walk_score":   np.nan,
                "transit_score": np.nan,
                "bike_score":   np.nan,
                "ws_description": "",
            })
            failed += 1
        time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.perf_counter() - t0
    new_df = pd.DataFrame(rows)

    # Merge with existing (resume support)
    if len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(str(out_path))

    logger.info(f"Walk Score fetch complete in {elapsed/60:.1f} min")
    logger.info(f"  Fetched:  {len(new_df):,}")
    logger.info(f"  Failed:   {failed:,}  ({100*failed/max(len(new_df),1):.1f}%)")
    logger.info(f"  Total saved: {len(combined):,}")
    logger.info(
        f"  Score distribution:\n"
        f"{combined['walk_score'].describe().round(1).to_string()}"
    )
    return combined


# ── Spatial cross-validation splits ──────────────────────────────────────────

def build_spatial_cv_splits(
    out_path: Path | None = None,
    n_train_folds: int = 4,
    min_fold_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Assign each hex cell to one of (n_train_folds + 1) spatial CV folds.
 
    Returns
    -------
    DataFrame with columns: h3_index, fold, fold_name, split, data_sparse
    """
    from sklearn.cluster import KMeans
 
    slug     = cfg.city.slug
    out_path = out_path or (cfg.paths.splits / f"{slug}_spatial_cv.parquet")
    cfg.paths.splits.mkdir(parents=True, exist_ok=True)
 
    net = gpd.read_parquet(
        str(cfg.paths.processed / f"{slug}_network_features.parquet")
    )[["h3_index", "centroid_lat", "centroid_lng", "centroid_x",
       "centroid_y", "data_sparse"]]
 
    # Load labeled hexes — KMeans runs only on hexes that will be used in modeling
    labels_path = cfg.paths.labels / f"{slug}_walk_scores.parquet"
    assert labels_path.exists(), (
        "Walk Score labels not found. Run: python -m src.features.labels --step scores"
    )
    labels = pd.read_parquet(str(labels_path))[["h3_index", "walk_score"]]
 
    labeled_dense = (
        net[net["data_sparse"] == 0]
        .merge(labels.dropna(subset=["walk_score"]), on="h3_index", how="inner")
    )
    logger.info(f"Labeled dense hexes for fold assignment: {len(labeled_dense):,}")
 
    # ── Step 1: Identify center fold ──────────────────────────────────────────
    center_wgs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([CHICAGO_CENTER_LNG], [CHICAGO_CENTER_LAT]),
        crs="EPSG:4326",
    ).to_crs(cfg.city.crs)
    cx = center_wgs.geometry.x.iloc[0]
    cy = center_wgs.geometry.y.iloc[0]
 
    dist_from_center = np.sqrt(
        (labeled_dense["centroid_x"] - cx)**2 +
        (labeled_dense["centroid_y"] - cy)**2
    )
    is_center = dist_from_center <= CENTER_RADIUS_M
 
    center_hexes = labeled_dense[is_center]["h3_index"].values
    train_hexes  = labeled_dense[~is_center].copy().reset_index(drop=True)
 
    logger.info(
        f"Center fold: {len(center_hexes):,} hexes  |  "
        f"Train pool: {len(train_hexes):,} hexes"
    )
 
    # ── Step 2: KMeans on train hex centroids ─────────────────────────────────
    # n_init=20 runs KMeans 20 times with different seeds and keeps the best
    coords = train_hexes[["centroid_x", "centroid_y"]].values
    kmeans = KMeans(
        n_clusters=n_train_folds,
        random_state=RANDOM_SEED,
        n_init=20,
    )
    cluster_labels = kmeans.fit_predict(coords)
 
    # Remap cluster IDs to folds 1..n_train_folds
    # Sort clusters by centroid northing (y) so fold 1 = northernmost cluster.
    # This makes fold names geographically interpretable.
    cluster_centers = kmeans.cluster_centers_
    north_order = np.argsort(cluster_centers[:, 1])[::-1]  # descending y
    remap = {old: new + 1 for new, old in enumerate(north_order)}
    train_hexes["fold"] = [remap[c] for c in cluster_labels]
 
    # ── Step 3: Build fold_name from cluster geographic character ────────────
    fold_name_map = {}
    for fold_id in range(1, n_train_folds + 1):
        fold_hexes = train_hexes[train_hexes["fold"] == fold_id]
        mean_lat = fold_hexes["centroid_lat"].mean()
        mean_lng = fold_hexes["centroid_lng"].mean()
        ns = "north" if mean_lat >= CHICAGO_CENTER_LAT else "south"
        ew = "east"  if mean_lng >= CHICAGO_CENTER_LNG else "west"
        fold_name_map[fold_id] = f"{ns}_{ew}_{fold_id}"
 
    # ── Step 4: Assign folds to ALL hexes (not just labeled dense ones) ───────
    # Non-labeled and sparse hexes get fold=-1 (excluded from modeling).
    # Center hexes get fold=0. Train hexes get fold 1..n_train_folds.
    h3_to_fold      = dict(zip(train_hexes["h3_index"], train_hexes["fold"]))
    center_set       = set(center_hexes)
    fold_name_map[0] = "center"
 
    all_folds      = []
    all_fold_names = []
    all_splits     = []
 
    for h3_id in net["h3_index"]:
        if h3_id in center_set:
            all_folds.append(0)
            all_fold_names.append("center")
            all_splits.append("test")
        elif h3_id in h3_to_fold:
            f = h3_to_fold[h3_id]
            all_folds.append(f)
            all_fold_names.append(fold_name_map[f])
            all_splits.append("train")
        else:
            all_folds.append(-1)
            all_fold_names.append("excluded")
            all_splits.append("excluded")
 
    result = pd.DataFrame({
        "h3_index":    net["h3_index"].values,
        "fold":        all_folds,
        "fold_name":   all_fold_names,
        "split":       all_splits,
        "data_sparse": net["data_sparse"].values,
    })
 
    # ── Step 5: Balance check ─────────────────────────────────────────────────
    train_only = result[result["split"] == "train"]
    fold_counts = train_only["fold"].value_counts().sort_index()
    total_train = len(train_only)
 
    logger.info("Spatial CV fold distribution (labeled dense hexes):")
    logger.info(f"  {'Fold':<6} {'Name':<20} {'Count':>6} {'Pct':>6}")
    logger.info(f"  {'0':<6} {'center (test)':<20} {len(center_hexes):>6}  "
                f"{100*len(center_hexes)/(len(center_hexes)+total_train):>5.1f}%")
    for fold_id, count in fold_counts.items():
        pct = 100 * count / total_train
        logger.info(f"  {fold_id:<6} {fold_name_map[fold_id]:<20} {count:>6}  {pct:>5.1f}%")
 
    # Enforce minimum fold size
    min_pct_actual = fold_counts.min() / total_train * 100
    if min_pct_actual < min_fold_pct:
        raise ValueError(
            f"Smallest train fold has only {min_pct_actual:.1f}% of train hexes "
            f"(minimum required: {min_fold_pct:.1f}%). "
            f"Increase CENTER_RADIUS_M or decrease n_train_folds."
        )
    logger.info(f"Balance check PASSED — smallest fold: {min_pct_actual:.1f}%")
 
    result.to_parquet(str(out_path))
    logger.info(f"Spatial CV splits saved → {out_path.relative_to(cfg.project_root)}")
    return result


# ── Verification helper ────────────────────────────────────────────────────────

def verify_labels(labels_path: Path | None = None) -> None:
    """Print a summary of Walk Score labels for review."""
    slug = cfg.city.slug
    labels_path = labels_path or (cfg.paths.labels / f"{slug}_walk_scores.parquet")

    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        return

    df = pd.read_parquet(str(labels_path))
    net = gpd.read_parquet(
        str(cfg.paths.processed / f"{slug}_network_features.parquet")
    )[["h3_index", "data_sparse"]]
    df = df.merge(net, on="h3_index", how="left")
    dense = df[df["data_sparse"] == 0]

    import h3
    spots = {
        "Loop":        h3.latlng_to_cell(41.8827, -87.6298, 9),
        "Lincoln Park": h3.latlng_to_cell(41.9214, -87.6513, 9),
        "Englewood":   h3.latlng_to_cell(41.7794, -87.6444, 9),
        "Hegewisch":   h3.latlng_to_cell(41.6497, -87.5525, 9),
    }

    print(f"\nWalk Score label summary ({len(dense):,} dense hexes):")
    print(f"  mean:   {dense['walk_score'].mean():.1f}")
    print(f"  median: {dense['walk_score'].median():.1f}")
    print(f"  std:    {dense['walk_score'].std():.1f}")
    print(f"  NaN%:   {100*dense['walk_score'].isna().mean():.1f}%")
    print()
    print("Spot checks (expect: Loop > Lincoln Park > Englewood > Hegewisch):")
    for name, idx in spots.items():
        row = df[df["h3_index"] == idx]
        if len(row):
            ws = row["walk_score"].iloc[0]
            print(f"  {name:<15} {ws:.0f}" if not pd.isna(ws) else f"  {name:<15} NaN")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/labels.log", rotation="10 MB", level="DEBUG")

    parser = argparse.ArgumentParser(description="Week 6: Walk Score labels + spatial CV splits.")
    parser.add_argument(
        "--step",
        choices=["scores", "splits", "both", "verify"],
        default="both",
        help="Which step to run (default: both)"
    )
    args = parser.parse_args()

    try:
        if args.step in ("scores", "both"):
            logger.info("=== WALK SCORE LABELS ===")
            build_walk_score_labels()

        if args.step in ("splits", "both"):
            logger.info("=== SPATIAL CV SPLITS ===")
            build_spatial_cv_splits()

        if args.step == "verify":
            verify_labels()

        if args.step in ("scores", "both", "splits"):
            logger.success("Week 6 complete.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)