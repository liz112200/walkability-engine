from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.config import cfg

# ── Features to always exclude ────────────────────────────────────────────────
EXCLUDE_COLS = {
    "h3_index", "geometry",
    "centroid_x", "centroid_y", "centroid_lat", "centroid_lng",
    "data_sparse", "n_real_edges", "n_nodes_in_hex",
    "poi_transit_kde",
    "safety_crash_count",
}

CENTER_FOLD = 0
RANDOM_SEED = 42


def load_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Load master features, Walk Score labels, and spatial CV splits.
    Returns (X, y, folds, feature_cols) filtered to dense hexes with valid labels.
    """
    slug = cfg.city.slug

    master = gpd.read_parquet(
        str(cfg.paths.processed.parent / "master_features.parquet")
    )
    labels = pd.read_parquet(
        str(cfg.paths.labels / f"{slug}_walk_scores.parquet")
    )[["h3_index", "walk_score"]]
    splits = pd.read_parquet(
        str(cfg.paths.splits / f"{slug}_spatial_cv.parquet")
    )[["h3_index", "fold"]]

    df = master.merge(labels, on="h3_index", how="inner")
    df = df.merge(splits,    on="h3_index", how="inner")
    df = df[df["data_sparse"] == 0].copy()
    df = df[df["walk_score"].notna()].copy()
    df = df.reset_index(drop=True)

    logger.info(
        f"Modeling dataset: {len(df):,} hexes  |  "
        f"label range: [{df['walk_score'].min():.0f}, {df['walk_score'].max():.0f}]  |  "
        f"folds: {df['fold'].value_counts().sort_index().to_dict()}"
    )

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and c not in {"walk_score", "fold", "data_sparse"}
        and not c.startswith("geometry")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16]
    ]

    X     = df[feature_cols].copy()
    y     = df["walk_score"].copy()
    folds = df["fold"].copy()

    logger.info(f"Feature matrix: {X.shape}  |  features: {len(feature_cols)}")
    return X, y, folds, feature_cols


def preprocess_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute NaN with training-set medians. Fit only on training data."""
    medians = X_train.median().fillna(0)
    X_train_filled = X_train.fillna(medians)
    X_val_filled   = X_val.fillna(medians)

    still_nan = X_train_filled.columns[X_train_filled.isna().all()].tolist()
    if still_nan:
        logger.warning(f"Dropping {len(still_nan)} all-NaN columns: {still_nan}")
        X_train_filled = X_train_filled.drop(columns=still_nan)
        X_val_filled   = X_val_filled.drop(columns=still_nan)

    return X_train_filled, X_val_filled


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "",
) -> dict:
    """Compute and log RMSE, R², MAE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    prefix = f"{label} " if label else ""
    logger.info(
        f"  {prefix}RMSE={rmse:.3f}  R²={r2:.3f}  MAE={mae:.3f}  n={len(y_true)}"
    )
    return {"rmse": rmse, "r2": r2, "mae": mae}


def generate_oof_predictions(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    train_folds: list[int],
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) predictions for stacking.

    For each fold, trains model_factory() on the other folds and predicts
    on the held-out fold. Returns predictions for all training hexes in
    their original row order. The center fold (0) is never touched.
    """
    train_mask  = folds.isin(train_folds)
    X_tr_all    = X[train_mask].reset_index(drop=True)
    y_tr_all    = y[train_mask].reset_index(drop=True)
    folds_tr    = folds[train_mask].reset_index(drop=True)

    oof_preds = np.zeros(len(X_tr_all))

    for val_fold in train_folds:
        tr_mask  = folds_tr != val_fold
        val_mask = folds_tr == val_fold

        X_t, X_v = preprocess_features(X_tr_all[tr_mask], X_tr_all[val_mask])
        y_t = y_tr_all[tr_mask]

        model = model_factory()
        model.fit(X_t, y_t)
        oof_preds[val_mask.values] = model.predict(X_v)

    logger.info(
        f"OOF predictions — RMSE={np.sqrt(mean_squared_error(y_tr_all, oof_preds)):.3f}  "
        f"n={len(oof_preds):,}"
    )
    return oof_preds