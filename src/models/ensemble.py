"""
Three ensemble strategies are evaluated
----------------------------------------
1. Simple average        : (xgb + lgbm) / 2
   No weights, equal trust in both models.

2. Weighted average      : w*xgb + (1-w)*lgbm
   Weight is optimised on training folds using Optuna (1D search).
   If w > 0.5, XGBoost contributes more. If w < 0.5, LightGBM contributes more.

3. Ridge meta-learner    : Ridge(xgb_pred, lgbm_pred) → walk_score
   Treats XGBoost and LightGBM predictions as features and fits a linear
   combination on top. More flexible than a fixed weight but risks overfitting
   with only 2 features.

The best strategy is selected by lowest validation RMSE on training folds.
The center fold is used for final evaluation of the chosen strategy.

Outputs
-------
    data/processed/predictions_ensemble.parquet
        Per-hex ensemble predictions with all three strategies stored
        for comparison in SHAP analysis and dashboard.

    outputs/models/ensemble_weights.json
        The chosen strategy and its parameters (weight for strategy 2,
        Ridge coefficients for strategy 3).

Usage
-----
    python -m src.models.ensemble
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.utils.config import cfg
from src.models.utils import (
    load_modeling_data,
    preprocess_features,
    regression_metrics,
    CENTER_FOLD,
)


# ── Load predictions from both models ─────────────────────────────────────────

def load_model_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    xgb_preds  : DataFrame with h3_index, predicted_score, walk_score, fold_name
    lgbm_preds : DataFrame with h3_index, predicted_score
    """
    slug = cfg.city.slug

    xgb_path  = cfg.paths.processed.parent / "predictions_xgb.parquet"
    lgbm_path = cfg.paths.processed.parent / "predictions_lgbm.parquet"

    assert xgb_path.exists(), (
        f"XGBoost predictions not found: {xgb_path}\n"
        "Run: python -m src.models.tabular"
    )
    assert lgbm_path.exists(), (
        f"LightGBM predictions not found: {lgbm_path}\n"
        "Run: python -m src.models.lgbm"
    )

    xgb_preds  = gpd.read_parquet(str(xgb_path))[
        ["h3_index", "predicted_score", "walk_score", "fold_name", "split"]
    ].rename(columns={"predicted_score": "xgb_score"})

    lgbm_preds = gpd.read_parquet(str(lgbm_path))[
        ["h3_index", "predicted_score"]
    ].rename(columns={"predicted_score": "lgbm_score"})

    return xgb_preds, lgbm_preds


# ── Ensemble strategies ────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def find_best_weight(
    xgb_train:  np.ndarray,
    lgbm_train: np.ndarray,
    y_train:    np.ndarray,
) -> float:
    """
    Find the optimal weight w for: w*xgb + (1-w)*lgbm
    using a simple grid search over w in [0, 1].
    """
    best_w    = 0.5
    best_rmse = float("inf")

    for w in np.linspace(0, 1, 101):
        blended = w * xgb_train + (1 - w) * lgbm_train
        rmse_val = _rmse(y_train, blended)
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_w    = w

    return float(best_w)


def fit_meta_learner(
    xgb_train:  np.ndarray,
    lgbm_train: np.ndarray,
    y_train:    np.ndarray,
) -> Ridge:
    """
    Fit a Ridge meta-learner on top of XGBoost and LightGBM predictions.
    Uses XGBoost and LightGBM predictions as the two input features.
    """
    X_meta = np.column_stack([xgb_train, lgbm_train])
    meta   = Ridge(alpha=1.0)
    meta.fit(X_meta, y_train)
    return meta


# ── Main ensemble pipeline ─────────────────────────────────────────────────────

def run_ensemble_pipeline() -> dict:
    """
    Combine XGBoost and LightGBM predictions into an ensemble.

    Steps:
    1. Load both prediction files
    2. Evaluate all three strategies on training folds
    3. Select the best strategy
    4. Evaluate the best strategy on the center fold
    5. Save ensemble predictions and weights
    """
    slug = cfg.city.slug
    cfg.paths.models.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("walkability-models")

    with mlflow.start_run(run_name=f"ensemble_{slug}"):

        t0 = time.perf_counter()

        # ── 1. Load predictions ────────────────────────────────────────────────
        logger.info("Loading XGBoost and LightGBM predictions…")
        xgb_preds, lgbm_preds = load_model_predictions()

        combined = xgb_preds.merge(lgbm_preds, on="h3_index", how="inner")
        logger.info(f"Combined predictions: {len(combined):,} hexes")

        # Log individual model performance on center fold
        center = combined[combined["split"] == "test"]
        logger.info("Individual model performance on center fold:")
        xgb_test_rmse  = _rmse(center["walk_score"].values, center["xgb_score"].values)
        lgbm_test_rmse = _rmse(center["walk_score"].values, center["lgbm_score"].values)
        logger.info(f"  XGBoost  RMSE: {xgb_test_rmse:.3f}")
        logger.info(f"  LightGBM RMSE: {lgbm_test_rmse:.3f}")

        mlflow.log_metrics({
            "xgb_test_rmse":  xgb_test_rmse,
            "lgbm_test_rmse": lgbm_test_rmse,
        })

        # ── 2. Load CV fold assignments for strategy selection ────────────────
        splits = pd.read_parquet(
            str(cfg.paths.splits / f"{slug}_spatial_cv.parquet")
        )[["h3_index", "fold"]]
        combined = combined.merge(splits, on="h3_index", how="left")

        train_folds = sorted(
            f for f in combined["fold"].unique() if f != CENTER_FOLD
        )
        train_mask  = combined["fold"].isin(train_folds)
        test_mask   = combined["fold"] == CENTER_FOLD

        xgb_train   = combined.loc[train_mask, "xgb_score"].values
        lgbm_train  = combined.loc[train_mask, "lgbm_score"].values
        y_train     = combined.loc[train_mask, "walk_score"].values

        xgb_test    = combined.loc[test_mask, "xgb_score"].values
        lgbm_test   = combined.loc[test_mask, "lgbm_score"].values
        y_test      = combined.loc[test_mask, "walk_score"].values

        # ── 3. Evaluate all three strategies on training data ─────────────────
        logger.info("=" * 55)
        logger.info("ENSEMBLE STRATEGY COMPARISON (train folds)")
        logger.info("=" * 55)

        # Strategy 1: Simple average
        simple_train_preds = (xgb_train + lgbm_train) / 2
        simple_rmse_train  = _rmse(y_train, simple_train_preds)
        logger.info(f"  Strategy 1 — Simple average      RMSE={simple_rmse_train:.3f}")

        # Strategy 2: Weighted average
        best_w = find_best_weight(xgb_train, lgbm_train, y_train)
        weighted_train_preds = best_w * xgb_train + (1 - best_w) * lgbm_train
        weighted_rmse_train  = _rmse(y_train, weighted_train_preds)
        logger.info(
            f"  Strategy 2 — Weighted average    RMSE={weighted_rmse_train:.3f}  "
            f"(w_xgb={best_w:.2f}, w_lgbm={1-best_w:.2f})"
        )

        # Strategy 3: Ridge meta-learner
        meta_model = fit_meta_learner(xgb_train, lgbm_train, y_train)
        meta_train_preds = meta_model.predict(
            np.column_stack([xgb_train, lgbm_train])
        )
        meta_rmse_train = _rmse(y_train, meta_train_preds)
        logger.info(
            f"  Strategy 3 — Ridge meta-learner  RMSE={meta_rmse_train:.3f}  "
            f"(coef: xgb={meta_model.coef_[0]:.3f}, lgbm={meta_model.coef_[1]:.3f})"
        )

        # ── 4. Select best strategy ────────────────────────────────────────────
        strategy_scores = {
            "simple":   simple_rmse_train,
            "weighted": weighted_rmse_train,
            "meta":     meta_rmse_train,
        }
        best_strategy = min(strategy_scores, key=strategy_scores.get)
        logger.info(f"  Best strategy on train: {best_strategy}")

        # ── 5. Evaluate best strategy on center fold ───────────────────────────
        logger.info("=" * 55)
        logger.info("FINAL ENSEMBLE — CENTER FOLD EVALUATION")
        logger.info("=" * 55)

        if best_strategy == "simple":
            test_preds = (xgb_test + lgbm_test) / 2
            weights    = {"strategy": "simple", "w_xgb": 0.5, "w_lgbm": 0.5}

        elif best_strategy == "weighted":
            test_preds = best_w * xgb_test + (1 - best_w) * lgbm_test
            weights    = {
                "strategy": "weighted",
                "w_xgb":    round(best_w, 4),
                "w_lgbm":   round(1 - best_w, 4),
            }

        else:  # meta
            test_preds = meta_model.predict(
                np.column_stack([xgb_test, lgbm_test])
            )
            weights = {
                "strategy":   "meta",
                "coef_xgb":   round(float(meta_model.coef_[0]), 4),
                "coef_lgbm":  round(float(meta_model.coef_[1]), 4),
                "intercept":  round(float(meta_model.intercept_), 4),
            }

        test_preds_clipped = np.clip(test_preds, 0, 100)
        ensemble_metrics = regression_metrics(
            y_test, test_preds_clipped, "Ensemble (center fold)"
        )

        # Improvement over both individual models
        xgb_improvement  = xgb_test_rmse  - ensemble_metrics["rmse"]
        lgbm_improvement = lgbm_test_rmse - ensemble_metrics["rmse"]
        logger.info(
            f"  Improvement over XGBoost:  {xgb_improvement:+.3f} RMSE points"
        )
        logger.info(
            f"  Improvement over LightGBM: {lgbm_improvement:+.3f} RMSE points"
        )

        mlflow.log_metrics({
            "ensemble_test_rmse":           ensemble_metrics["rmse"],
            "ensemble_test_r2":             ensemble_metrics["r2"],
            "ensemble_test_mae":            ensemble_metrics["mae"],
            "ensemble_improvement_vs_xgb":  xgb_improvement,
            "ensemble_improvement_vs_lgbm": lgbm_improvement,
        })
        mlflow.log_params(weights)

        # ── 6. Build full prediction set (all hexes, all strategies) ──────────
        # Store all three strategies in the output file for comparison
        all_simple   = np.clip((combined["xgb_score"] + combined["lgbm_score"]) / 2, 0, 100)
        all_weighted = np.clip(
            best_w * combined["xgb_score"] + (1 - best_w) * combined["lgbm_score"],
            0, 100
        )
        if best_strategy == "meta":
            all_meta = np.clip(
                meta_model.predict(
                    np.column_stack([combined["xgb_score"].values,
                                     combined["lgbm_score"].values])
                ),
                0, 100
            )
        else:
            all_meta = all_weighted  # fallback if meta wasn't best

        combined["ensemble_score"]   = (
            all_simple if best_strategy == "simple" else
            all_weighted if best_strategy == "weighted" else
            all_meta
        )
        combined["simple_score"]     = all_simple
        combined["weighted_score"]   = all_weighted
        combined["residual"]         = combined["walk_score"] - combined["ensemble_score"]
        combined["abs_residual"]     = combined["residual"].abs()
        combined["model_agreement"]  = (
            combined["xgb_score"] - combined["lgbm_score"]
        ).abs()

        # ── 7. Attach geometry and save ───────────────────────────────────────
        master = gpd.read_parquet(
            str(cfg.paths.processed.parent / "master_features.parquet")
        )[["h3_index", "geometry", "centroid_lat", "centroid_lng"]]

        result = master.merge(
            combined[[
                "h3_index", "ensemble_score", "simple_score", "weighted_score",
                "xgb_score", "lgbm_score", "walk_score",
                "residual", "abs_residual", "model_agreement",
                "fold_name", "split",
            ]],
            on="h3_index", how="right"
        )
        result = gpd.GeoDataFrame(result, crs=cfg.city.crs)

        out_path = cfg.paths.processed.parent / "predictions_ensemble.parquet"
        result.to_parquet(str(out_path))
        logger.info(f"Ensemble predictions saved → {out_path.name}  ({len(result):,} hexes)")

        # Save weights
        weights_path = cfg.paths.models / "ensemble_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        logger.info(f"Ensemble weights saved → {weights_path.name}")
        mlflow.log_artifact(str(weights_path))

        # ── 8. Final summary ───────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        logger.info("=" * 55)
        logger.info("WEEK 8 ENSEMBLE SUMMARY")
        logger.info("=" * 55)
        logger.info(f"  XGBoost   RMSE : {xgb_test_rmse:.3f}")
        logger.info(f"  LightGBM  RMSE : {lgbm_test_rmse:.3f}")
        logger.info(f"  Ensemble  RMSE : {ensemble_metrics['rmse']:.3f}  ({best_strategy})")
        logger.info(f"  Ensemble  R²   : {ensemble_metrics['r2']:.3f}")
        logger.info(f"  Strategy       : {best_strategy}  {weights}")
        logger.info(f"  Runtime        : {elapsed:.1f}s")
        logger.info("=" * 55)

        return {
            "xgb_test_rmse":    xgb_test_rmse,
            "lgbm_test_rmse":   lgbm_test_rmse,
            "ensemble_rmse":    ensemble_metrics["rmse"],
            "ensemble_r2":      ensemble_metrics["r2"],
            "best_strategy":    best_strategy,
            "weights":          weights,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/ensemble.log", rotation="10 MB", level="DEBUG")

    try:
        results = run_ensemble_pipeline()
        logger.success(
            f"Week 8 ensemble complete — "
            f"RMSE={results['ensemble_rmse']:.3f}  "
            f"R²={results['ensemble_r2']:.3f}  "
            f"strategy={results['best_strategy']}"
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)