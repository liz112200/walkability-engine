from __future__ import annotations

import argparse
import sys
import mlflow
import time
import optuna
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb

from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.config import cfg
# Remove these three function definitions from tabular.py entirely,
# and replace the imports section with:

from src.models.utils import (
    load_modeling_data,
    preprocess_features,
    regression_metrics,
    CENTER_FOLD,
    RANDOM_SEED
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OPTUNA_TRIALS = 100

# ── Linear baseline ────────────────────────────────────────────────────────────
def train_linear_baseline(
        X: pd.DataFrame,
        y: pd.Series,
        folds: pd.Series
) -> dict:
    """
    Ridge regression baseline using spatial CV.
    """
    logger.info("Training linear baseline (Ridge Regression)")
    scaler = StandardScaler()

    fold_metrics = []
    for test_fold in sorted(folds.unique()):
        train_mask = folds != test_fold
        test_mask = folds == test_fold

        X_tr, X_te = preprocess_features(X[train_mask], X[test_mask])
        y_tr = y[train_mask]
        y_te = y[test_mask]

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_te_scaled)

        m = regression_metrics(y_te.values, preds, f"Ridge fold={test_fold}")
        fold_metrics.append(m)

    mean_rmse = np.mean([m["rmse"] for m in fold_metrics])
    mean_r2 = np.mean([m["r2"] for m in fold_metrics])
    logger.info(f"  Baseline CV mean — RMSE={mean_rmse:.3f}  R²={mean_r2:.3f}")
    return {"cv_rmse": mean_rmse, "cv_r2": mean_r2}

# ── XGBoost with Optuna ────────────────────────────────────────────────────────

def _xgb_cv_score(
        params: dict,
        X: pd.DataFrame, 
        y: pd.Series,
        folds: pd.Series, 
        train_folds: list[int]
) -> float:
    
    fold_rmses = []
    for val_fold in train_folds:
        tr_mask = folds.isin([f for f in train_folds if f != val_fold])
        val_mask = folds == val_fold

        X_tr, X_val = preprocess_features(X[tr_mask], X[val_mask])
        y_tr = y[tr_mask]
        y_val = y[val_mask]

        model = xgb.XGBRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
            eval_metric="rmse"
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        preds = model.predict(X_val)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    return float(np.mean(fold_rmses))


def tune_xgboost(X: pd.DataFrame, y: pd.Series, folds: pd.Series, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    train_folds = [f for f in folds.unique() if f != CENTER_FOLD]
    logger.info(
        f"Tuning XGBoost — {n_trials} Optuna trials  |  "
        f"spatial CV on folds {train_folds}"
    )

    # trial object is Optuna's interface for proposing values
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 800),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
        }
        return _xgb_cv_score(params, X, y, folds, train_folds)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    logger.info(
        f"Best CV RMSE: {study.best_value:.3f}  |  "
        f"best params: {best}"
    )
    return best

# ── Final model training ───────────────────────────────────────────────────────
def train_final_model(
        X: pd.DataFrame,
        y: pd.Series,
        folds: pd.Series,
        best_params: dict,
        feature_cols: list[str]
) -> tuple[xgb.XGBRegressor, dict]:
    
    train_mask = folds != CENTER_FOLD
    test_mask = folds == CENTER_FOLD

    X_train, X_test = preprocess_features(X[train_mask], X[test_mask])
    y_train = y[train_mask]
    y_test = y[test_mask]

    logger.info(
        f"Training final model - "
        f"train: {train_mask.sum():,}   test (center fold): {test_mask.sum():,}"
    )

    model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    logger.info(f"Training complete in {time.perf_counter() - t0:.1f}s")

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    logger.info("Final model performance:")
    train_metrics = regression_metrics(y_train.values, train_preds, "Train")
    test_metrics = regression_metrics(y_test.values, test_preds, "Test")

    # Overfitting check
    overfit_gap = train_metrics["rmse"] - test_metrics["rmse"]
    if test_metrics["rmse"] > train_metrics["rmse"] * 1.5:
        logger.warning(
            f"Large train/test gap (train RMSE={train_metrics['rmse']:.2f}, "
            f"test RMSE={test_metrics['rmse']:.2f}) — possible overfitting"
        )

    return model, {
        "train_rmse": train_metrics["rmse"],
        "train_r2":   train_metrics["r2"],
        "test_rmse":  test_metrics["rmse"],
        "test_r2":    test_metrics["r2"],
        "test_mae":   test_metrics["mae"],
        "n_train":    int(train_mask.sum()),
        "n_test":     int(test_mask.sum()),
        "n_features": len(feature_cols),
    }

# ── Save outputs ───────────────────────────────────────────────────────────────
def save_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    model: xgb.XGBRegressor,
    feature_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Generate predictions for all hexes and save with residuals + fold info.
    Useful for mapping and SHAP analysis in Weeks 11–12.
    """
    slug = cfg.city.slug
    out_path = cfg.paths.processed.parent / "predictions_xgb.parquet"
 
    # Predict on full dataset (train + test)
    X_filled = X.fillna(X.median())
    all_preds = model.predict(X_filled)
 
    # Load hex geometries for spatial output
    master = gpd.read_parquet(
        str(cfg.paths.processed.parent / "master_features.parquet")
    )[["h3_index", "geometry", "centroid_lat", "centroid_lng"]]
 
    labels = pd.read_parquet(
        str(cfg.paths.labels / f"{slug}_walk_scores.parquet")
    )[["h3_index", "walk_score"]]
    splits = pd.read_parquet(
        str(cfg.paths.splits / f"{slug}_spatial_cv.parquet")
    )[["h3_index", "fold", "fold_name", "split"]]
 
    net = gpd.read_parquet(
        str(cfg.paths.processed / f"{slug}_network_features.parquet")
    )[["h3_index", "data_sparse"]]
    base = net[net["data_sparse"] == 0].merge(labels, on="h3_index", how="inner")
    base = base.merge(splits, on="h3_index", how="inner")
    base = base[base["walk_score"].notna()].reset_index(drop=True)
 
    base["predicted_score"] = all_preds.clip(0, 100)
    base["residual"]        = base["walk_score"] - base["predicted_score"]
    base["abs_residual"]    = base["residual"].abs()
 
    # Merge geometry
    result = master.merge(base[["h3_index", "predicted_score", "residual",
                                "abs_residual", "walk_score", "fold_name", "split"]],
                          on="h3_index", how="right")
    result = gpd.GeoDataFrame(result, crs=cfg.city.crs)
 
    result.to_parquet(str(out_path))
    logger.info(f"Predictions saved → {out_path.name}  ({len(result):,} hexes)")
 
    # Summary
    logger.info(
        f"Prediction range: [{result['predicted_score'].min():.0f}, "
        f"{result['predicted_score'].max():.0f}]  |  "
        f"mean residual: {result['residual'].mean():.2f}  |  "
        f"mean |residual|: {result['abs_residual'].mean():.2f}"
    )
    return result
     



# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_tabular_pipeline(n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """
    Pipeline: baseline -> tune -> train -> evaluate -> save
    """
    slug = cfg.city.slug
    cfg.paths.models.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("walkability-models")

    with mlflow.start_run(run_name=f"week7_{slug}"):
        # ── 1. Load data ───────────────────────────────────────────────────────
        logger.info("Loading modeling data")
        X, y, folds, feature_cols = load_modeling_data()
        mlflow.log_params({
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "city": slug
        })

        # ── 2. Linear baseline ─────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("LINEAR BASELINE")
        logger.info("=" * 55)
        baseline_metrics = train_linear_baseline(X, y, folds)
        mlflow.log_metrics({
            "baseline_cv_rmse": baseline_metrics["cv_rmse"],
            "baseline_cv_r2":   baseline_metrics["cv_r2"],
        })

        # ── 3. XGBoost tuning ─────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("XGBOOST TUNING")
        logger.info("=" * 55)
        t_tune = time.perf_counter()
        best_params = tune_xgboost(X, y, folds, n_trials=n_trials)
        logger.info(f"Tunning complete in {(time.perf_counter() - t_tune)/60:.1f} min")
        mlflow.log_params(best_params)

        # ── 4. Final model ────────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("FINAL MODEL - CENTER FOLD EVALUATION")
        logger.info("=" * 55)
        model, final_metrics = train_final_model(X, y, folds, best_params, feature_cols)
        mlflow.log_metrics(final_metrics)

        # ── 5. Improvement over baseline ──────────────────────────────────────
        improvement = baseline_metrics["cv_rmse"] - final_metrics["test_rmse"]
        logger.info(
            f"XGBoost vs baseline: "
            f"RMSE {final_metrics['test_rmse']:.3f} vs {baseline_metrics['cv_rmse']:.3f} "
            f"(improvement: {improvement:+.3f})"
        )
        mlflow.log_metric("rmse_improvement_over_baseline", improvement)

        # ── 6. Save model + feature list ──────────────────────────────────────
        model_path = cfg.paths.models / "xgb_best.json"
        model.save_model(str(model_path))
        logger.info(f"Model saved -> {model_path.name}")

        feat_path = cfg.paths.models / "feature_columns.json"
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"Feature list saved → {feat_path.name}")
 
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feat_path))

        # ── 7. Save predictions ───────────────────────────────────────────────
        logger.info("Saving predictions…")
        save_predictions(X, y, folds, model, feature_cols)
 
        # ── 8. Final summary ──────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("WEEK 7 SUMMARY")
        logger.info("=" * 55)
        logger.info(f"  Baseline  RMSE : {baseline_metrics['cv_rmse']:.3f}")
        logger.info(f"  XGBoost   RMSE : {final_metrics['test_rmse']:.3f}  (center fold)")
        logger.info(f"  XGBoost   R²   : {final_metrics['test_r2']:.3f}  (center fold)")
        logger.info(f"  Improvement    : {improvement:+.3f} RMSE points over baseline")
        logger.info("=" * 55)
 
        mlflow.log_text(
            f"baseline_rmse={baseline_metrics['cv_rmse']:.3f}\n"
            f"xgb_test_rmse={final_metrics['test_rmse']:.3f}\n"
            f"xgb_test_r2={final_metrics['test_r2']:.3f}\n",
            "results_summary.txt"
        )
 
        return {**baseline_metrics, **final_metrics, "best_params": best_params}

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/tabular.log", rotation="10 MB", level="DEBUG")
    
    parser = argparse.ArgumentParser(description="Week 7: XGBoost walkability model.")
    parser.add_argument("--no-tune",  action="store_true",
                        help="Skip Optuna tuning, use default XGBoost params")
    parser.add_argument("--fast",     action="store_true",
                        help="Use 20 Optuna trials instead of 100 (for testing)")
    args = parser.parse_args()

    n_trials = 1 if args.no_tune else (20 if args.fast else N_OPTUNA_TRIALS)
    try:
        results = run_tabular_pipeline(n_trials=n_trials)
        logger.success(
            f"Week 7 complete — "
            f"test RMSE={results['test_rmse']:.3f}  "
            f"R²={results['test_r2']:.3f}"
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)
