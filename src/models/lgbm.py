from __future__ import annotations

import sys
import argparse
import mlflow
import optuna
import time
import json
import pandas as pd
import lightgbm as lgb
import numpy as np
import geopandas as gpd

from pathlib import Path
from loguru import logger
from sklearn.metrics import mean_squared_error


from src.utils.config import cfg
from src.models.utils import (
    load_modeling_data,
    preprocess_features,
    regression_metrics,
    CENTER_FOLD,
    RANDOM_SEED
)
 
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OPTUNA_TRIALS = 100

# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _lgbm_cv_score(
    params:      dict,
    X:           pd.DataFrame,
    y:           pd.Series,
    folds:       pd.Series,
    train_folds: list[int],
) -> float:
    """
    Train LightGBM with given params on each train fold and return mean RMSE.
    """
    fold_rmses = []
    for val_fold in train_folds:
        tr_mask  = folds.isin([f for f in train_folds if f != val_fold])
        val_mask = folds == val_fold
 
        X_tr, X_val = preprocess_features(X[tr_mask], X[val_mask])
        y_tr  = y[tr_mask].values
        y_val = y[val_mask].values
 
        model = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        preds = model.predict(X_val)
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
 
    return float(np.mean(fold_rmses))

def tune_lgbm(X: pd.DataFrame, y: pd.Series, folds: pd.Series, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    train_folds = [f for f in sorted(folds.unique()) if f != CENTER_FOLD]
    logger.info(
        f"Tuning LightGBM — {n_trials} Optuna trials  |  "
        f"spatial CV on folds {train_folds}"
    )
 
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tree structure — leaf-wise growth
            "num_leaves":        trial.suggest_int("num_leaves", 15, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
 
            # Boosting
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
 
            # Randomisation (reduces overfitting)
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
 
            # Regularisation
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
 
            # Objective
            "objective":         "regression",
            "metric":            "rmse",
        }
        return _lgbm_cv_score(params, X, y, folds, train_folds)
 
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
 
    best = study.best_params
    best["objective"] = "regression"
    best["metric"]    = "rmse"
 
    logger.info(
        f"Best CV RMSE: {study.best_value:.3f}  |  "
        f"best params: {best}"
    )
    return best


# ── Final model ────────────────────────────────────────────────────────────────
 
def train_final_lgbm(
    X:            pd.DataFrame,
    y:            pd.Series,
    folds:        pd.Series,
    best_params:  dict,
    feature_cols: list[str],
) -> tuple[lgb.LGBMRegressor, dict]:

    train_mask = folds != CENTER_FOLD
    test_mask  = folds == CENTER_FOLD
 
    X_train, X_test = preprocess_features(X[train_mask], X[test_mask])
    y_train = y[train_mask].values
    y_test  = y[test_mask].values
 
    logger.info(
        f"Training final LightGBM — "
        f"train: {train_mask.sum():,}  test (center fold): {test_mask.sum():,}"
    )
 
    model = lgb.LGBMRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
 
    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    logger.info(f"Training complete in {time.perf_counter()-t0:.1f}s")
 
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)
 
    logger.info("Final LightGBM performance:")
    train_metrics = regression_metrics(y_train, train_preds, "Train")
    test_metrics  = regression_metrics(y_test,  test_preds,  "Test (center fold)")
 
    if test_metrics["rmse"] > train_metrics["rmse"] * 1.5:
        logger.warning(
            f"Large train/test gap — "
            f"train RMSE={train_metrics['rmse']:.2f}, "
            f"test RMSE={test_metrics['rmse']:.2f}"
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
 
 
# ── Save predictions ───────────────────────────────────────────────────────────
 
def save_lgbm_predictions(
    X:            pd.DataFrame,
    y:            pd.Series,
    folds:        pd.Series,
    model:        lgb.LGBMRegressor,
    feature_cols: list[str],
) -> gpd.GeoDataFrame:
    """Save per-hex LightGBM predictions with residuals and geometry."""
    slug     = cfg.city.slug
    out_path = cfg.paths.processed.parent / "predictions_lgbm.parquet"
 
    X_filled  = X.fillna(X.median())
    all_preds = model.predict(X_filled)
 
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
 
    base["predicted_score"] = np.clip(all_preds, 0, 100)
    base["residual"]        = base["walk_score"] - base["predicted_score"]
    base["abs_residual"]    = base["residual"].abs()
 
    result = master.merge(
        base[["h3_index", "predicted_score", "residual",
              "abs_residual", "walk_score", "fold_name", "split"]],
        on="h3_index", how="right"
    )
    result = gpd.GeoDataFrame(result, crs=cfg.city.crs)
    result.to_parquet(str(out_path))
 
    logger.info(f"LightGBM predictions saved → {out_path.name}  ({len(result):,} hexes)")
    logger.info(
        f"Prediction range: [{result['predicted_score'].min():.0f}, "
        f"{result['predicted_score'].max():.0f}]  |  "
        f"mean |residual|: {result['abs_residual'].mean():.2f}"
    )
    return result
 

 

# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_lgbm_pipeline(n_trials: int = N_OPTUNA_TRIALS) -> dict:
    slug = cfg.city.slug
    cfg.paths.models.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("walkability-models")

    with mlflow.start_run(run_name=f"lgbm_{slug}"):
        # ── 1. Load data ───────────────────────────────────────────────────────
        logger.info("Loading modeling data…")
        X, y, folds, feature_cols = load_modeling_data()
        mlflow.log_params({
            "model": "lightgbm",
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "city": slug
        })

        # ── 2. LGBM tuning ───────────────────────────────────────────────────────

        logger.info("=" * 55)
        logger.info("LIGHTGBM TUNING")
        logger.info("=" * 55)
        t_tune = time.perf_counter()
        best_params = tune_lgbm(X, y, folds, n_trials=n_trials)
        logger.info(f"Tunning complete in {(time.perf_counter() - t_tune)/60:.1f} min")
        mlflow.log_params(best_params)

        # ── 3. Final model ───────────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("FINAL MODEL")
        logger.info("=" * 55)
        model, final_metrics = train_final_lgbm(
            X, y, folds, best_params, feature_cols
        )
        mlflow.log_metrics(final_metrics)
    
        # Save model
        model_path = cfg.paths.models / "lgbm_best.txt"
        model.booster_.save_model(str(model_path))
        logger.info(f"Model saved → {model_path.name}")
        mlflow.log_artifact(str(model_path))
 
        # Save feature list
        feat_path = cfg.paths.models / "feature_columns.json"
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(str(feat_path))
 
        # Save predictions
        save_lgbm_predictions(X, y, folds, model, feature_cols)
 
        logger.info("=" * 55)
        logger.info("LGBM SUMMARY")
        logger.info("=" * 55)
        logger.info(f"  LightGBM  RMSE : {final_metrics['test_rmse']:.3f}  (center fold)")
        logger.info(f"  LightGBM  R²   : {final_metrics['test_r2']:.3f}  (center fold)")
        logger.info("=" * 55)
 
        return {**final_metrics, "best_params": best_params}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/lgbm.log", rotation="10 MB", level="DEBUG")

    parser = argparse.ArgumentParser(description="Week 8 Part 1: LightGBM model.")
    parser.add_argument("--fast", action="store_true",
                        help="20 Optuna trials instead of 100")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna, use default params")
    args = parser.parse_args()

    n_trials = 1 if args.no_tune else (20 if args.fast else N_OPTUNA_TRIALS)

    try:
        results = run_lgbm_pipeline(n_trials=n_trials)
        
        logger.success(
            f"LightGBM complete — "
            f"test RMSE={results['test_rmse']:.3f}  "
            f"R²={results['test_r2']:.3f}"
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)
