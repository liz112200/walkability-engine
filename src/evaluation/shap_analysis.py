from __future__ import annotations

import sys
import argparse
import h3
import json
import shap
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
import geopandas as gpd
import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger

from src.utils.config import cfg
from src.models.utils import load_modeling_data, preprocess_features, CENTER_FOLD

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_xgb_model_and_data():
    slug = cfg.city.slug

    model_path = cfg.paths.models / "xgb_best.json"
    feat_path = cfg.paths.models / "feature_columns.json"
    assert model_path.exists(), f"XGBoost model not found: {model_path}"

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    with open(feat_path) as f:
        feature_cols = json.load(f)

    logger.info(f"XGBoost model loaded: {len(feature_cols)} features")
    
    X, y, folds, _ = load_modeling_data()

    train_mask = folds != CENTER_FOLD
    X_train = X[feature_cols][train_mask]
    X_all = X[feature_cols]
    _, X_filled = preprocess_features(X_train, X_all)

    pred_path = cfg.paths.processed.parent / "predictions_xgb.parquet"
    predictions = gpd.read_parquet(str(pred_path))

    logger.info(
        f"Data loaded: {len(X_filled):,} hexes |  "
        f"{len(feature_cols)} features"
    )

    return model, X_filled, feature_cols, predictions, folds, y


def compute_shap_values(
        model: xgb.XGBRegressor,
        X_filled: pd.DataFrame,
        feature_cols: list[str]
) -> tuple[np.ndarray, float]:
    
    """
    Compute SHAP values using TreeSHAP
    - Run in O(TLD^2) T=trees, L=leaves, D=depth
                      765               3
    - Return exact values, not predictions

    Returns:
    - shap_values: np.ndarray shape (n_hexes, n_features)
                    shap_values[i, j] = contribution of feature j to hex i's pred, in Walk Score point

    - expected_value: float - the baseline 
    
    """

    logger.info("Computing SHAP values using TreeSHAP")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_filled)
    expected_val = float(explainer.expected_value)

    logger.info(
        f"SHAP values computed: shape {shap_values.shape}  |  "
        f"baseline (expected value): {expected_val:.2f}"
    )

    # Sanity check: sum of SHAP values + expected_val = model pred
    preds_direct = model.predict(X_filled)
    preds_from_shap = shap_values.sum(axis=1) + expected_val
    max_diff = np.abs(preds_direct - preds_from_shap).max()

    if max_diff > 0.01:
        logger.warning(f"SHAP additivity check failed: max diff = {max_diff:.4f}")
    else:
        logger.info(f"SHAP additivity check PASSED: max diff = {max_diff:.6f}")

    return shap_values, expected_val


def save_shap_parquet(
        shap_values: np.ndarray,
        X_filled: pd.DataFrame, 
        feature_cols: list[str],
        predictions: gpd.GeoDataFrame,
        expected_val: float
) -> pd.DataFrame:
    
    """
    One row per hex, one col per feature (prefixed with shap_)
    plus h3_index, predicted_score, walk_score, fold_name, geometry
    
    """

    slug = cfg.city.slug
    out_path = cfg.paths.processed.parent / "shap_values.parquet"

    shap_cols = {f"shap_{col}": shap_values[:, i]
                 for i, col in enumerate(feature_cols)}
    
    shap_df = pd.DataFrame(shap_cols)

    shap_df["h3_index"] = predictions["h3_index"].values
    shap_df["shap_baseline"] = expected_val
    shap_df["predicted_score"] = model.predict(X_filled)
    shap_df["walk_score"] = predictions["walk_score"].values
    shap_df["fold_name"] = predictions["fold_name"].values

    shap_df.to_parquet(str(out_path))
    logger.info(f"SHAP values saved to {out_path.name}   shape: {shap_df.shape}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top5 = sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])[:5]
    logger.info("Top 5 features by mean |SHAP|:")
    for feat, val in top5:
        logger.info(f"  {feat:<40} {val:.3f}")

    return shap_df


def plot_beeswarm(
        shap_values: np.ndarray,
        X_filled: pd.DataFrame,
        feature_cols: list[str],
        n_features: int = 20
) -> None:
    """
    Beeswarm plot showing global feature importance.

    Each dot is one hex cell. X position = SHAP value (impact on prediction).
    Colour = feature value (red = high, blue = low).

    This plot answers: "which features matter most for walkability
    predictions across Chicago, and in what direction?"

    """

    logger.info(f"Generating beeswarm plot (top {n_features} features)...")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-n_features:]

    top_shap = shap_values[:, top_idx]
    top_feat = [feature_cols[i] for i in top_idx]
    top_X    = X_filled.iloc[:, top_idx]

    # Clean up feature names for display
    def clean_name(name: str) -> str:
        return (name
                .replace("lw_prop_", "% ")
                .replace("lw_avg_", "avg ")
                .replace("poi_nearest_", "nearest ")
                .replace("poi_", "")
                .replace("transit_", "transit: ")
                .replace("census_", "census: ")
                .replace("safety_", "safety: ")
                .replace("terrain_", "terrain: ")
                .replace("_kde", " density")
                .replace("_m", " (m)")
                .replace("_per_km2", "/km²")
                .replace("_", " "))

    display_names = [clean_name(f) for f in top_feat]

    fig, ax = plt.subplots(figsize=(12, 9))

    order = np.argsort(mean_abs[top_idx])
    sorted_shap  = top_shap[:, order]
    sorted_names = [display_names[i] for i in order]
    sorted_X     = top_X.iloc[:, order]

    cmap = plt.cm.RdBu_r   # red = high feature value, blue = low
    for feat_idx in range(len(sorted_names)):
        y_pos    = feat_idx
        sv       = sorted_shap[:, feat_idx]             # 5023 SHAP values for this feature
        fv       = sorted_X.iloc[:, feat_idx].values    # 5023 feature values

        # Normalise feature values to [0, 1] for colour mapping
        fv_norm  = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)

        # Add vertical jitter to spread overlapping dots
        jitter   = np.random.uniform(-0.3, 0.3, size=len(sv))
        y_jitter = y_pos + jitter * 0.4

        colours  = cmap(fv_norm)
        ax.scatter(sv, y_jitter, c=colours, alpha=0.4, s=8, linewidths=0)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP value (impact on Walk Score prediction)", fontsize=11)
    ax.set_title(
        "Feature Impact on Walkability Predictions\n"
        "Each dot = one Chicago hex cell  |  "
        "Red = high feature value, Blue = low",
        fontsize=12, fontweight="bold"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Feature value (normalised)", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "shap_beeswarm.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Beeswarm saved → {out.name}")


# ── Waterfall plots ────────────────────────────────────────────────────────────

def plot_waterfall(
    shap_values:  np.ndarray,
    X_filled:     pd.DataFrame,
    feature_cols: list[str],
    expected_val: float,
    predictions:  gpd.GeoDataFrame,
    hex_id:       str,
    label:        str,
    n_features:   int = 12,
) -> None:
    """
    Waterfall plot for a single hex cell.

    Shows the baseline prediction, then each feature's contribution
    as a bar extending left (negative) or right (positive) until the
    final predicted score is reached.

    This plot answers: "why does THIS specific neighbourhood score THIS way?"
    """
    # Find row index for this hex
    hex_rows = predictions[predictions["h3_index"] == hex_id]
    if len(hex_rows) == 0:
        logger.warning(f"Hex {hex_id} ({label}) not found in predictions")
        return

    row_idx  = hex_rows.index[0]
    sv       = shap_values[row_idx]   # SHAP values for this hex
    fv       = X_filled.iloc[row_idx]  # Feature values for this hex
    pred     = float(predictions.loc[row_idx, "predicted_score"])
    actual   = float(predictions.loc[row_idx, "walk_score"])

    # Select top n features by absolute SHAP value for this hex
    abs_sv   = np.abs(sv)
    top_idx  = np.argsort(abs_sv)[-n_features:]
    rest_sum = sv[np.argsort(abs_sv)[:-n_features]].sum()

    # Build waterfall data
    items = []
    for i in top_idx:
        items.append((feature_cols[i], sv[i], float(fv.iloc[i])))
    items.sort(key=lambda x: x[1])   # sort by SHAP value

    if abs(rest_sum) > 0.01:
        items.insert(0, (f"other {len(feature_cols)-n_features} features", rest_sum, 0))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw waterfall bars
    running = expected_val
    for feat_name, sv_val, _ in items:
        colour = "#d73027" if sv_val < 0 else "#1a9850"  # red=negative, green=positive
        ax.barh(feat_name, sv_val, left=running, color=colour, alpha=0.8, height=0.6)
        # Value label
        x_label = running + sv_val/2
        ax.text(x_label, feat_name, f"{sv_val:+.1f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        running += sv_val

    # Baseline and final prediction markers
    ax.axvline(x=expected_val, color="navy", linestyle="--", linewidth=1.5,
               label=f"Baseline: {expected_val:.1f}")
    ax.axvline(x=pred, color="black", linestyle="-", linewidth=2,
               label=f"Predicted: {pred:.1f}")
    ax.axvline(x=actual, color="orange", linestyle=":", linewidth=2,
               label=f"Actual WalkScore: {actual:.0f}")

    ax.set_xlabel("Walk Score", fontsize=11)
    ax.set_title(
        f"SHAP Waterfall — {label}\n"
        f"Predicted: {pred:.1f}  |  Actual Walk Score: {actual:.0f}  |  "
        f"Residual: {actual-pred:+.1f}",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe_label = label.lower().replace(" ", "_").replace("/", "_")
    out = FIGURES_DIR / f"shap_waterfall_{safe_label}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Waterfall saved → {out.name}  (pred={pred:.1f}, actual={actual:.0f})")


# ── Feature summary statistics ─────────────────────────────────────────────────

def compute_shap_summary(
    shap_values:  np.ndarray,
    feature_cols: list[str],
    folds:        pd.Series,
) -> pd.DataFrame:
    """
    Compute per-feature SHAP summary statistics for the full dataset
    and for the test fold separately.

    Returns a DataFrame used in the equity audit (Week 12).
    """
    summary = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap":     shap_values.mean(axis=0),
        "std_shap":      shap_values.std(axis=0),
        "max_shap":      shap_values.max(axis=0),
        "min_shap":      shap_values.min(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    summary["rank"] = summary.index + 1

    out_path = Path("outputs") / "shap_feature_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(out_path), index=False)
    logger.info(f"SHAP feature summary saved → {out_path.name}")

    return summary




def run_shap_analysis():
    slug = cfg.city.slug
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model 
    model, X_filled, feature_cols, predictions, folds, y = \
        load_xgb_model_and_data()
    
    # 2. Compute shap 
    shap_values, expected_val = compute_shap_values(model, X_filled, feature_cols)

    # 3. Save SHAP parquet
    shap_df = save_shap_parquet(
        shap_values, X_filled, feature_cols, predictions, expected_val
    )
    
    # 4. Global beeswarm
    plot_beeswarm(shap_values, X_filled, feature_cols, n_features=20)

    # 5. Waterfall plots for key neighbourhoods
    spots = {
        "Loop (downtown)":         h3.latlng_to_cell(41.8827, -87.6298, 9),
        "Lincoln Park (wealthy)":  h3.latlng_to_cell(41.9214, -87.6513, 9),
        "Englewood (low-income)":  h3.latlng_to_cell(41.7794, -87.6444, 9),
        "Hegewisch (industrial)":  h3.latlng_to_cell(41.6497, -87.5525, 9),
    }
    for label, hex_id in spots.items():
        plot_waterfall(
            shap_values, X_filled, feature_cols, expected_val,
            predictions, hex_id, label,
        )

    # 6. Feature summary 
    summary = compute_shap_summary(shap_values, feature_cols, folds)

    # 7. Print headline findings 
    logger.info("=" * 60)
    logger.info("WEEK 10 SHAP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Baseline (mean prediction) : {expected_val:.2f}")
    logger.info(f"  Features analysed          : {len(feature_cols)}")
    logger.info(f"  Hexes analysed             : {len(X_filled):,}")
    logger.info("")
    logger.info("  Top 10 features by mean |SHAP value|:")
    for _, row in summary.head(10).iterrows():
        direction = "↑" if row["mean_shap"] > 0 else "↓"
        logger.info(
            f"  {row['rank']:>2}. {row['feature']:<40} "
            f"{row['mean_abs_shap']:>6.3f}  {direction}"
        )
    logger.info("=" * 60)

    return {
        "expected_value":   expected_val,
        "n_features":       len(feature_cols),
        "n_hexes":          len(X_filled),
        "top_feature":      summary.iloc[0]["feature"],
        "top_feature_shap": summary.iloc[0]["mean_abs_shap"],
    }


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/shap_analysis.log", rotation="10 MB", level="DEBUG")

    try:
        results = run_shap_analysis()
        logger.success(
            f"Week 10 complete — "
            f"top feature: {results['top_feature']}  "
            f"mean |SHAP|: {results['top_feature_shap']:.3f}"
        )

        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)



        