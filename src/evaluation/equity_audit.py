from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from loguru import logger
from pathlib import Path
from scipy import stats

from src.utils.config import cfg
 
# ── Output directories ─────────────────────────────────────────────────────────
EQUITY_DIR  = Path("outputs/equity")
FIGURES_DIR = Path("outputs/figures")
EQUITY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
# Demographic variables to audit
DEMO_VARS = {
    "census_median_income":  "Median income ($)",
    "census_pct_poverty":    "% below poverty line",
    "census_pct_minority":   "% minority",
    "census_pct_black":      "% Black",
    "census_pct_hispanic":   "% Hispanic",
    "census_pop_density":    "Population density",
}
 

# ── Data loading ───────────────────────────────────────────────────────────────

def load_audit_data() -> pd.Dataframe:
    slug = cfg.city.slug

    shap_path = cfg.paths.processed.parent / "shap_values.parquet"
    assert shap_path.exists(), (
        "shap_values.parquet not found"
    )
    shap_df = pd.read_parquet(str(shap_path))

    master = gpd.read_parquet(
        str(cfg.paths.processed.parent / "master_features.parquet")
    )

    df = shap_df.merge(
        master[["h3_index", "geometry"] + list(DEMO_VARS.keys())],
        on="h3_index",
        how="left",
    )

    df = df[df["walk_score"].notna()].reset_index(drop=True)

    logger.info(
        f"Audit dataset: {len(df):,} hexes  |  "
        f"{len([c for c in df.columns if c.startswith('shap_')] )} SHAP cols  |  "
        f"{len(DEMO_VARS)} demographic vars"
    )
    return df

# ── Analysis 1: Moran's I ──────────────────────────────────────────────────────
def compute_morans_i(df: pd.DataFrame) -> dict:
    """
 
    Moran's I measures spatial autocorrelation:
        I = 1   → perfect clustering (similar values near each other)
        I = 0   → random spatial arrangement
        I = -1  → perfect dispersion (dissimilar values near each other)
 
    Uses H3 adjacency as the spatial weights matrix.
    Significance tested via z-score under normality assumption.
 
    Returns dict with I, expected_I, z_score, p_value, interpretation.
    """

    logger.info("Computing Moran's I spatial autocorrelation…")

    try:
        import h3
        from scipy.sparse import lil_matrix

        hex_ids = df["h3_index"].tolist()
        hex_to_idx = {hid: i for i, hid in enumerate(hex_ids)}
        valid_set = set(hex_ids)
        n = len(hex_ids)

        # Build spatial weights matrix W (row-standardised)
        # W[i,j] = 1/degree(i) if i and j are adjacent, else 0
        W = lil_matrix((n, n), dtype=np.float64)
        for hid in hex_ids:
            i = hex_to_idx[hid]
            neighbors = set(h3.grid_disk(hid, 1)) - {hid}
            valid_nb = [hex_to_idx[nb] for nb in neighbors if nb in valid_set]
            if valid_nb:
                w = 1.0 / len(valid_nb)
                for j in valid_nb:
                    W[i, j] = w
        W = W.tocsr()

        # Moran's I formula
        # z = (x - mean(x)) / std(x)  — mean-centred values
        # I = (n / S0) × (z^T W z) / (z^T z)
        # where S0 = sum of all weights
        x = df["predicted_score"].values
        z = x - x.mean()
        S0 = W.sum()
        Wz = W.dot(z)
        I = (n / S0) * (z @ Wz) / (z @ z)

        E_I   = -1.0 / (n - 1)
        S1    = 0.5 * ((W + W.T).power(2)).sum()
        S2    = ((np.array(W.sum(axis=1)) + np.array(W.sum(axis=0).T))**2).sum()
        n2    = n * n
        A     = n * (n2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2
        B     = (z**4).mean() / ((z**2).mean()**2)
        C     = (n2 - n) * S1 - 2*n*S2 + 6*S0**2
        D     = (B - 1) * C
        E     = (n - 1) * (n2 - 3*n + 3) * S1 - (n2 - n) * S2 + 3 * (n-1) * S0**2
        E_I2  = (A - D) / (E * (n-2) * (n-3) / n)
        Var_I = max(E_I2 - E_I**2, 1e-10)
        z_score = (I - E_I) / np.sqrt(Var_I)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
 
        if I > 0.3 and p_value < 0.05:
            interpretation = "Strong positive spatial clustering — low-walkability hexes cluster together"
        elif I > 0.1 and p_value < 0.05:
            interpretation = "Moderate positive spatial clustering"
        elif p_value >= 0.05:
            interpretation = "No significant spatial autocorrelation"
        else:
            interpretation = "Spatial dispersion"
 
        result = {
            "morans_i":       round(float(I), 4),
            "expected_i":     round(float(E_I), 4),
            "z_score":        round(float(z_score), 3),
            "p_value":        round(float(p_value), 6),
            "significant":    p_value < 0.05,
            "interpretation": interpretation,
        }
 
        logger.info(f"  Moran's I = {I:.4f}  (expected {E_I:.4f})")
        logger.info(f"  z-score   = {z_score:.3f}  p-value = {p_value:.6f}")
        logger.info(f"  {interpretation}")

        return result
    
    except ImportError as e:
        logger.warning(f"Could not compute Moran's I: {e}")
        return {"morans_i": None, "error": str(e)}


# ── Analysis 2: Demographic correlations ──────────────────────────────────────

def compute_demographic_correlations(df: pd.DataFrame, n_permutations: int = 100) -> pd.DataFrame:
    logger.info(
        f"Computing demographic correlations "
        f"({n_permutations} permutation tests)"
    )

    scores = df["predicted_score"].values
    rows = []

    for var, label in DEMO_VARS.items():
        if var not in df.columns:
            logger.warning(f"  {var} not in dataset — skipping")
            continue

        mask = df[var].notna().values
        s = scores[mask]
        v = df[var][mask].values

        if len(v) < 50:
            logger.warning(f"  {var}: too few non-null values ({len(v)})")
            continue

        pearson_r,  pearson_p  = stats.pearsonr(s, v)
        spearman_r, spearman_p = stats.spearmanr(s, v)

        perm_rs = np.array([
            stats.pearsonr(s, np.random.permutation(v))[0]
            for _ in range(n_permutations)
        ])

        perm_p = np.mean(np.abs(perm_rs) >= abs(pearson_r))

        rows.append({
            "variable":    var,
            "label":       label,
            "pearson_r":   round(float(pearson_r),  4),
            "pearson_p":   round(float(pearson_p),  6),
            "spearman_r":  round(float(spearman_r), 4),
            "spearman_p":  round(float(spearman_p), 6),
            "perm_p":      round(float(perm_p),     4),
            "significant": perm_p < 0.05,
            "n_hexes":     int(len(v)),
        })

        dir_sym = "↑" if pearson_r > 0 else "↓"
        logger.info(
            f"  {var:<30} r={pearson_r:+.3f}  "
            f"perm_p={perm_p:.4f}  "
            f"{'✓' if perm_p < 0.05 else '✗'}  {dir_sym}"
        )

    return pd.DataFrame(rows)

# ── Analysis 3: SHAP by demographic quartile ──────────────────────────────────
def compute_shap_by_quartile(
    df:           pd.DataFrame,
    grouping_var: str = "census_pct_minority",
    n_features:   int = 15,
) -> pd.DataFrame:

    """
    Split hexes into quartiles by minority percentage.
    For each quartile compute mean SHAP per feature.
 
    This reveals which specific features drive the walkability gap
    between high-minority and low-minority areas.
 
    Returns a DataFrame with columns:
        feature, Q1_mean, Q2_mean, Q3_mean, Q4_mean, gap (Q4 - Q1)
    """

    logger.info(
        f"Computing SHAP by {grouping_var} quartile…"
    )
 
    if grouping_var not in df.columns:
        logger.warning(f"{grouping_var} not found — skipping quartile analysis")
        return pd.DataFrame()
    
    df = df.copy()
    df["quartile"] = pd.qcut(
        df[grouping_var],
        q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
        duplicates="drop"        
    )

    shap_cols = [c for c in df.columns if c.startswith("shap_") and c != "shap_baseline"]

    quartile_means = df.groupby("quartile", observed=True)[shap_cols].mean()

    rows = []
    for col in shap_cols:
        feat_name = col.replace("shap_", "")
        q_vals = quartile_means[col].values
        if len(q_vals) < 4:
            continue
        gap = float(q_vals[3] - q_vals[0])
        rows.append({
            "feature": feat_name,
            "Q1_mean": round(float(q_vals[0]), 3),
            "Q2_mean": round(float(q_vals[1]), 3),
            "Q3_mean": round(float(q_vals[2]), 3),
            "Q4_mean": round(float(q_vals[3]), 3),
            "gap_Q4_minus_Q1": round(gap, 3),
            "abs_gap": round(abs(gap), 3),
        })

    result = pd.DataFrame(rows).sort_values("abs_gap", ascending=False)

    sizes = df["quartile"].value_counts().sort_index()
    logger.info("Quartile sizes:")
    for q, n in sizes.items():
        mean_score = df[df["quartile"] == q]["predicted_score"].mean()
        logger.info(f"  {q}: {n:,} hexes  mean_score={mean_score:.1f}")
 
    logger.info(f"\nTop 10 features by Q4-Q1 SHAP gap:")
    for _, row in result.head(10).iterrows():
        direction = "▼" if row["gap_Q4_minus_Q1"] < 0 else "▲"
        logger.info(
            f"  {row['feature']:<40} "
            f"Q1={row['Q1_mean']:+.2f}  Q4={row['Q4_mean']:+.2f}  "
            f"gap={row['gap_Q4_minus_Q1']:+.3f} {direction}"
        )
 
    return result

# ── Visualisation 1: Correlation scatter plots ────────────────────────────────
 
def plot_correlation_scatter(
    df:       pd.DataFrame,
    corr_df:  pd.DataFrame,
) -> None:
    """
    2×3 grid of scatter plots: Walk Score vs each demographic variable.
    Regression line and r/p annotated on each panel.
    """
    logger.info("Generating correlation scatter plots…")
 
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
 
    vars_to_plot = list(DEMO_VARS.items())
 
    for idx, (var, label) in enumerate(vars_to_plot):
        if var not in df.columns:
            continue
 
        ax   = fig.add_subplot(gs[idx // 3, idx % 3])
        mask = df[var].notna()
        x    = df.loc[mask, var].values
        y    = df.loc[mask, "predicted_score"].values
 
        # Scatter with low alpha for overplotting
        ax.scatter(x, y, alpha=0.15, s=6, color="#2166ac", linewidths=0)
 
        # Regression line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m*x_line + b, color="#d73027", linewidth=2, alpha=0.8)
 
        # Annotate with r value
        row = corr_df[corr_df["variable"] == var]
        if len(row):
            r   = row.iloc[0]["pearson_r"]
            p   = row.iloc[0]["perm_p"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.annotate(
                f"r = {r:.3f} {sig}",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, fontweight="bold",
                color="#d73027" if abs(r) > 0.3 else "black",
            )
 
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Predicted Walk Score" if idx % 3 == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
 
    fig.suptitle(
        "Walkability Predictions vs Demographic Variables\n"
        "* p<0.05  ** p<0.01  *** p<0.001  (permutation test)",
        fontsize=13, fontweight="bold", y=1.01,
    )
 
    out = FIGURES_DIR / "equity_correlation.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation plot saved → {out.name}")
 
 
# ── Visualisation 2: SHAP gap decomposition ───────────────────────────────────
 
def plot_shap_gap_decomposition(
    quartile_df: pd.DataFrame,
    n_features:  int = 15,
) -> None:
    """
    Horizontal bar chart showing the SHAP gap between Q4 (highest minority)
    and Q1 (lowest minority) for the top N features.
 
    Negative gap = feature contributes less to Q4 hexes' scores than Q1.
    Positive gap = feature contributes more to Q4 hexes' scores than Q1.
 
    This is the equity audit's headline plot — it shows which specific
    features create or close the walkability gap between demographic groups.
    """
    if quartile_df.empty:
        logger.warning("Quartile DataFrame is empty — skipping gap plot")
        return
 
    logger.info(f"Generating SHAP gap decomposition plot…")
 
    # Select top N by absolute gap
    top = quartile_df.head(n_features).copy()
    top = top.sort_values("gap_Q4_minus_Q1", ascending=True)
 
    def clean(name):
        return (name
                .replace("census_", "census: ")
                .replace("poi_nearest_", "nearest ")
                .replace("poi_", "")
                .replace("transit_", "transit: ")
                .replace("safety_", "safety: ")
                .replace("terrain_", "terrain: ")
                .replace("lw_prop_", "% ")
                .replace("lw_avg_", "avg ")
                .replace("_kde", " density")
                .replace("_m$", " (m)")
                .replace("_", " "))
 
    labels = [clean(f) for f in top["feature"]]
    gaps   = top["gap_Q4_minus_Q1"].values
    colours = ["#d73027" if g < 0 else "#1a9850" for g in gaps]
 
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(labels, gaps, color=colours, alpha=0.85, height=0.7)
 
    # Value labels on bars
    for bar, gap in zip(bars, gaps):
        x = bar.get_width()
        ax.text(
            x + (0.02 if x >= 0 else -0.02),
            bar.get_y() + bar.get_height()/2,
            f"{gap:+.2f}",
            va="center", ha="left" if x >= 0 else "right",
            fontsize=9, fontweight="bold",
        )
 
    ax.axvline(x=0, color="black", linewidth=1.0, linestyle="-")
    ax.set_xlabel(
        "SHAP gap: Q4 (highest minority) minus Q1 (lowest minority)\n"
        "Negative = feature hurts high-minority hexes more than low-minority hexes",
        fontsize=10,
    )
    ax.set_title(
        "Which Features Drive the Walkability Gap\n"
        "Between High-Minority and Low-Minority Neighbourhoods",
        fontsize=12, fontweight="bold",
    )
 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", alpha=0.85,
              label="Disadvantages high-minority areas"),
        Patch(facecolor="#1a9850", alpha=0.85,
              label="Advantages high-minority areas"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
 
    plt.tight_layout()
    out = FIGURES_DIR / "shap_gap_decomposition.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Gap decomposition plot saved → {out.name}")
 
 
# ── Save reports ───────────────────────────────────────────────────────────────
 
def save_equity_reports(
    morans:      dict,
    corr_df:     pd.DataFrame,
    quartile_df: pd.DataFrame,
) -> None:
    """Save all equity audit results to outputs/equity/."""
 
    # Correlation table
    corr_path = EQUITY_DIR / "equity_summary.csv"
    corr_df.to_csv(str(corr_path), index=False)
    logger.info(f"Correlation summary → {corr_path.name}")
 
    # SHAP by quartile
    if not quartile_df.empty:
        q_path = EQUITY_DIR / "shap_by_quartile.csv"
        quartile_df.to_csv(str(q_path), index=False)
        logger.info(f"SHAP quartile analysis → {q_path.name}")
 
    # Moran's I text report
    report_path = EQUITY_DIR / "morans_i_report.txt"
    with open(str(report_path), "w") as f:
        f.write("SPATIAL AUTOCORRELATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        for k, v in morans.items():
            f.write(f"{k:<20}: {v}\n")
    logger.info(f"Moran's I report → {report_path.name}")
 
 
# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_equity_audit(n_permutations: int = 100) -> dict:
    logger.info("=" * 60)
    logger.info("WEEK 11- EQUITY AUDIT")
    logger.info("=" * 60)

    # 1. Load data
    df = load_audit_data()

    # 2. Moran's I
    logger.info("\n── Analysis 1: Spatial Autocorrelation ──")
    morans = compute_morans_i(df)

    # 3. Demographic correlations
    logger.info("\n── Analysis 2: Demographic Correlations ──")
    corr_df = compute_demographic_correlations(df, n_permutations=n_permutations)

    # 4. SHAP by quartile
    logger.info("\n── Analysis 3: SHAP by Minority Quartile ──")
    quartile_df = compute_shap_by_quartile(df, grouping_var="census_pct_minority")

    # 5. Plots
    logger.info("\n── Generating visualisations ──")
    plot_correlation_scatter(df, corr_df)
    plot_shap_gap_decomposition(quartile_df)
 
    # 6. Save reports
    save_equity_reports(morans, corr_df, quartile_df)
 
    # 7. Summary 
    logger.info("")
    logger.info("=" * 60)
    logger.info("WEEK 11 EQUITY AUDIT SUMMARY")
    logger.info("=" * 60)
 
    if morans.get("morans_i"):
        logger.info(
            f"  Moran's I            : {morans['morans_i']:.4f}  "
            f"(p={morans['p_value']:.4f})"
        )
 
    logger.info("\n  Demographic correlations with predicted Walk Score:")
    for _, row in corr_df.iterrows():
        sig = "✓" if row["significant"] else "✗"
        logger.info(
            f"  {sig} {row['label']:<28} "
            f"r={row['pearson_r']:+.3f}  "
            f"perm_p={row['perm_p']:.4f}"
        )
 
    if not quartile_df.empty:
        top_gap = quartile_df.iloc[0]
        logger.info(
            f"\n  Largest SHAP gap feature: {top_gap['feature']}"
            f"  gap={top_gap['gap_Q4_minus_Q1']:+.3f}"
        )
 
    logger.info("=" * 60)
 
    return {
        "morans_i":        morans.get("morans_i"),
        "morans_p":        morans.get("p_value"),
        "n_hexes":         len(df),
        "top_corr_var":    corr_df.iloc[0]["variable"] if len(corr_df) else None,
        "top_corr_r":      corr_df.iloc[0]["pearson_r"] if len(corr_df) else None,
    }
 
 


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/equity_audit.log", rotation="10 MB", level="DEBUG")

    try:
        results = run_equity_audit(n_permutations=1000)
        logger.success(
            f"Week 11 complete - "
            f"Moran's I={results['morans_i']}"
            f"top_r={results['top_corr_r']}"
        )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)
