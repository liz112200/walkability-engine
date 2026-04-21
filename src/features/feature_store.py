from __future__ import annotations

import argparse
import geopandas as gpd
import pandas as pd

from pathlib import Path
from loguru import logger

from src.utils.config import cfg


FEATURE_FILES: list[tuple[str, str]] = [
    ("network_features",  "Week 2 — network topology"),
    ("poi_features",      "Week 3 — POI density"),
    ("transit_features",  "Week 3 — transit accessibility"),
    ("terrain_features",  "Week 4 — elevation / slope"),
    # ("safety_features",   "Week 4 — pedestrian safety"),
    ("census_features",   "Week 5 — Census ACS demographics"),
]
 
MASTER_PATH = cfg.paths.processed.parent / "master_features.parquet"

def load_feature_file(name: str) -> gpd.GeoDataFrame | pd.DataFrame | None:
    path = cfg.paths.processed / f"{cfg.city.slug}_{name}.parquet"
    if not path.exists():
        return None
    try:
        try:
            df = gpd.read_parquet(str(path))
        except Exception:
            df = pd.read_parquet(str(path))
        logger.debug(f"Loaded {name}: {df.shape}")
        return df
    except Exception as e:
        logger.warning(f"Failed to load {name}: {e}")
        return None



def merge_all_features(rebuild: bool = False) -> gpd.GeoDataFrame:

    if MASTER_PATH.exists() and not rebuild:
        logger.info(f"Loading cached master features from {MASTER_PATH.name}")
        return gpd.read_parquet(str(MASTER_PATH))

    logger.info("Building master feature table")
    master: gpd.GeoDataFrame | None = None
    loaded_count = 0

    for name, description in FEATURE_FILES:
        df = load_feature_file(name)
        if df is None:
            logger.info(f"  Skipping {name} (not yet generated — {description})")
            continue
        loaded_count += 1
        if master is None:
            if "h3_index" not in df.columns:
                raise ValueError(f"{name} is missing 'h3_index' column")
            master = df.copy() if isinstance(df, gpd.GeoDataFrame) else gpd.GeoDataFrame(df)

            logger.info(f"  Base table from {name}: {master.shape}")
        else:
            merge_cols = [c for c in df.columns
                          if c != "geometry" and c not in master.columns]
            merge_cols = ["h3_index"] + merge_cols

            df_slim = df[merge_cols]
            before = len(master)
            master = master.merge(df_slim, on="h3_index", how="left")
            assert len(master) == before, "Merge changed row count — check h3_index uniqueness"
            logger.info(f"  Merged {name}: +{len(merge_cols)-1} features → {master.shape}")

    if master is None:
        raise RuntimeError(
            "No feature files found. Run Week 2 pipeline first:\n"
            "  python -m src.features.network_features"
        )
    
    # Save master
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(str(MASTER_PATH))
    logger.info(f"Master features saved → {MASTER_PATH.name}")
    logger.info(f"  Shape: {master.shape}")
    logger.info(f"  Feature files merged: {loaded_count}")
    logger.info(f"  Sparse cells: {int(master.get('data_sparse', pd.Series([0])).sum())}")
 
    return master
 
def feature_report(master: gpd.GeoDataFrame) -> None:
    exclude = {"h3_index", "geometry", "centroid_x", "centroid_y",
               "centroid_lat", "centroid_lng", "data_sparse",
               "n_edges_in_hex", "n_nodes_in_hex"}
    feature_cols = [c for c in master.columns if c not in exclude]
 
    non_sparse = master[master.get("data_sparse", pd.Series(0, index=master.index)) == 0]
 
    logger.info("=" * 70)
    logger.info(f"MASTER FEATURE REPORT  ({len(feature_cols)} features, {len(master):,} hex cells)")
    logger.info(f"{'Feature':<38} {'NaN%':>6} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    logger.info("-" * 70)
    for col in sorted(feature_cols):
        series = non_sparse[col] if col in non_sparse.columns else master[col]
        nan_pct = 100 * series.isna().mean()
        clean = series.dropna()
        if len(clean) > 0:
            logger.info(
                f"{col:<38} {nan_pct:>5.1f}% {clean.mean():>9.2f} "
                f"{clean.std():>9.2f} {clean.min():>9.2f} {clean.max():>9.2f}"
            )
        else:
            logger.info(f"{col:<38} {nan_pct:>5.1f}% {'(all NaN)':>9}")
    logger.info("=" * 70)



# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all feature Parquet files into master.")
    parser.add_argument("--rebuild", action="store_true", 
                        help="Force rebuild even if master already exists")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/feature_store.log", rotation="10 MB", level="DEBUG")

    master = merge_all_features(rebuild=args.rebuild)
    feature_report(master)

    logger.success("Feature store up to date.")