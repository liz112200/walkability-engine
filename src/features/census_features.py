"""
src/features/census_features.py
─────────────────────────────────
Week 5: Census ACS demographic features via areal interpolation.

Writes: data/processed/features/chicago_census_features.parquet

Census data is published at the census tract level (~4,000 people per tract).
H3 hex cells don't align with tract boundaries, so we use areal interpolation
(tobler library) to disaggregate tract-level values down to hex cells.

The key principle: a hex cell gets a weighted average of the tract values
it overlaps, weighted by the fraction of the hex's area that falls in each tract.

Features (8)
------------
    census_median_income        Median household income (USD)
    census_pct_poverty          % households below poverty line
    census_pct_minority         % non-white population
    census_pct_black            % Black or African American population
    census_pct_hispanic         % Hispanic or Latino population
    census_pct_over_65          % population aged 65+
    census_pct_under_18         % population aged under 18
    census_pop_density          Population per km²

These features drive the equity audit in Week 12:
    - Do low-walkability hexes correlate with low income?
    - Are minority neighbourhoods systematically underserved by walkable infrastructure?
    - Are elderly residents (high walk-dependency) in the least walkable areas?

Usage
-----
    python -m src.features.census_features

Requirements
------------
    CENSUS_API_KEY in .env — free at https://api.census.gov/data/key_signup.html
    pip install census-data-downloader tobler
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

from src.utils.config import cfg

# ── ACS variable codes ────────────────────────────────────────────────────────
# ACS 5-year estimates, 2019–2023 vintage
# Full variable list: https://api.census.gov/data/2022/acs/acs5/variables.json

ACS_VARIABLES = {
    "B19013_001E": "median_income",        
    "B17001_001E": "total_poverty_denom",  
    "B17001_002E": "below_poverty",       
    "B02001_001E": "total_race",        
    "B02001_002E": "pop_white_alone",     
    "B03003_001E": "total_hispanic_denom",
    "B03003_003E": "pop_hispanic",         
    "B02001_003E": "pop_black",           
    "B01001_001E": "total_pop",            
    "B01001_020E": "male_65_66",          
    "B01001_021E": "male_67_69",
    "B01001_022E": "male_70_74",
    "B01001_023E": "male_75_79",
    "B01001_024E": "male_80_84",
    "B01001_025E": "male_85_plus",
    "B01001_044E": "female_65_66",
    "B01001_045E": "female_67_69",
    "B01001_046E": "female_70_74",
    "B01001_047E": "female_75_79",
    "B01001_048E": "female_80_84",
    "B01001_049E": "female_85_plus",
    "B01001_003E": "male_under_5",
    "B01001_004E": "male_5_9",
    "B01001_005E": "male_10_14",
    "B01001_006E": "male_15_17",
    "B01001_027E": "female_under_5",
    "B01001_028E": "female_5_9",
    "B01001_029E": "female_10_14",
    "B01001_030E": "female_15_17",
}

# Census API base URL
ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"

STATE_FIPS  = "17"
COUNTY_FIPS = "031"   


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_acs_tracts(api_key: str, cache_dir: Path) -> pd.DataFrame:
    """
    Fetch ACS 5-year estimates for all census tracts in Cook County.
    Cached as acs_cook_county.parquet after first download.
    """
    cache_path = cache_dir / "acs_cook_county.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Loading ACS data from cache…")
        return pd.read_parquet(str(cache_path))

    vars_str = ",".join(ACS_VARIABLES.keys())
    url = (
        f"{ACS_BASE}?get={vars_str}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}+county:{COUNTY_FIPS}"
        f"&key={api_key}"
    )

    logger.info("Downloading ACS 5-year estimates for Cook County tracts…")
    r = requests.get(url, timeout=60)
    if r.status_code == 401:
        raise ValueError(
            "Census API key rejected (401). "
            "Verify CENSUS_API_KEY in .env is correct and activated.\n"
            "Keys take ~1 hour to activate after signup."
        )
    r.raise_for_status()

    data = r.json()
    header = data[0]
    rows   = data[1:]
    df = pd.DataFrame(rows, columns=header)

    num_cols = list(ACS_VARIABLES.keys())
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace(-666666666, np.nan, inplace=True)
    df.replace(-999999999, np.nan, inplace=True)

    df = df.rename(columns=ACS_VARIABLES)

    df.to_parquet(str(cache_path))
    logger.info(f"ACS data cached → {cache_path.name}  ({len(df):,} tracts)")
    return df


def _fetch_tract_geometries(api_key: str, cache_dir: Path) -> gpd.GeoDataFrame:
    """
    Download census tract boundary shapefiles for Cook County.
    Uses Census TIGER/Line API.
    """
    cache_path = cache_dir / "cook_county_tracts.gpkg"

    if cache_path.exists():
        logger.info("Loading tract geometries from cache…")
        return gpd.read_file(cache_path)

    logger.info("Downloading Cook County census tract geometries…")
    # Census TIGER/Line shapefile — most reliable source for tract boundaries
    # URL pattern: https://www2.census.gov/geo/tiger/TIGER2022/TRACT/
    tiger_url = (
        "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/"
        f"tl_2022_{STATE_FIPS}_tract.zip"
    )

    import io, zipfile
    logger.info(f"Downloading TIGER tract shapefile for state {STATE_FIPS}…")
    r = requests.get(tiger_url, timeout=180, stream=True)
    r.raise_for_status()

    # Read shapefile directly from the zip bytes
    zip_bytes = io.BytesIO(r.content)
    gdf = gpd.read_file(zip_bytes)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Filter to Cook County only
    gdf = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    logger.info(f"Cook County tracts after filter: {len(gdf):,}")

    gdf.to_file(cache_path, driver="GPKG")
    logger.info(f"Tract geometries cached → {cache_path.name}  ({len(gdf):,} tracts)")
    return gdf


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage columns from ACS counts.
    Accepts either friendly column names or raw Census variable codes.
    All percentages are 0–100.
    """
    df = df.copy()

    # Apply rename if raw codes are present (handles stale cache or un-renamed df)
    if "B17001_002E" in df.columns:
        df = df.rename(columns=ACS_VARIABLES)

    df["pct_poverty"] = (
        df["below_poverty"] / df["total_poverty_denom"].clip(lower=1) * 100
    )

    df["pct_minority"] = (
        (df["total_race"] - df["pop_white_alone"]) /
        df["total_race"].clip(lower=1) * 100
    )

    df["pct_black"] = df["pop_black"] / df["total_race"].clip(lower=1) * 100

    df["pct_hispanic"] = (
        df["pop_hispanic"] / df["total_hispanic_denom"].clip(lower=1) * 100
    )

    over65_cols = [
        "male_65_66", "male_67_69", "male_70_74", "male_75_79",
        "male_80_84", "male_85_plus", "female_65_66", "female_67_69",
        "female_70_74", "female_75_79", "female_80_84", "female_85_plus",
    ]
    df["pop_over_65"] = df[[c for c in over65_cols if c in df.columns]].sum(axis=1)
    df["pct_over_65"] = df["pop_over_65"] / df["total_pop"].clip(lower=1) * 100

    under18_cols = [
        "male_under_5", "male_5_9", "male_10_14", "male_15_17",
        "female_under_5", "female_5_9", "female_10_14", "female_15_17",
    ]
    df["pop_under_18"] = df[[c for c in under18_cols if c in df.columns]].sum(axis=1)
    df["pct_under_18"] = df["pop_under_18"] / df["total_pop"].clip(lower=1) * 100

    return df


# ── Areal interpolation ───────────────────────────────────────────────────────

def _areal_interpolate(
    tracts_gdf: gpd.GeoDataFrame,
    hex_gdf: gpd.GeoDataFrame,
    extensive_cols: list[str],
    intensive_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Interpolate tract-level values to H3 hex cells using areal weighting.

    Extensive variables (counts like population) are split proportionally
    by area overlap. Intensive variables (rates/percentages) are weighted
    averages by area overlap.

    Falls back to a manual spatial-join-based approach if tobler is unavailable.
    """
    try:
        from tobler.area_weighted import area_interpolate
        logger.info("Running areal interpolation with tobler…")

        result = area_interpolate(
            source_df=tracts_gdf,
            target_df=hex_gdf,
            extensive_variables=extensive_cols,
            intensive_variables=intensive_cols,
        )
        return result

    except ImportError:
        logger.warning(
            "tobler not installed — falling back to centroid-based assignment.\n"
            "Install tobler for more accurate interpolation: pip install tobler"
        )
        return _centroid_fallback(tracts_gdf, hex_gdf, extensive_cols + intensive_cols)


def _centroid_fallback(
    tracts_gdf: gpd.GeoDataFrame,
    hex_gdf: gpd.GeoDataFrame,
    value_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Fallback: assign each hex the value of the tract containing its centroid.
    Less accurate than areal interpolation but requires no additional dependencies.
    """
    logger.info("Centroid-based tract assignment…")

    hex_centroids = hex_gdf.copy()
    hex_centroids.geometry = hex_gdf.geometry.centroid

    joined = gpd.sjoin(
        hex_centroids[["h3_index", "geometry"]],
        tracts_gdf[["GEOID"] + value_cols + ["geometry"]],
        how="left",
        predicate="within",
    )

    result = hex_gdf[["h3_index", "geometry"]].copy()
    for col in value_cols:
        if col in joined.columns:
            result[col] = joined[col].values
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_census_features(
    hex_gdf: gpd.GeoDataFrame | None = None,
    out_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Fetch ACS data, interpolate to H3 grid, and save features.
    """
    load_dotenv()
    api_key = os.getenv("CENSUS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "CENSUS_API_KEY not found in .env\n"
            "Get a free key at: https://api.census.gov/data/key_signup.html\n"
            "Keys activate within ~1 hour of signup."
        )

    slug     = cfg.city.slug
    out_path = out_path or (cfg.paths.processed / f"{slug}_census_features.parquet")
    cache_dir = cfg.paths.raw_census

    if hex_gdf is None:
        net_path = cfg.paths.processed / f"{slug}_network_features.parquet"
        assert net_path.exists(), "Run Week 2 first."
        hex_gdf = gpd.read_parquet(str(net_path))[
            ["h3_index", "geometry", "centroid_lat", "centroid_lng",
             "centroid_x", "centroid_y"]
        ]

    logger.info(f"Building Census features for {len(hex_gdf):,} hex cells…")
    t0 = time.perf_counter()

    # ── 1. Fetch ACS data ──────────────────────────────────────────────────────
    acs_df = _fetch_acs_tracts(api_key, cache_dir)
    acs_df = _compute_derived_columns(acs_df)

    # Build GEOID matching Census TIGER format: 2-digit state + 3-digit county + 6-digit tract
    # TIGER GEOIDs are 11 characters: e.g. "17031010100"
    acs_df["GEOID"] = (
        acs_df["state"].astype(str).str.strip().str.zfill(2) +
        acs_df["county"].astype(str).str.strip().str.zfill(3) +
        acs_df["tract"].astype(str).str.strip().str.zfill(6)
    )
    logger.info(f"ACS GEOID sample: {acs_df['GEOID'].head(3).tolist()}")

    # ── 2. Fetch tract geometries ──────────────────────────────────────────────
    tracts_geo = _fetch_tract_geometries(api_key, cache_dir)

    # ── 3. Join ACS values onto tract geometries ───────────────────────────────
    value_cols = [
        "median_income", "pct_poverty", "pct_minority", "pct_black",
        "pct_hispanic", "pct_over_65", "pct_under_18", "total_pop",
    ]
    # Normalise GEOID to 11 digits on both sides before merging.
    # TIGER shapefiles sometimes produce 12-digit GEOIDs with a leading digit;
    # ACS API always returns 11-digit GEOIDs (2+3+6). Taking the last 11
    # characters handles both formats safely.
    tracts_geo = tracts_geo.copy()
    tracts_geo["GEOID11"] = tracts_geo["GEOID"].astype(str).str[-11:]
    acs_df["GEOID11"]     = acs_df["GEOID"].astype(str).str[-11:]

    tracts_with_data = tracts_geo.merge(
        acs_df[["GEOID11"] + value_cols],
        on="GEOID11",
        how="left",
    )
    n_matched = tracts_with_data["median_income"].notna().sum()
    logger.info(f"GEOID merge: {n_matched:,} of {len(tracts_with_data):,} tracts matched ACS data")

    tracts_utm = tracts_with_data.to_crs(cfg.city.crs)
    hex_utm    = hex_gdf.to_crs(cfg.city.crs)

    logger.info(
        f"Tracts loaded: {len(tracts_utm):,}  |  "
        f"Tracts with ACS data: {tracts_utm['median_income'].notna().sum():,}"
    )

    # ── 4. Areal interpolation ─────────────────────────────────────────────────
    extensive = ["total_pop"]
    intensive = [c for c in value_cols if c != "total_pop"]

    interpolated = _areal_interpolate(
        tracts_gdf=tracts_utm,
        hex_gdf=hex_utm,
        extensive_cols=extensive,
        intensive_cols=intensive,
    )

    # ── 5. Build output GeoDataFrame ───────────────────────────────────────────
    result = hex_gdf[["h3_index", "geometry",
                       "centroid_lat", "centroid_lng",
                       "centroid_x", "centroid_y"]].copy()

    col_map = {
        "median_income": "census_median_income",
        "pct_poverty":   "census_pct_poverty",
        "pct_minority":  "census_pct_minority",
        "pct_black":     "census_pct_black",
        "pct_hispanic":  "census_pct_hispanic",
        "pct_over_65":   "census_pct_over_65",
        "pct_under_18":  "census_pct_under_18",
        "total_pop":     "census_pop_density",   
    }

    for src_col, dst_col in col_map.items():
        if src_col in interpolated.columns:
            result[dst_col] = interpolated[src_col].values
        else:
            result[dst_col] = np.nan
            logger.warning(f"Column {src_col} missing from interpolation output")

    hex_areas_km2 = result.geometry.area / 1e6
    result["census_pop_density"] = (
        result["census_pop_density"] / hex_areas_km2.clip(lower=1e-6)
    )

    # Clip percentages to [0, 100]
    pct_cols = [c for c in result.columns if "pct" in c]
    for col in pct_cols:
        result[col] = result[col].clip(lower=0, upper=100)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info(f"Census features complete in {elapsed:.1f}s")

    result.to_parquet(str(out_path))
    logger.info(f"Saved → {out_path.relative_to(cfg.project_root)}")

    net_dense = gpd.read_parquet(
        str(cfg.paths.processed / f"{slug}_network_features.parquet")
    )
    dense_ids = net_dense[net_dense["data_sparse"] == 0]["h3_index"]
    dense = result[result["h3_index"].isin(dense_ids)]

    census_cols = [c for c in result.columns if c.startswith("census_")]
    logger.info(f"Census feature summary ({len(census_cols)} columns, dense cells):")
    for col in census_cols:
        s = dense[col].dropna()
        if len(s) > 0:
            logger.info(
                f"  {col:<30} median={s.median():>10.1f}  "
                f"NaN%={100*dense[col].isna().mean():.1f}%"
            )
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/census_features.log", rotation="10 MB", level="DEBUG")

    try:
        build_census_features()

        logger.info("Merging into master feature table…")
        from src.features.feature_store import merge_all_features
        merge_all_features(rebuild=True)

        logger.success("Week 5 complete.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)