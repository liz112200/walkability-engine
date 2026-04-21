"""
src/features/terrain_safety_features.py
─────────────────────────────────────────
Week 4: Terrain, sidewalk, and pedestrian safety features.

Computes two feature groups and writes one Parquet file:
    data/processed/features/chicago_terrain_features.parquet

Merged into master_features.parquet by feature_store.py.

Terrain Features (5)
--------------------
Derived from NASA SRTM 30m elevation raster tiles.

    terrain_elevation_mean_m    Mean elevation in metres (MSL)
    terrain_elevation_std_m     Std dev of elevation — topographic roughness
    terrain_slope_mean_pct      Mean slope in percent — walking effort proxy
    terrain_slope_max_pct       Max slope in hex — identifies steep segments
    terrain_flat_ratio          Fraction of raster cells with slope < 2%
                                (near-flat = easier walking)

Safety Features (4)
-------------------
Derived from Vision Zero Chicago pedestrian crash dataset and the
Chicago sidewalk inventory (CDOT open data).

    safety_crash_density        Pedestrian crashes per km² (5-year window)
    safety_crash_count          Raw crash count within hex boundary
    safety_sidewalk_coverage    Fraction of street edges with mapped sidewalk #not working
    safety_lit_street_ratio     Fraction of edges tagged with street lighting

Usage
-----
    # Download SRTM tiles first (see instructions below), then:
    python -m src.features.terrain_safety_features

SRTM Download Instructions
---------------------------
    Use the `elevation` Python package (pip install elevation)
    which downloads SRTM tiles automatically:
    eio clip -o data/raw/elevation/chicago_srtm.tif \\
       --bounds -88.0 41.6 -87.5 42.1

Vision Zero Data
----------------
Downloaded automatically from Chicago Data Portal (no API key needed).
URL: https://data.cityofchicago.org/resource/85ca-t3if.json
Updates monthly. Cached in data/raw/crash/

Sidewalk Data (not working)
-------------
Downloaded automatically from Chicago Data Portal.
URL: https://data.cityofchicago.org/resource/ii9m-5gg9.json
CDOT sidewalk inventory by block face.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import box
from tqdm import tqdm

from src.utils.config import cfg

# ── Constants ──────────────────────────────────────────────────────────────────

# Chicago Data Portal — Vision Zero crashes (pedestrian involved)
CRASH_URL = (
    "https://data.cityofchicago.org/resource/85ca-t3if.json"
    "?$where=first_crash_type=%27PEDESTRIAN%27"
    "&$limit=200000"
    "&$select=latitude,longitude,crash_date"
)

# Chicago sidewalk inventory
SIDEWALK_URL = (
    "https://data.cityofchicago.org/resource/77cn-6x4c.json"
    "?$limit=200000"
    "&$select=the_geom"
)

# Slope threshold for "essentially flat" classification
FLAT_SLOPE_PCT = 2.0


# ── SRTM elevation processing ─────────────────────────────────────────────────

def _load_elevation_raster(elevation_dir: Path) -> tuple:
    """
    Load SRTM elevation data from the elevation directory.

    Tries three approaches in order:
    1. A pre-merged GeoTIFF (chicago_srtm.tif) — fastest
    2. Individual .hgt tiles — merged on the fly
    3. The `elevation` package auto-download — if installed

    Returns (data_array, transform, crs) or raises if nothing found.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge
    except ImportError:
        raise ImportError(
            "Install with: conda install -c conda-forge rasterio"
        )

    # Option 1: pre-merged GeoTIFF
    merged_path = elevation_dir / "chicago_srtm.tif"
    if merged_path.exists():
        logger.info(f"Loading elevation from {merged_path.name}")
        with rasterio.open(merged_path) as src:
            return src.read(1), src.transform, src.crs

    # Option 2: individual SRTM .hgt or .tif tiles
    tiles = list(elevation_dir.glob("*.hgt")) + list(elevation_dir.glob("N4*.tif"))
    if tiles:
        logger.info(f"Merging {len(tiles)} SRTM tiles…")
        datasets = [rasterio.open(t) for t in tiles]
        mosaic, transform = rio_merge(datasets)
        crs = datasets[0].crs
        for d in datasets:
            d.close()

        # Cache the merged raster for future runs
        from rasterio.transform import from_bounds
        import rasterio.crs
        profile = {
            "driver": "GTiff", "dtype": mosaic.dtype,
            "width": mosaic.shape[2], "height": mosaic.shape[1],
            "count": 1, "crs": crs, "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(merged_path, "w", **profile) as dst:
            dst.write(mosaic[0], 1)
        logger.info(f"Merged raster cached → {merged_path.name}")
        return mosaic[0], transform, crs

    # Option 3: auto-download via `elevation` package
    try:
        import elevation
        logger.info("Downloading SRTM via elevation package (~100MB)…")
        output = str(merged_path)
        minx, miny, maxx, maxy = -88.1, 41.5, -87.4, 42.2
        elevation.clip(bounds=(minx, miny, maxx, maxy), output=output, product="SRTM3")
        elevation.clean()
        with rasterio.open(output) as src:
            return src.read(1), src.transform, src.crs
    except ImportError:
        pass

    raise FileNotFoundError(
        "No SRTM elevation data found in data/raw/elevation/\n\n"
        "Option A — Auto-download (easiest):\n"
        "  pip install elevation\n"
        "  eio clip -o data/raw/elevation/chicago_srtm.tif "
        "--bounds -88.1 41.5 -87.4 42.2\n\n"
        "Option B — Manual download:\n"
        "  Visit https://dwtkns.com/srtm30m/\n"
        "  Download tiles N41W088, N41W089, N42W088, N42W089\n"
        "  Place .hgt files in data/raw/elevation/"
    )


def _compute_slope(elevation: np.ndarray, transform) -> np.ndarray:
    """
    Compute slope in percent from an elevation array.

    Uses numpy gradient with cell size derived from the raster transform.
    Returns slope array in same shape as elevation.
    """
    # Cell size in metres — SRTM is in geographic degrees, convert approx
    # At Chicago's latitude (~41.8°), 1° lat ≈ 111,200m, 1° lon ≈ 83,100m
    cell_size_y = abs(transform.e) * 111_200   # degrees → metres (N-S)
    cell_size_x = abs(transform.a) * 83_100    # degrees → metres (E-W)

    dz_dy, dz_dx = np.gradient(elevation.astype(float),
                                cell_size_y, cell_size_x)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_pct = np.tan(slope_rad) * 100
    return slope_pct


def _sample_raster_in_hex(
    hex_poly,
    elevation: np.ndarray,
    slope: np.ndarray,
    transform,
    raster_crs,
    hex_crs_wgs84_geom,
) -> dict[str, float]:
    """
    Sample elevation and slope values within a hex polygon.
    """
    try:
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.windows import from_bounds

        # Get bounding box of hex in raster pixel coordinates
        bounds = hex_poly.bounds 
        window = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            transform=transform
        )

        # Read the window
        row_off = max(0, int(window.row_off))
        col_off = max(0, int(window.col_off))
        row_end = min(elevation.shape[0], int(window.row_off + window.height) + 1)
        col_end = min(elevation.shape[1], int(window.col_off + window.width) + 1)

        if row_off >= row_end or col_off >= col_end:
            return _empty_terrain()

        elev_crop  = elevation[row_off:row_end, col_off:col_end].flatten()
        slope_crop = slope[row_off:row_end, col_off:col_end].flatten()

        # Filter nodata (SRTM nodata = -32768)
        valid_mask = elev_crop > -32000
        elev_valid  = elev_crop[valid_mask]
        slope_valid = slope_crop[valid_mask]

        if len(elev_valid) == 0:
            return _empty_terrain()

        return {
            "terrain_elevation_mean_m": float(np.mean(elev_valid)),
            "terrain_elevation_std_m":  float(np.std(elev_valid)),
            "terrain_slope_mean_pct":   float(np.mean(slope_valid)),
            "terrain_slope_max_pct":    float(np.max(slope_valid)),
            "terrain_flat_ratio":       float(np.mean(slope_valid < FLAT_SLOPE_PCT)),
        }

    except Exception as e:
        logger.debug(f"Raster sampling failed for hex: {e}")
        return _empty_terrain()


def _empty_terrain() -> dict[str, float]:
    return {
        "terrain_elevation_mean_m": np.nan,
        "terrain_elevation_std_m":  np.nan,
        "terrain_slope_mean_pct":   np.nan,
        "terrain_slope_max_pct":    np.nan,
        "terrain_flat_ratio":       np.nan,
    }


# ── Crash data ────────────────────────────────────────────────────────────────

def _fetch_crash_data(crash_dir: Path) -> gpd.GeoDataFrame:
    """
    Download Vision Zero Chicago pedestrian crash data.
    Cached as crash_pedestrian.gpkg after first download.
    """
    cache_path = crash_dir / "crash_pedestrian.gpkg"
    crash_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Loading crash data from cache…")
        return gpd.read_file(cache_path)

    logger.info("Downloading Vision Zero pedestrian crash data…")
    try:
        r = requests.get(CRASH_URL, timeout=120)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        logger.info(f"Downloaded {len(df):,} pedestrian crash records")

        # Drop rows with missing coordinates
        df = df.dropna(subset=["latitude", "longitude"])
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        ).to_crs(cfg.city.crs)

        gdf[["geometry", "crash_date"]].to_file(cache_path, driver="GPKG")
        logger.info(f"Crash data cached → {cache_path.name}  ({len(gdf):,} records)")
        return gdf

    except Exception as e:
        logger.warning(f"Crash download failed: {e}")
        logger.warning("Returning empty crash dataset — safety_crash_* features will be NaN")
        return gpd.GeoDataFrame(columns=["geometry"], crs=cfg.city.crs)


# ── Sidewalk data ─────────────────────────────────────────────────────────────

def _fetch_sidewalk_data(crash_dir: Path) -> gpd.GeoDataFrame:
    """
    Download CDOT sidewalk inventory from Chicago Data Portal.
    Cached as sidewalk_inventory.gpkg after first download.
    """
    cache_path = crash_dir / "sidewalk_inventory.gpkg"

    if cache_path.exists():
        logger.info("Loading sidewalk data from cache…")
        return gpd.read_file(cache_path)

    logger.info("Downloading sidewalk inventory from Chicago Data Portal…")
    try:
        import os
        token = os.getenv("CHICAGO_DATA_TOKEN", "")
        headers = {"X-App-Token": token} if token else {}
        if not token:
            logger.warning(
                "CHICAGO_DATA_TOKEN not set in .env — sidewalk download may return 403.\n"
                "Register free at: https://data.cityofchicago.org/profile/app_tokens"
            )
        r = requests.get(SIDEWALK_URL, headers=headers, timeout=120)
        if r.status_code == 403:
            logger.warning(
                "Sidewalk API returned 403 — app token required.\n"
                "Set CHICAGO_DATA_TOKEN in .env and re-run.\n"
                "safety_sidewalk_coverage will be NaN for now."
            )
            return gpd.GeoDataFrame(columns=["geometry"], crs=cfg.city.crs)
        r.raise_for_status()
        data = r.json()

        if not data:
            logger.warning("Sidewalk API returned empty response")
            return gpd.GeoDataFrame(columns=["geometry"], crs=cfg.city.crs)

        gdf = gpd.GeoDataFrame.from_features(
            [{"type": "Feature",
              "geometry": row["the_geom"],
              "properties": {"sidewalk_location": row.get("sidewalk_location", "")}}
             for row in data if "the_geom" in row],
            crs="EPSG:4326",
        ).to_crs(cfg.city.crs)

        gdf.to_file(cache_path, driver="GPKG")
        logger.info(f"Sidewalk data cached → {cache_path.name}  ({len(gdf):,} segments)")
        return gdf

    except Exception as e:
        logger.warning(f"Sidewalk download failed: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=cfg.city.crs)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_terrain_features(
    hex_gdf: gpd.GeoDataFrame | None = None,
    out_path: Path | None = None,
    skip_elevation: bool = False,
) -> gpd.GeoDataFrame:
    """
    Compute terrain and safety features for all H3 hex cells.
    """
    slug = cfg.city.slug
    out_path = out_path or (cfg.paths.processed / f"{slug}_terrain_features.parquet")

    if hex_gdf is None:
        net_path = cfg.paths.processed / f"{slug}_network_features.parquet"
        assert net_path.exists(), "Run Week 2 first."
        hex_gdf = gpd.read_parquet(str(net_path))[
            ["h3_index", "geometry", "centroid_lat", "centroid_lng",
             "centroid_x", "centroid_y"]
        ]

    logger.info(f"Building terrain + safety features for {len(hex_gdf):,} hex cells…")
    t0 = time.perf_counter()

    result = hex_gdf[["h3_index", "geometry",
                       "centroid_lat", "centroid_lng",
                       "centroid_x", "centroid_y"]].copy()

    # ── 1. Elevation and slope (SRTM) ─────────────────────────────────────────
    if skip_elevation:
        logger.warning("Skipping elevation — terrain_* columns will be NaN")
        for col in ["terrain_elevation_mean_m", "terrain_elevation_std_m",
                    "terrain_slope_mean_pct", "terrain_slope_max_pct",
                    "terrain_flat_ratio"]:
            result[col] = np.nan
    else:
        try:
            elev_data, transform, raster_crs = _load_elevation_raster(
                cfg.paths.raw_elevation
            )
            slope_data = _compute_slope(elev_data, transform)
            logger.info(
                f"Raster loaded: {elev_data.shape}  |  "
                f"elevation range: {elev_data[elev_data > -32000].min():.0f}–"
                f"{elev_data[elev_data > -32000].max():.0f}m"
            )

            # Hex grid must be in WGS-84 for raster sampling
            hex_wgs = hex_gdf.to_crs("EPSG:4326")

            terrain_rows = []
            for _, row in tqdm(hex_wgs.iterrows(), total=len(hex_wgs),
                               desc="Terrain sampling"):
                terrain_rows.append(
                    _sample_raster_in_hex(row.geometry, elev_data, slope_data, transform, raster_crs, row.geometry)
                )

            terrain_df = pd.DataFrame(terrain_rows)
            for col in terrain_df.columns:
                result[col] = terrain_df[col].values

        except FileNotFoundError as e:
            logger.warning(str(e))
            logger.warning("Continuing without elevation — terrain_* columns will be NaN")
            for col in ["terrain_elevation_mean_m", "terrain_elevation_std_m",
                        "terrain_slope_mean_pct", "terrain_slope_max_pct",
                        "terrain_flat_ratio"]:
                result[col] = np.nan

    # ── 2. Pedestrian crash density (Vision Zero) ─────────────────────────────
    crashes = _fetch_crash_data(cfg.paths.raw_crash)

    if len(crashes) > 0:
        logger.info(f"Computing crash density for {len(hex_gdf):,} hexes…")

        # Spatial join: assign each crash to its hex
        joined = gpd.sjoin(
            crashes[["geometry"]],
            hex_gdf[["h3_index", "geometry"]],
            how="left",
            predicate="within",
        )
        crash_counts = (
            joined.dropna(subset=["h3_index"])
            .groupby("h3_index")
            .size()
            .rename("safety_crash_count")
        )
        result = result.merge(crash_counts, on="h3_index", how="left")
        result["safety_crash_count"] = result["safety_crash_count"].fillna(0).astype(int)

        # Density = crashes per km²
        hex_areas_km2 = result.geometry.area / 1e6
        result["safety_crash_density"] = (
            result["safety_crash_count"] / hex_areas_km2.clip(lower=1e-6)
        )

        n_hexes_with_crashes = (result["safety_crash_count"] > 0).sum()
        logger.info(
            f"Crash data: {len(crashes):,} crashes assigned  |  "
            f"{n_hexes_with_crashes:,} hexes with ≥1 crash"
        )
    else:
        result["safety_crash_count"]   = np.nan
        result["safety_crash_density"] = np.nan

    # ── 3. Sidewalk coverage ──────────────────────────────────────────────────
    sidewalks = _fetch_sidewalk_data(cfg.paths.raw_crash)

    if len(sidewalks) > 0:
        logger.info("Computing sidewalk coverage per hex…")

        # Spatial join sidewalk segments → hexes
        joined_sw = gpd.sjoin(
            sidewalks[["geometry"]],
            hex_gdf[["h3_index", "geometry"]],
            how="left",
            predicate="intersects",
        )
        sw_count = (
            joined_sw.dropna(subset=["h3_index"])
            .groupby("h3_index")
            .size()
            .rename("_sw_segments")
        )

        # Load edge counts from network features for normalisation
        try:
            net = gpd.read_parquet(
                str(cfg.paths.processed / f"{slug}_network_features.parquet")
            )[["h3_index", "n_real_edges"]]
            result = result.merge(net, on="h3_index", how="left")
            result = result.merge(sw_count, on="h3_index", how="left")
            result["_sw_segments"] = result["_sw_segments"].fillna(0)
            # Sidewalk coverage = sidewalk segments / total edges (capped at 1.0)
            result["safety_sidewalk_coverage"] = (
                result["_sw_segments"] / result["n_real_edges"].clip(lower=1)
            ).clip(upper=1.0)
            result = result.drop(columns=["_sw_segments", "n_real_edges"])
        except Exception as e:
            logger.warning(f"Sidewalk coverage computation failed: {e}")
            result["safety_sidewalk_coverage"] = np.nan
    else:
        logger.warning("No sidewalk data — safety_sidewalk_coverage will be NaN")
        result["safety_sidewalk_coverage"] = np.nan

    # ── 4. Street lighting (from OSM edge tags) ───────────────────────────────
    # OSM lit=yes tag on edges — sparse but directionally useful
    try:
        edges = gpd.read_file(
            str(cfg.paths.raw_osm / f"{slug}_walk_edges.gpkg"), layer="edges"
        )
        if "lit" in edges.columns:
            is_lit = edges["lit"].isin(["yes", "24/7", "automatic"])
            lit_edges = edges[is_lit][["geometry"]].copy()
            joined_lit = gpd.sjoin(
                lit_edges,
                hex_gdf[["h3_index", "geometry"]],
                how="left",
                predicate="within",
            )
            lit_count = (
                joined_lit.dropna(subset=["h3_index"])
                .groupby("h3_index")
                .size()
                .rename("_lit_count")
            )
            net2 = gpd.read_parquet(
                str(cfg.paths.processed / f"{slug}_network_features.parquet")
            )[["h3_index", "n_real_edges"]]
            result = result.merge(lit_count, on="h3_index", how="left")
            result = result.merge(net2, on="h3_index", how="left")
            result["_lit_count"] = result["_lit_count"].fillna(0)
            result["safety_lit_street_ratio"] = (
                result["_lit_count"] / result["n_real_edges"].clip(lower=1)
            ).clip(upper=1.0)
            result = result.drop(columns=["_lit_count", "n_real_edges"])
        else:
            logger.info("No 'lit' column in OSM edges — safety_lit_street_ratio = 0")
            result["safety_lit_street_ratio"] = 0.0
    except Exception as e:
        logger.warning(f"Lighting feature failed: {e}")
        result["safety_lit_street_ratio"] = np.nan

    # ── Save ───────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info(f"Terrain + safety features complete in {elapsed:.1f}s")

    result.to_parquet(str(out_path))
    logger.info(f"Saved → {out_path.relative_to(cfg.project_root)}")

    # Summary
    feature_cols = [c for c in result.columns
                    if c.startswith(("terrain_", "safety_"))]
    logger.info(f"Feature summary ({len(feature_cols)} columns):")
    net_dense = gpd.read_parquet(
        str(cfg.paths.processed / f"{slug}_network_features.parquet")
    )
    dense_ids = net_dense[net_dense["data_sparse"] == 0]["h3_index"]
    dense = result[result["h3_index"].isin(dense_ids)]
    for col in feature_cols:
        s = dense[col].dropna()
        if len(s) > 0:
            logger.info(
                f"  {col:<35} median={s.median():.2f}  NaN%={100*dense[col].isna().mean():.1f}%"
            )
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/terrain_safety_features.log", rotation="10 MB", level="DEBUG")

    parser = argparse.ArgumentParser(
        description="Week 4: terrain and safety features."
    )
    parser.add_argument(
        "--skip-elevation",
        action="store_true",
        help="Skip SRTM elevation processing (if tiles not yet downloaded). "
             "Safety features are still computed.",
    )
    args = parser.parse_args()

    try:
        build_terrain_features(skip_elevation=args.skip_elevation)

        logger.info("Merging into master feature table…")
        from src.features.feature_store import merge_all_features
        merge_all_features(rebuild=True)

        logger.success("Week 4 complete.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)