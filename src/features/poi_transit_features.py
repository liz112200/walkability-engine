from __future__ import annotations

import time
import argparse
import requests
import geopandas as gpd
import osmnx as ox
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from scipy.stats import gaussian_kde
from tqdm import tqdm

from src.utils.config import cfg
from src.utils.h3_grid import add_h3_index

 
POI_CATEGORIES: dict[str, dict] = {
    "grocery":    {"shop": ["supermarket", "grocery", "convenience", "greengrocer"]},
    "pharmacy":   {"amenity": ["pharmacy"], "shop": ["chemist"]},
    "park":       {"leisure": ["park", "nature_reserve", "garden"],
                   "landuse": ["recreation_ground", "grass"]},
    "school":     {"amenity": ["school", "university", "college", "kindergarten"]},
    "restaurant": {"amenity": ["restaurant", "cafe", "fast_food", "food_court", "bar"]},
    "healthcare": {"amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"]},
    "transit":    {"highway": ["bus_stop"],
                   "railway": ["station", "tram_stop", "subway_entrance"]},
    "retail":     {"shop": True},  
}

KDE_BANDWIDTH_M = 150.0

TRANSIT_RADIUS_M = 400.0
CTA_GTFS_URL = (
    "https://www.transit.land/api/v2/rest/feeds/f-dp3wjv-chicagotransitauthority"
    "/download_latest_feed_version"
)
CTA_GTFS_FALLBACK = "https://www.transitchicago.com/downloads/sch_data/google_transit.zip"



# ── POI fetching ──────────────────────────────────────────────────────────────

def fetch_pois(city: str, out_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, gpd.GeoDataFrame] = {}

    for category, tags in POI_CATEGORIES.items():  
        cache_path = out_dir / f"poi_{category}.gpkg"
        if cache_path.exists():
            logger.info(f"  {category:<15} loading from cache")
            gdf = gpd.read_file(cache_path)
            results[category] = gdf
            continue

        logger.info(f"  {category:<15} fetching from OSM…")
        try:
            gdf = ox.features_from_place(city, tags=tags)
            gdf = gdf[gdf.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon"])]
            gdf = gdf.copy().to_crs(cfg.city.crs)
            gdf.geometry = gdf.geometry.centroid
            gdf = gdf.to_crs("EPSG:4326")
            gdf.to_file(cache_path, driver="GPKG")
            logger.info(f"  {category:<15} {len(gdf):,} POIs cached")
            results[category] = gdf

        except Exception as e:
            logger.warning(f"  {category:<15} fetch failed: {e} — skipping")
            results[category] = gpd.GeoDataFrame(
                columns=["geometry"], geometry="geometry", crs="EPSG:4326"
            )

    return results


# ── KDE density computation ────────────────────────────────────────────────────
# How much stuff is here

def _kde_density(
        poi_gdf: gpd.GeoDataFrame, 
        hex_centroids_utm: gpd.GeoDataFrame,
        bandwidth_m: float = KDE_BANDWIDTH_M
) -> np.ndarray:
    
    if len(poi_gdf) < 2:
        return np.zeros(len(hex_centroids_utm))
    
    poi_xy = np.column_stack([
        poi_gdf.geometry.x.values,
        poi_gdf.geometry.y.values
    ])
    query_xy = np.column_stack([
        hex_centroids_utm.geometry.x.values,
        hex_centroids_utm.geometry.y.values
    ])

    bw_factor = bandwidth_m / np.std(poi_xy, axis=0).mean()
    try:
        kde = gaussian_kde(poi_xy.T, bw_method=bw_factor)
        return kde(query_xy.T)
    except Exception as e:
        logger.warning(f"KDE failed: {e}")
        return np.zeros(len(hex_centroids_utm))

# How balanced is the stuff
def _shannon_entropy(category_counts: pd.Series) -> float:
    total = category_counts.sum()
    if total == 0:
        return 0.0
    probs = category_counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))



# ── Main POI feature builder ───────────────────────────────────────────────────
def build_poi_features(
        hex_gdf: gpd.GeoDataFrame | None = None,
        out_path: Path | None = None
) -> gpd.GeoDataFrame:

    slug = cfg.city.slug
    out_path = out_path or (cfg.paths.processed / f"{slug}_poi_features.parquet")
    poi_raw_dir = cfg.paths.raw_poi

    if hex_gdf is None:
        net_path = cfg.paths.processed / f"{slug}_network_features.parquet"
        assert net_path.exists(), (
            f"Run Week 2 first: {net_path} not found.\n"
            "python -m src.features.network_features"
        )
        hex_gdf = gpd.read_parquet(str(net_path))[
            ["h3_index", "geometry", "centroid_lat", "centroid_lng",
             "centroid_x", "centroid_y"]
        ]
    
    logger.info(f"Building POI features for {len(hex_gdf):,} hex cells…")
    t0 = time.perf_counter()

    hex_centroids_utm = gpd.GeoDataFrame(
        {"h3_index": hex_gdf["h3_index"]},
        geometry=gpd.points_from_xy(hex_gdf["centroid_x"], hex_gdf["centroid_y"]),
        crs=cfg.city.crs
    )

    logger.info("Fetching POIs from OSM (cached after first run)")
    pois = fetch_pois(cfg.city.name, poi_raw_dir)

    pois_utm = {
        cat: gdf.to_crs(cfg.city.crs) if len(gdf) > 0 else gdf
        for cat, gdf in pois.items()
    }

    # ── KDE features ──────────────────────────────────────────────────────────
    result = hex_gdf[["h3_index", "geometry",
                       "centroid_lat", "centroid_lng",
                       "centroid_x", "centroid_y"]].copy()

    for category, gdf_utm in pois_utm.items():
        col = f"poi_{category}_kde"
        logger.info(f"  KDE: {category} ({len(gdf_utm):,} POIs)…")
        # result[col] = _kde_density(gdf_utm, hex_centroids_utm)
        raw_kde = _kde_density(gdf_utm, hex_centroids_utm)
        kde_max = raw_kde.max()
        if kde_max > 0:
            result[col] = raw_kde / kde_max
        else:
            result[col] = raw_kde

    # ── Raw count and category diversity ──────────────────────────────────────
    logger.info("Computing POI counts and diversity per hex…")
    all_poi_rows = []
    for category, gdf_utm in pois_utm.items():
        if len(gdf_utm) == 0:
            continue
        tmp = gdf_utm[["geometry"]].copy()
        tmp["category"] = category
        tmp["centroid_lat"] = tmp.geometry.to_crs("EPSG:4326").y
        tmp["centroid_lng"] = tmp.geometry.to_crs("EPSG:4326").x
        all_poi_rows.append(tmp)
    
    if all_poi_rows:
        all_pois_list = []
        for tmp in all_poi_rows:
            all_pois_list.append(tmp[["geometry", "category"]].copy())
        all_pois = gpd.GeoDataFrame(
            pd.concat(all_pois_list, ignore_index=True),
            geometry="geometry",
            crs=cfg.city.crs,
        )

        hex_polys = hex_gdf[["h3_index", "geometry"]].copy()
        joined = gpd.sjoin(
            all_pois,
            hex_polys,
            how="left",
            predicate="within",
        )
 
        count_by_hex = (
            joined.dropna(subset=["h3_index"])
            .groupby("h3_index")
            .size()
            .rename("poi_total_count")
        )


        # all_pois = pd.concat(all_poi_rows, ignore_index=True)
        # all_pois = gpd.GeoDataFrame(all_pois, crs=cfg.city.crs)
        # all_pois = add_h3_index(all_pois, resolution=cfg.h3.resolution)

        # count_by_hex = all_pois.groupby("h3_index").size().rename("poi_total_count")
        result = result.merge(count_by_hex, on="h3_index", how="left")
        result["poi_total_count"] = result["poi_total_count"].fillna(0).astype(int)

        # poi category diversity
        # entropy_by_hex = (
        #     all_pois.groupby(["h3_index", "category"])      
        #     .size()                                         # count pois by hex and category
        #     .unstack(fill_value=0)                          # 1 list of counts -> table with category as column
        #     .apply(_shannon_entropy, axis=1)                # run diversity formula on every row
        #     .rename("poi_category_diversity")              
        # )

        entropy_by_hex = (
            joined.dropna(subset=["h3_index"])
            .groupby(["h3_index", "category"])
            .size()
            .unstack(fill_value=0)
            .apply(_shannon_entropy, axis=1)
            .rename("poi_category_diversity")
        )


        result = result.merge(entropy_by_hex, on="h3_index", how="left")
        result["poi_category_diversity"] = result["poi_category_diversity"].fillna(0.0)

    else:
        result["poi_total_count"] = 0
        result["poi_category_diversity"] = 0.0

    # ── Nearest grocery and park distance ─────────────────────────────────────
    # Simple Euclidean nearest-neighbour in UTM (metres).
    # True network distance is computed in Week 9 when we have routing set up.
    from scipy.spatial import cKDTree
    NEAREST_CATEGORIES = [
        ("grocery",    "poi_nearest_grocery_m"),
        ("pharmacy",   "poi_nearest_pharmacy_m"),
        ("park",       "poi_nearest_park_m"),
        ("school",     "poi_nearest_school_m"),
        ("healthcare", "poi_nearest_healthcare_m"),
    ]
 
    logger.info(f"Computing nearest-destination distances for {len(NEAREST_CATEGORIES)} categories…")
    query_xy = np.column_stack([
        hex_centroids_utm.geometry.x.values,
        hex_centroids_utm.geometry.y.values,
    ])

    for cat, col in NEAREST_CATEGORIES:
        gdf_utm = pois_utm.get(cat)
        if gdf_utm is None or len(gdf_utm) == 0:
            logger.warning(f"  {cat}: no POIs found — {col} set to NaN")
            result[col] = np.nan
            continue
        poi_xy = np.column_stack([
            gdf_utm.geometry.x.values,
            gdf_utm.geometry.y.values,
        ])
        tree = cKDTree(poi_xy)
        distances, _ = tree.query(query_xy, k=1, workers=-1)
        result[col] = distances.astype(float)
        logger.info(f"  {cat}: median nearest = {float(np.median(distances)):.0f}m")
 
    elapsed = time.perf_counter() - t0
    logger.info(f"POI features complete in {elapsed:.1f}s")
 
    result.to_parquet(str(out_path))
    logger.info(f"Saved → {out_path.relative_to(cfg.project_root)}")
    return result

# ── Transit features ──────────────────────────────────────────────────────────

def _download_gtfs(out_dir: Path) -> Path:
    zip_path = out_dir / "cta_gtfs.zip"
    if zip_path.exists():
        logger.info(f"GTFS already downloaded at {zip_path}")
        return zip_path
    
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CTA GTFS feed.")

    for url in [CTA_GTFS_FALLBACK]:
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"GTFS downloaded: {zip_path.stat().st_size/1e6:.1f} MB")
            return zip_path
        except Exception as e:
            logger.warning(f"Download from {url} failed: {e}")

    raise RuntimeError(
        "GTFS download failed. Download manually from:\n"
        "  https://www.transitchicago.com/downloads/sch_data/google_transit.zip\n"
        f"and place at: {zip_path}"
    )

def build_transit_features(
        hex_gdf: gpd.GeoDataFrame | None = None,
        out_path: Path | None = None,
) -> gpd.GeoDataFrame:
    
    import zipfile
    import io

    slug = cfg.city.slug
    out_path = out_path or (cfg.paths.processed / f"{slug}_transit_features.parquet")
    gtfs_dir = cfg.paths.raw_gtfs

    if hex_gdf is None:
        net_path = cfg.paths.processed / f"{slug}_network_features.parquet"
        assert net_path.exists(), "Run Week 2 first."
        hex_gdf = gpd.read_parquet(str(net_path))[
            ["h3_index", "geometry", "centroid_lat", "centroid_lng",
             "centroid_x", "centroid_y"]
        ]

    logger.info(f"Building transit features for {len(hex_gdf):,} hex cells…")
    t0 = time.perf_counter()
 
    zip_path = _download_gtfs(gtfs_dir)

    with zipfile.ZipFile(zip_path) as zf:
        stops_df = pd.read_csv(io.BytesIO(zf.read("stops.txt")))
        stop_times_df = pd.read_csv(
            io.BytesIO(zf.read("stop_times.txt")),
            usecols=["trip_id", "stop_id", "departure_time"],
        )
        trips_df = pd.read_csv(
            io.BytesIO(zf.read("trips.txt")),
            usecols=["trip_id", "route_id"],
        )
        routes_df = pd.read_csv(
            io.BytesIO(zf.read("routes.txt")),
            usecols=["route_id", "route_type"],
        )
 
    logger.info(
        f"GTFS loaded: {len(stops_df):,} stops, "
        f"{len(stop_times_df):,} stop_times"
    )

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(cfg.city.crs)

    # Compute average headway per stop (minutes between successive trips)
    def _avg_headway(group: pd.DataFrame) -> float:
        times = []
        for t in group["departure_time"]:
            try:
                h, m, s = map(int, str(t).split(":"))
                times.append(h * 3600 + m * 60 + s)
            except Exception:
                continue
        if len(times) < 2:
            return np.nan
        times_sorted = sorted(times)
        gaps = [times_sorted[i+1] - times_sorted[i]
                for i in range(len(times_sorted)-1)
                if 0 < times_sorted[i+1] - times_sorted[i] < 7200]
        return float(np.mean(gaps) / 60) if gaps else np.nan  # → minutes
 
    logger.info("Computing per-stop headways (this takes ~2–3 min)…")
    headways = stop_times_df.groupby("stop_id").apply(_avg_headway)

    # stop_times_df: trip A stops at B
    # trips_df: trip A belongs to route C
    # routes_df: route C is a bus
    # -> every arrival at a stop is labeled with its specific route ID and mode
    # aggregation: n_routes: unique transit lines serving this stop
    #              n_modes: unique types of transit
    stop_routes = (
        stop_times_df
        .merge(trips_df, on="trip_id")
        .merge(routes_df, on="route_id")
        .groupby("stop_id")
        .agg(
            n_routes=("route_id", "nunique"),
            n_modes=("route_type", "nunique"),
        )
    )
    
    # After this code runs, every point in stops_gdf is now a high-dimensional feature vector that tells the AI:
    # "This stop has a bus every 8 minutes."
    # "This stop serves 4 different routes."
    # "This stop provides access to 2 different modes (Rail and Bus)."
    stops_gdf = stops_gdf.merge(
        headways.rename("avg_headway_min"), left_on="stop_id", right_index=True, how="left"
    ).merge(stop_routes, on="stop_id", how="left")

    # ── Per-hex transit aggregation ────────────────────────────────────────────
    logger.info(f"Aggregating transit metrics to {len(hex_gdf):,} hex cells…")
 
    hex_centroids_utm = gpd.GeoDataFrame(
        {"h3_index": hex_gdf["h3_index"]},
        geometry=gpd.points_from_xy(hex_gdf["centroid_x"], hex_gdf["centroid_y"]),
        crs=cfg.city.crs,
    )

    from scipy.spatial import cKDTree
 
    stop_xy = np.column_stack([stops_gdf.geometry.x.values, stops_gdf.geometry.y.values])
 
    hex_xy  = np.column_stack([
        hex_centroids_utm.geometry.x.values,
        hex_centroids_utm.geometry.y.values,
    ])
    stop_tree = cKDTree(stop_xy)
 
    # Query 1: nearest stop distance for every hex (vectorised, instant)
    nearest_dists, nearest_idxs = stop_tree.query(hex_xy, k=1, workers=-1)
 
    # Query 2: all stops within TRANSIT_RADIUS_M of each hex
    # ball_point_query returns a list-of-lists — one list of stop indices per hex
    within_idxs = stop_tree.query_ball_point(hex_xy, r=TRANSIT_RADIUS_M, workers=-1)
 
    logger.info(f"cKDTree queries complete — aggregating {len(hex_xy):,} hexes…")
 
    stops_arr_n_routes  = stops_gdf["n_routes"].fillna(0).values
    stops_arr_headway   = stops_gdf["avg_headway_min"].values
    stops_arr_n_modes   = stops_gdf["n_modes"].fillna(0).values
 
    transit_rows = []
    for i, idx_list in enumerate(within_idxs):
        nearest_m = float(nearest_dists[i])
        if len(idx_list) == 0:
            transit_rows.append({
                "transit_stops_400m":      0,
                "transit_routes_400m":     0,
                "transit_avg_headway_min": np.nan,
                "transit_nearest_stop_m":  nearest_m,
                "transit_mode_diversity":  0,
            })
        else:
            nearby_headways = stops_arr_headway[idx_list]
            valid_hw = nearby_headways[~np.isnan(nearby_headways)]
            transit_rows.append({
                "transit_stops_400m":      len(idx_list),
                "transit_routes_400m":     int(stops_arr_n_routes[idx_list].sum()),
                "transit_avg_headway_min": float(valid_hw.mean()) if len(valid_hw) > 0 else np.nan,
                "transit_nearest_stop_m":  nearest_m,
                "transit_mode_diversity":  int(stops_arr_n_modes[idx_list].max()),
            })
 
    result = hex_gdf[["h3_index", "geometry",
                       "centroid_lat", "centroid_lng",
                       "centroid_x", "centroid_y"]].copy()
    transit_df = pd.DataFrame(transit_rows)
    for col in transit_df.columns:
        result[col] = transit_df[col].values
 
    elapsed = time.perf_counter() - t0
    logger.info(f"Transit features complete in {elapsed:.1f}s")
    result.to_parquet(str(out_path))
    logger.info(f"Saved → {out_path.relative_to(cfg.project_root)}")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/poi_transit_features.log", rotation="10 MB", level="DEBUG")

    parser = argparse.ArgumentParser(
        description="Week 3: compute POI and transit features."
    )
    parser.add_argument(
        "--step",
        choices=["poi", "transit", "both"],
        default="both",
        help="Which feature group to compute (default: both)"
    )
    args = parser.parse_args()


    try:
        if args.step in ("poi", "both"):
            logger.info("=== POI FEATURES ===")
            build_poi_features()

        if args.step in ("transit", "both"):
            logger.info("=== TRANSIT FEATURES ===")
            build_transit_features()

        logger.info("Merging into master feature table…")
        from src.features.feature_store import merge_all_features
        merge_all_features(rebuild=True)
 
        logger.success("Week 3 complete.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)


