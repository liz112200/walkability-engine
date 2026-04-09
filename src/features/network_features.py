"""
Key design principles (revised after data exploration findings)
---------------------------------------------------------------
1. LENGTH-WEIGHTED aggregation everywhere
   Simple edge counts are misleading when the median edge is 19m but some
   edges are 2,500m. Every proportion and every mean is weighted by
   edge length so that a 100m street segment counts 100x more than a 1m
   artifact toward the hex's feature values.

2. No imputed continuous values
   maxspeed / lanes / width remain NaN for untagged edges. We never fill
   them. Instead, we aggregate the SPEED REGIME (1–5 ordinal, assigned by
   edge_preprocessing.py) using a length-weighted mean — giving a genuine
   "what is the typical pedestrian speed environment here?" signal.

3. Edge roles drive feature groups
   After preprocessing, every edge carries an edge_role:
     'road', 'true_alley', 'true_footway', 'road_fragment', 'artifact'
   Aggregation excludes artifacts and uses role-specific breakdowns.

4. "Other" category is decomposed, not lumped
   The ~25% "other" edges are classified in preprocessing. By the time
   we aggregate, there is no mystery bucket — only named roles.

Feature catalogue (25 features after revision)
------------------------------------------------
Topology / connectivity (7)
    intersection_density       Real intersections per km²
    dead_end_ratio             Fraction of nodes that are dead ends
    avg_node_degree            Mean node degree
    prop_4way_intersections    Fraction of intersections that are 4-way
    prop_3way_intersections    Fraction of intersections that are 3-way
    node_density_per_km2       Nodes per km²
    avg_circuity               Mean route/straight-line distance ratio

Street geometry — length-weighted (5)
    lw_avg_edge_length_m       Length-weighted mean edge length
    total_street_length_m      Total street length (excl. artifacts) in metres
    street_density_m_per_km2   Total length / hex area in m/km²
    edge_length_cv             Coefficient of variation (heterogeneity index)
    avg_block_length_m         Estimated block length

Edge type composition — length-weighted proportions (6)
    lw_prop_footway            Fraction of street-metres that are footway/path/ped
    lw_prop_residential        Fraction of street-metres that are residential
    lw_prop_alley              Fraction of street-metres that are true alleys
    lw_prop_road_fragment      Fraction of metres that are road fragments
    lw_prop_named              Fraction of street-metres with a name tag
    lw_prop_oneway             Fraction of street-metres that are one-way

Speed environment — length-weighted (4)
    lw_avg_speed_regime        Length-weighted mean speed regime (1=hostile, 5=safe)
    lw_prop_regime_5           Fraction of metres in regime 5 (dedicated ped space)
    lw_prop_regime_3plus       Fraction of metres in regime 3–5 (safe or moderate)
    lw_prop_measured_speed     Fraction of metres with actual OSM maxspeed tag

Hazard presence — edge-count proportions (3)
    prop_has_speed_limit       Fraction of edges with OSM maxspeed (arterial signal)
    prop_is_bridge             Fraction of edges tagged bridge
    prop_is_tunnel             Fraction of edges tagged tunnel
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.features.edge_preprocessing import preprocess_edges
from src.utils.config import cfg
from src.utils.h3_grid import city_hex_grid

MIN_STREET_LENGTH_M = 10.0
MIN_EDGES_PER_HEX   = 3

FOOTWAY_TYPES  = frozenset({'footway', 'path', 'pedestrian', 'steps', 'cycleway', 'track'})
RESIDENTIAL_T  = frozenset({'residential', 'living_street', 'unclassified', 'service'})


# ── Length-weighted aggregation helpers ───────────────────────────────────────

def _lw_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0)
    total_w = w.sum()
    if total_w == 0:
        return np.nan
    return float((values * w).sum() / total_w)


def _lw_proportion(mask: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0)
    total_w = w.sum()
    if total_w == 0:
        return np.nan
    return float(w[mask.fillna(False)].sum() / total_w)


# ── Per-hex feature computation ────────────────────────────────────────────────

def _compute_hex_features(
    e: gpd.GeoDataFrame,
    n_df: gpd.GeoDataFrame,
    hex_area_km2: float,
) -> dict[str, float]:

    feat: dict[str, float] = {}

    n_edges = len(e)
    n_nodes = len(n_df)
    total_length = float(e['length'].sum()) if n_edges > 0 else 0.0

    feat['n_real_edges']          = float(n_edges)
    feat['n_nodes_in_hex']        = float(n_nodes)
    feat['total_street_length_m'] = total_length
    feat['data_sparse'] = float(
        n_edges < MIN_EDGES_PER_HEX or total_length < MIN_STREET_LENGTH_M
    )

    _ALL_FEATURES = [
        'intersection_density', 'dead_end_ratio', 'avg_node_degree',
        'prop_4way_intersections', 'prop_3way_intersections',
        'node_density_per_km2', 'avg_circuity',
        'lw_avg_edge_length_m', 'street_density_m_per_km2',
        'edge_length_cv', 'avg_block_length_m',
        'lw_prop_footway', 'lw_prop_residential', 'lw_prop_alley',
        'lw_prop_road_fragment', 'lw_prop_named', 'lw_prop_oneway',
        'lw_avg_speed_regime', 'lw_prop_regime_5',
        'lw_prop_regime_3plus', 'lw_prop_measured_speed',
        'prop_has_speed_limit', 'prop_is_bridge', 'prop_is_tunnel',
    ]
    if feat['data_sparse']:
        for f in _ALL_FEATURES:
            feat[f] = np.nan
        return feat

    lengths = e['length'].fillna(0.0)

    # ── 1. Topology / connectivity ────────────────────────────────────────────
    if n_nodes > 0:
        degrees  = n_df['_degree'].values if '_degree' in n_df.columns else                    [0] * n_nodes
        n_dead   = int((degrees == 1).sum())
        n_4way   = int((degrees >= 4).sum())
        n_3way   = int((degrees == 3).sum())
        n_inters = int((degrees >= 3).sum())

        feat['intersection_density']    = n_inters / max(hex_area_km2, 1e-9)
        feat['dead_end_ratio']          = n_dead / n_nodes
        feat['avg_node_degree']         = float(np.mean(degrees))
        feat['prop_4way_intersections'] = n_4way / max(n_inters, 1)
        feat['prop_3way_intersections'] = n_3way / max(n_inters, 1)
        feat['node_density_per_km2']    = n_nodes / max(hex_area_km2, 1e-9)
    else:
        for f in ['intersection_density', 'dead_end_ratio', 'avg_node_degree',
                  'prop_4way_intersections', 'prop_3way_intersections',
                  'node_density_per_km2']:
            feat[f] = np.nan

    if '_circuity' in e.columns:
        valid = e[e['_circuity'].notna()]
        feat['avg_circuity'] = (
            _lw_mean(valid['_circuity'], valid['length'])
            if len(valid) > 0 else np.nan
        )
    else:
        feat['avg_circuity'] = np.nan

    # ── 2. Street geometry ────────────────────────────────────────────────────
    feat['lw_avg_edge_length_m']     = _lw_mean(lengths, lengths)
    feat['street_density_m_per_km2'] = total_length / max(hex_area_km2, 1e-9)

    # Coefficient of variation = std/mean — measures how heterogeneous the
    # street network is. High CV = mix of micro-paths and long arterials.
    mean_len = lengths.mean()
    feat['edge_length_cv'] = (
        float(lengths.std() / mean_len)
        if len(lengths) > 1 and mean_len > 0 else 0.0
    )

    n_int_approx = max(
        feat.get('intersection_density', 0) * hex_area_km2, 1.0
    )
    feat['avg_block_length_m'] = total_length / n_int_approx

    # ── 3. Edge type composition — length-weighted ────────────────────────────
    roles    = e.get('edge_role', pd.Series('road', index=e.index))
    hw_types = e.get('hw_norm',
        e['highway'].apply(lambda x: str(x).split('|')[0].strip()
                           if pd.notna(x) else 'unknown'))

    feat['lw_prop_footway']     = _lw_proportion(hw_types.isin(FOOTWAY_TYPES), lengths)
    feat['lw_prop_residential'] = _lw_proportion(
        hw_types.isin(RESIDENTIAL_T) & (roles != 'true_alley'), lengths
    )
    feat['lw_prop_alley']         = _lw_proportion(roles == 'true_alley', lengths)
    feat['lw_prop_road_fragment'] = _lw_proportion(roles == 'road_fragment', lengths)

    if 'name' in e.columns:
        has_name = (
            e['name'].notna() &
            (e['name'].astype(str).str.strip() != '') &
            (e['name'].astype(str).str.lower() != 'nan')
        )
        feat['lw_prop_named'] = _lw_proportion(has_name, lengths)
    else:
        feat['lw_prop_named'] = np.nan

    if 'oneway' in e.columns:
        is_oneway = e['oneway'].astype(str).isin(['True', 'true', '1', 'yes'])
        feat['lw_prop_oneway'] = _lw_proportion(is_oneway, lengths)
    else:
        feat['lw_prop_oneway'] = np.nan

    # ── 4. Speed environment — length-weighted ────────────────────────────────
    if 'speed_regime' in e.columns:
        regimes = e['speed_regime'].fillna(3).astype(float)
        feat['lw_avg_speed_regime']   = _lw_mean(regimes, lengths)
        feat['lw_prop_regime_5']      = _lw_proportion(regimes == 5, lengths)
        feat['lw_prop_regime_3plus']  = _lw_proportion(regimes >= 3, lengths)
        conf = e.get('speed_confidence', pd.Series('default', index=e.index))
        feat['lw_prop_measured_speed'] = _lw_proportion(
            conf == 'measured', lengths
        )
    else:
        for f in ['lw_avg_speed_regime', 'lw_prop_regime_5',
                  'lw_prop_regime_3plus', 'lw_prop_measured_speed']:
            feat[f] = np.nan

    # ── 5. Hazard presence — edge-count proportions ───────────────────────────
    # These use count (not length) because the question is
    # "does any edge in this hex have a speed sign posted?" — not
    # "how many metres of speed-signed road are there?"
    n = max(n_edges, 1)
    feat['prop_has_speed_limit'] = (
        e['maxspeed'].notna().sum() / n
        if 'maxspeed' in e.columns else 0.0
    )
    feat['prop_is_bridge'] = (
        e['bridge'].notna().sum() / n
        if 'bridge' in e.columns else 0.0
    )
    feat['prop_is_tunnel'] = (
        e['tunnel'].notna().sum() / n
        if 'tunnel' in e.columns else 0.0
    )

    return feat


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_network_features(
    graphml_path: Path | None = None,
    out_path:     Path | None = None,
    resolution:   int  | None = None,
) -> gpd.GeoDataFrame:
    """Full pipeline: load → preprocess → H3 grid → features → save."""
    slug = cfg.city.slug
    res  = resolution or cfg.h3.resolution

    graphml_path = graphml_path or (cfg.paths.raw_osm / f"{slug}_walk_graph.graphml")
    out_path     = out_path     or (cfg.paths.processed / f"{slug}_network_features.parquet")
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)

    assert graphml_path.exists(), (
        f"GraphML not found: {graphml_path}\n"
        "Run src/ingestion/fetch_osm_network.py first."
    )

    logger.info(f"Loading graph from {graphml_path.name}…")
    t0 = time.perf_counter()
    G  = ox.load_graphml(str(graphml_path))
    logger.info(
        f"Loaded in {time.perf_counter()-t0:.1f}s — "
        f"{len(G.nodes):,} nodes, {len(G.edges):,} edges"
    )

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    def _first_value(x) -> str:
        """Return the first element of a list, pipe-string, or plain string.
        Handles all OSMnx output formats safely."""
        if x is None:
            return ''
        if isinstance(x, list):
            return str(x[0]).strip() if x else ''
        s = str(x).strip()
        if s in ('nan', 'None'):
            return ''
        return s.split('|')[0].strip()

    # Normalise multi-value columns — OSMnx may return lists OR pipe-strings
    for col in ['highway', 'name', 'service', 'access']:
        if col in edges_gdf.columns:
            edges_gdf[col] = edges_gdf[col].apply(_first_value)

    node_degrees = dict(G.degree())

    logger.info("Running edge preprocessing…")
    edges_proc  = preprocess_edges(edges_gdf, node_degrees)
    edges_clean = edges_proc[~edges_proc['is_artifact']].copy()
    logger.info(
        f"Clean edges: {len(edges_clean):,} "
        f"({edges_proc['is_artifact'].sum():,} artifacts excluded)"
    )

    logger.info(f"Building H3 res-{res} grid…")
    hex_gdf = city_hex_grid(edges_clean, resolution=res)
    hex_areas_km2 = hex_gdf.geometry.area / 1e6
    logger.info(f"Grid: {len(hex_gdf):,} cells")

    # ── Spatial join: assign every edge and node to its hex ONCE ────────────────
    logger.info("Spatial join: assigning edges to hex cells…")
    t_feat = time.perf_counter()

    # Use edge centroids for the join (faster than full geometry intersection)
    edges_for_join = edges_clean.copy()
    edges_for_join.geometry = edges_clean.geometry.centroid
    edge_to_hex = gpd.sjoin(
        edges_for_join[["geometry"]],
        hex_gdf[["h3_index", "geometry"]],
        how="left",
        predicate="within",
    )["h3_index"]
    edges_clean = edges_clean.copy()
    edges_clean["h3_index"] = edge_to_hex.values

    nodes_for_join = nodes_gdf.copy()
    node_to_hex = gpd.sjoin(
        nodes_for_join[["geometry"]],
        hex_gdf[["h3_index", "geometry"]],
        how="left",
        predicate="within",
    )["h3_index"]
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["h3_index"] = node_to_hex.values

    logger.info(
        f"Spatial join complete in {time.perf_counter()-t_feat:.1f}s  |  "
        f"Edges assigned: {edges_clean['h3_index'].notna().sum():,}  |  "
        f"Nodes assigned: {nodes_gdf['h3_index'].notna().sum():,}"
    )

    # Pre-vectorise circuity for all edges (avoid per-row geometry iteration)
    logger.info("Pre-computing circuity for all edges…")
    def _edge_circuity(geom):
        if geom is None:
            return np.nan
        coords = list(geom.coords)
        if len(coords) < 2:
            return np.nan
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        straight = (dx*dx + dy*dy) ** 0.5
        return float(geom.length / straight) if straight > 0 else np.nan

    edges_clean = edges_clean.copy()
    edges_clean["_circuity"] = edges_clean.geometry.apply(_edge_circuity)

    # Pre-map node degrees onto the node GDF
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["_degree"] = nodes_gdf.index.map(node_degrees).fillna(0).astype(int)

    # ── Per-hex feature computation (now over grouped subsets, not full GDF) ───
    logger.info("Computing per-hex features (length-weighted)…")
    t_loop = time.perf_counter()

    rows = []
    for i, (_, hr) in enumerate(
        tqdm(hex_gdf.iterrows(), total=len(hex_gdf), desc="Hex cells")
    ):
        hid = hr["h3_index"]
        e   = edges_clean[edges_clean["h3_index"] == hid]
        n_df = nodes_gdf[nodes_gdf["h3_index"] == hid]
        rows.append(_compute_hex_features(
            e            = e,
            n_df         = n_df,
            hex_area_km2 = float(hex_areas_km2.iloc[i]),
        ))

    feats_df = pd.DataFrame(rows)
    logger.info(f"Features computed in {time.perf_counter()-t_loop:.1f}s")

    result = hex_gdf.copy()
    for col in feats_df.columns:
        result[col] = feats_df[col].values

    result.to_parquet(str(out_path))
    logger.info(
        f"Saved → {out_path.relative_to(cfg.project_root)}  "
        f"({out_path.stat().st_size/1e6:.1f} MB)"
    )
    _summary(result)
    return result


def _summary(gdf: gpd.GeoDataFrame) -> None:
    exclude = {'h3_index', 'geometry', 'centroid_x', 'centroid_y',
               'centroid_lat', 'centroid_lng', 'data_sparse',
               'n_real_edges', 'n_nodes_in_hex'}
    cols  = [c for c in gdf.columns if c not in exclude]
    dense = gdf[gdf['data_sparse'] == 0]
    logger.info("=" * 68)
    logger.info("FEATURE SUMMARY  (dense cells only)")
    logger.info(f"{'Feature':<38} {'Mean':>8} {'Median':>8} {'NaN%':>6}")
    logger.info("-" * 68)
    for col in cols:
        s = dense[col].dropna()
        nan_pct = 100 * dense[col].isna().mean()
        if len(s):
            logger.info(
                f"{col:<38} {s.mean():>8.3f} {s.median():>8.3f} {nan_pct:>5.1f}%"
            )
    logger.info("=" * 68)


if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/network_features.log", rotation="10 MB", level="DEBUG")
    try:
        result = build_network_features()
        logger.success(f"Week 2 complete — {len(result):,} hex cells")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)
