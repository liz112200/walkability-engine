from __future__ import annotations

import pandas as pd
import geopandas as gpd

from loguru import logger

MIN_EDGE_M = 2.0
HOSTILE_TYPES = frozenset({
    'motorway', 'motorway_link', 'trunk', 'trunk_link'
})

SPEED_REGIME_MAP: dict[str, tuple[int, float, str]] = {
    # highway_type → (regime, inferred_kph, confidence_tier)
    # ── Pedestrian-primary (regime 5) ────────────────────────────────────────
    'pedestrian':   (5, 5.0,  'type_inferred'),
    'footway':      (5, 5.0,  'type_inferred'),
    'path':         (5, 8.0,  'type_inferred'),
    'steps':        (5, 3.0,  'type_inferred'),
    'cycleway':     (5, 15.0, 'type_inferred'),  
    'track':        (4, 15.0, 'type_inferred'),

    # ── Low-speed shared space (regime 4) ────────────────────────────────────
    'living_street':(4, 15.0, 'type_inferred'),
    'service':      (4, 15.0, 'type_inferred'),  
    # ── Standard residential (regime 3) ──────────────────────────────────────
    'residential':  (3, 25.0, 'type_inferred'),
    'unclassified': (3, 25.0, 'type_inferred'),
    'road':         (3, 30.0, 'type_inferred'),
    # ── Collector / minor arterial (regime 2) ────────────────────────────────
    'tertiary':     (2, 40.0, 'type_inferred'),
    'tertiary_link':(2, 40.0, 'type_inferred'),
    'secondary':    (2, 50.0, 'type_inferred'),
    'secondary_link':(2,50.0, 'type_inferred'),
    'primary':      (2, 55.0, 'type_inferred'),
    'primary_link': (2, 55.0, 'type_inferred'),
    # ── Hostile (should be removed before this point) ────────────────────────
    'trunk':        (1, 80.0, 'type_inferred'),
    'trunk_link':   (1, 80.0, 'type_inferred'),
    'motorway':     (1, 100.0,'type_inferred'),
    'motorway_link':(1, 100.0,'type_inferred'),
}


# Map real maxspeed to its regime
def _kph_to_regime(kph: float) -> int:
    if kph <= 10:  return 5
    if kph <= 20:  return 5
    if kph <= 30:  return 4
    if kph <= 48:  return 3   # 30mph
    if kph <= 65:  return 2   # 40mph
    return 1
 
def _parse_maxspeed(raw) -> float | None:
    """Parse OSM maxspeed to kph float. Returns None if unparseable.
 
    Handles all OSMnx output formats:
      - plain string:  '30 mph', '50'
      - list of strings: ['30 mph', '35 mph']  (multi-way edges)
      - None / NaN
    When a list is received we take the minimum value — conservative
    (pedestrian-safety) estimate for a segment with mixed speed limits.
    """
    if isinstance(raw, list):
        parsed = [_parse_maxspeed(v) for v in raw]
        valid  = [v for v in parsed if v is not None]
        return min(valid) if valid else None
 
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass 
 
    s = str(raw).strip().lower()
    if not s or s in ('nan', 'none', 'walk', 'signals', 'variable'):
        return None
    if 'mph' in s:
        try:
            return float(s.replace('mph', '').strip()) * 1.60934
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None
    

# ── Operation 1: Remove hostile edges ─────────────────────────────────────────
def remove_hostile_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    hw = edges['hw_norm'] if 'hw_norm' in edges.columns else \
            edges['highway'].apply(lambda x: str(x).split('|')[0].strip())
    
    hostile_mask = hw.isin(HOSTILE_TYPES)
    n_hostiles = hostile_mask.sum()

    if n_hostiles > 0:
        logger.info(
            f"Removing {n_hostiles:,} hostile edges "
            f"({hostile_mask.value_counts().to_dict()}) — pedestrian-inaccessible"
        )
    else:
        logger.info("No hostile edges found.")

    clean = edges[~hostile_mask].copy()
    clean['is_hostile'] = False
    return clean


# ── Operation 2: Micro-artifact detection ─────────────────────────────────────
def detect_artifacts(
    edges: gpd.GeoDataFrame,
    node_degrees: dict,
    min_length_m: float = MIN_EDGE_M,
) -> gpd.GeoDataFrame:
    
    edges = edges.copy()
 
    # ── Expose u and v as plain columns ───────────────────────────────────────
    # OSMnx sets (u, v, key) as a MultiIndex. reset_index() promotes them to
    # regular columns so vectorised .map() can reach them.
    original_index_names = list(edges.index.names)
    had_uv_index = 'u' in original_index_names
    if had_uv_index:
        edges = edges.reset_index()
 
    short_mask = edges['length'] < min_length_m
    n_short = short_mask.sum()
    logger.info(
        f"Edges shorter than {min_length_m}m: {n_short:,} "
        f"({100*n_short/len(edges):.1f}%)"
    )
 
    # ── Vectorised artifact detection (replaces slow row-wise .apply()) ───────
    logger.info("Classifying short edges by topological role…")
    u_deg = edges['u'].map(node_degrees).fillna(0).astype(int)
    v_deg = edges['v'].map(node_degrees).fillna(0).astype(int)
    # Artifact = short AND neither endpoint is a real intersection (degree >= 3)
    edges['is_artifact'] = short_mask & (u_deg < 3) & (v_deg < 3)
 
    n_artifact = int(edges['is_artifact'].sum())
    n_legitimate_short = n_short - n_artifact
    logger.info(f"  Artifacts (short + no intersection endpoint): {n_artifact:,}")
    logger.info(f"  Legitimate short edges (at real intersections): {n_legitimate_short:,}")
 
    # ── Restore original MultiIndex ────────────────────────────────────────────
    if had_uv_index:
        edges = edges.set_index(original_index_names)
 
    return edges
 

# ── Operation 3: Ghost road / alley classification ────────────────────────────
def classify_edge_roles(
    edges: gpd.GeoDataFrame,
    node_degrees: dict,
    snap_distance_m: float = 8.0,
) -> gpd.GeoDataFrame:
 
    edges = edges.copy()
 
    hw = edges['hw_norm'] if 'hw_norm' in edges.columns else \
         edges['highway'].apply(lambda x: str(x).split('|')[0].strip())
 
    has_name = (
        edges['name'].notna() &
        (edges['name'].astype(str).str.strip() != '') &
        (edges['name'].astype(str).str.lower() != 'nan')
    ) if 'name' in edges.columns else pd.Series(False, index=edges.index)
 
    # ── Expose u and v as plain columns (same pattern as detect_artifacts) ──────
    original_index_names = list(edges.index.names)
    had_uv_index = 'u' in original_index_names
    if had_uv_index:
        edges = edges.reset_index()
 
    # ── Build node → set-of-edge-names lookup ────────────────────────────────
    logger.info("Building node → edge-name index for ghost road detection…")
    node_to_names: dict[int, set[str]] = {}
    named_edges = edges[has_name.values if had_uv_index else has_name]
    for _, row in named_edges.iterrows():
        name_val = str(row['name']).strip()
        for nid in [row['u'], row['v']]:
            if pd.notna(nid):
                node_to_names.setdefault(int(nid), set()).add(name_val)
 
    # ── Vectorised role assignment ─────────────────────────────────────────────
    # Build Series for each component, then combine with np.select — much faster
    # than row-wise .apply() on 1M edges.
    import numpy as np
 
    hw_types  = edges['hw_norm'] if 'hw_norm' in edges.columns else                 edges['highway'].apply(lambda x: str(x).split('|')[0].strip())
    edge_name = edges['name'].fillna('') if 'name' in edges.columns                 else pd.Series('', index=edges.index)
    is_named  = edge_name.astype(str).str.strip().str.lower().replace('nan', '').ne('')
    is_artifact = edges.get('is_artifact', pd.Series(False, index=edges.index))
 
    u_names_series = edges['u'].map(lambda n: node_to_names.get(int(n), set())
                                    if pd.notna(n) else set())
    v_names_series = edges['v'].map(lambda n: node_to_names.get(int(n), set())
                                    if pd.notna(n) else set())
 
    has_u_names  = u_names_series.map(bool)
    has_v_names  = v_names_series.map(bool)
    shared_names = [bool(u & v) for u, v in zip(u_names_series, v_names_series)]
    shared_names = pd.Series(shared_names, index=edges.index)
 
    is_footway_type = hw_types.isin(
        {'footway', 'path', 'pedestrian', 'steps', 'cycleway', 'track'}
    )
    is_service_type = hw_types == 'service'
 
    conditions = [
        is_artifact.astype(bool),                              # artifact
        is_named,                                              # road
        shared_names,                                          # road_fragment
        is_service_type,                                       # true_alley
        is_footway_type,                                       # true_footway
        (~is_named) & has_u_names & has_v_names & ~shared_names, # road_fragment (unnamed residential)
    ]
    choices = [
        'artifact',
        'road',
        'road_fragment',
        'true_alley',
        'true_footway',
        'road_fragment',
    ]
    logger.info("Classifying edge roles (road / true_alley / road_fragment / artifact)…")
    edges['edge_role'] = np.select(conditions, choices, default='road')
 
    # ── Restore original MultiIndex ────────────────────────────────────────────
    if had_uv_index:
        edges = edges.set_index(original_index_names)
 
    role_counts = edges['edge_role'].value_counts()
    logger.info("Edge role breakdown:")
    for role, count in role_counts.items():
        logger.info(f"  {role:<20} {count:>9,}  ({100*count/len(edges):.1f}%)")
 
    return edges
 
         
# ── Operation 4: Speed regime inference ───────────────────────────────────────
 
def infer_speed_regimes(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    edges = edges.copy()
    hw = edges['hw_norm'] if 'hw_norm' in edges.columns else \
         edges['highway'].apply(lambda x: str(x).split('|')[0].strip())
 
    regimes = []
    speeds  = []
    confidences = []
 
    maxspeed_col = edges.get('maxspeed', pd.Series(None, index=edges.index))
 
    for idx, row in edges.iterrows():
        hw_type = str(hw.loc[idx]).split('|')[0].strip()
        raw_speed = maxspeed_col.loc[idx] if idx in maxspeed_col.index else None
        edge_role = row.get('edge_role', 'road')
 
        # Priority 1: measured tag
        kph = _parse_maxspeed(raw_speed)
        if kph is not None:
            regime = _kph_to_regime(kph)
            regimes.append(regime)
            speeds.append(kph)
            confidences.append('measured')
            continue
 
        # Priority 2: type taxonomy
        if hw_type in SPEED_REGIME_MAP:
            regime, inferred_kph, conf = SPEED_REGIME_MAP[hw_type]
 
            if hw_type == 'service' and edge_role == 'true_alley':
                regime = 4         
                inferred_kph = 16.0
 
            regimes.append(regime)
            speeds.append(inferred_kph)
            confidences.append(conf)
            continue
 
        # Priority 3: default
        regimes.append(3)
        speeds.append(25.0)
        confidences.append('default')
 
    edges['speed_regime']       = regimes
    edges['inferred_speed_kph'] = speeds
    edges['speed_confidence']   = confidences
 
    # Summary
    conf_counts = pd.Series(confidences).value_counts()
    logger.info("Speed regime assignment:")
    for conf, count in conf_counts.items():
        logger.info(f"  {conf:<20} {count:>9,}  ({100*count/len(edges):.1f}%)")
    logger.info(f"  Regime distribution:\n{pd.Series(regimes).value_counts().sort_index().to_string()}")
 
    return edges
 




# ── Master preprocessing pipeline ─────────────────────────────────────────────
def preprocess_edges(
        edges: gpd.GeoDataFrame,
        node_degrees: dict
) -> gpd.GeoDataFrame:
    
    logger.info(f"Starting edge preprocessing — {len(edges):,} input edges")


    # if 'hw_norm' not in edges:
    #     edges = edges.copy()
    #     edges['hw_norm'] = edges['highway'].apply(
    #         lambda x: str(x).split('|')[0].strip() if pd.notna(x) else 'unknown'
    #     )

    if 'hw_norm' not in edges.columns:
        edges = edges.copy()
 
        def _normalise_highway(x) -> str:
            if x is None:
                return 'unknown'
            if isinstance(x, list):
                return str(x[0]).strip() if x else 'unknown'
            s = str(x).strip()
            if s in ('', 'nan', 'None'):
                return 'unknown'
            return s.split('|')[0].strip()
 
        edges['hw_norm'] = edges['highway'].apply(_normalise_highway)


    edges = remove_hostile_edges(edges)
    edges = detect_artifacts(edges, node_degrees)
    edges = classify_edge_roles(edges, node_degrees)
    edges = infer_speed_regimes(edges)

    logger.info(
        f"Preprocessing complete — {len(edges):,} edges remain "
        f"({edges['is_artifact'].sum():,} artifacts flagged, "
        f"kept in table but excluded from feature aggregation)"
    )

    return edges