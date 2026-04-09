"""
tests/test_week2_revised.py
────────────────────────────
Tests for the revised Week 2 pipeline:
  - edge_preprocessing: artifact detection, ghost road classification, speed regimes
  - network_features: length-weighted aggregation correctness
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_edges(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a minimal edge GDF from a list of dicts."""
    defaults = {
        'u': 0, 'v': 1, 'length': 50.0,
        'highway': 'residential', 'hw_norm': 'residential',
        'name': None, 'oneway': False, 'maxspeed': None,
        'bridge': None, 'tunnel': None, 'service': None,
        'geometry': LineString([(0, 0), (50, 0)]),
    }
    records = [{**defaults, **r} for r in rows]
    return gpd.GeoDataFrame(records, crs='EPSG:26916')


# ── edge_preprocessing tests ───────────────────────────────────────────────────

class TestSpeedRegimes:
    def test_footway_gets_regime_5(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        e = _make_edges([{'highway': 'footway', 'hw_norm': 'footway', 'edge_role': 'true_footway'}])
        result = infer_speed_regimes(e)
        assert result['speed_regime'].iloc[0] == 5

    def test_residential_gets_regime_3(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        e = _make_edges([{'highway': 'residential', 'hw_norm': 'residential', 'edge_role': 'road'}])
        result = infer_speed_regimes(e)
        assert result['speed_regime'].iloc[0] == 3

    def test_measured_maxspeed_overrides_type(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        # 64 kph (40mph) → regime 2
        e = _make_edges([{
            'highway': 'residential', 'hw_norm': 'residential',
            'maxspeed': '40 mph', 'edge_role': 'road'
        }])
        result = infer_speed_regimes(e)
        assert result['speed_confidence'].iloc[0] == 'measured'
        assert result['speed_regime'].iloc[0] == 2

    def test_low_speed_measured_gives_high_regime(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        e = _make_edges([{
            'highway': 'service', 'hw_norm': 'service',
            'maxspeed': '10', 'edge_role': 'true_alley'   # 10 kph → regime 5
        }])
        result = infer_speed_regimes(e)
        assert result['speed_regime'].iloc[0] == 5
        assert result['speed_confidence'].iloc[0] == 'measured'

    def test_alley_gets_regime_4(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        e = _make_edges([{'highway': 'service', 'hw_norm': 'service', 'edge_role': 'true_alley'}])
        result = infer_speed_regimes(e)
        assert result['speed_regime'].iloc[0] == 4
        assert abs(result['inferred_speed_kph'].iloc[0] - 16.0) < 0.1

    def test_trunk_gets_regime_1(self):
        from src.features.edge_preprocessing import infer_speed_regimes
        e = _make_edges([{'highway': 'trunk', 'hw_norm': 'trunk', 'edge_role': 'road'}])
        result = infer_speed_regimes(e)
        assert result['speed_regime'].iloc[0] == 1

    def test_mph_string_parsed_correctly(self):
        from src.features.edge_preprocessing import _parse_maxspeed
        assert abs(_parse_maxspeed('25 mph') - 40.234) < 0.1
        assert _parse_maxspeed('walk') is None
        assert _parse_maxspeed(None) is None
        assert _parse_maxspeed(float('nan')) is None
        assert abs(_parse_maxspeed('50') - 50.0) < 0.1


class TestArtifactDetection:
    def test_very_short_edge_at_non_intersection_is_artifact(self):
        from src.features.edge_preprocessing import detect_artifacts
        e = _make_edges([{
            'u': 10, 'v': 11,
            'length': 0.5,   # short
            'geometry': LineString([(0, 0), (0.5, 0)]),
        }])
        # Both nodes have degree 2 — not real intersections
        node_degrees = {10: 2, 11: 2}
        result = detect_artifacts(e, node_degrees, min_length_m=2.0)
        assert result['is_artifact'].iloc[0] == True

    def test_short_edge_at_intersection_is_kept(self):
        from src.features.edge_preprocessing import detect_artifacts
        e = _make_edges([{
            'u': 10, 'v': 11,
            'length': 1.0,
            'geometry': LineString([(0, 0), (1, 0)]),
        }])
        # Node 10 has degree 4 — real intersection
        node_degrees = {10: 4, 11: 2}
        result = detect_artifacts(e, node_degrees, min_length_m=2.0)
        assert result['is_artifact'].iloc[0] == False

    def test_long_edge_never_artifact(self):
        from src.features.edge_preprocessing import detect_artifacts
        e = _make_edges([{'u': 10, 'v': 11, 'length': 100.0}])
        node_degrees = {10: 1, 11: 1}
        result = detect_artifacts(e, node_degrees, min_length_m=2.0)
        assert result['is_artifact'].iloc[0] == False


class TestEdgeRoles:
    def test_named_edge_is_road(self):
        from src.features.edge_preprocessing import classify_edge_roles
        e = _make_edges([{
            'u': 1, 'v': 2,
            'name': 'Michigan Avenue',
            'hw_norm': 'residential',
            'is_artifact': False,
        }])
        result = classify_edge_roles(e, {1: 3, 2: 3})
        assert result['edge_role'].iloc[0] == 'road'

    def test_unnamed_service_with_named_neighbors_is_alley(self):
        from src.features.edge_preprocessing import classify_edge_roles
        # Edge 1→2 unnamed service. Node 1 connects to 'State St', node 2 to 'Wabash Ave'
        e_main = _make_edges([
            {'u': 0, 'v': 1, 'name': 'State Street',  'hw_norm': 'residential', 'is_artifact': False},
            {'u': 2, 'v': 3, 'name': 'Wabash Avenue', 'hw_norm': 'residential', 'is_artifact': False},
            {'u': 1, 'v': 2, 'name': None, 'hw_norm': 'service', 'is_artifact': False},
        ])
        node_degrees = {0: 3, 1: 3, 2: 3, 3: 3}
        result = classify_edge_roles(e_main, node_degrees)
        # The unnamed service edge (row 2) should be classified as true_alley
        assert result.iloc[2]['edge_role'] == 'true_alley'

    def test_unnamed_footway_is_true_footway(self):
        from src.features.edge_preprocessing import classify_edge_roles
        e = _make_edges([{
            'u': 10, 'v': 11,
            'name': None,
            'hw_norm': 'footway',
            'is_artifact': False,
        }])
        result = classify_edge_roles(e, {10: 1, 11: 1})
        assert result['edge_role'].iloc[0] == 'true_footway'


# ── network_features: length-weighted aggregation tests ───────────────────────

class TestLengthWeightedAggregation:
    def test_lw_proportion_weights_by_length(self):
        from src.features.network_features import _lw_proportion
        # 10 short edges (1m) + 1 long edge (100m), only long one is footway
        mask    = pd.Series([False]*10 + [True])
        weights = pd.Series([1.0]*10 + [100.0])
        prop = _lw_proportion(mask, weights)
        # footway = 100 / (10 + 100) = 0.909
        assert abs(prop - 100/110) < 0.001

    def test_lw_proportion_vs_naive_count(self):
        """Demonstrate that LW and naive proportions differ when lengths vary."""
        from src.features.network_features import _lw_proportion
        # 9 short non-footway + 1 long footway
        mask    = pd.Series([False]*9 + [True])
        weights = pd.Series([1.0]*9 + [1000.0])
        lw_prop    = _lw_proportion(mask, weights)
        naive_prop = mask.mean()
        # naive = 0.1; lw = 1000/1009 ≈ 0.991
        assert lw_prop > 0.9
        assert naive_prop == pytest.approx(0.1)
        assert lw_prop > naive_prop * 5   # lw correctly shows footway dominates by length

    def test_lw_mean_returns_nan_on_zero_weight(self):
        from src.features.network_features import _lw_mean
        result = _lw_mean(pd.Series([1.0, 2.0]), pd.Series([0.0, 0.0]))
        assert np.isnan(result)

    def test_lw_mean_simple(self):
        from src.features.network_features import _lw_mean
        # 1 long edge (100m, regime 5) + 1 short edge (10m, regime 1)
        values  = pd.Series([5.0, 1.0])
        weights = pd.Series([100.0, 10.0])
        result  = _lw_mean(values, weights)
        expected = (5*100 + 1*10) / 110
        assert abs(result - expected) < 0.001