"""
src/dashboard/api_client.py
----------------------------
All data-fetching functions for the dashboard.

  fetch_*        — call the FastAPI backend (cached per Streamlit session)
  geocode        — Nominatim (OpenStreetMap) address → (lat, lng, display_name)
  fetch_by_latlng — thin wrapper around /predict/{lat}/{lng}

Named api_client (not api) to avoid shadowing the src/api/ package.
"""
from __future__ import annotations

import requests
import pandas as pd
import streamlit as st
from typing import Optional

from src.dashboard.config import API_URL


# ── Backend endpoints ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_city_summary() -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/city/summary", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Cannot reach API at {API_URL}. Is the server running?  ({e})")
        return None


@st.cache_data(ttl=300)
def fetch_hex(h3_index: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/hex/{h3_index}", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Could not fetch hex {h3_index}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_neighbourhood(name: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}/neighbourhood/{name}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Could not fetch neighbourhood '{name}': {e}")
        return None


@st.cache_data(ttl=300)
def fetch_compare(hex_a: str, hex_b: str) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_URL}/compare",
            json={"hex_a": hex_a, "hex_b": hex_b},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Could not compare hexes: {e}")
        return None


@st.cache_data(ttl=600)
def fetch_all_hex_scores() -> pd.DataFrame:
    """
    Load predicted scores for all hexes directly from parquet.
    The backend has no bulk endpoint; calling /hex 5 000+ times is too slow.
    """
    try:
        df = pd.read_parquet("data/processed/shap_values.parquet")
        master = pd.read_parquet("data/processed/master_features.parquet")[
            ["h3_index", "centroid_lat", "centroid_lng"]
        ]
        df = df.merge(master, on="h3_index", how="left")
        df = df[df["centroid_lat"].notna()]
        return df[["h3_index", "predicted_score", "walk_score",
                   "centroid_lat", "centroid_lng"]].copy()
    except Exception as e:
        st.error(f"Could not load hex scores: {e}")
        return pd.DataFrame()


# ── Geocoding ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def geocode(query: str) -> Optional[tuple[float, float, str]]:
    """
    Convert a free-text address / landmark name to (lat, lng, display_name).

    Uses Nominatim (OpenStreetMap) — free, no API key required.
    Appends ", Chicago IL" to bias results toward Chicago.
    Cached for 1 hour so repeat searches don't re-hit the API.

    Returns None if the query resolves to nothing.
    """
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q":      f"{query}, Chicago, IL",
                "format": "json",
                "limit":  1,
            },
            headers={"User-Agent": "ChicagoWalkabilityEngine/1.0"},
            timeout=6,
        )
        results = r.json()
        if not results:
            return None
        hit = results[0]
        return float(hit["lat"]), float(hit["lon"]), hit.get("display_name", query)
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_by_latlng(lat: float, lng: float) -> Optional[dict]:
    """
    Resolve a lat/lng to its H3 hex and return full prediction + SHAP data.
    Delegates to the backend's existing /predict/{lat}/{lng} endpoint.
    """
    try:
        r = requests.get(f"{API_URL}/predict/{lat}/{lng}", timeout=10)
        if r.status_code == 400:
            return None   # outside Chicago bounds
        if r.status_code == 404:
            return None   # hex not in dataset
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Could not fetch location ({lat:.4f}, {lng:.4f}): {e}")
        return None
