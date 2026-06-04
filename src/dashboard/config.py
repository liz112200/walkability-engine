"""
src/dashboard/config.py
-----------------------
All constants for the dashboard.  No Streamlit or heavy-library imports here
so this module loads instantly and can be imported anywhere without side effects.
"""
import os


def _resolve_api_url() -> str:
    """
    Priority order:
      1. st.secrets["API_URL"]  — Streamlit Community Cloud secrets UI
      2. os.getenv("API_URL")   — env var (Railway, Docker, local .env)
      3. localhost fallback      — local dev without secrets file
    """
    try:
        import streamlit as st  # lazy import — already loaded when dashboard runs
        val = st.secrets.get("API_URL")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv("API_URL", "http://localhost:8001")


# ── API ────────────────────────────────────────────────────────────────────────
API_URL = _resolve_api_url()

# ── Neighbourhoods ─────────────────────────────────────────────────────────────
NEIGHBOURHOODS = [
    "loop", "lincoln-park", "englewood", "hegewisch",
    "pilsen", "wicker-park", "hyde-park",
    "rogers-park", "humboldt-park", "austin",
]

# Representative centroid for each neighbourhood — matches API NEIGHBOURHOOD_COORDS
NEIGHBOURHOOD_COORDS: dict[str, tuple[float, float]] = {
    "loop":          (41.8827, -87.6298),
    "lincoln-park":  (41.9214, -87.6513),
    "englewood":     (41.7794, -87.6444),
    "hegewisch":     (41.6497, -87.5525),
    "pilsen":        (41.8535, -87.6650),
    "wicker-park":   (41.9083, -87.6783),
    "hyde-park":     (41.7943, -87.5962),
    "rogers-park":   (42.0118, -87.6683),
    "humboldt-park": (41.8990, -87.7228),
    "austin":        (41.8956, -87.7647),
}

# H3 grid_disk radius used when highlighting a neighbourhood (~1 km at res-9)
NEIGHBOURHOOD_K = 6

# ── Landmarks ──────────────────────────────────────────────────────────────────
# Shown as amber pins on the city map for orientation
CHICAGO_LANDMARKS: list[dict] = [
    {"name": "Willis Tower",    "lat": 41.8789, "lng": -87.6359},
    {"name": "Millennium Park", "lat": 41.8826, "lng": -87.6233},
    {"name": "Navy Pier",       "lat": 41.8917, "lng": -87.6086},
    {"name": "Wrigley Field",   "lat": 41.9484, "lng": -87.6553},
    {"name": "Lincoln Park",    "lat": 41.9274, "lng": -87.6392},
    {"name": "O'Hare Airport",  "lat": 41.9742, "lng": -87.9073},
    {"name": "Midway Airport",  "lat": 41.7868, "lng": -87.7522},
    {"name": "UChicago",        "lat": 41.7886, "lng": -87.5987},
    {"name": "United Center",   "lat": 41.8806, "lng": -87.6742},
]

# ── Navigation ─────────────────────────────────────────────────────────────────
# (session_state key, emoji, display label)
NAV_PANELS: list[tuple[str, str, str]] = [
    ("city_map",   "🗺️", "City Map"),
    ("hex_detail", "🔍", "Explore Location"),
    ("compare",    "⚖️", "Compare"),
    ("equity",     "📊", "Equity"),
]
