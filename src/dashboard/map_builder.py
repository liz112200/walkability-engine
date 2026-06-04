"""
src/dashboard/map_builder.py
-----------------------------
Builds the pydeck Deck object for the city walkability map.

Layers (bottom → top):
  1. H3HexagonLayer   — walk-score coloured hex cells
  2. ScatterplotLayer — neighbourhood centroid dots (white)
  3. TextLayer        — neighbourhood name labels (white)
  4. ScatterplotLayer — famous landmark pins (amber)
  5. TextLayer        — landmark name labels (amber, below pin)
  6. ScatterplotLayer — search-result pin (sky blue, shown only after a search)
  7. TextLayer        — search-result label

Base map: Carto Dark Matter (free, no Mapbox token, renders street names at all zooms).
"""
from __future__ import annotations

import pandas as pd
import pydeck as pdk
from typing import Optional

from src.dashboard.config import NEIGHBOURHOOD_COORDS, CHICAGO_LANDMARKS


# ── Hex fill colours ───────────────────────────────────────────────────────────

def _hex_color(score: float, highlighted: bool) -> list[int]:
    if highlighted:
        return [255, 215, 0, 245]       # amber — neighbourhood highlight
    if score >= 90:   return [0,   180,  80, 210]
    elif score >= 70: return [80,  200,  80, 200]
    elif score >= 50: return [255, 200,   0, 200]
    elif score >= 25: return [255, 120,   0, 200]
    else:             return [220,  50,  50, 200]


# ── Public builder ─────────────────────────────────────────────────────────────

def build_hex_map(
    df: pd.DataFrame,
    selected_h3_set: Optional[set] = None,
    selected_nb: Optional[str] = None,
    search_pin: Optional[dict] = None,
) -> pdk.Deck:
    """
    Parameters
    ----------
    df              : hex-level dataframe (h3_index, predicted_score, walk_score)
    selected_h3_set : set of H3 IDs to highlight in amber (neighbourhood ring)
    selected_nb     : neighbourhood key used to zoom the view
    search_pin      : dict with keys lat, lng, label — renders a sky-blue pin
    """
    if df.empty:
        return pdk.Deck()

    df = df.copy()
    df["color"] = df.apply(
        lambda r: _hex_color(
            r["predicted_score"],
            bool(selected_h3_set and r["h3_index"] in selected_h3_set),
        ),
        axis=1,
    )
    # Pre-format tooltip values — pydeck does not evaluate Python format specs
    df["predicted_str"] = df["predicted_score"].apply(lambda x: f"{x:.1f}")
    df["walk_str"]      = df["walk_score"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "—"
    )

    # ── Layer 1: hex data ──────────────────────────────────────────────────────
    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=df,
        get_hexagon="h3_index",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 18],
        line_width_min_pixels=0.4,
        pickable=True,
        extruded=False,
        filled=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 55],
    )

    # ── Layers 2 & 3: neighbourhood dots + labels ──────────────────────────────
    nb_data = [
        {"name": n.replace("-", " ").title(), "lat": lat, "lng": lng}
        for n, (lat, lng) in NEIGHBOURHOOD_COORDS.items()
    ]
    nb_dot = pdk.Layer(
        "ScatterplotLayer",
        data=nb_data,
        get_position=["lng", "lat"],
        get_radius=110,
        get_fill_color=[255, 255, 255, 190],
        get_line_color=[30, 30, 60, 160],
        stroked=True,
        line_width_min_pixels=1.2,
        pickable=False,
    )
    nb_text = pdk.Layer(
        "TextLayer",
        data=nb_data,
        get_position=["lng", "lat"],
        get_text="name",
        get_size=12,
        get_color=[235, 240, 255, 215],
        get_background_color=[15, 23, 42, 190],
        background=True,
        get_pixel_offset=[0, -20],
        font_family="Inter, sans-serif",
        font_weight="bold",
        pickable=False,
    )

    # ── Layers 4 & 5: landmark pins + labels (amber) ───────────────────────────
    lm_dot = pdk.Layer(
        "ScatterplotLayer",
        data=CHICAGO_LANDMARKS,
        get_position=["lng", "lat"],
        get_radius=90,
        get_fill_color=[255, 190, 40, 230],
        get_line_color=[180, 120, 0, 220],
        stroked=True,
        line_width_min_pixels=1.5,
        pickable=True,
    )
    lm_text = pdk.Layer(
        "TextLayer",
        data=CHICAGO_LANDMARKS,
        get_position=["lng", "lat"],
        get_text="name",
        get_size=11,
        get_color=[255, 210, 80, 220],
        get_background_color=[40, 28, 0, 200],
        background=True,
        get_pixel_offset=[0, 18],    # label below the dot
        font_family="Inter, sans-serif",
        font_weight="600",
        pickable=False,
    )

    layers = [hex_layer, nb_dot, nb_text, lm_dot, lm_text]

    # ── Layers 6 & 7: search-result pin (sky blue) — optional ─────────────────
    if search_pin:
        pin_data = [search_pin]   # expects keys: lat, lng, label
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pin_data,
            get_position=["lng", "lat"],
            get_radius=130,
            get_fill_color=[14, 165, 233, 240],    # sky-400
            get_line_color=[255, 255, 255, 200],
            stroked=True,
            line_width_min_pixels=2,
            pickable=False,
        ))
        layers.append(pdk.Layer(
            "TextLayer",
            data=pin_data,
            get_position=["lng", "lat"],
            get_text="label",
            get_size=13,
            get_color=[14, 165, 233, 240],
            get_background_color=[10, 30, 50, 210],
            background=True,
            get_pixel_offset=[0, -24],
            font_family="Inter, sans-serif",
            font_weight="bold",
            pickable=False,
        ))

    # ── View state ─────────────────────────────────────────────────────────────
    if search_pin:
        view = pdk.ViewState(
            latitude=search_pin["lat"], longitude=search_pin["lng"],
            zoom=14, pitch=0, bearing=0, transition_duration=600,
        )
    elif selected_nb and selected_nb in NEIGHBOURHOOD_COORDS:
        clat, clng = NEIGHBOURHOOD_COORDS[selected_nb]
        # FlyToInterpolator arcs through a higher altitude (zooms out then in)
        # giving the two-stage animation effect requested
        view = pdk.ViewState(
            latitude=clat, longitude=clng,
            zoom=13, pitch=18, bearing=0,
            transition_duration=1500,
            transition_interpolator={"@@type": "FlyToInterpolator", "speed": 1.2},
        )
    else:
        view = pdk.ViewState(
            latitude=41.85, longitude=-87.65,
            zoom=10, pitch=0, bearing=0,
            transition_duration=600,
        )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={
            "html": (
                "<b style='color:#7dd3fc'>H3:</b> {h3_index}<br/>"
                "<b style='color:#86efac'>Predicted:</b> {predicted_str}<br/>"
                "<b style='color:#fde68a'>Walk Score:</b> {walk_str}"
            ),
            "style": {
                "backgroundColor": "#252526",
                "color":           "#cccccc",
                "border":          "1px solid #3e3e3e",
                "borderRadius":    "8px",
                "padding":         "8px 12px",
                "fontSize":        "13px",
                "fontFamily":      "Inter, sans-serif",
            },
        },
        # Carto Dark Matter — free, no Mapbox token, shows street + district labels
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
