"""
src/dashboard/app.py
---------------------
Chicago Walkability Engine — Streamlit entry point.

Responsibilities (ONLY):
  - st.set_page_config
  - Page header + sidebar layout
  - Panel routing
  - render_* functions (pure UI orchestration — no data, no styling logic)

All other concerns live in dedicated modules:
  config.py      — constants
  styles.py      — CSS overrides, Tailwind inject, score colour/label
  api_client.py  — fetch functions, geocoder
  charts.py      — Plotly figure builders
  map_builder.py — pydeck Deck builder

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# When Streamlit runs this file directly (`streamlit run src/dashboard/app.py`)
# the project root is NOT on sys.path, so `src.*` imports fail.
# Insert it here — before any src.* import — so all sibling modules resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h3
import pandas as pd
import streamlit as st
from typing import Optional

# Absolute project root — used for data file paths so CWD doesn't matter
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.dashboard.config import (
    NEIGHBOURHOODS,
    NEIGHBOURHOOD_COORDS,
    NEIGHBOURHOOD_K,
    NAV_PANELS,
)
from src.dashboard.styles import inject_styles, score_color, score_label
from src.dashboard.api_client import (
    fetch_city_summary,
    fetch_hex,
    fetch_neighbourhood,
    fetch_compare,
    fetch_all_hex_scores,
    geocode,
    fetch_by_latlng,
)
from src.dashboard.charts import (
    plot_shap_waterfall,
    plot_score_gauge,
    plot_comparison_bars,
    plot_score_distribution,
    plot_top_features,
)
from src.dashboard.map_builder import build_hex_map


# ── Page setup ──────────────────────────────────────────────────────────────────

def render_header(summary: Optional[dict]) -> None:
    """Slim topbar — dark, minimal, stats on the right."""
    n_hexes    = f"{summary['n_hexes']:,}" if summary else "—"
    mean_score = f"{summary['mean_score']:.1f}" if summary else "—"
    dot_color  = "#10b981" if summary else "#ef4444"
    api_text   = "Connected" if summary else "Offline"

    # Build pieces as plain strings — no multi-line, no HTML comments
    icon = (
        '<div style="width:36px;height:36px;border-radius:8px;flex-shrink:0;'
        'background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);'
        'display:flex;align-items:center;justify-content:center;font-size:18px">🚶</div>'
    )
    brand = (
        '<div>'
        '<div style="font-size:1.1rem;font-weight:700;color:#e4e4e7;letter-spacing:-0.02em;line-height:1.15">Chicago Walkability Engine</div>'
        '<div style="font-size:0.71rem;color:#52525a;margin-top:3px">ML predictions &amp; SHAP attribution &middot; H3 resolution-9 hex grid</div>'
        '</div>'
    )
    chip_s = (
        '<div style="background:#161618;border:1px solid #2d2d32;border-radius:8px;padding:6px 14px;text-align:center;min-width:64px">'
        f'<div style="font-size:1rem;font-weight:700;color:#818cf8;letter-spacing:-0.01em">{mean_score}</div>'
        '<div style="font-size:0.57rem;color:#52525a;margin-top:2px;text-transform:uppercase;letter-spacing:0.07em">Avg score</div>'
        '</div>'
    )
    chip_h = (
        '<div style="background:#161618;border:1px solid #2d2d32;border-radius:8px;padding:6px 14px;text-align:center;min-width:64px">'
        f'<div style="font-size:1rem;font-weight:700;color:#818cf8;letter-spacing:-0.01em">{n_hexes}</div>'
        '<div style="font-size:0.57rem;color:#52525a;margin-top:2px;text-transform:uppercase;letter-spacing:0.07em">Hex cells</div>'
        '</div>'
    )
    chip_api = (
        '<div style="background:#161618;border:1px solid #2d2d32;border-radius:8px;padding:6px 14px;display:flex;align-items:center;gap:7px">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>'
        f'<span style="font-size:0.71rem;color:#71717a;white-space:nowrap">{api_text}</span>'
        '</div>'
    )

    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;padding-bottom:1.25rem;border-bottom:1px solid #2d2d32;margin-bottom:1.75rem">'
        f'<div style="display:flex;align-items:center;gap:12px">{icon}{brand}</div>'
        f'<div style="display:flex;align-items:center;gap:8px">{chip_s}{chip_h}{chip_api}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar(summary: Optional[dict]) -> str:
    """Dark sidebar — indigo active pills, muted inactive text."""
    if "panel" not in st.session_state:
        st.session_state.panel = "city_map"

    with st.sidebar:
        # Brand
        st.markdown(
            '<div style="padding:1.2rem 1rem 1rem;border-bottom:1px solid #1a1a1e">'
            '<div style="display:flex;align-items:center;gap:9px">'
            '<div style="width:28px;height:28px;border-radius:7px;flex-shrink:0;'
            'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
            'display:flex;align-items:center;justify-content:center;font-size:13px">🚶</div>'
            '<div>'
            '<div style="font-size:0.82rem;font-weight:700;color:#d4d4d8;letter-spacing:-0.01em">CWE</div>'
            '<div style="font-size:0.59rem;color:#3f3f46;text-transform:uppercase;letter-spacing:0.09em;margin-top:1px">Chicago &middot; Week 13</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        # Navigate label
        st.markdown(
            '<div style="padding:14px 16px 5px;font-size:0.59rem;color:#3f3f46;text-transform:uppercase;letter-spacing:0.13em;font-weight:600">Navigate</div>',
            unsafe_allow_html=True,
        )

        for key, icon, label in NAV_PANELS:
            active = st.session_state.panel == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.panel = key
                st.rerun()

        # Status label
        st.markdown(
            '<div style="margin-top:10px;border-top:1px solid #1a1a1e;padding:13px 16px 5px;'
            'font-size:0.59rem;color:#3f3f46;text-transform:uppercase;letter-spacing:0.13em;font-weight:600">Status</div>',
            unsafe_allow_html=True,
        )

        if summary:
            n_h = f"{summary['n_hexes']:,}"
            st.markdown(
                f'<div style="padding:0 8px 10px">'
                f'<div style="display:flex;align-items:center;gap:7px;'
                f'background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.18);'
                f'border-radius:7px;padding:6px 11px">'
                f'<div style="width:6px;height:6px;border-radius:50%;background:#10b981;flex-shrink:0"></div>'
                f'<span style="font-size:0.72rem;color:#71717a">Online &middot; {n_h}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            ca, cb = st.columns(2)
            ca.metric("Mean", f"{summary['mean_score']:.1f}")
            cb.metric("Std",  f"{summary['std_score']:.1f}")
        else:
            st.markdown(
                '<div style="padding:0 8px 10px">'
                '<div style="display:flex;align-items:center;gap:7px;'
                'background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.18);'
                'border-radius:7px;padding:6px 11px">'
                '<div style="width:6px;height:6px;border-radius:50%;background:#ef4444;flex-shrink:0"></div>'
                '<span style="font-size:0.72rem;color:#71717a">Offline</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.code("uvicorn src.api.main:app --port 8001")

        # Footer
        st.markdown(
            '<div style="padding:14px 16px 12px;border-top:1px solid #1a1a1e;margin-top:6px;'
            'font-size:0.6rem;color:#27272a">Chicago Walkability &middot; Week 13</div>',
            unsafe_allow_html=True,
        )

    return st.session_state.panel


# ── Shared component: hex stats card ───────────────────────────────────────────

def _render_hex_stats(hex_data: dict, heading: str = "") -> None:
    """
    Score banner + gauges + SHAP waterfall + demographics.
    Reused by Explore Location panel and city-map search results.
    """
    pred   = hex_data["predicted_score"]
    actual = hex_data.get("walk_score") or 0
    c      = score_color(pred)

    # Score banner
    residual_color = "#10b981" if (actual - pred) >= 0 else "#f43f5e"
    h3_id = hex_data["h3_index"]
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:2rem;border:1px solid {c}35;border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;background:{c}0d">'
        f'<div style="flex-shrink:0">'
        f'<div style="font-size:2.75rem;font-weight:800;color:{c};line-height:1;letter-spacing:-0.03em">{pred:.1f}</div>'
        f'<div style="font-size:0.72rem;color:#52525a;margin-top:5px">Predicted Walk Score</div>'
        f'</div>'
        f'<div style="flex:1">'
        f'<span style="background:{c}20;border:1px solid {c}40;color:{c};padding:3px 12px;border-radius:999px;font-size:0.76rem;font-weight:600">{score_label(pred)}</span>'
        f'<div style="font-size:0.84rem;color:#9898a0;margin-top:10px">Actual: <b style="color:#e4e4e7">{actual:.0f}</b> &middot; Residual: <b style="color:{residual_color}">{actual - pred:+.1f}</b></div>'
        f'<div style="font-size:0.72rem;color:#52525a;margin-top:5px">H3: <code style="background:#1e1e21;padding:2px 7px;border-radius:4px;font-size:0.7rem;color:#818cf8">{h3_id}</code></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Gauges
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_score_gauge(pred, "Predicted Score"), use_container_width=True)
    with g2:
        if actual:
            st.plotly_chart(plot_score_gauge(actual, "Actual Walk Score"), use_container_width=True)

    # SHAP waterfall
    st.divider()
    st.markdown(
        '<div style="font-size:0.67rem;font-weight:600;color:#52525a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">SHAP Feature Attribution</div>'
        '<p style="font-size:0.82rem;color:#71717a;margin-bottom:8px">Positive → pushes score above city baseline (~66). Negative → pulls it below.</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_shap_waterfall(hex_data, title=heading), use_container_width=True)

    # Demographics
    demo = hex_data.get("demographics", {})
    if demo:
        st.divider()
        st.markdown(
            '<div style="font-size:0.67rem;font-weight:600;color:#52525a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">Demographics</div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        # Census pct values are stored as 0–100 (e.g. 52.1 = 52.1%), not 0–1 decimals
        d1.metric("Median income",   f"${demo.get('census_median_income', 0):,.0f}"  if demo.get('census_median_income') else "—")
        d2.metric("% minority",      f"{demo.get('census_pct_minority', 0):.1f}%"   if demo.get('census_pct_minority') else "—")
        d3.metric("% poverty",       f"{demo.get('census_pct_poverty', 0):.1f}%"    if demo.get('census_pct_poverty')  else "—")
        d4, d5, d6 = st.columns(3)
        d4.metric("% Black",         f"{demo.get('census_pct_black', 0):.1f}%"      if demo.get('census_pct_black')    else "—")
        d5.metric("% Hispanic",      f"{demo.get('census_pct_hispanic', 0):.1f}%"   if demo.get('census_pct_hispanic') else "—")
        d6.metric("Pop density/km²", f"{demo.get('census_pop_density', 0):,.0f}"    if demo.get('census_pop_density')  else "—")


# ── Panel 1: City Map ───────────────────────────────────────────────────────────

def render_city_map() -> None:
    st.markdown("## 🗺️ City Walkability Map")

    # ── Search bar (full width of map column) ──────────────────────────────────
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "search",
            placeholder="Search address or landmark in Chicago…  e.g. Willis Tower, 1234 N Clark St",
            label_visibility="collapsed",
            key="map_search",
        )
    with btn_col:
        do_search = st.button("Search", use_container_width=True, type="primary")

    search_pin:   Optional[dict] = None
    search_result: Optional[dict] = None

    if do_search and query.strip():
        with st.spinner("Geocoding…"):
            geo = geocode(query.strip())
        if geo is None:
            st.warning(f"Could not find **{query}** in Chicago. Try a more specific address.")
        else:
            lat, lng, display = geo
            with st.spinner("Fetching walk score…"):
                search_result = fetch_by_latlng(lat, lng)
            if search_result is None:
                st.warning("This location is outside Chicago's walkable area or not in the dataset.")
            else:
                search_pin = {"lat": lat, "lng": lng, "label": query.strip()[:40]}

    # ── Controls + map layout ──────────────────────────────────────────────────
    ctrl, map_col = st.columns([1, 5])

    with ctrl:
        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-4 mb-3">'
            '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Neighbourhood</div>',
            unsafe_allow_html=True,
        )
        selected_nb = st.selectbox(
            "nb", ["None"] + NEIGHBOURHOODS,
            format_func=lambda x: x.replace("-", " ").title() if x != "None" else "— All Chicago —",
            key="map_neighbourhood",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-4 mb-3">'
            '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Score Range</div>',
            unsafe_allow_html=True,
        )
        score_range = st.slider(
            "sr", 0, 100, (0, 100),
            key="score_filter", label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-4 mb-3">'
            '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Legend</div>',
            unsafe_allow_html=True,
        )
        _legend_rows = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0">'
            f'<div style="width:10px;height:10px;border-radius:2px;'
            f'background:{bg};flex-shrink:0"></div>'
            f'<span style="font-size:0.76rem;color:#9898a0;white-space:nowrap">'
            f'<b style="color:#c4c4c8;font-weight:600">{rng}</b>'
            f'&nbsp;<span style="color:#52525a">{lbl}</span></span>'
            f'</div>'
            for rng, lbl, bg in [
                ("90–100", "Walker's paradise", "#16a34a"),
                ("70–89",  "Very walkable",     "#22c55e"),
                ("50–69",  "Somewhat walkable", "#ca8a04"),
                ("25–49",  "Some errands",      "#ea580c"),
                ("0–24",   "Car dependent",     "#dc2626"),
            ]
        )
        st.markdown(
            f'<div style="display:flex;flex-direction:column;gap:1px">{_legend_rows}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        with st.spinner("Loading hex data…"):
            df = fetch_all_hex_scores()

        if df.empty:
            st.warning("Could not load hex data. Check that parquet files exist.")
            return

        df = df[
            (df["predicted_score"] >= score_range[0]) &
            (df["predicted_score"] <= score_range[1])
        ]

        selected_h3_set: Optional[set] = None
        if selected_nb != "None" and selected_nb in NEIGHBOURHOOD_COORDS:
            lat_nb, lng_nb = NEIGHBOURHOOD_COORDS[selected_nb]
            center_cell     = h3.latlng_to_cell(lat_nb, lng_nb, 9)
            ring_cells      = set(h3.grid_disk(center_cell, NEIGHBOURHOOD_K))
            selected_h3_set = ring_cells & set(df["h3_index"].values)

        # ── Two-phase neighbourhood zoom: city overview → neighbourhood ──────────
        # Phase "zoom_out": send city-level ViewState so deck.gl begins zooming out.
        # st.rerun() immediately follows, sending the neighbourhood ViewState while
        # deck.gl is mid-flight → FlyToInterpolator smoothly re-routes to target.
        _nb_prev  = st.session_state.get("_map_nb_prev",  "None")
        _nb_phase = st.session_state.get("_map_nb_phase", "idle")

        if selected_nb != _nb_prev:
            st.session_state._map_nb_prev  = selected_nb
            _nb_phase = "zoom_out" if selected_nb != "None" else "idle"
            st.session_state._map_nb_phase = _nb_phase

        _nb_for_map = None if _nb_phase == "zoom_out" else (
            selected_nb if selected_nb != "None" else None
        )
        _h3_for_map = None if _nb_phase == "zoom_out" else selected_h3_set

        deck = build_hex_map(
            df,
            selected_h3_set=_h3_for_map,
            selected_nb=_nb_for_map,
            search_pin=search_pin,
        )
        st.pydeck_chart(deck, width="stretch", height=555, key="city_map_pydeck")

        if _nb_phase == "zoom_out":
            st.session_state._map_nb_phase = "zoom_in"
            # Sleep matches the zoom-out transition_duration (600 ms) so deck.gl
            # fully reaches the city overview before the zoom-in is triggered.
            time.sleep(0.65)
            st.rerun()
        elif _nb_phase == "zoom_in":
            st.session_state._map_nb_phase = "idle"
        st.caption(f"{len(df):,} hexes  ·  street labels from Carto Dark Matter  ·  amber = landmark pins")

    # ── Quick stats row ────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hexes shown", f"{len(df):,}")
    c2.metric("Mean",        f"{df['predicted_score'].mean():.1f}")
    c3.metric("Median",      f"{df['predicted_score'].median():.1f}")
    c4.metric("Min",         f"{df['predicted_score'].min():.1f}")
    c5.metric("Max",         f"{df['predicted_score'].max():.1f}")

    if selected_h3_set:
        nb_df = df[df["h3_index"].isin(selected_h3_set)]
        st.info(
            f"**{selected_nb.replace('-', ' ').title()}** — "
            f"{len(selected_h3_set):,} hexes highlighted  ·  "
            f"mean {nb_df['predicted_score'].mean():.1f}  ·  "
            f"median {nb_df['predicted_score'].median():.1f}"
        )

    # ── Search result card ─────────────────────────────────────────────────────
    if search_result:
        st.divider()
        st.markdown(
            f'<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">'
            f'📍 Search Result — {query.strip()}</div>',
            unsafe_allow_html=True,
        )
        _render_hex_stats(search_result, heading=f"SHAP — {query.strip()[:40]}")


# ── Panel 2: Explore Location ───────────────────────────────────────────────────

def render_hex_detail() -> None:
    st.markdown("## 🔍 Explore Location")

    # Address search is primary
    st.markdown(
        '<div class="bg-dark-surface border border-dark-border rounded-2xl p-5 mb-4">'
        '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Search by Address or Landmark</div>',
        unsafe_allow_html=True,
    )
    addr_col, btn_col = st.columns([5, 1])
    with addr_col:
        addr_query = st.text_input(
            "addr",
            placeholder="e.g. Willis Tower, 606 W Jackson Blvd, Hyde Park…",
            key="detail_address",
            label_visibility="collapsed",
        )
    with btn_col:
        addr_search = st.button("Go", key="addr_go", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Secondary options
    with st.expander("Or pick a neighbourhood / H3 index"):
        col1, col2 = st.columns(2)
        with col1:
            selected_nb = st.selectbox(
                "Neighbourhood",
                ["(select)"] + NEIGHBOURHOODS,
                format_func=lambda x: x.replace("-", " ").title() if x != "(select)" else "— pick one —",
                key="detail_neighbourhood",
            )
        with col2:
            manual_h3 = st.text_input(
                "H3 index",
                placeholder="892664c1e17ffff",
                key="manual_h3",
            )

    # Resolve to hex_data
    hex_data: Optional[dict] = None
    heading  = ""

    if addr_search and addr_query.strip():
        with st.spinner("Geocoding…"):
            geo = geocode(addr_query.strip())
        if geo is None:
            st.warning(f"Could not locate **{addr_query}** in Chicago.")
        else:
            lat, lng, _ = geo
            with st.spinner("Fetching walk score…"):
                hex_data = fetch_by_latlng(lat, lng)
            heading = addr_query.strip()[:60]
            if hex_data is None:
                st.warning("Location is outside Chicago's dataset coverage.")

    elif selected_nb != "(select)":
        with st.spinner(f"Loading {selected_nb}…"):
            hex_data = fetch_neighbourhood(selected_nb)
        heading = selected_nb.replace("-", " ").title()

    elif manual_h3.strip():
        with st.spinner(f"Loading {manual_h3.strip()}…"):
            hex_data = fetch_hex(manual_h3.strip())
        heading = manual_h3.strip()

    if hex_data is None and not (addr_search and addr_query.strip()):
        st.info("Search an address, pick a neighbourhood, or enter an H3 index above.")
        return

    if hex_data:
        st.divider()
        _render_hex_stats(hex_data, heading=f"SHAP — {heading}")


# ── Panel 3: Compare ────────────────────────────────────────────────────────────

def render_compare() -> None:
    st.markdown("## ⚖️ Compare")

    mode = st.radio(
        "Compare by",
        ["Neighbourhood", "Address"],
        horizontal=True,
        key="compare_mode",
    )

    if mode == "Neighbourhood":
        _compare_by_neighbourhood()
    else:
        _compare_by_address()


def _compare_by_neighbourhood() -> None:
    s1, s2 = st.columns(2)
    with s1:
        nb_a = st.selectbox(
            "Neighbourhood A", NEIGHBOURHOODS, index=0, key="compare_nb_a",
            format_func=lambda x: x.replace("-", " ").title(),
        )
    with s2:
        nb_b = st.selectbox(
            "Neighbourhood B", NEIGHBOURHOODS, index=2, key="compare_nb_b",
            format_func=lambda x: x.replace("-", " ").title(),
        )

    if nb_a == nb_b:
        st.warning("Select two different neighbourhoods.")
        return

    with st.spinner("Fetching data…"):
        data_a = fetch_neighbourhood(nb_a)
        data_b = fetch_neighbourhood(nb_b)

    if not data_a or not data_b:
        st.error("Could not fetch neighbourhood data.")
        return

    _render_comparison(
        data_a, data_b,
        label_a=nb_a.replace("-", " ").title(),
        label_b=nb_b.replace("-", " ").title(),
    )


def _compare_by_address() -> None:
    st.markdown(
        '<p style="font-size:0.83rem;color:#9e9e9e;margin-bottom:1rem">'
        'Enter any Chicago address, intersection, or landmark for each side.</p>',
        unsafe_allow_html=True,
    )
    s1, s2 = st.columns(2)
    with s1:
        addr_a = st.text_input("Location A", placeholder="e.g. Willis Tower", key="compare_addr_a")
    with s2:
        addr_b = st.text_input("Location B", placeholder="e.g. 63rd & Cottage Grove", key="compare_addr_b")

    if not (addr_a.strip() and addr_b.strip()):
        st.info("Enter both locations to compare.")
        # Clear stale results if inputs are cleared
        st.session_state.pop("addr_compare_result", None)
        return

    if st.button("Compare locations", type="primary"):
        geo_a, geo_b = None, None
        with st.spinner("Geocoding…"):
            geo_a = geocode(addr_a.strip())
            geo_b = geocode(addr_b.strip())

        if not geo_a:
            st.error(f"Could not locate **{addr_a}** in Chicago.")
            return
        if not geo_b:
            st.error(f"Could not locate **{addr_b}** in Chicago.")
            return

        with st.spinner("Fetching walk scores…"):
            data_a = fetch_by_latlng(geo_a[0], geo_a[1])
            data_b = fetch_by_latlng(geo_b[0], geo_b[1])

        if not data_a:
            st.error(f"**{addr_a}** is outside Chicago's dataset coverage.")
            return
        if not data_b:
            st.error(f"**{addr_b}** is outside Chicago's dataset coverage.")
            return

        # Persist results in session_state so they survive subsequent reruns
        st.session_state["addr_compare_result"] = {
            "data_a":  data_a,
            "data_b":  data_b,
            "label_a": addr_a.strip()[:40],
            "label_b": addr_b.strip()[:40],
        }

    # Render persisted results (survives reruns triggered by other widgets)
    result = st.session_state.get("addr_compare_result")
    if result:
        _render_comparison(
            result["data_a"], result["data_b"],
            label_a=result["label_a"],
            label_b=result["label_b"],
        )


def _render_comparison(
    data_a: dict, data_b: dict,
    label_a: str, label_b: str,
) -> None:
    """Render the full comparison view for two hex data dicts."""
    with st.spinner("Computing SHAP differences…"):
        compare_data = fetch_compare(data_a["h3_index"], data_b["h3_index"])

    score_a = data_a["predicted_score"]
    score_b = data_b["predicted_score"]
    diff    = score_a - score_b
    ca      = score_color(score_a)
    cb      = score_color(score_b)
    d_col   = "#16a34a" if diff > 0 else "#dc2626"
    d_bg    = "#0d2818" if diff > 0 else "#2d1212"
    winner  = label_a if diff > 0 else label_b

    st.divider()
    col_a, col_mid, col_b = st.columns([5, 2, 5])

    with col_a:
        st.markdown(
            f'<div class="bg-dark-surface border border-dark-border rounded-2xl p-5 text-center mb-2"'
            f' style="border-top:4px solid {ca}">'
            f'<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-2">{label_a}</div>'
            f'<div class="text-[2.6rem] font-black leading-none" style="color:{ca}">{score_a:.1f}</div>'
            f'<div class="text-[0.8rem] text-dark-muted mt-1">{score_label(score_a)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_score_gauge(score_a, ""), use_container_width=True)

    with col_mid:
        st.markdown(
            f'<div class="rounded-2xl p-5 text-center mt-6" style="background:{d_bg}">'
            f'<div class="text-[2.2rem] font-black leading-none" style="color:{d_col}">{diff:+.1f}</div>'
            f'<div class="text-[0.72rem] font-semibold mt-1" style="color:{d_col}">pts gap</div>'
            f'<div class="text-[0.68rem] text-dark-sub mt-2">{"▲" if diff > 0 else "▼"} {winner} leads</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f'<div class="bg-dark-surface border border-dark-border rounded-2xl p-5 text-center mb-2"'
            f' style="border-top:4px solid {cb}">'
            f'<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-2">{label_b}</div>'
            f'<div class="text-[2.6rem] font-black leading-none" style="color:{cb}">{score_b:.1f}</div>'
            f'<div class="text-[0.8rem] text-dark-muted mt-1">{score_label(score_b)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_score_gauge(score_b, ""), use_container_width=True)

    # SHAP waterfalls
    st.divider()
    st.markdown(
        '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">SHAP Feature Attribution</div>',
        unsafe_allow_html=True,
    )
    w1, w2 = st.columns(2)
    with w1:
        st.plotly_chart(plot_shap_waterfall(data_a, title=label_a), use_container_width=True)
    with w2:
        st.plotly_chart(plot_shap_waterfall(data_b, title=label_b), use_container_width=True)

    # Feature gap bar chart
    if compare_data:
        st.divider()
        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-5">'
            '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Feature Contribution Gap</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_comparison_bars(compare_data, name_a=label_a, name_b=label_b),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Demographics
    demo_a = data_a.get("demographics", {})
    demo_b = data_b.get("demographics", {})
    if demo_a and demo_b:
        st.divider()
        st.markdown(
            '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Demographics</div>',
            unsafe_allow_html=True,
        )
        # Census pct values are stored as 0–100 (e.g. 52.1 = 52.1%), not 0–1 decimals
        demo_df = pd.DataFrame({
            "Variable": ["Median income", "% minority", "% poverty", "% Black", "Pop density/km²"],
            label_a: [
                f"${demo_a.get('census_median_income', 0):,.0f}",
                f"{demo_a.get('census_pct_minority', 0):.1f}%",
                f"{demo_a.get('census_pct_poverty', 0):.1f}%",
                f"{demo_a.get('census_pct_black', 0):.1f}%",
                f"{demo_a.get('census_pop_density', 0):,.0f}",
            ],
            label_b: [
                f"${demo_b.get('census_median_income', 0):,.0f}",
                f"{demo_b.get('census_pct_minority', 0):.1f}%",
                f"{demo_b.get('census_pct_poverty', 0):.1f}%",
                f"{demo_b.get('census_pct_black', 0):.1f}%",
                f"{demo_b.get('census_pop_density', 0):,.0f}",
            ],
        })
        st.dataframe(demo_df, use_container_width=True, hide_index=True)


# ── Panel 4: Equity ─────────────────────────────────────────────────────────────

def render_equity(summary: Optional[dict]) -> None:
    st.markdown("## 📊 Equity Overview")

    if not summary:
        st.error("Cannot load city summary. Check API connection.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total hexes",   f"{summary['n_hexes']:,}")
    m2.metric("Mean score",    f"{summary['mean_score']:.1f}")
    m3.metric("Median score",  f"{summary['median_score']:.1f}")
    m4.metric("Std deviation", f"{summary['std_score']:.1f}")
    m5.metric("Score range",   f"{summary['min_score']:.0f}–{summary['max_score']:.0f}")

    st.divider()

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-4">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_score_distribution(summary), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with e2:
        st.markdown(
            '<div class="bg-dark-surface border border-dark-border rounded-2xl p-4">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_top_features(summary), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-4">Key Equity Findings</div>',
        unsafe_allow_html=True,
    )

    findings = [
        ("0.866",  "Moran's I",           "Spatial clustering (p < 0.0001)", "#3b82f6"),
        ("+0.317", "Income → Score",      "Pearson r",                       "#16a34a"),
        ("−0.259", "% Black → Score",     "Pearson r (negative)",            "#dc2626"),
        ("−0.231", "% Minority → Score",  "Pearson r (negative)",            "#ea580c"),
        ("+0.650", "Density → Score",     "Pop density correlation",         "#8b5cf6"),
        ("−3.42",  "Restaurant SHAP Gap", "Q4 vs Q1 minority quartile",      "#ec4899"),
    ]

    cols = st.columns(3)
    for i, (val, title, desc, accent) in enumerate(findings):
        with cols[i % 3]:
            st.markdown(
                f'<div class="bg-dark-surface2 border border-dark-border rounded-xl p-4 text-center mb-3"'
                f' style="border-left:4px solid {accent}">'
                f'<div class="text-[1.35rem] font-bold" style="color:{accent}">{val}</div>'
                f'<div class="text-[0.82rem] font-semibold text-dark-text mt-1">{title}</div>'
                f'<div class="text-[0.7rem] text-dark-sub mt-0.5">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        '<div class="bg-dark-surface border border-dark-border rounded-2xl p-5">'
        '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Interpretation</div>'
        '<p class="text-[0.88rem] text-dark-text leading-relaxed m-0">'
        "Chicago's walkability gap is driven primarily by <b>restaurant and retail density</b> "
        "differences between high-minority and low-minority neighbourhoods — not safety infrastructure. "
        "High-minority areas have better transit access (<b>+0.33 SHAP gap</b>) but significantly "
        "fewer restaurants (<b>−3.42 pts</b>) and lower retail density (<b>−0.76</b>)."
        "</p></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="text-[0.67rem] font-bold text-dark-sub uppercase tracking-widest mb-3">Audit Data</div>',
        unsafe_allow_html=True,
    )
    _equity_dir = _PROJECT_ROOT / "outputs" / "equity"

    try:
        corr_df = pd.read_csv(_equity_dir / "equity_summary.csv")
        st.dataframe(
            corr_df[["label", "pearson_r", "spearman_r", "perm_p", "significant"]],
            use_container_width=True, hide_index=True,
        )
    except FileNotFoundError:
        st.info("Run `python -m src.evaluation.equity_audit` to generate the full audit.")
    except Exception as e:
        st.error(f"Could not load equity_summary.csv: {e}")

    try:
        gap_df = pd.read_csv(_equity_dir / "shap_by_quartile.csv")
        st.markdown("**Top 10 SHAP gaps — Q4 (highest minority) vs Q1 (lowest):**")
        st.dataframe(
            gap_df[["feature", "Q1_mean", "Q4_mean", "gap_Q4_minus_Q1"]].head(10),
            use_container_width=True, hide_index=True,
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        st.error(f"Could not load shap_by_quartile.csv: {e}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Chicago Walkability Engine",
        page_icon="🚶",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject CSS before any content renders — must come immediately after set_page_config
    inject_styles()

    summary = fetch_city_summary()
    render_header(summary)
    panel = render_sidebar(summary)

    if panel == "city_map":
        render_city_map()
    elif panel == "hex_detail":
        render_hex_detail()
    elif panel == "compare":
        render_compare()
    elif panel == "equity":
        render_equity(summary)


if __name__ == "__main__":
    main()
