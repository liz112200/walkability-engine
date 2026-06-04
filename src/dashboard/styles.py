"""
src/dashboard/styles.py
-----------------------
All visual configuration for the dashboard.

Dark theme palette
------------------
  #0f0f11  background (deepest)
  #161618  surface    (cards, panels)
  #1e1e21  surface+   (hover, elevated)
  #2d2d32  border
  #3a3a40  border+
  #e4e4e7  text-1  (primary)
  #9898a0  text-2  (secondary)
  #52525a  text-3  (tertiary)
  #6366f1  indigo  (accent / active nav)
  #818cf8  indigo-2
  #10b981  emerald (positive / online)
  #f43f5e  rose    (negative)
  #f59e0b  amber   (highlight / search pin)
  #38bdf8  sky     (links / info)

Public API
----------
  inject_styles()       — call once right after st.set_page_config()
  score_color(score)    — hex colour for a walk-score value
  score_label(score)    — tier label string
"""
import streamlit as st


# ── Score helpers ───────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 90:   return "#10b981"   # emerald
    elif score >= 70: return "#22c55e"   # green
    elif score >= 50: return "#f59e0b"   # amber
    elif score >= 25: return "#f97316"   # orange
    else:             return "#ef4444"   # red


def score_label(score: float) -> str:
    if score >= 90:   return "Walker's Paradise"
    elif score >= 70: return "Very Walkable"
    elif score >= 50: return "Somewhat Walkable"
    elif score >= 25: return "Some Errands"
    else:             return "Car Dependent"


# ── CSS — Streamlit component overrides ────────────────────────────────────────
# config.toml sets the base dark theme; this file fine-tunes every widget.
# No <script> tags — Streamlit strips them and breaks subsequent rendering.

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global font ── */
html, body, [class*="css"], button, input, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Hide default Streamlit header / footer ── */
/* stExpandSidebarButton lives inside stToolbar → stHeader.
   We cannot use display:none on either ancestor — that would
   permanently hide the re-open button. Instead we collapse them
   to zero height with overflow:visible so the fixed-positioned
   expand button can still escape the container. */
[data-testid="stHeader"] {
    background:    transparent !important;
    border-bottom: none !important;
    height:        0 !important;
    min-height:    0 !important;
    overflow:      visible !important;
    padding:       0 !important;
    margin-bottom: 0 !important;
}
[data-testid="stToolbar"] {
    background:    transparent !important;
    height:        0 !important;
    min-height:    0 !important;
    overflow:      visible !important;
    padding:       0 !important;
}
/* Hide toolbar content we don't want (deploy, settings, host items) */
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"] { display: none !important; }
footer                     { display: none !important; }

/* ── Sidebar expand button (shown when sidebar is closed) ── */
/* Pull it out of the 0-height header via position:fixed so it's
   always reachable. In Streamlit 1.55 the testid is stExpandSidebarButton. */
[data-testid="stExpandSidebarButton"] {
    position:        fixed !important;
    top:             0.6rem !important;
    left:            0.6rem !important;
    z-index:         99999 !important;
    display:         flex !important;
    align-items:     center !important;
    justify-content: center !important;
    width:           2rem !important;
    height:          2rem !important;
    background:      #161618 !important;
    border:          1px solid #2d2d32 !important;
    border-radius:   8px !important;
    color:           #71717a !important;
    cursor:          pointer !important;
}
[data-testid="stExpandSidebarButton"]:hover {
    background:   #1e1e21 !important;
    border-color: #3a3a40 !important;
    color:        #e4e4e7 !important;
}

/* ── Sidebar collapse button (inside sidebar, < arrow) ── */
/* The broad stSidebar div rule sets color:#3f3f46 which makes
   the icon near-invisible on the dark background — override it. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] span { color: #71717a !important; }
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover button { color: #a1a1aa !important; }

/* ── Main content padding ── */
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #0a0a0c !important;
    border-right: 1px solid #1e1e22 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Sidebar all text defaults */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #3f3f46 !important; }

/* ── Sidebar nav buttons — inactive ── */
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background:    transparent !important;
    border:        none !important;
    border-radius: 6px !important;
    color:         #71717a !important;
    width:         calc(100% - 16px) !important;
    margin:        1px 8px !important;
    padding:       8px 12px !important;
    font-size:     0.84rem !important;
    font-weight:   400 !important;
    text-align:    left !important;
    box-shadow:    none !important;
    letter-spacing: 0.005em;
    transition:    background 0.1s ease, color 0.1s ease;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(255,255,255,0.05) !important;
    color:      #a1a1aa !important;
}

/* ── Sidebar nav buttons — active ── */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background:    rgba(99,102,241,0.14) !important;
    border:        1px solid rgba(99,102,241,0.28) !important;
    border-radius: 6px !important;
    color:         #c7d2fe !important;
    width:         calc(100% - 16px) !important;
    margin:        1px 8px !important;
    padding:       8px 12px !important;
    font-size:     0.84rem !important;
    font-weight:   600 !important;
    text-align:    left !important;
    box-shadow:    0 0 0 1px rgba(99,102,241,0.12), 0 1px 4px rgba(0,0,0,0.5) !important;
    letter-spacing: 0.005em;
}

/* Sidebar metric mini-cards */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background:    rgba(255,255,255,0.03) !important;
    border:        1px solid #1e1e22 !important;
    border-radius: 8px !important;
    padding:       0.6rem 0.9rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #3f3f46 !important; font-size: 0.64rem !important;
    text-transform: uppercase; letter-spacing: 0.07em;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #a1a1aa !important; font-size: 1.05rem !important; font-weight: 700 !important;
}

/* ── Main area — metric cards ── */
[data-testid="stMetric"] {
    background:    #161618 !important;
    border:        1px solid #2d2d32 !important;
    border-radius: 12px !important;
    padding:       1rem 1.2rem !important;
    transition:    box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 0 0 1px #3a3a40 !important;
}
[data-testid="stMetricLabel"] {
    font-size:      0.66rem !important;
    font-weight:    600 !important;
    color:          #52525a !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    font-size:      1.5rem !important;
    font-weight:    800 !important;
    color:          #e4e4e7 !important;
    letter-spacing: -0.02em;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background:    #161618 !important;
    border-color:  #2d2d32 !important;
    border-radius: 8px !important;
    color:         #e4e4e7 !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] > div > div {
    background:    #161618 !important;
    border-color:  #2d2d32 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input {
    color: #e4e4e7 !important;
}
[data-testid="stTextInput"] input::placeholder { color: #52525a !important; }

/* ── Slider ── */
[data-testid="stSlider"] .stSlider { color: #6366f1 !important; }

/* ── Radio ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p { color: #9898a0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background:    #161618 !important;
    border:        1px solid #2d2d32 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: #9898a0 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #6366f1 !important; }

/* ── Code ── */
[data-testid="stCode"],
.stCode { background: #161618 !important; border: 1px solid #2d2d32 !important; border-radius: 8px !important; }
code { background: #1e1e21; padding: 1px 5px; border-radius: 4px; font-size: 0.82em; color: #818cf8; }

/* ── Divider ── */
hr { border-color: #2d2d32 !important; margin: 1.25rem 0 !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #52525a !important; font-size: 0.76rem !important; }

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p  { color: #9898a0; line-height: 1.65; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #e4e4e7; font-weight: 700; }
[data-testid="stMarkdownContainer"] a  { color: #818cf8; }
[data-testid="stMarkdownContainer"] strong { color: #e4e4e7; }

/* ── Tables / DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Page section headers (## in panels) ── */
h2 { color: #e4e4e7 !important; font-weight: 700 !important; letter-spacing: -0.015em !important; }
h3 { color: #c4c4c8 !important; font-weight: 600 !important; }

/* ── Button (primary — non-nav context) ── */
button[data-testid="baseButton-primary"]:not([data-testid="stSidebar"] *) {
    background: #6366f1 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.4) !important;
}
button[data-testid="baseButton-primary"]:hover:not([data-testid="stSidebar"] *) {
    background: #818cf8 !important;
}
</style>
"""


def inject_styles() -> None:
    """
    Inject CSS immediately after st.set_page_config().
    config.toml handles the base dark theme; this handles fine-grained widgets.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
