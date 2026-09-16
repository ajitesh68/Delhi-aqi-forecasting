"""Glassmorphism dark theme: Streamlit CSS plus a shared Plotly template."""

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0B1120"
SURFACE = "rgba(30, 41, 59, 0.55)"
BORDER = "rgba(148, 163, 184, 0.16)"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
ACCENT = "#38BDF8"
GRID = "rgba(148, 163, 184, 0.13)"

FONT = ("Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, Helvetica, Arial, sans-serif")

SERIES = ["#38BDF8", "#A78BFA", "#F472B6", "#FBBF24", "#34D399", "#FB923C"]


def register_plotly_template():
    pio.templates["aqi_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT, color=TEXT, size=13),
            colorway=SERIES,
            xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                       tickfont=dict(color=MUTED, size=11),
                       title=dict(font=dict(color=MUTED, size=12))),
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
                       tickfont=dict(color=MUTED, size=11),
                       title=dict(font=dict(color=MUTED, size=12))),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                        orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=8, r=8, t=36, b=8),
            hoverlabel=dict(bgcolor="#1E293B", bordercolor=BORDER,
                            font=dict(color=TEXT, family=FONT, size=12)),
            title=dict(font=dict(color=TEXT, size=15), x=0, xanchor="left"),
        )
    )
    pio.templates.default = "aqi_dark"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{
    background:
        radial-gradient(1100px 620px at 12% -8%, rgba(56,189,248,0.10), transparent 60%),
        radial-gradient(900px 520px at 88% 0%, rgba(167,139,250,0.10), transparent 58%),
        {BG};
    font-family: {FONT};
    color: {TEXT};
}}
footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stMainMenuButton"] {{ visibility: hidden; }}
[data-testid="stExpandSidebarButton"] {{ visibility: visible !important; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}

h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.02em; }}

.glass {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 28px rgba(2,6,23,0.42);
    transition: transform .18s ease, border-color .18s ease;
    flex: 1;
    width: 100%;
    box-sizing: border-box;
}}
.glass:hover {{ transform: translateY(-2px); border-color: rgba(148,163,184,0.3); }}

[data-testid="stHorizontalBlock"]:has(.glass) {{ align-items: stretch; }}
[data-testid="stHorizontalBlock"]:has(.glass) [data-testid="stColumn"] > div,
[data-testid="stHorizontalBlock"]:has(.glass) [data-testid="stElementContainer"]:has(.glass),
[data-testid="stHorizontalBlock"]:has(.glass) [data-testid="stMarkdown"]:has(.glass),
[data-testid="stHorizontalBlock"]:has(.glass) [data-testid="stMarkdown"]:has(.glass) > div,
[data-testid="stHorizontalBlock"]:has(.glass) [data-testid="stMarkdownContainer"]:has(.glass) {{
    height: 100%;
    display: flex;
    flex: 1;
    flex-direction: column;
    min-height: 0;
}}

.eyebrow {{
    font-size: .68rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 8px;
}}
.metric-value {{ font-size: 2.5rem; font-weight: 800; line-height: 1.05; }}
.metric-sub {{ font-size: .82rem; color: {MUTED}; margin-top: 6px; line-height: 1.45; }}

.hero-aqi {{ font-size: 4.6rem; font-weight: 800; line-height: 1; letter-spacing: -0.04em; }}
.hero-cat {{ font-size: 1.35rem; font-weight: 700; margin-top: 2px; }}

.pill {{
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .7rem; font-weight: 700; letter-spacing: .04em;
    border: 1px solid transparent; white-space: nowrap;
}}
.pill-live {{ background: rgba(34,197,94,.14); color:#4ADE80; border-color: rgba(34,197,94,.35); }}
.pill-stale {{ background: rgba(251,191,36,.14); color:#FBBF24; border-color: rgba(251,191,36,.35); }}
.pill-muted {{ background: rgba(148,163,184,.12); color:{MUTED}; border-color: rgba(148,163,184,.26); }}

.rank-row {{
    display:flex; align-items:center; gap:12px;
    padding:9px 12px; border-radius:10px; margin-bottom:5px;
    background: rgba(15,23,42,.45); border:1px solid {BORDER};
}}
.rank-num {{ font-size:.72rem; color:{MUTED}; width:20px; font-weight:700; }}
.rank-name {{ flex:1; font-size:.86rem; font-weight:500; }}
.rank-aqi {{ font-size:1.02rem; font-weight:800; min-width:46px; text-align:right; }}

.note {{
    font-size:.76rem; color:{MUTED}; line-height:1.55;
    border-left:2px solid {BORDER}; padding-left:10px; margin-top:6px;
}}

div[data-testid="stMetricValue"] {{ font-size: 1.7rem; font-weight: 800; }}
section[data-testid="stSidebar"] {{
    background: rgba(15,23,42,.96);
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption *,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] p {{
    color: #A9B6C8 !important;
    font-size: .78rem !important;
    line-height: 1.6 !important;
    opacity: 1 !important;
}}
section[data-testid="stSidebar"] strong {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label * {{
    color: #CBD5E1 !important; font-weight: 600; font-size: .82rem;
}}
section[data-testid="stSidebar"] h3 {{ color: {TEXT}; margin-bottom: .2rem; }}
section[data-testid="stSidebar"] hr {{ border-color: {BORDER}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 9px 9px 0 0;
    color: {MUTED}; font-weight: 600; font-size: .88rem; padding: 9px 16px;
}}
.stTabs [aria-selected="true"] {{ background: {SURFACE}; color: {TEXT}; }}

div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; }}
</style>
"""


def glow(hex_color, strength=0.4):
    """Severity-tinted shadow, used to make AQI cards read at a glance."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"0 0 26px rgba({r},{g},{b},{strength})"


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)
    register_plotly_template()
