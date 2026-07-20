"""
Design system for the Streamlit UI: CSS injected once from streamlit_app.py,
plus small HTML-snippet helpers for the custom "field report" card, section
headers, step badges, and disease-library tags. Kept separate from
streamlit_app.py so the page logic stays readable.

Palette: a field-green / harvest-gold identity (not the generic
cream-and-terracotta AI-default look) — deliberately chosen for a crop
diagnosis tool used by farmers.
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg: #F5F6F0;
    --surface: #FFFFFF;
    --surface-alt: #EFF2E9;
    --border: #E1E2D6;
    --ink: #1B2B20;
    --ink-muted: #5C6B5E;
    --brand: #2D6A4F;
    --brand-dark: #1F4D39;
    --brand-tint: #E7F0EA;
    --gold: #B9800E;
    --gold-tint: #FBF1DD;
    --rust: #B3452F;
    --rust-tint: #FAEAE5;
    --radius: 12px;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp { background: var(--bg); color: var(--ink); }

/* ---- kill default chrome ---- */
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 880px; }

/* ---- headings ---- */
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; color: var(--ink); letter-spacing: -0.01em; }
h1 { font-weight: 600; }
h2, h3 { font-weight: 600; }

/* ---- custom app header banner ---- */
.app-banner {
    display: flex; align-items: center; gap: 16px;
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%);
    border-radius: var(--radius);
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(31,77,57,0.18);
}
.app-banner .mark {
    width: 52px; height: 52px; border-radius: 14px;
    background: rgba(255,255,255,0.16);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
}
.app-banner .title { color: #FFFFFF; font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.55rem; line-height: 1.15; margin: 0; }
.app-banner .subtitle { color: rgba(255,255,255,0.82); font-size: 0.88rem; margin: 3px 0 0 0; }

/* ---- eyebrow / section labels ---- */
.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.09em; text-transform: uppercase; color: var(--brand);
    margin-bottom: 4px; display: block;
}

/* ---- step badges (Diagnose tab) ---- */
.step-row { display: flex; align-items: center; gap: 10px; margin: 22px 0 10px 0; }
.step-num {
    width: 26px; height: 26px; border-radius: 50%;
    background: var(--brand); color: #fff; font-weight: 600; font-size: 0.82rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-label { font-weight: 600; font-size: 1.02rem; color: var(--ink); }

/* ---- field report card (diagnosis result) ---- */
.field-report {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 5px solid var(--tier-color, var(--brand));
    border-radius: var(--radius);
    padding: 20px 22px;
    margin: 14px 0 6px 0;
    box-shadow: 0 2px 8px rgba(27,43,32,0.05);
}
.field-report .eyebrow { color: var(--tier-color, var(--brand)); }
.field-report .diagnosis-name { font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; color: var(--ink); margin: 2px 0 10px 0; }
.confidence-track { background: var(--surface-alt); border-radius: 6px; height: 10px; width: 100%; overflow: hidden; }
.confidence-fill { background: var(--tier-color, var(--brand)); height: 100%; border-radius: 6px; }
.confidence-text { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--ink-muted); margin-top: 6px; }

/* ---- tag pills (disease library) ---- */
.tag-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.01em;
}
.tag-diagnosable { background: var(--brand-tint); color: var(--brand-dark); }
.tag-reference { background: var(--gold-tint); color: var(--gold); border: 1px solid #EAD198; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.sidebar-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; margin-bottom: 14px;
}
.history-chip {
    border-left: 3px solid var(--brand); background: var(--surface);
    border-radius: 8px; padding: 7px 10px; margin-bottom: 6px; font-size: 0.78rem;
    color: var(--ink-muted);
}

/* ---- tabs as pill nav ---- */
[data-testid="stTabs"] [role="tablist"] {
    gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px 10px 0 0; font-weight: 600; color: var(--ink-muted);
    padding: 10px 18px; font-size: 0.94rem;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--brand-tint); color: var(--brand-dark);
}

/* ---- buttons ---- */
.stButton button, .stDownloadButton button {
    border-radius: 9px; font-weight: 600; border: 1px solid var(--border);
}
.stButton button[kind="primary"] {
    background: var(--brand); border-color: var(--brand);
}
.stButton button[kind="primary"]:hover {
    background: var(--brand-dark); border-color: var(--brand-dark);
}

/* ---- inputs ---- */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 9px !important;
}

/* ---- expanders (disease library entries) ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
    background: var(--surface); margin-bottom: 8px;
}

/* ---- metrics (fertilizer calculator) ---- */
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-muted); }
[data-testid="stMetricValue"] { color: var(--brand-dark); }

/* ---- alerts ---- */
[data-testid="stAlert"] { border-radius: var(--radius); border: 1px solid var(--border); }

/* ---- file uploader / camera / audio inputs ---- */
[data-testid="stFileUploader"], [data-testid="stCameraInput"], [data-testid="stAudioInput"] {
    border-radius: var(--radius);
}

/* ---- radio as segmented control ---- */
[data-testid="stRadio"] > div { gap: 6px; }
"""


def banner_html(icon: str, title: str, subtitle: str) -> str:
    return (
        '<div class="app-banner">'
        f'<div class="mark">{icon}</div>'
        f'<div><p class="title">{title}</p><p class="subtitle">{subtitle}</p></div>'
        '</div>'
    )


def step_html(number: int, label: str) -> str:
    return (
        '<div class="step-row">'
        f'<div class="step-num">{number}</div>'
        f'<div class="step-label">{label}</div>'
        '</div>'
    )


def field_report_html(eyebrow: str, name: str, confidence: float | None, tier_color: str) -> str:
    conf_block = ""
    if confidence is not None:
        pct = round(confidence * 100)
        conf_block = (
            f'<div class="confidence-track"><div class="confidence-fill" style="width:{pct}%;"></div></div>'
            f'<div class="confidence-text">CONFIDENCE&nbsp;&nbsp;{pct}%</div>'
        )
    return (
        f'<div class="field-report" style="--tier-color:{tier_color};">'
        f'<span class="eyebrow">{eyebrow}</span>'
        f'<div class="diagnosis-name">{name}</div>'
        f'{conf_block}'
        '</div>'
    )


def tag_pill_html(is_diagnosable: bool) -> str:
    if is_diagnosable:
        return '<span class="tag-pill tag-diagnosable">📷&nbsp;PHOTO-DIAGNOSABLE</span>'
    return '<span class="tag-pill tag-reference">📖&nbsp;REFERENCE ONLY</span>'


TIER_COLORS = {
    "high": "#2D6A4F",
    "medium": "#B9800E",
    "low": "#B3452F",
}
