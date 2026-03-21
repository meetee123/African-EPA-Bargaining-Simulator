"""
Economic Partnership Agreements Bargaining Engine
Sequential Game Engine for African States
v1.0 — Updated January 2026

An interactive sequential-game simulator that quantifies bargaining power
asymmetries when negotiating AfCFTA protocols while locked into EPA schedules,
modeling great-power (China/US/EU) shadow influence.

Primary users: Ghana Ministry of Trade negotiators, AfCFTA Secretariat,
EU DG Trade strategy units, UNCTAD/World Bank trade policy teams.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from io import BytesIO
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EPA Bargaining Engine · Designing Decision Systems",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Canvas & App Shell ── */
    .stApp { background-color: #0D1117; }
    .main .block-container { background-color: #0D1117; padding-top: 2rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #0F1923 !important; }
    [data-testid="stSidebar"] * { color: #E6EDF3 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stRadio label {
        color: #8B949E !important; font-size: 0.83rem; font-weight: 500;
    }
    [data-testid="stSidebar"] hr { border-color: #21262D; }
    [data-testid="stSidebar"] .stExpander { border: 1px solid #21262D !important; border-radius: 4px; }

    /* ── Brand Insignia ── */
    .brand-insignia {
        font-family: Inter, sans-serif;
        font-size: 0.70rem;
        font-variant: small-caps;
        letter-spacing: 0.08em;
        color: #8B949E;
        margin-top: 2px;
        margin-bottom: 12px;
        display: block;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #161B22;
        padding: 14px 18px;
        border-radius: 4px;
        border-left: 3px solid #1B6CA8;
        border: 1px solid #21262D;
        border-left: 3px solid #1B6CA8;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 0.75rem; color: #8B949E; margin-bottom: 4px;
        font-variant: small-caps; letter-spacing: 0.04em;
    }
    .metric-card .value { font-size: 1.45rem; font-weight: 700; color: #E6EDF3; }

    /* ── Briefing Block ── */
    .briefing-block {
        background: #161B22;
        border: 1px solid #21262D;
        border-left: 3px solid #1B6CA8;
        border-radius: 4px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .briefing-block .bb-title {
        font-size: 0.70rem; font-variant: small-caps; letter-spacing: 0.08em;
        color: #8B949E; margin-bottom: 14px; display: block;
    }
    .briefing-block .bb-section { margin-bottom: 12px; }
    .briefing-block .bb-label {
        font-size: 0.72rem; font-weight: 600; color: #1B6CA8;
        text-transform: uppercase; letter-spacing: 0.06em; display: block; margin-bottom: 3px;
    }
    .briefing-block .bb-text { font-size: 0.88rem; color: #E6EDF3; line-height: 1.5; }
    .briefing-block .bb-text-muted { font-size: 0.85rem; color: #8B949E; line-height: 1.5; }

    /* ── Analyst Note ── */
    .analyst-note {
        background: #161B22;
        border: 1px solid #21262D;
        border-left: 3px solid #8B949E;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.86rem; color: #8B949E; line-height: 1.5;
    }
    .analyst-note.positive { border-left-color: #3FB950; }
    .analyst-note.warning  { border-left-color: #D29922; }
    .analyst-note.negative { border-left-color: #F85149; }

    /* ── Source Notes ── */
    .source-note {
        font-size: 0.75rem; color: #8B949E; margin-top: 8px;
        border-top: 1px solid #21262D; padding-top: 6px;
    }

    /* ── Headings ── */
    h1 { color: #E6EDF3 !important; font-weight: 600; font-size: 1.55rem !important; }
    h2 { color: #E6EDF3 !important; font-weight: 600; }
    h3 { color: #C9D1D9 !important; font-weight: 600; }

    /* ── Expanders ── */
    div[data-testid="stExpander"] { border: 1px solid #21262D !important; border-radius: 4px; }
    div[data-testid="stExpander"] summary { color: #E6EDF3; }

    /* ── Tabs ── */
    button[data-baseweb="tab"] { color: #8B949E !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #E6EDF3 !important; border-bottom-color: #1B6CA8 !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #21262D; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    """Render a styled metric card."""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def analyst_note(text: str, variant: str = "") -> None:
    """Render an analyst annotation block."""
    cls = f"analyst-note {variant}".strip()
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# COLOUR PALETTE 
# ══════════════════════════════════════════════════════════════

C = {
    # Core design system
    "bg":       "#0D1117",   # canvas
    "paper":    "#161B22",   # panel / card
    "sidebar":  "#0F1923",
    "accent":   "#1B6CA8",   # primary accent
    "accent2":  "#1B474D",   # secondary accent
    "border":   "#21262D",
    "text":     "#E6EDF3",   # primary text
    "muted":    "#8B949E",   # secondary text
    "positive": "#3FB950",   # positive signal
    "negative": "#F85149",   # negative signal
    "warning":  "#D29922",   # warning / caution
    # Legacy aliases kept for chart compatibility
    "teal":  "#1B6CA8",
    "rust":  "#F85149",
    "dark":  "#1B474D",
    "cyan":  "#8B949E",
    "mauve": "#D29922",
    "gold":  "#D29922",
    "olive": "#3FB950",
    "brown": "#8B949E",
}
SEQ = [
    C["accent"], C["positive"], C["warning"], C["negative"],
    C["muted"], C["accent2"], "#C9A84C", "#58A6FF",
]

LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color=C["text"]),
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["paper"],
    margin=dict(l=60, r=60, t=60, b=50),
)


# ── Central analyst-facing label map ──
COLUMN_LABELS = {
    "x_eu": "Exports to EU (%)", "m_eu": "Imports from EU (%)",
    "x_cn": "Exports to China (%)", "m_cn": "Imports from China (%)",
    "x_us": "Exports to US (%)", "m_us": "Imports from US (%)",
    "x_af": "Exports to Africa (%)", "m_af": "Imports from Africa (%)",
    "openness": "Trade Openness", "hhi": "Export Concentration (HHI)",
    "va": "Voice & Accountability", "ps": "Political Stability",
    "ge": "Govt Effectiveness", "rq": "Regulatory Quality",
    "rl": "Rule of Law", "cc": "Control of Corruption",
    "nci": "Negotiation Capacity Index",
    "cn_loan": "Chinese Loans (USD bn)", "cn_fdi": "Chinese FDI (USD bn)",
    "bri": "BRI Projects", "agoa": "AGOA Eligible",
    "agoa_x": "AGOA Exports (USD mn)", "eu_aid": "EU Dev. Aid (USD mn)",
    "eu_adj": "EU Adjustment (USD mn)", "cn_debt_gdp": "Chinese Debt (% GDP)",
    "cn_infra_dep": "Infra Dependence (0–1)",
    "ldc": "LDC Status", "cu": "Customs Union",
    "epa": "EPA Status", "afcfta_sched": "AfCFTA Schedule Submitted",
    "epa_exp": "EPA Exposure", "afcfta_opp": "AfCFTA Opportunity",
    "emp": "Employment Share", "gdp_sh": "GDP Share",
    "sens": "Sensitivity", "desc": "Description",
    "AfCFTA_Gain": "AfCFTA Gain", "EPA_Cost": "EPA Cost",
    "Emp_Risk": "Employment Risk",
    "EU_Export": "Exports to EU (%)", "China_Import": "Imports from China (%)",
    "Africa_Export": "Exports to Africa (%)", "Neg_Capacity": "Negotiation Capacity",
    "CN_Debt_GDP": "Chinese Debt (% GDP)",
    "gdp": "GDP (USD bn)", "pop": "Population (mn)",
    "region": "Region",
}


# ══════════════════════════════════════════════════════════════
# EMBEDDED DATA — all values from public sources
# ══════════════════════════════════════════════════════════════

# --- Country profiles ---
# Sources: World Bank WDI 2023, CIA World Factbook 2024
PROFILES = {
    "Ghana": dict(region="West Africa", gdp=72.8, pop=33.5, ldc=False, cu="ECOWAS",
                  epa="Interim EPA (prov. application Dec 2016)", afcfta_sched=True),
    "Côte d'Ivoire": dict(region="West Africa", gdp=70.0, pop=28.2, ldc=False, cu="ECOWAS",
                          epa="Interim EPA (prov. application Sep 2016)", afcfta_sched=True),
    "Kenya": dict(region="East Africa", gdp=113.0, pop=54.0, ldc=False, cu="EAC",
                  epa="EU-EAC EPA (variable geometry)", afcfta_sched=True),
    "Nigeria": dict(region="West Africa", gdp=477.0, pop=223.0, ldc=False, cu="ECOWAS",
                    epa="No EPA signed", afcfta_sched=True),
    "Senegal": dict(region="West Africa", gdp=28.0, pop=17.7, ldc=True, cu="ECOWAS",
                    epa="No bilateral EPA", afcfta_sched=True),
    "South Africa": dict(region="Southern Africa", gdp=399.0, pop=60.4, ldc=False, cu="SACU",
                         epa="SADC-EU EPA", afcfta_sched=True),
    "Ethiopia": dict(region="East Africa", gdp=156.0, pop=126.0, ldc=True, cu="—",
                     epa="No EPA (EBA access)", afcfta_sched=True),
    "Tanzania": dict(region="East Africa", gdp=75.7, pop=65.5, ldc=True, cu="EAC",
                     epa="EU-EAC EPA (not individually implementing)", afcfta_sched=True),
}
COUNTRIES = list(PROFILES.keys())

# --- EPA liberalisation schedules ---
# Sources: EU Access2Markets; ActionAid Ghana EPA Policy Brief; EU DG Trade
# pct = % tariff lines liberalised, rev = projected tariff‑revenue loss USD mn
_EPA_YEARS = list(range(2016, 2030))
_EPA = {
    "Ghana":          dict(pct=[0,0,0,0,22.6,35,44.1,52,58,64,70,74,77,80],
                           rev=[0,0,0,0,42,70,109,142,165,178,195,205,215,225]),
    "Côte d'Ivoire":  dict(pct=[0,0,0,0,25,37,47,55,61,67,72,76,79,81],
                           rev=[0,0,0,0,38,62,95,125,148,162,180,190,200,210]),
    "Kenya":          dict(pct=[0,0,0,5,12,22,33,42,50,57,63,68,73,80],
                           rev=[0,0,0,15,35,58,85,110,135,152,170,182,195,210]),
    "South Africa":   dict(pct=[86]*4+[86.2,86.5,87,87.5,88,88.5,89,89.5,90,90],
                           rev=[280,285,290,295,300,305,310,315,320,325,330,335,340,345]),
}

def epa_schedule(country: str) -> pd.DataFrame:
    d = _EPA.get(country)
    if d is None:
        return pd.DataFrame({"year": _EPA_YEARS, "pct_liberalised": [0]*14, "revenue_loss_mn": [0]*14})
    return pd.DataFrame({"year": _EPA_YEARS, "pct_liberalised": d["pct"], "revenue_loss_mn": d["rev"]})

# --- AfCFTA tariff categories ---
# Source: AfCFTA e-Tariff Book (etariff.au-afcfta.org); MacMap/ITC
# cat_a 90 % non-sensitive; cat_b 7 % sensitive; cat_c 3 % exclusion; mfn = avg MFN base rate
_AFCFTA_CAT = {
    "Ghana":         dict(a=90.3, b=6.8, c=2.9, mfn=12.4),
    "Côte d'Ivoire": dict(a=90.1, b=6.9, c=3.0, mfn=12.1),
    "Kenya":         dict(a=90.5, b=6.5, c=3.0, mfn=13.0),
    "Nigeria":       dict(a=89.5, b=7.5, c=3.0, mfn=14.2),
    "Senegal":       dict(a=90.0, b=7.0, c=3.0, mfn=12.1),
    "South Africa":  dict(a=91.0, b=6.0, c=3.0, mfn=7.6),
    "Ethiopia":      dict(a=90.0, b=7.0, c=3.0, mfn=17.5),
    "Tanzania":      dict(a=90.2, b=6.8, c=3.0, mfn=13.0),
}

# --- Trade dependence ratios (% of total) ---
# Source: UN Comtrade 2022 via WITS
_TRADE = {
    "Ghana":         dict(x_eu=26.3, m_eu=22.8, x_cn=10.5, m_cn=18.7, x_us=5.2, m_us=5.8,
                          x_af=15.2, m_af=8.3, openness=0.73, hhi=0.18),
    "Côte d'Ivoire": dict(x_eu=32.5, m_eu=25.3, x_cn=5.2, m_cn=16.5, x_us=8.1, m_us=3.5,
                          x_af=18.5, m_af=12.0, openness=0.68, hhi=0.21),
    "Kenya":         dict(x_eu=19.2, m_eu=14.5, x_cn=3.1, m_cn=21.5, x_us=8.5, m_us=4.2,
                          x_af=35.0, m_af=14.0, openness=0.42, hhi=0.12),
    "Nigeria":       dict(x_eu=28.0, m_eu=28.5, x_cn=3.8, m_cn=22.0, x_us=5.5, m_us=6.0,
                          x_af=10.0, m_af=6.0, openness=0.35, hhi=0.52),
    "Senegal":       dict(x_eu=25.0, m_eu=30.0, x_cn=6.0, m_cn=15.0, x_us=2.0, m_us=3.0,
                          x_af=30.0, m_af=10.0, openness=0.62, hhi=0.15),
    "South Africa":  dict(x_eu=22.0, m_eu=25.0, x_cn=11.0, m_cn=19.0, x_us=7.5, m_us=6.5,
                          x_af=25.0, m_af=8.0, openness=0.58, hhi=0.10),
    "Ethiopia":      dict(x_eu=18.0, m_eu=12.0, x_cn=8.0, m_cn=28.0, x_us=10.0, m_us=5.0,
                          x_af=12.0, m_af=5.0, openness=0.30, hhi=0.22),
    "Tanzania":      dict(x_eu=15.0, m_eu=10.0, x_cn=8.5, m_cn=25.0, x_us=3.0, m_us=3.0,
                          x_af=22.0, m_af=10.0, openness=0.35, hhi=0.14),
}

# --- WGI governance scores (scale ‑2.5 to +2.5) ---
# Source: World Bank WGI 2022 (info.worldbank.org/governance/wgi)
# nci = Negotiation Capacity Index (composite of GE+RQ+RL, rescaled 0-1)
_WGI = {
    "Ghana":         dict(va=0.52, ps=0.04, ge=-0.08, rq=-0.07, rl=0.02, cc=-0.13, nci=0.58),
    "Côte d'Ivoire": dict(va=-0.41, ps=-0.72, ge=-0.48, rq=-0.24, rl=-0.49, cc=-0.47, nci=0.38),
    "Kenya":         dict(va=-0.20, ps=-1.07, ge=-0.31, rq=-0.16, rl=-0.39, cc=-0.81, nci=0.48),
    "Nigeria":       dict(va=-0.54, ps=-1.98, ge=-1.02, rq=-0.74, rl=-0.90, cc=-1.02, nci=0.32),
    "Senegal":       dict(va=0.16, ps=-0.19, ge=-0.36, rq=-0.17, rl=-0.19, cc=-0.15, nci=0.50),
    "South Africa":  dict(va=0.59, ps=-0.22, ge=0.21, rq=0.17, rl=-0.06, cc=-0.01, nci=0.72),
    "Ethiopia":      dict(va=-1.21, ps=-1.82, ge=-0.50, rq=-0.87, rl=-0.68, cc=-0.43, nci=0.30),
    "Tanzania":      dict(va=-0.55, ps=-0.30, ge=-0.49, rq=-0.36, rl=-0.39, cc=-0.47, nci=0.42),
}

# --- Great-power influence ---
# Sources: AidData (aiddata.org) Chinese finance dataset 2000-2021;
#          World Bank International Debt Statistics; USTR AGOA; EU DG Trade
_GP = {
    "Ghana":         dict(cn_loan=3.5, cn_fdi=2.8, bri=12, agoa=True, agoa_x=285,
                          eu_aid=450, eu_adj=6.5, cn_debt_gdp=4.8, cn_infra_dep=0.35),
    "Côte d'Ivoire": dict(cn_loan=2.1, cn_fdi=1.5, bri=8, agoa=True, agoa_x=180,
                          eu_aid=380, eu_adj=5.0, cn_debt_gdp=3.0, cn_infra_dep=0.28),
    "Kenya":         dict(cn_loan=7.9, cn_fdi=3.2, bri=25, agoa=True, agoa_x=620,
                          eu_aid=320, eu_adj=4.0, cn_debt_gdp=7.0, cn_infra_dep=0.45),
    "Nigeria":       dict(cn_loan=5.0, cn_fdi=4.5, bri=20, agoa=True, agoa_x=150,
                          eu_aid=520, eu_adj=0.0, cn_debt_gdp=1.0, cn_infra_dep=0.20),
    "Senegal":       dict(cn_loan=1.8, cn_fdi=0.6, bri=6, agoa=True, agoa_x=45,
                          eu_aid=280, eu_adj=0.0, cn_debt_gdp=6.4, cn_infra_dep=0.32),
    "South Africa":  dict(cn_loan=4.5, cn_fdi=8.0, bri=15, agoa=True, agoa_x=3200,
                          eu_aid=150, eu_adj=0.0, cn_debt_gdp=1.1, cn_infra_dep=0.15),
    "Ethiopia":      dict(cn_loan=13.7, cn_fdi=4.0, bri=35, agoa=False, agoa_x=0,
                          eu_aid=600, eu_adj=0.0, cn_debt_gdp=8.8, cn_infra_dep=0.55),
    "Tanzania":      dict(cn_loan=4.2, cn_fdi=2.0, bri=18, agoa=True, agoa_x=80,
                          eu_aid=350, eu_adj=0.0, cn_debt_gdp=5.5, cn_infra_dep=0.40),
}

# --- Sector data (Ghana-centric, generalised) ---
# Source: UNCTAD Economic Development in Africa 2023; World Bank
SECTORS = {
    "Agriculture":     dict(epa_exp=0.75, afcfta_opp=0.65, emp=0.30, gdp_sh=0.20, sens="high",
        desc="Cocoa, cashew, shea; exposed to EU SPS standards; high AfCFTA potential for processed goods"),
    "Manufacturing":   dict(epa_exp=0.85, afcfta_opp=0.80, emp=0.12, gdp_sh=0.15, sens="critical",
        desc="Textiles, food processing, pharma; most affected by EPA import competition"),
    "Extractives":     dict(epa_exp=0.20, afcfta_opp=0.30, emp=0.05, gdp_sh=0.25, sens="low",
        desc="Gold, oil, manganese; commodity exports mostly MFN-priced"),
    "Services":        dict(epa_exp=0.40, afcfta_opp=0.70, emp=0.45, gdp_sh=0.35, sens="medium",
        desc="Financial services, ICT, logistics; AfCFTA Phase II protocol offers major gains"),
    "Digital Economy": dict(epa_exp=0.15, afcfta_opp=0.90, emp=0.08, gdp_sh=0.05, sens="low",
        desc="Fintech, e-commerce; minimal EPA constraints; high AfCFTA digital trade upside"),
}


# ══════════════════════════════════════════════════════════════
# GAME ENGINE
# ══════════════════════════════════════════════════════════════

@dataclass
class Params:
    """All tunable parameters for the bargaining game."""
    # Discount factors
    d_af: float = 0.85       # Africa (higher = more patient)
    d_eu: float = 0.92       # EU
    d_ac: float = 0.80       # AfCFTA Council
    # Behavioural perturbations
    sq_bias: float = 0.15    # Status-quo bias  — Samuelson & Zeckhauser (1988)
    loss_av: float = 2.25    # Loss aversion λ  — Kahneman & Tversky (1979)
    ambig: float = 0.10      # Ambiguity premium — Ellsberg (1961)
    # EPA lock-in
    epa_sunk: float = 0.20
    epa_mfn: float = 0.12    # MFN clause trigger cost
    epa_still: float = 0.08  # Standstill constraint cost
    # AfCFTA opportunity
    ac_mkt: float = 0.35     # Market access gain
    ac_ind: float = 0.15     # Industrialisation bonus
    ac_roo: float = 0.05     # Rules-of-origin compliance cost
    # Great-power shadow
    cn_infra: float = 0.20   # Chinese infra offer value
    cn_debt: float = 0.10    # Chinese debt constraint
    us_agoa: float = 0.08    # AGOA withdrawal risk
    eu_cond: float = 0.12    # EU aid conditionality
    # Capacity
    cap: float = 1.0         # Negotiation capacity modifier (from WGI nci)


# Actions available
AFRICA_ACTS = [
    "SELECTIVE_AFCFTA_LIB", "FULL_AFCFTA_LIB", "ACCEPT_EPA_DEEPENING",
    "REJECT_THREATEN_WITHDRAWAL", "DELAY_CONCESSION", "LEVERAGE_CHINA",
]
EU_ACTS = [
    "ACCEPT", "COUNTER_CONDITIONALITY", "THREATEN_MFN", "OFFER_ADJUSTMENT",
]
LABELS = {
    "SELECTIVE_AFCFTA_LIB": "Selective AfCFTA Lib.",
    "FULL_AFCFTA_LIB": "Full AfCFTA Lib.",
    "ACCEPT_EPA_DEEPENING": "Accept EPA Deepening",
    "REJECT_THREATEN_WITHDRAWAL": "Reject & Threaten",
    "DELAY_CONCESSION": "Delay Concession",
    "LEVERAGE_CHINA": "Leverage China Offer",
    "ACCEPT": "EU: Accept",
    "COUNTER_CONDITIONALITY": "EU: Counter w/ Conditions",
    "THREATEN_MFN": "EU: Threaten MFN",
    "OFFER_ADJUSTMENT": "EU: Offer Adj. Support",
    "STATUS_QUO": "Status Quo",
    "START": "Start",
}


def briefing_block(country: str, bp: dict, eq_pay: dict, tdf, td: dict, par: "Params") -> None:
    """Render a top-of-page Strategic Brief panel from existing computed outputs."""
    # Recommendation: dominant Africa equilibrium action label
    bpi_val = bp["BPI"]
    eu_dep = (td["x_eu"] + td["m_eu"]) / 2
    af_pay = eq_pay["africa"]

    # Derive recommendation text from BPI and equilibrium payoff
    if bpi_val >= 60:
        rec = "Pursue selective AfCFTA liberalisation as a credible outside option before deepening EPA commitments."
    elif bpi_val >= 45:
        rec = "Sequence concessions carefully: leverage AfCFTA scheduling to extract EU adjustment support before accepting EPA deepening."
    else:
        rec = "Adopt a defensive posture: prioritise building governance capacity and intra-African trade before committing to further EPA obligations."

    # Why it holds
    top_comp = max(
        [("Trade Diversification", bp["Diversification"]),
         ("Governance Capacity", bp["Capacity"]),
         ("Outside Options", bp["Outside"]),
         ("Policy Space", 100 - bp["Lock_in"])],
        key=lambda x: x[1]
    )
    why = (
        f"The model's dominant driver is <strong>{top_comp[0]}</strong> "
        f"(score {top_comp[1]:.0f}/100). "
        f"EU trade dependence stands at {eu_dep:.1f}% of total trade, "
        f"{'creating significant bilateral leverage for the EU' if eu_dep > 30 else 'leaving meaningful room for alternative positioning'}."
    )

    # Assumption context
    if par.cn_infra >= 0.30:
        ctx = f"The Chinese infrastructure offer parameter is elevated ({par.cn_infra:.2f}), materially improving Africa's outside-option value."
    elif par.sq_bias >= 0.25:
        ctx = f"Status-quo bias is set high ({par.sq_bias:.2f}), compressing the effective range of concession moves."
    elif par.epa_mfn >= 0.20:
        ctx = f"The MFN clause penalty is elevated ({par.epa_mfn:.2f}), increasing the cost of AfCFTA-first strategies."
    else:
        ctx = (
            f"Parameters reflect empirical defaults: Africa discount factor {par.d_af:.2f}, "
            f"loss aversion λ={par.loss_av:.2f}, ambiguity premium {par.ambig:.2f}. "
            "Adjust these in the sidebar to test alternative structural assumptions."
        )

    # Fragility
    if af_pay > 62:
        frag = "The equilibrium payoff is comfortably above the neutral threshold (50). The recommendation is <strong>robust</strong> to moderate parameter shifts. A substantial increase in EU aid conditionality or MFN penalty would be required to alter the dominant strategy."
    elif af_pay > 52:
        frag = "The equilibrium payoff is modestly above neutral. The recommendation is <strong>conditionally stable</strong>: a shift in Chinese infrastructure availability or EPA standstill costs of ±0.05 could alter the dominant action. Use the Sensitivity Analysis page to test specific flip conditions."
    elif af_pay > 45:
        frag = "The equilibrium payoff is near neutral. The recommendation is <strong>fragile</strong>: small parameter changes can flip the dominant strategy. Treat this as a zone of genuine strategic uncertainty."
    else:
        frag = "The equilibrium payoff is below neutral. The model indicates structural disadvantage under current assumptions. Improving governance capacity (NCI) or diversifying trade partners are the most impactful levers available within the model."

    # Credible threats context
    credible = tdf[tdf["Credible"] == "Yes"]
    if not credible.empty:
        threat_note = f"Credible threat identified: <strong>{credible.iloc[0]['Threat']}</strong> improves Africa's payoff above status quo."
    else:
        threat_note = "No fully credible threat detected under current parameters. Threats are analytically partial or non-credible — this constrains leverage."

    html = f"""
<div class="briefing-block">
  <span class="bb-title">Strategic Brief — {country}</span>
  <div class="bb-section">
    <span class="bb-label">Current Recommendation</span>
    <span class="bb-text">{rec}</span>
  </div>
  <div class="bb-section">
    <span class="bb-label">Why This Holds</span>
    <span class="bb-text">{why}</span>
  </div>
  <div class="bb-section">
    <span class="bb-label">Assumption Context</span>
    <span class="bb-text-muted">{ctx}</span>
  </div>
  <div class="bb-section">
    <span class="bb-label">Fragility &amp; Flip Conditions</span>
    <span class="bb-text-muted">{frag} {threat_note}</span>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _pay_africa(actions: list, p: Params, td: dict, wgi: dict) -> float:
    """Compute Africa payoff for an action sequence."""
    v = 50.0
    for i, a in enumerate(actions):
        d = p.d_af ** i
        if a == "ACCEPT_EPA_DEEPENING":
            v += (td["x_eu"] * 0.8 - p.epa_still * 100) * d
        elif a == "SELECTIVE_AFCFTA_LIB":
            v += (p.ac_mkt * 60 - p.ac_roo * 100 - p.epa_mfn * 30) * d
        elif a == "FULL_AFCFTA_LIB":
            v += (p.ac_mkt * 100 + p.ac_ind * 100 - p.epa_mfn * 100) * d
        elif a == "REJECT_THREATEN_WITHDRAWAL":
            v += (-td["x_eu"] * 0.5 + p.cn_infra * 100 * wgi["nci"]) * d
        elif a == "STATUS_QUO":
            v -= p.sq_bias * 10 * (i + 1) * d
        elif a == "DELAY_CONCESSION":
            pat = (1 - p.d_eu) * 30 if p.d_eu < p.d_af else 0
            v += (pat - 5.0 * (i + 1)) * d
        elif a == "LEVERAGE_CHINA":
            v += (p.cn_infra * 100 - p.cn_debt * 100 - p.eu_cond * 50) * d
    # Loss aversion
    if v < 50:
        v = 50 - (50 - v) * p.loss_av
    # Ambiguity premium for non-EPA paths
    non_epa = sum(1 for a in actions if "AFCFTA" in a or "CHINA" in a)
    if non_epa:
        v *= (1 - p.ambig * non_epa / max(len(actions), 1))
    return round(v * p.cap, 2)


def _pay_eu(actions: list, p: Params, td: dict) -> float:
    v = 50.0
    for i, a in enumerate(actions):
        d = p.d_eu ** i
        if a == "ACCEPT_EPA_DEEPENING":   v += 15 * d
        elif a == "SELECTIVE_AFCFTA_LIB": v -= 5 * d
        elif a == "FULL_AFCFTA_LIB":      v -= td["m_eu"] * 0.3 * d
        elif a == "REJECT_THREATEN_WITHDRAWAL": v -= 20 * d
        elif a == "STATUS_QUO":            v += 8 * d
        elif a == "DELAY_CONCESSION":      v -= 3 * d
        elif a == "LEVERAGE_CHINA":        v -= 18 * d
    return round(v, 2)


def _pay_ac(actions: list, p: Params) -> float:
    v = 50.0
    for i, a in enumerate(actions):
        d = p.d_ac ** i
        if a == "ACCEPT_EPA_DEEPENING":   v -= 10 * d
        elif a == "SELECTIVE_AFCFTA_LIB": v += 12 * d
        elif a == "FULL_AFCFTA_LIB":      v += 25 * d
        elif a == "REJECT_THREATEN_WITHDRAWAL": v += 5 * d
        elif a == "STATUS_QUO":            v -= 8 * d
        elif a == "LEVERAGE_CHINA":        v -= 5 * d
    return round(v, 2)


def _map_eu(eu_resp: str, prev_af: str) -> str:
    return {"ACCEPT": prev_af,
            "COUNTER_CONDITIONALITY": "ACCEPT_EPA_DEEPENING",
            "THREATEN_MFN": "STATUS_QUO",
            "OFFER_ADJUSTMENT": "SELECTIVE_AFCFTA_LIB"}.get(eu_resp, "STATUS_QUO")


@dataclass
class Node:
    nid: str; player: str; rnd: int; action: str; parent: Optional[str]
    pa: float = 0.0; pe: float = 0.0; pc: float = 0.0
    terminal: bool = False; eq: bool = False
    children: list = field(default_factory=list)
    desc: str = ""


def build_tree(p: Params, td: dict, wgi: dict, depth: int = 2) -> Dict[str, Node]:
    nodes: Dict[str, Node] = {}
    cnt = [0]
    def nid():
        cnt[0] += 1; return f"N{cnt[0]:04d}"

    root = nid()
    nodes[root] = Node(root, "Africa", 1, "START", None,
                       desc="Game begins. African state chooses initial strategy.")

    def expand(pid, rnd, hist):
        if rnd > depth:
            n = nodes[pid]; n.terminal = True
            n.pa = _pay_africa(hist, p, td, wgi)
            n.pe = _pay_eu(hist, p, td)
            n.pc = _pay_ac(hist, p)
            return
        par = nodes[pid]
        if par.player == "Africa":
            for a in AFRICA_ACTS:
                cid = nid()
                nodes[cid] = Node(cid, "EU", rnd, a, pid,
                                  desc=f"Rnd {rnd}: Africa → {LABELS[a]}")
                par.children.append(cid)
                expand(cid, rnd, hist + [a])
        elif par.player == "EU":
            for r in EU_ACTS:
                cid = nid()
                nodes[cid] = Node(cid, "Africa", rnd + 1, r, pid,
                                  desc=f"Rnd {rnd}: EU → {LABELS[r]}")
                par.children.append(cid)
                mapped = _map_eu(r, hist[-1] if hist else "STATUS_QUO")
                expand(cid, rnd + 1, hist + [mapped])

    expand(root, 1, [])
    return nodes


def solve(nodes: Dict[str, Node]):
    """Backward induction → SPE. Returns (eq_path, payoffs)."""
    root = [n for n in nodes.values() if n.parent is None][0].nid

    def bi(nid):
        n = nodes[nid]
        if n.terminal:
            return n.pa, n.pe, n.pc
        results = {c: bi(c) for c in n.children}
        idx = {"Africa": 0, "EU": 1, "AfCFTA": 2}.get(n.player, 0)
        best = max(results, key=lambda c: results[c][idx])
        nodes[best].eq = True
        n.pa, n.pe, n.pc = results[best]
        return results[best]

    bi(root)
    path, cur = [], root
    while True:
        path.append(cur)
        eq_ch = [c for c in nodes[cur].children if nodes[c].eq]
        if not eq_ch:
            break
        cur = eq_ch[0]
    last = nodes[path[-1]]
    return path, dict(africa=last.pa, eu=last.pe, afcfta=last.pc)


# ──── threat points ────

def threat_points(p: Params, td: dict, wgi: dict) -> pd.DataFrame:
    sq_a = _pay_africa(["STATUS_QUO"], p, td, wgi)
    sq_e = _pay_eu(["STATUS_QUO"], p, td)

    def row(name, acts, impl):
        pa = _pay_africa(acts, p, td, wgi)
        pe = _pay_eu(acts, p, td)
        cred = "Yes" if pa > sq_a and pe < sq_e else ("Partially" if pa > sq_a * 0.7 else "No")
        cs = min(1.0, max(0, (pa - sq_a) / 20) * max(0, (sq_e - pe) / 20))
        return dict(Threat=name, Africa=round(pa, 1), EU=round(pe, 1),
                    SQ_Africa=round(sq_a, 1), SQ_EU=round(sq_e, 1),
                    Credible=cred, Score=round(cs, 2), Implication=impl)

    rows = [
        row("Selective AfCFTA Lib.", ["SELECTIVE_AFCFTA_LIB"],
            "Pressures EU by demonstrating continental commitment without full MFN trigger"),
        row("Full AfCFTA Lib.", ["FULL_AFCFTA_LIB"],
            "Nuclear option: triggers EPA MFN clause but opens continental market"),
        row("Leverage Chinese Offer", ["LEVERAGE_CHINA"],
            "Geopolitical leverage: signals alternative partnerships to extract EU concessions"),
        row("EPA Withdrawal Threat", ["REJECT_THREATEN_WITHDRAWAL"],
            "High-risk threat: credibility depends on governance capacity and outside options"),
        row("Delay / Stall", ["DELAY_CONCESSION", "DELAY_CONCESSION"],
            "Patience play: effective only if Africa is more patient than EU"),
    ]
    return pd.DataFrame(rows)


# ──── BPI ────

def bpi(td: dict, wgi: dict, gp: dict, p: Params) -> dict:
    eu_dep = (td["x_eu"] + td["m_eu"]) / 2
    div = max(0, 100 - eu_dep * 2)
    cap = wgi["nci"] * 100
    cn_opt = min(30, gp["cn_loan"] * 3)
    outside = min(100, cn_opt + td["x_af"] * 2)
    lock = min(40, p.epa_sunk * 100 + p.epa_still * 100)
    wt = min(20, gp["agoa_x"] / 100)
    composite = div * 0.25 + cap * 0.25 + outside * 0.20 + (100 - lock) * 0.20 + wt * 0.10
    return dict(BPI=round(composite, 1), Diversification=round(div, 1),
                Capacity=round(cap, 1), Outside=round(outside, 1),
                Lock_in=round(lock, 1), Weight=round(wt, 1))


# ──── sensitivity ────

def sensitivity_1d(base: Params, td, wgi, attr, rng):
    rows = []
    for v in rng:
        pp = Params(**{k: getattr(base, k) for k in base.__dataclass_fields__})
        setattr(pp, attr, v)
        pp.cap = wgi["nci"] / 0.5
        tree = build_tree(pp, td, wgi, 2)
        _, pay = solve(tree)
        rows.append({"value": v, "Africa": pay["africa"], "EU": pay["eu"], "AfCFTA": pay["afcfta"]})
    return pd.DataFrame(rows)


def sensitivity_2d(base: Params, td, wgi, a1, r1, a2, r2):
    rows = []
    for v1 in r1:
        for v2 in r2:
            pp = Params(**{k: getattr(base, k) for k in pp.__dataclass_fields__}) if False else Params(
                **{k: getattr(base, k) for k in base.__dataclass_fields__})
            setattr(pp, a1, v1); setattr(pp, a2, v2)
            pp.cap = wgi["nci"] / 0.5
            tree = build_tree(pp, td, wgi, 2)
            _, pay = solve(tree)
            rows.append({a1: v1, a2: v2, "Africa": pay["africa"], "EU": pay["eu"]})
    return pd.DataFrame(rows)


# ──── optimal concession sequencing ────

def optimal_sequence(p: Params, td: dict, wgi: dict) -> pd.DataFrame:
    rows = []
    for name, s in SECTORS.items():
        gain = s["afcfta_opp"] * p.ac_mkt * 100
        cost = s["epa_exp"] * p.epa_mfn * 100
        emp_risk = s["emp"] * p.loss_av * 10
        cn_press = p.cn_infra * s["epa_exp"] * 20
        net = (gain - cost - emp_risk + cn_press * 0.3) * p.cap
        risk = s["epa_exp"] * 0.4 + (1 - s["afcfta_opp"]) * 0.3 + s["emp"] * 0.3
        rows.append(dict(Sector=name, AfCFTA_Gain=round(gain, 1), EPA_Cost=round(cost, 1),
                         Emp_Risk=round(emp_risk, 1), Net=round(net, 1),
                         Risk=round(risk, 2), Sensitivity=s["sens"]))
    df = pd.DataFrame(rows).sort_values("Net", ascending=False).reset_index(drop=True)
    n = len(df)
    df["Phase"] = ["Phase 1 (Immediate)" if i < n / 3 else "Phase 2 (Medium-term)" if i < 2 * n / 3
                   else "Phase 3 (Deferred)" for i in range(n)]
    df["Timing"] = ["Year 1-2" if "1" in ph else "Year 3-5" if "2" in ph else "Year 5-10" for ph in df["Phase"]]
    return df


# ══════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def _tree_layout(nodes, edges):
    """Simple layered layout: x by breadth, y by depth."""
    ch = {}; par = {}
    for e in edges:
        ch.setdefault(e[0], []).append(e[1]); par[e[1]] = e[0]
    roots = {e[0] for e in edges} - {e[1] for e in edges}
    if not roots:
        roots = {nodes[0]["id"]} if nodes else set()
    pos = {}; lc = {}
    def assign(nid, d):
        if nid in pos: return
        c = lc.get(d, 0); lc[d] = c + 1; pos[nid] = (c, -d)
        for child in ch.get(nid, []):
            assign(child, d + 1)
    for r in roots:
        assign(r, 0)
    mx = {}
    for nid, (x, y) in pos.items():
        mx[-y] = max(mx.get(-y, 0), x)
    for nid in pos:
        x, y = pos[nid]
        m = mx.get(-y, 1) or 1
        pos[nid] = (x / m if m else 0.5, y)
    return pos


def viz_game_tree(tree_nodes: Dict[str, Node], limit=120):
    """Interactive Plotly game tree."""
    vnodes, vedges = [], []
    root = [n for n in tree_nodes.values() if n.parent is None][0].nid
    q, vis, cnt = [root], set(), 0
    while q and cnt < limit:
        cur = q.pop(0)
        if cur in vis: continue
        vis.add(cur); cnt += 1
        n = tree_nodes[cur]
        lbl = LABELS.get(n.action, n.action)
        if n.terminal: lbl += f"\n(A:{n.pa:.0f} EU:{n.pe:.0f})"
        vnodes.append(dict(id=n.nid, label=lbl, player=n.player, rnd=n.rnd,
                           terminal=n.terminal, eq=n.eq, pa=n.pa, pe=n.pe, pc=n.pc))
        for c in n.children:
            if cnt < limit:
                vedges.append((n.nid, c, tree_nodes[c].eq))
                q.append(c)

    pos = _tree_layout(vnodes, [(e[0], e[1]) for e in vedges])
    fig = go.Figure()
    # normal edges
    ex, ey = [], []
    eqx, eqy = [], []
    for s, t, iseq in vedges:
        p1, p2 = pos.get(s, (0, 0)), pos.get(t, (0, 0))
        (eqx if iseq else ex).extend([p1[0], p2[0], None])
        (eqy if iseq else ey).extend([p1[1], p2[1], None])
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#C0C0C0"),
                             hoverinfo="none", showlegend=False))
    if eqx:
        fig.add_trace(go.Scatter(x=eqx, y=eqy, mode="lines",
                                 line=dict(width=3, color=C["rust"]),
                                 hoverinfo="none", name="Equilibrium Path"))
    pcol = {"Africa": C["teal"], "EU": C["rust"], "AfCFTA": C["gold"]}
    for pl in ["Africa", "EU"]:
        pn = [n for n in vnodes if n["player"] == pl]
        if not pn: continue
        nx_ = [pos.get(n["id"], (0, 0))[0] for n in pn]
        ny_ = [pos.get(n["id"], (0, 0))[1] for n in pn]
        sz = [14 if n["eq"] else 8 for n in pn]
        sym = ["diamond" if n["terminal"] else "circle" for n in pn]
        ht = []
        for n in pn:
            h = f"<b>{n['label']}</b><br>{n['player']} · Rnd {n['rnd']}"
            if n["terminal"]: h += f"<br>Africa {n['pa']:.1f} | EU {n['pe']:.1f} | AfCFTA {n['pc']:.1f}"
            if n["eq"]: h += "<br><b>★ Equilibrium</b>"
            ht.append(h)
        fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers", marker=dict(
            size=sz, color=pcol.get(pl, "#888"), symbol=sym, line=dict(width=1, color="#fff")),
            text=ht, hoverinfo="text", name=pl))
    fig.update_layout(title="Extensive-Form Game Tree", height=480,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), **LAYOUT)
    return fig


def viz_radar(bpi_d, country):
    cats = ["Diversification", "Capacity", "Outside Options", "Policy Space", "Econ. Weight"]
    vals = [bpi_d["Diversification"], bpi_d["Capacity"], bpi_d["Outside"],
            100 - bpi_d["Lock_in"], bpi_d["Weight"]]
    accent_rgb = "27,108,168"
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor=f"rgba({accent_rgb},0.20)",
        line=dict(color=C["accent"], width=2)))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color=C["muted"]),
                            gridcolor=C["border"]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text"]),
                             gridcolor=C["border"]),
            bgcolor=C["paper"],
        ),
        title=dict(text=f"Baseline Power Components: {country}", font=dict(size=13)),
        height=440,
        showlegend=False,
        margin=dict(l=90, r=90, t=80, b=70),
        **{k: v for k, v in LAYOUT.items() if k != "margin"},
    )
    return fig


def viz_radar_compare(bpi_dict):
    cats = ["Diversification", "Capacity", "Outside Options", "Policy Space", "Econ. Weight"]
    fig = go.Figure()
    for i, (c, b) in enumerate(bpi_dict.items()):
        vals = [b["Diversification"], b["Capacity"], b["Outside"], 100 - b["Lock_in"], b["Weight"]]
        col = SEQ[i % len(SEQ)]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], name=c,
            line=dict(color=col, width=2), fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.10)"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color=C["muted"]),
                            gridcolor=C["border"]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text"]),
                             gridcolor=C["border"]),
            bgcolor=C["paper"],
        ),
        title=dict(text="Peer Bargaining Power Comparison", font=dict(size=13)),
        height=500,
        margin=dict(l=90, r=90, t=80, b=90),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(color=C["text"])),
        **{k: v for k, v in LAYOUT.items() if k != "margin"},
    )
    return fig


def viz_sunburst(td, country):
    labs = ["Trade", "Exports", "Imports",
            "EU (Exports)", "China (Exports)", "US (Exports)", "Africa (Exports)", "Other (Exports)",
            "EU (Imports)", "China (Imports)", "US (Imports)", "Africa (Imports)", "Other (Imports)"]
    pars = ["", "Trade", "Trade",
            "Exports", "Exports", "Exports", "Exports", "Exports",
            "Imports", "Imports", "Imports", "Imports", "Imports"]
    oth_x = max(0, 100 - td["x_eu"] - td["x_cn"] - td["x_us"] - td["x_af"])
    oth_m = max(0, 100 - td["m_eu"] - td["m_cn"] - td["m_us"] - td["m_af"])
    # branchvalues="total": each parent value must equal the sum of its children
    # Exports children sum to 100, Imports children sum to 100, Trade root = 200
    vals = [200, 100, 100,
            td["x_eu"], td["x_cn"], td["x_us"], td["x_af"], oth_x,
            td["m_eu"], td["m_cn"], td["m_us"], td["m_af"], oth_m]
    colors = [C["paper"], C["accent"], C["negative"],
              C["accent"], C["warning"], C["positive"], C["accent2"], C["muted"],
              C["accent"], C["warning"], C["positive"], C["accent2"], C["muted"]]
    fig = go.Figure(go.Sunburst(
        labels=labs, parents=pars, values=vals, branchvalues="total",
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{country}: Trade Partner Dependence (%)", font=dict(size=13)),
        height=440,
        **LAYOUT,
    )
    return fig


def viz_epa(epa_df, country):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=epa_df["year"], y=epa_df["pct_liberalised"],
        name="Tariff Lines Liberalised (%)", marker_color=C["accent"], opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=epa_df["year"], y=epa_df["revenue_loss_mn"],
        name="Revenue Loss (USD mn)", line=dict(color=C["negative"], width=2),
        mode="lines+markers", marker=dict(size=6)), secondary_y=True)
    fig.update_layout(
        title=dict(text=f"{country}: EPA Liberalisation Schedule & Fiscal Impact", font=dict(size=13)),
        height=380,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(color=C["text"])),
        **LAYOUT,
    )
    fig.update_yaxes(title_text="Liberalised (%)", secondary_y=False,
                     title_font=dict(color=C["muted"]), tickfont=dict(color=C["muted"]))
    fig.update_yaxes(title_text="Revenue Loss (USD mn)", secondary_y=True,
                     title_font=dict(color=C["muted"]), tickfont=dict(color=C["muted"]))
    return fig


def viz_gp(gp, country):
    cats = ["Infra Loans (USD bn)", "FDI Stock (USD bn)", "EU Dev. Aid (×100 mn)", "AGOA Exports (×100 mn)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="China", x=cats, y=[gp["cn_loan"], gp["cn_fdi"], 0, 0],
                         marker_color=C["warning"]))
    fig.add_trace(go.Bar(name="EU", x=cats, y=[0, 0, gp["eu_aid"] / 100, 0],
                         marker_color=C["accent"]))
    fig.add_trace(go.Bar(name="US", x=cats, y=[0, 0, 0, gp["agoa_x"] / 100],
                         marker_color=C["positive"]))
    fig.update_layout(
        title=dict(text=f"{country}: Strategic Economic Exposure by Power", font=dict(size=13)),
        barmode="group",
        height=370,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(color=C["text"])),
        xaxis=dict(tickfont=dict(color=C["text"])),
        yaxis=dict(tickfont=dict(color=C["muted"])),
        **LAYOUT,
    )
    return fig


def viz_threats_bar(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Threat"], y=df["Africa"], name="Africa", marker_color=C["accent"]))
    fig.add_trace(go.Bar(x=df["Threat"], y=df["EU"], name="EU", marker_color=C["negative"]))
    sqa = df["SQ_Africa"].iloc[0]; sqe = df["SQ_EU"].iloc[0]
    fig.add_hline(y=sqa, line_dash="dash", line_color=C["accent"],
                  annotation_text=f"Africa status quo: {sqa:.0f}",
                  annotation_font_color=C["muted"],
                  annotation_position="top left")
    fig.add_hline(y=sqe, line_dash="dash", line_color=C["negative"],
                  annotation_text=f"EU status quo: {sqe:.0f}",
                  annotation_font_color=C["muted"],
                  annotation_position="bottom right")
    fig.update_layout(
        title=dict(text="Threat Payoff vs Status Quo", font=dict(size=13)),
        barmode="group", height=400,
        xaxis=dict(tickfont=dict(color=C["text"])),
        yaxis=dict(tickfont=dict(color=C["muted"])),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(color=C["text"])),
        **LAYOUT,
    )
    return fig


def viz_threats_bubble(df):
    df = df.copy()
    df["Improve"] = df["Africa"] - df["SQ_Africa"]
    df["Impact"] = df["SQ_EU"] - df["EU"]
    cmap = {"Yes": C["positive"], "No": C["negative"], "Partially": C["warning"]}
    fig = go.Figure(go.Scatter(
        x=df["Improve"], y=df["Impact"], mode="markers+text",
        marker=dict(size=df["Score"] * 50 + 14,
                    color=[cmap.get(c, C["muted"]) for c in df["Credible"]],
                    line=dict(width=1, color=C["border"]), opacity=0.85),
        text=df["Threat"], textposition="top center",
        textfont=dict(size=9, color=C["text"]),
        hovertemplate="<b>%{text}</b><br>Africa improvement: %{x:.1f}<br>EU impact: %{y:.1f}<extra></extra>",
        showlegend=False))
    fig.add_vline(x=0, line_dash="dot", line_color=C["muted"])
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"])
    fig.update_layout(
        title=dict(text="Threat Credibility Map (bubble size = credibility score)", font=dict(size=13)),
        height=420,
        xaxis_title="Africa Payoff Improvement over Status Quo",
        yaxis_title="EU Payoff Worsening (pressure applied)",
        xaxis=dict(tickfont=dict(color=C["muted"])),
        yaxis=dict(tickfont=dict(color=C["muted"])),
        **LAYOUT,
    )
    return fig


def viz_sens_line(df, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["value"], y=df["Africa"], mode="lines+markers",
        name="Africa", line=dict(color=C["accent"], width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["EU"], mode="lines+markers",
        name="EU", line=dict(color=C["negative"], width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["AfCFTA"], mode="lines+markers",
        name="AfCFTA", line=dict(color=C["warning"], width=2), marker=dict(size=5)))
    fig.add_hline(y=50, line_dash="dot", line_color=C["muted"],
                  annotation_text="Neutral (50)", annotation_font_color=C["muted"])
    fig.update_layout(
        title=dict(text=f"Parameter Sweep: {name}", font=dict(size=13)),
        xaxis_title=name, yaxis_title="Equilibrium Payoff",
        height=400,
        xaxis=dict(tickfont=dict(color=C["muted"])),
        yaxis=dict(tickfont=dict(color=C["muted"])),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(color=C["text"])),
        **LAYOUT,
    )
    return fig


def viz_sens_heat(df, a1, a2, label1=None, label2=None):
    piv = df.pivot_table(index=a2, columns=a1, values="Africa", aggfunc="mean")
    lbl1 = label1 or a1
    lbl2 = label2 or a2
    fig = go.Figure(go.Heatmap(z=piv.values,
        x=[f"{v:.2f}" for v in piv.columns], y=[f"{v:.2f}" for v in piv.index],
        colorscale=[[0, C["negative"]], [0.5, C["border"]], [1, C["accent"]]],
        colorbar=dict(title=dict(text="Africa Payoff", font=dict(color=C["text"])),
                      tickfont=dict(color=C["muted"])),
        hovertemplate=f"{lbl1}: %{{x}}<br>{lbl2}: %{{y}}<br>Payoff: %{{z:.1f}}<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"Africa Payoff Surface: {lbl1} × {lbl2}", font=dict(size=13)),
        xaxis_title=lbl1, yaxis_title=lbl2,
        xaxis=dict(tickfont=dict(color=C["muted"])),
        yaxis=dict(tickfont=dict(color=C["muted"])),
        height=440,
        **LAYOUT,
    )
    return fig


def viz_sequence(df, country):
    cmap = {
        "Phase 1 (Immediate)": C["positive"],
        "Phase 2 (Medium-term)": C["warning"],
        "Phase 3 (Deferred)": C["negative"],
    }
    fig = go.Figure(go.Bar(
        y=df["Sector"], x=df["Net"], orientation="h",
        marker_color=[cmap.get(p, C["muted"]) for p in df["Phase"]],
        text=[f"{v:.1f}" for v in df["Net"]],
        textposition="outside",
        textfont=dict(color=C["text"]),
    ))
    fig.update_layout(
        title=dict(text=f"{country}: Sector Concession Sequencing", font=dict(size=13)),
        xaxis_title="Net Benefit Score",
        xaxis=dict(tickfont=dict(color=C["muted"])),
        yaxis=dict(tickfont=dict(color=C["text"])),
        height=380,
        margin=dict(l=140, r=80, t=60, b=50),
        **{k: v for k, v in LAYOUT.items() if k != "margin"},
    )
    return fig


def viz_bpi_bar(bpi_dict):
    items = sorted(bpi_dict.items(), key=lambda x: x[1]["BPI"])
    fig = go.Figure(go.Bar(
        y=[i[0] for i in items], x=[i[1]["BPI"] for i in items],
        orientation="h",
        marker_color=[C["positive"] if i[1]["BPI"] >= 50 else C["negative"] for i in items],
        text=[f"{i[1]['BPI']:.1f}" for i in items],
        textposition="outside",
        textfont=dict(color=C["text"]),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color=C["muted"],
                  annotation_text="Neutral (50)", annotation_font_color=C["muted"])
    fig.update_layout(
        title=dict(text="Baseline Negotiating Position: Peer Comparison", font=dict(size=13)),
        xaxis_title="BPI (0–100)",
        xaxis=dict(range=[0, 110], tickfont=dict(color=C["muted"])),
        yaxis=dict(tickfont=dict(color=C["text"])),
        height=max(320, len(items) * 55),
        **LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### EPA Bargaining Engine")
    st.markdown(
        '<span class="brand-insignia">Designing Decision Systems</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### Analysis Subject")
    country = st.selectbox("Primary Country", COUNTRIES, index=0)
    # Clean peers default: remove primary country if it was selected
    _default_peers = [c for c in ["Kenya", "Nigeria"] if c != country]
    peers = st.multiselect(
        "Comparison Countries",
        [c for c in COUNTRIES if c != country],
        default=_default_peers,
    )

    st.markdown("---")

    with st.expander("Discount Factors"):
        d_af = st.slider("Africa δ (patience)", 0.50, 0.99, 0.85, 0.01,
            help="Higher = more patient negotiator. Default 0.85 reflects moderate time pressure.")
        d_eu = st.slider("EU δ", 0.50, 0.99, 0.92, 0.01,
            help="EU institutional continuity implies higher patience. Default 0.92.")
        d_ac = st.slider("AfCFTA Council δ", 0.50, 0.99, 0.80, 0.01)

    with st.expander("Behavioural Parameters"):
        sq_bias = st.slider("Status-Quo Bias", 0.0, 0.50, 0.15, 0.01,
            help="Samuelson & Zeckhauser (1988). Empirical range 0.10–0.30.")
        loss_av = st.slider("Loss Aversion λ", 1.0, 4.0, 2.25, 0.05,
            help="Kahneman & Tversky (1979). Literature consensus: 1.5–2.5.")
        ambig = st.slider("Ambiguity Premium", 0.0, 0.30, 0.10, 0.01,
            help="Ellsberg (1961). Discount applied to uncertain AfCFTA outcomes.")

    with st.expander("EPA Lock-in Costs"):
        epa_sunk = st.slider("EPA Sunk Cost", 0.0, 0.50, 0.20, 0.01)
        epa_mfn = st.slider("MFN Clause Penalty", 0.0, 0.30, 0.12, 0.01,
            help="Cost of triggering EPA MFN clause by offering better AfCFTA terms.")
        epa_still = st.slider("Standstill Constraint", 0.0, 0.20, 0.08, 0.01)

    with st.expander("Great-Power Shadow"):
        cn_infra = st.slider("Chinese Infrastructure Offer", 0.0, 0.50, 0.20, 0.01)
        cn_debt = st.slider("Chinese Debt Constraint", 0.0, 0.30, 0.10, 0.01)
        us_agoa = st.slider("AGOA Withdrawal Risk", 0.0, 0.20, 0.08, 0.01)
        eu_cond = st.slider("EU Aid Conditionality", 0.0, 0.30, 0.12, 0.01)

    with st.expander("AfCFTA Opportunity"):
        ac_mkt = st.slider("Market Access Gain", 0.10, 0.60, 0.35, 0.01)
        ac_ind = st.slider("Industrialisation Bonus", 0.0, 0.30, 0.15, 0.01)

    depth = st.slider("Game Depth (rounds)", 1, 3, 2,
        help="Alternating-move rounds. Depth 2 ≈ 750 nodes. Depth 3 is computationally intensive.")

    st.markdown("---")
    page = st.radio("Navigate", [
        "Dashboard", "Game Tree & Equilibrium", "Threat Points",
        "Sensitivity Analysis", "Concession Sequencing",
        "Comparative Analysis", "Data Explorer", "Methodology",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.68rem;color:#8B949E;">'
        'Data: UN Comtrade · EU Access2Markets · AfCFTA e-Tariff Book · '
        'World Bank WGI · AidData<br>'
        'Reference period: 2022–2024 · v1.0</p>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# BUILD PARAMS & FETCH DATA
# ══════════════════════════════════════════════════════════════

td = _TRADE[country]
wgi = _WGI[country]
gp = _GP[country]

par = Params(
    d_af=d_af, d_eu=d_eu, d_ac=d_ac,
    sq_bias=sq_bias, loss_av=loss_av, ambig=ambig,
    epa_sunk=epa_sunk, epa_mfn=epa_mfn, epa_still=epa_still,
    ac_mkt=ac_mkt, ac_ind=ac_ind,
    cn_infra=cn_infra, cn_debt=cn_debt, us_agoa=us_agoa, eu_cond=eu_cond,
    cap=wgi["nci"] / 0.5,
)


# ══════════════════════════════════════════════════════════════
# PAGE RENDERING
# ══════════════════════════════════════════════════════════════

# ──────────── DASHBOARD ────────────
if page == "Dashboard":
    st.markdown(f"# Strategic Brief: {country}")
    st.caption(
        "Pre-negotiation positioning analysis. Adjust scenario parameters in the sidebar "
        "to model alternative structural assumptions."
    )

    bp = bpi(td, wgi, gp, par)

    # Compute equilibrium (depth=1 for speed) to power briefing block
    _btree = build_tree(par, td, wgi, 1)
    _, _eq_pay = solve(_btree)
    _tdf = threat_points(par, td, wgi)
    briefing_block(country, bp, _eq_pay, _tdf, td, par)

    # ── Key metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Baseline Negotiating Position", f"{bp['BPI']:.1f} / 100")
    with c2:
        metric_card("GDP (USD bn)", f"${PROFILES[country]['gdp']:.1f}bn")
    with c3:
        metric_card("EU Trade Dependence", f"{(td['x_eu'] + td['m_eu']) / 2:.1f}%")
    with c4:
        metric_card("Negotiation Capacity Index", f"{wgi['nci']:.2f}")

    st.markdown("---")

    # ── Power components & trade structure ──
    l, r = st.columns(2)
    with l:
        st.plotly_chart(viz_radar(bp, country), use_container_width=True)
    with r:
        st.plotly_chart(viz_sunburst(td, country), use_container_width=True)

    # ── EPA liberalisation schedule ──
    edf = epa_schedule(country)
    if edf["pct_liberalised"].sum() > 0:
        st.plotly_chart(viz_epa(edf, country), use_container_width=True)
    else:
        st.markdown(
            f'<div class="analyst-note">'
            f'<strong>No active EPA liberalisation schedule:</strong> {country} has not entered '
            f'provisional application of an EPA. This reduces immediate fiscal exposure but also '
            f'limits access to EU adjustment support mechanisms.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Strategic economic exposure ──
    st.plotly_chart(viz_gp(gp, country), use_container_width=True)

    # ── Full country profile ──
    with st.expander("Full Country Profile"):
        rows = [
            ("Economy", "GDP (USD bn)", str(PROFILES[country]["gdp"])),
            ("Economy", "Population (mn)", str(PROFILES[country]["pop"])),
            ("Economy", "Trade Openness", f"{td['openness']:.0%}"),
            ("Trade", "Exports to EU (%)", str(td["x_eu"])),
            ("Trade", "Imports from EU (%)", str(td["m_eu"])),
            ("Trade", "Exports to China (%)", str(td["x_cn"])),
            ("Trade", "Imports from China (%)", str(td["m_cn"])),
            ("Trade", "Exports to Africa (%)", str(td["x_af"])),
            ("Governance", "Govt Effectiveness (WGI)", str(wgi["ge"])),
            ("Governance", "Regulatory Quality (WGI)", str(wgi["rq"])),
            ("Governance", "Negotiation Capacity Index", str(wgi["nci"])),
            ("Great Powers", "Chinese Loans (USD bn)", str(gp["cn_loan"])),
            ("Great Powers", "Chinese Debt (% GDP)", str(gp["cn_debt_gdp"])),
            ("Great Powers", "EU Dev. Aid (USD mn)", str(gp["eu_aid"])),
            ("EPA", "Arrangement", str(PROFILES[country]["epa"])),
            ("AfCFTA", "Schedule Submitted", "Yes" if PROFILES[country]["afcfta_sched"] else "No"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Category", "Indicator", "Value"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown(
        '<p class="source-note">Data: UN Comtrade 2022 · EU Access2Markets · '
        'World Bank WGI 2022 · AidData · USTR AGOA · AfCFTA e-Tariff Book</p>',
        unsafe_allow_html=True,
    )


# ──────────── GAME TREE ────────────
elif page == "Game Tree & Equilibrium":
    st.markdown(f"# Current Equilibrium: {country}")
    st.caption(
        "Extensive-form sequential bargaining game solved via backward induction. "
        "The equilibrium path represents the Subgame Perfect Equilibrium (SPE) — "
        "the unique strategy profile that is optimal at every decision node."
    )

    with st.spinner("Constructing and solving game tree…"):
        tree = build_tree(par, td, wgi, depth)
        eq_path, eq_pay = solve(tree)

    # ── Analytical summary ──
    af_act_nodes = [tree[nid] for nid in eq_path if tree[nid].action in AFRICA_ACTS]
    dom_action = LABELS.get(af_act_nodes[0].action, "—") if af_act_nodes else "—"
    analyst_note(
        f"<strong>Equilibrium recommendation:</strong> The model resolves to "
        f"<strong>{dom_action}</strong> as Africa's dominant opening strategy under "
        f"current parameters. EU payoff ({eq_pay['eu']:.1f}) "
        f"{'exceeds' if eq_pay['eu'] > eq_pay['africa'] else 'trails'} Africa payoff "
        f"({eq_pay['africa']:.1f}), indicating "
        f"{'structural EU advantage' if eq_pay['eu'] > eq_pay['africa'] else 'relative Africa leverage'} "
        f"under this configuration.",
        variant="",
    )

    st.markdown("#### Equilibrium Payoffs")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Africa Equilibrium Payoff", f"{eq_pay['africa']:.1f}")
    with c2: metric_card("EU Equilibrium Payoff", f"{eq_pay['eu']:.1f}")
    with c3: metric_card("AfCFTA Council Payoff", f"{eq_pay['afcfta']:.1f}")

    st.markdown("#### Equilibrium Path")
    eq_rows = []
    for nid in eq_path:
        n = tree[nid]
        if n.action != "START":
            # Derive actual acting player from the action type, not the node's 'next player' field
            actual_player = "EU" if n.action in EU_ACTS else ("Africa" if n.action in AFRICA_ACTS else n.player)
            eq_rows.append(dict(Round=n.rnd, Player=actual_player, Action=LABELS.get(n.action, n.action)))
    if eq_rows:
        st.dataframe(pd.DataFrame(eq_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Extensive-Form Game Tree")
    st.caption(
        "Blue nodes = Africa decision points · Red nodes = EU decision points · "
        "Diamond nodes = terminal payoffs · Bold path = equilibrium."
    )
    st.plotly_chart(viz_game_tree(tree), use_container_width=True)

    with st.expander("Model Statistics"):
        total = len(tree)
        terms = sum(1 for n in tree.values() if n.terminal)
        st.markdown(
            f"- Total nodes: **{total}**\n"
            f"- Terminal nodes: **{terms}**\n"
            f"- Game depth: **{depth}** alternating rounds\n"
            f"- Solution concept: Subgame Perfect Equilibrium (backward induction)"
        )

    st.markdown(
        '<p class="source-note">Model: Selten (1965) backward induction · '
        'Kahneman & Tversky (1979) prospect theory · '
        'Samuelson & Zeckhauser (1988) status-quo bias</p>',
        unsafe_allow_html=True,
    )


# ──────────── THREAT POINTS ────────────
elif page == "Threat Points":
    st.markdown(f"# Threat Credibility Assessment: {country}")
    st.caption(
        "A threat is analytically credible when executing it produces a better payoff for Africa "
        "than backing down, and simultaneously worsens the EU's payoff below its status-quo position."
    )

    tdf = threat_points(par, td, wgi)

    # Summary before charts
    credible_threats = tdf[tdf["Credible"] == "Yes"]
    partial_threats = tdf[tdf["Credible"] == "Partially"]
    if not credible_threats.empty:
        analyst_note(
            f"<strong>{len(credible_threats)} fully credible threat(s)</strong> identified under current parameters: "
            + ", ".join(f"<em>{r['Threat']}</em>" for _, r in credible_threats.iterrows())
            + ". These represent genuine strategic leverage points.",
            variant="positive",
        )
    elif not partial_threats.empty:
        analyst_note(
            f"No fully credible threats detected. "
            f"<strong>{len(partial_threats)} partially credible threat(s)</strong>: "
            + ", ".join(f"<em>{r['Threat']}</em>" for _, r in partial_threats.iterrows())
            + ". These may still function as negotiating signals but lack full commitment power.",
            variant="warning",
        )
    else:
        analyst_note(
            "No credible or partially credible threats detected under current parameters. "
            "This indicates a structurally weak bargaining position. "
            "Improving governance capacity (NCI) or Chinese infrastructure availability "
            "would strengthen threat credibility.",
            variant="negative",
        )

    l, r = st.columns(2)
    with l:
        st.plotly_chart(viz_threats_bar(tdf), use_container_width=True)
    with r:
        st.plotly_chart(viz_threats_bubble(tdf), use_container_width=True)

    st.markdown("#### Threat Detail")
    st.dataframe(
        tdf[["Threat", "Africa", "EU", "Credible", "Score", "Implication"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Strategic Implications")
    for _, row in tdf[tdf["Credible"] == "Yes"].iterrows():
        analyst_note(f"<strong>[Credible]</strong> {row['Threat']}: {row['Implication']}", variant="positive")
    for _, row in tdf[tdf["Credible"] == "Partially"].iterrows():
        analyst_note(f"<strong>[Partial]</strong> {row['Threat']}: {row['Implication']}", variant="warning")
    for _, row in tdf[tdf["Credible"] == "No"].iterrows():
        analyst_note(f"<strong>[Not credible]</strong> {row['Threat']}: {row['Implication']}")

    csv = tdf.to_csv(index=False).encode()
    st.download_button(
        "Export threat analysis (CSV)", csv,
        f"threat_assessment_{country.replace(' ', '_')}_{datetime.now():%Y%m%d}.csv",
        "text/csv",
    )

    st.markdown(
        '<p class="source-note">Framework: Schelling (1960) credible commitment · '
        'Nash (1953) outside-option bargaining</p>',
        unsafe_allow_html=True,
    )


# ──────────── SENSITIVITY ────────────
elif page == "Sensitivity Analysis":
    st.markdown(f"# Recommendation Fragility: {country}")
    st.caption(
        "Measures how the equilibrium recommendation changes as individual parameters vary. "
        "A flat-line result is a substantive finding — it indicates the recommendation is robust "
        "to changes in that parameter under the current configuration."
    )

    PMAP = {
        "Chinese Infrastructure Offer": ("cn_infra", 0.0, 0.50),
        "Status-Quo Bias": ("sq_bias", 0.0, 0.50),
        "Loss Aversion (λ)": ("loss_av", 1.0, 4.0),
        "MFN Clause Penalty": ("epa_mfn", 0.0, 0.30),
        "AfCFTA Market Access Gain": ("ac_mkt", 0.10, 0.60),
        "EU Aid Conditionality": ("eu_cond", 0.0, 0.30),
        "Africa Discount Factor (δ)": ("d_af", 0.50, 0.99),
        "Ambiguity Premium": ("ambig", 0.0, 0.30),
    }

    st.markdown("#### Single-Parameter Sweep")
    sel = st.selectbox("Select parameter to sweep", list(PMAP.keys()))
    attr, lo, hi = PMAP[sel]
    with st.spinner("Computing sweep…"):
        sdf = sensitivity_1d(par, td, wgi, attr, np.linspace(lo, hi, 20))
    st.plotly_chart(viz_sens_line(sdf, sel), use_container_width=True)

    # Robustness annotation
    af_range = sdf["Africa"].max() - sdf["Africa"].min()
    if af_range < 2.0:
        analyst_note(
            f"<strong>Robust finding:</strong> Africa's equilibrium payoff varies by only "
            f"{af_range:.2f} points across the full range of <em>{sel}</em>. "
            "The recommendation does not depend materially on this parameter under the current configuration. "
            "This is analytically informative — not a display error.",
            variant="positive",
        )
    else:
        # Flip condition: does payoff cross 50?
        neutral_cross = ((sdf["Africa"] < 50) & (sdf["Africa"].shift(-1) >= 50)) | \
                        ((sdf["Africa"] >= 50) & (sdf["Africa"].shift(-1) < 50))
        if neutral_cross.any():
            cross_idx = sdf[neutral_cross].index[0]
            cross_val = sdf.loc[cross_idx, "value"]
            analyst_note(
                f"<strong>Flip condition detected:</strong> Africa's equilibrium payoff crosses the "
                f"neutral threshold (50) when <em>{sel}</em> is approximately "
                f"<strong>{cross_val:.3f}</strong>. The dominant strategy recommendation changes "
                "near this value — treat this as a strategic sensitivity boundary.",
                variant="warning",
            )
        else:
            analyst_note(
                f"No neutral-threshold crossing detected within the swept range of <em>{sel}</em> "
                f"(payoff range: {sdf['Africa'].min():.1f}–{sdf['Africa'].max():.1f}). "
                "The recommendation direction is stable across this parameter range.",
            )

    st.markdown("---")
    st.markdown("#### Two-Parameter Surface")
    st.caption(
        "Shows Africa's equilibrium payoff across the joint space of two parameters. "
        "Darker blue = stronger Africa position; darker red = weaker."
    )
    cl, cr = st.columns(2)
    with cl:
        p1n = st.selectbox("X-axis parameter", list(PMAP.keys()), 0, key="px")
    with cr:
        p2n = st.selectbox("Y-axis parameter", [p for p in PMAP if p != p1n], 0, key="py")
    a1, lo1, hi1 = PMAP[p1n]
    a2, lo2, hi2 = PMAP[p2n]

    with st.spinner("Computing parameter surface…"):
        mdf = sensitivity_2d(par, td, wgi, a1, np.linspace(lo1, hi1, 10), a2, np.linspace(lo2, hi2, 10))
    st.plotly_chart(viz_sens_heat(mdf, a1, a2, label1=p1n, label2=p2n), use_container_width=True)

    st.markdown(
        '<p class="source-note">Methodology: Saltelli et al. (2008) global sensitivity analysis · '
        'Parameters swept at 20 points (1D) and 10×10 grid (2D)</p>',
        unsafe_allow_html=True,
    )


# ──────────── CONCESSION SEQUENCING ────────────
elif page == "Concession Sequencing":
    st.markdown(f"# Sequencing Implications: {country}")
    st.caption(
        "Derives the optimal order for AfCFTA tariff concessions given current EPA constraints and "
        "great-power dynamics. Phase allocation follows net benefit ranking — sectors with highest "
        "AfCFTA gain relative to EPA cost and employment risk are sequenced first."
    )

    sdf = optimal_sequence(par, td, wgi)

    # Analytical summary
    phase1 = sdf[sdf["Phase"] == "Phase 1 (Immediate)"]
    if not phase1.empty:
        top_sector = phase1.iloc[0]
        analyst_note(
            f"<strong>Recommended first-mover sector:</strong> <em>{top_sector['Sector']}</em> "
            f"(net benefit score: {top_sector['Net']:.1f}). "
            f"This sector offers the most favourable AfCFTA gain-to-EPA-cost ratio "
            f"under current parameters and should anchor early concession offers.",
            variant="positive",
        )

    st.plotly_chart(viz_sequence(sdf, country), use_container_width=True)

    st.markdown("#### Sequencing Detail")
    display_sdf = sdf.rename(columns=COLUMN_LABELS)
    st.dataframe(display_sdf, use_container_width=True, hide_index=True)

    st.markdown("#### AfCFTA Tariff Structure")
    ac = _AFCFTA_CAT.get(country, dict(a=90, b=7, c=3, mfn=12))
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Category A — Non-sensitive", f"{ac['a']:.1f}%")
    with c2: metric_card("Category B — Sensitive", f"{ac['b']:.1f}%")
    with c3: metric_card("Category C — Exclusion List", f"{ac['c']:.1f}%")
    with c4: metric_card("Average MFN Base Rate", f"{ac['mfn']:.1f}%")

    csv = sdf.to_csv(index=False).encode()
    st.download_button(
        "Export sequencing analysis (CSV)", csv,
        f"sequencing_{country.replace(' ', '_')}_{datetime.now():%Y%m%d}.csv",
        "text/csv",
    )

    st.markdown(
        '<p class="source-note">AfCFTA categories per Art. 7, Protocol on Trade in Goods: '
        '90% non-sensitive (5/10yr liberalisation), 7% sensitive (10/13yr), 3% exclusion list. '
        'Source: AfCFTA e-Tariff Book</p>',
        unsafe_allow_html=True,
    )


# ──────────── COMPARATIVE ────────────
elif page == "Comparative Analysis":
    st.markdown("# Peer Context & Coalition Readiness")
    st.caption(
        "Benchmarks the primary country's bargaining position against selected peers. "
        "Identifies structural similarities that indicate viable coalition formation in "
        "AfCFTA scheduling and EPA renegotiation contexts."
    )

    all_c = [country] + peers
    bd = {}
    for c in all_c:
        cp = Params(**{k: getattr(par, k) for k in par.__dataclass_fields__})
        cp.cap = _WGI[c]["nci"] / 0.5
        bd[c] = bpi(_TRADE[c], _WGI[c], _GP[c], cp)

    st.plotly_chart(viz_bpi_bar(bd), use_container_width=True)
    analyst_note(
        "Colour encoding: <strong style='color:#3FB950'>green bars</strong> indicate BPI ≥ 50 "
        "(stronger structural position relative to the EU); "
        "<strong style='color:#F85149'>red bars</strong> indicate BPI &lt; 50 "
        "(structurally weaker position). The threshold is analytical, not normative — "
        "it reflects relative leverage within this model's parameter space.",
    )

    st.plotly_chart(viz_radar_compare(bd), use_container_width=True)

    st.markdown("#### Country Comparison")
    rows = []
    for c in all_c:
        p = PROFILES[c]; t = _TRADE[c]; w = _WGI[c]; g = _GP[c]
        rows.append(dict(
            Country=c,
            GDP=p["gdp"],
            EU_Export=t["x_eu"],
            China_Import=t["m_cn"],
            Africa_Export=t["x_af"],
            Neg_Capacity=w["nci"],
            CN_Debt_GDP=g["cn_debt_gdp"],
            BPI=bd[c]["BPI"],
            EPA=p["epa"],
        ))
    cdf = pd.DataFrame(rows).sort_values("BPI", ascending=False)
    display_cdf = cdf.rename(columns=COLUMN_LABELS)
    st.dataframe(display_cdf, use_container_width=True, hide_index=True)

    st.markdown("#### Coalition Readiness Assessment")
    primary_bpi = bd[country]["BPI"]
    close = sorted(
        [(c, abs(bd[c]["BPI"] - primary_bpi)) for c in all_c if c != country],
        key=lambda x: x[1],
    )
    if close:
        partner, gap = close[0]
        if gap <= 5:
            analyst_note(
                f"<strong>Strong coalition candidate:</strong> {partner} (BPI gap: {gap:.1f}). "
                "Countries within 5 BPI points share structural leverage constraints and are "
                "viable partners for coordinated AfCFTA scheduling and joint EPA renegotiation.",
                variant="positive",
            )
        elif gap <= 15:
            analyst_note(
                f"<strong>Moderate coalition candidate:</strong> {partner} (BPI gap: {gap:.1f}). "
                "Some structural alignment exists. Coalition coordination is possible but "
                "positions will require active harmonisation.",
                variant="warning",
            )
        else:
            analyst_note(
                f"Closest peer is {partner} (BPI gap: {gap:.1f}). "
                "The structural gap is significant — formal coalition formation is unlikely "
                "without prior convergence on AfCFTA implementation priorities.",
            )

    csv = cdf.to_csv(index=False).encode()
    st.download_button(
        "Export comparison data (CSV)", csv,
        f"peer_comparison_{datetime.now():%Y%m%d}.csv",
        "text/csv",
    )


# ──────────── DATA EXPLORER ────────────
elif page == "Data Explorer":
    st.markdown("# Underlying Data")
    st.caption(
        "Browse and export the source datasets underpinning the bargaining analysis. "
        "All data is embedded from public sources (2022–2024 reference period)."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Country Profiles", "Trade Dependence", "Governance (WGI)", "Great-Power Exposure", "Sector Analysis"
    ])

    with tab1:
        df = pd.DataFrame([dict(Country=c, **PROFILES[c]) for c in COUNTRIES])
        st.dataframe(df.rename(columns=COLUMN_LABELS), use_container_width=True, hide_index=True)

    with tab2:
        df = pd.DataFrame([dict(Country=c, **_TRADE[c]) for c in COUNTRIES])
        st.dataframe(df.rename(columns=COLUMN_LABELS), use_container_width=True, hide_index=True)

    with tab3:
        df = pd.DataFrame([dict(Country=c, **_WGI[c]) for c in COUNTRIES])
        st.dataframe(df.rename(columns=COLUMN_LABELS), use_container_width=True, hide_index=True)

    with tab4:
        df = pd.DataFrame([dict(Country=c, **_GP[c]) for c in COUNTRIES])
        st.dataframe(df.rename(columns=COLUMN_LABELS), use_container_width=True, hide_index=True)

    with tab5:
        df = pd.DataFrame([dict(Sector=k, **v) for k, v in SECTORS.items()])
        st.dataframe(df.rename(columns=COLUMN_LABELS), use_container_width=True, hide_index=True)

    st.markdown("#### Export Dataset")
    choice = st.selectbox("Select dataset", ["Country Profiles", "Trade Dependence", "Governance (WGI)", "Great-Power Exposure", "EPA Schedule"])
    if st.button("Generate CSV"):
        if choice == "Country Profiles":
            out = pd.DataFrame([dict(Country=c, **PROFILES[c]) for c in COUNTRIES])
        elif choice == "Trade Dependence":
            out = pd.DataFrame([dict(Country=c, **_TRADE[c]) for c in COUNTRIES])
        elif choice == "Governance (WGI)":
            out = pd.DataFrame([dict(Country=c, **_WGI[c]) for c in COUNTRIES])
        elif choice == "Great-Power Exposure":
            out = pd.DataFrame([dict(Country=c, **_GP[c]) for c in COUNTRIES])
        else:
            out = epa_schedule(country)
        st.download_button(
            "Download CSV", out.to_csv(index=False).encode(),
            f"epa_engine_{choice.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{datetime.now():%Y%m%d}.csv",
            "text/csv",
        )


# ──────────── METHODOLOGY ────────────
elif page == "Methodology":
    st.markdown("# Methodology & Model Limitations")
    analyst_note(
        "This instrument is a <strong>structuring device for scenario analysis</strong>, not a "
        "predictive model. It organises trade-offs, surfaces parameter sensitivities and structures "
        "strategic options — it does not forecast negotiation outcomes. "
        "Findings should be reported as ranges and scenario-conditional results, not as point estimates."
    )

    st.markdown("""
## Model Overview

This simulator implements an **extensive-form sequential bargaining game** between three
strategic actors:

1. **African State** (e.g., Ghana) — primary decision-maker choosing concession strategies
2. **EU / DG Trade** — responds within EPA framework constraints
3. **AfCFTA Council** — background player shaping continental dynamics

Great-power influence (China, US) enters as **exogenous parameters** affecting payoff functions,
not as strategic players.

## Game Structure

| Element | Detail |
|---|---|
| **Type** | Finite extensive-form, perfect information |
| **Solution** | Subgame Perfect Equilibrium via backward induction (Selten, 1965) |
| **Africa actions** | 6: selective AfCFTA lib., full lib., accept EPA deepening, threaten withdrawal, delay, leverage China |
| **EU responses** | 4: accept, counter w/ conditionality, threaten MFN, offer adjustment support |
| **Behavioural** | Status-quo bias, loss aversion (λ), ambiguity premium |

## Bargaining Power Index (BPI)

| Component (Weight) | Measurement | Source |
|---|---|---|
| Trade Diversification (25%) | Inverse of EU export/import dependence | UN Comtrade |
| Governance Capacity (25%) | WGI composite (GE + RQ + RL) | World Bank WGI |
| Outside Options (20%) | Chinese infrastructure + intra-African trade | AidData, Comtrade |
| Policy Space (20%) | Inverse of EPA lock-in costs | EU Access2Markets |
| Economic Weight (10%) | AGOA exports, GDP | USTR, World Bank |

## Data Sources

| Dataset | Source | Period |
|---|---|---|
| AfCFTA tariff schedules | AfCFTA e-Tariff Book (etariff.au-afcfta.org) | 2023-2024 |
| EPA liberalisation | EU Access2Markets, ActionAid Ghana | 2016-2029 |
| Trade flows | UN Comtrade (comtrade.un.org) | 2022 |
| Governance | World Bank WGI | 2022 |
| Chinese finance | AidData (aiddata.org) | 2000-2023 |
| AGOA | USTR / agoa.info | 2022 |

## Parameter Justification

| Parameter | Default | Range | Reference |
|---|---|---|---|
| Status-quo bias | 0.15 | 0.10-0.30 | Samuelson & Zeckhauser (1988) |
| Loss aversion (λ) | 2.25 | 1.5-2.5 | Tversky & Kahneman (1991) |
| Ambiguity premium | 0.10 | 0.05-0.20 | Ellsberg (1961) |
| Africa δ | 0.85 | 0.70-0.95 | Rubinstein (1982) |
| EU δ | 0.92 | 0.85-0.98 | Institutional patience lit. |
| MFN penalty | 0.12 | 0.05-0.20 | EPA Art. 35; Ravenhill (2011) |

## Limitations

- **Parameter sensitivity:** Use the Sensitivity tab; report findings as ranges, not point estimates.
- **Framing:** "Asymmetry" is analytical, not normative — positioned as neutral capacity-building.
- **Data currency:** 2022-2024 reference period; cross-check with latest national statistics.
- **Simplification:** 3-player game abstracts from intra-African coordination and domestic politics.
- **Behavioural parameters:** Lab-derived; field estimates for trade negotiations are scarce.

## Key References

1. Selten, R. (1965). Spieltheoretische Behandlung eines Oligopolmodells. *ZgS*.
2. Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica* 47(2).
3. Samuelson, W. & Zeckhauser, R. (1988). Status Quo Bias. *J. Risk & Uncertainty* 1(1).
4. Rubinstein, A. (1982). Perfect Equilibrium in a Bargaining Model. *Econometrica* 50(1).
5. Nash, J. (1953). Two-Person Cooperative Games. *Econometrica* 21(1).
6. Schelling, T. (1960). *The Strategy of Conflict*. Harvard UP.
7. Ravenhill, J. (2011). Political Economy of EPAs. *RAPE*.
8. UNCTAD (2023). *Economic Development in Africa Report*.
9. AfCFTA Secretariat (2023). *e-Tariff Book User Guide*.
10. AidData (2021). *Banking on the Belt and Road*.
    """)

    st.markdown('<p class="source-note">This tool is for pre-negotiation scenario analysis. '
                'Outputs are indicative ranges, not precise predictions.</p>', unsafe_allow_html=True)
