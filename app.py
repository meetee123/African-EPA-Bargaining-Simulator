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
    page_title="Economic Partnership Agreements Bargaining Engine",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #13343B; }
    [data-testid="stSidebar"] * { color: #F3F3EE !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #BCE2E7 !important; font-weight: 500;
    }
    .metric-card {
        background: #F3F3EE; padding: 14px 18px; border-radius: 8px;
        border-left: 4px solid #20808D; margin-bottom: 8px;
    }
    .metric-card .label { font-size: 0.82rem; color: #2E565D; margin-bottom: 2px; }
    .metric-card .value { font-size: 1.55rem; font-weight: 700; color: #13343B; }
    .source-note {
        font-size: 0.78rem; color: #2E565D; margin-top: 6px;
        border-top: 1px solid #E5E3D4; padding-top: 6px;
    }
    h1, h2, h3 { color: #13343B !important; }
    div[data-testid="stExpander"] { border: 1px solid #E5E3D4; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    """Render a styled metric card."""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# COLOUR PALETTE 
# ══════════════════════════════════════════════════════════════

C = {
    "teal": "#20808D",
    "rust": "#A84B2F",
    "dark": "#1B474D",
    "cyan": "#BCE2E7",
    "mauve": "#944454",
    "gold": "#FFC553",
    "olive": "#848456",
    "brown": "#6E522B",
    "bg": "#FCFAF6",
    "paper": "#F3F3EE",
    "text": "#13343B",
    "muted": "#2E565D",
}
SEQ = [C["teal"], C["rust"], C["dark"], C["cyan"], C["mauve"], C["gold"], C["olive"], C["brown"]]

LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color=C["text"]),
    paper_bgcolor=C["bg"], plot_bgcolor=C["paper"],
    margin=dict(l=40, r=40, t=60, b=40),
)


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
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(32,128,141,0.25)", line=dict(color=C["teal"], width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]),
                                 bgcolor=C["paper"]),
                      title=f"Bargaining Power: {country}", height=420,
                      showlegend=False, **LAYOUT)
    return fig


def viz_radar_compare(bpi_dict):
    cats = ["Diversification", "Capacity", "Outside Options", "Policy Space", "Econ. Weight"]
    fig = go.Figure()
    for i, (c, b) in enumerate(bpi_dict.items()):
        vals = [b["Diversification"], b["Capacity"], b["Outside"], 100 - b["Lock_in"], b["Weight"]]
        col = SEQ[i % len(SEQ)]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], name=c,
            line=dict(color=col, width=2), fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.1)"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]),
                                 bgcolor=C["paper"]),
                      title="Comparative Bargaining Power", height=480,
                      legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"), **LAYOUT)
    return fig


def viz_sunburst(td, country):
    labs = ["Trade", "Exports", "Imports",
            "EU(X)", "China(X)", "US(X)", "Africa(X)", "Other(X)",
            "EU(M)", "China(M)", "US(M)", "Africa(M)", "Other(M)"]
    pars = ["", "Trade", "Trade",
            "Exports", "Exports", "Exports", "Exports", "Exports",
            "Imports", "Imports", "Imports", "Imports", "Imports"]
    oth_x = max(0, 100 - td["x_eu"] - td["x_cn"] - td["x_us"] - td["x_af"])
    oth_m = max(0, 100 - td["m_eu"] - td["m_cn"] - td["m_us"] - td["m_af"])
    vals = [0, 50, 50, td["x_eu"], td["x_cn"], td["x_us"], td["x_af"], oth_x,
            td["m_eu"], td["m_cn"], td["m_us"], td["m_af"], oth_m]
    fig = go.Figure(go.Sunburst(labels=labs, parents=pars, values=vals, branchvalues="total",
        marker=dict(colors=[C["paper"], C["teal"], C["rust"],
            C["dark"], C["mauve"], C["gold"], C["teal"], C["olive"],
            C["dark"], C["mauve"], C["gold"], C["teal"], C["olive"]]),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"))
    fig.update_layout(title=f"{country}: Trade Partner Dependence (%)", height=440, **LAYOUT)
    return fig


def viz_epa(epa_df, country):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=epa_df["year"], y=epa_df["pct_liberalised"],
        name="Lines Liberalised (%)", marker_color=C["teal"], opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=epa_df["year"], y=epa_df["revenue_loss_mn"],
        name="Revenue Loss (USD mn)", line=dict(color=C["rust"], width=2),
        mode="lines+markers", marker=dict(size=6)), secondary_y=True)
    fig.update_layout(title=f"{country}: EPA Liberalisation & Revenue Impact", height=380,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), **LAYOUT)
    fig.update_yaxes(title_text="Liberalised (%)", secondary_y=False)
    fig.update_yaxes(title_text="Revenue Loss (USD mn)", secondary_y=True)
    return fig


def viz_gp(gp, country):
    cats = ["Infra Loans\n(USD bn)", "FDI Stock\n(USD bn)", "EU Dev Aid\n(×100 mn)", "AGOA Exports\n(×100 mn)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="China", x=cats, y=[gp["cn_loan"], gp["cn_fdi"], 0, 0], marker_color=C["mauve"]))
    fig.add_trace(go.Bar(name="EU", x=cats, y=[0, 0, gp["eu_aid"]/100, 0], marker_color=C["teal"]))
    fig.add_trace(go.Bar(name="US", x=cats, y=[0, 0, 0, gp["agoa_x"]/100], marker_color=C["gold"]))
    fig.update_layout(title=f"{country}: Great-Power Economic Footprint", barmode="group",
                      height=370, legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), **LAYOUT)
    return fig


def viz_threats_bar(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Threat"], y=df["Africa"], name="Africa", marker_color=C["teal"]))
    fig.add_trace(go.Bar(x=df["Threat"], y=df["EU"], name="EU", marker_color=C["rust"]))
    sqa = df["SQ_Africa"].iloc[0]; sqe = df["SQ_EU"].iloc[0]
    fig.add_hline(y=sqa, line_dash="dash", line_color=C["teal"],
                  annotation_text=f"Africa SQ: {sqa:.0f}", annotation_position="top left")
    fig.add_hline(y=sqe, line_dash="dash", line_color=C["rust"],
                  annotation_text=f"EU SQ: {sqe:.0f}", annotation_position="bottom right")
    fig.update_layout(title="Threat Payoff Comparison", barmode="group", height=400,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), **LAYOUT)
    return fig


def viz_threats_bubble(df):
    df = df.copy()
    df["Improve"] = df["Africa"] - df["SQ_Africa"]
    df["Impact"] = df["SQ_EU"] - df["EU"]
    cmap = {"Yes": C["teal"], "No": C["rust"], "Partially": C["gold"]}
    fig = go.Figure(go.Scatter(
        x=df["Improve"], y=df["Impact"], mode="markers+text",
        marker=dict(size=df["Score"] * 50 + 12, color=[cmap.get(c, C["muted"]) for c in df["Credible"]],
                    line=dict(width=1, color="#fff"), opacity=0.85),
        text=df["Threat"], textposition="top center", textfont=dict(size=9),
        hovertemplate="<b>%{text}</b><br>Improve: %{x:.1f}<br>EU Impact: %{y:.1f}<extra></extra>",
        showlegend=False))
    fig.add_vline(x=0, line_dash="dot", line_color="#999")
    fig.add_hline(y=0, line_dash="dot", line_color="#999")
    fig.update_layout(title="Threat Credibility Map", height=400,
        xaxis_title="Africa Payoff Improvement over SQ",
        yaxis_title="EU Payoff Worsening (higher = more pressure)", **LAYOUT)
    return fig


def viz_sens_line(df, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["value"], y=df["Africa"], mode="lines+markers",
        name="Africa", line=dict(color=C["teal"], width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["EU"], mode="lines+markers",
        name="EU", line=dict(color=C["rust"], width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["value"], y=df["AfCFTA"], mode="lines+markers",
        name="AfCFTA", line=dict(color=C["gold"], width=2), marker=dict(size=5)))
    fig.update_layout(title=f"Sensitivity: {name}", xaxis_title=name,
        yaxis_title="Equilibrium Payoff", height=400,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), **LAYOUT)
    return fig


def viz_sens_heat(df, a1, a2):
    piv = df.pivot_table(index=a2, columns=a1, values="Africa", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=piv.values,
        x=[f"{v:.2f}" for v in piv.columns], y=[f"{v:.2f}" for v in piv.index],
        colorscale=[[0, C["rust"]], [0.5, C["paper"]], [1, C["teal"]]],
        colorbar=dict(title="Africa Payoff"),
        hovertemplate=f"{a1}: %{{x}}<br>{a2}: %{{y}}<br>Payoff: %{{z:.1f}}<extra></extra>"))
    fig.update_layout(title=f"Africa Payoff: {a1} vs {a2}", xaxis_title=a1, yaxis_title=a2,
                      height=440, **LAYOUT)
    return fig


def viz_sequence(df, country):
    cmap = {"Phase 1 (Immediate)": C["teal"], "Phase 2 (Medium-term)": C["gold"], "Phase 3 (Deferred)": C["rust"]}
    fig = go.Figure(go.Bar(y=df["Sector"], x=df["Net"], orientation="h",
        marker_color=[cmap.get(p, C["muted"]) for p in df["Phase"]],
        text=[f"{v:.1f}" for v in df["Net"]], textposition="outside"))
    fig.update_layout(title=f"{country}: Optimal Sector Concession Sequence",
                      xaxis_title="Net Benefit Score", height=370, **LAYOUT)
    return fig


def viz_bpi_bar(bpi_dict):
    items = sorted(bpi_dict.items(), key=lambda x: x[1]["BPI"])
    fig = go.Figure(go.Bar(y=[i[0] for i in items], x=[i[1]["BPI"] for i in items],
        orientation="h", marker_color=[C["teal"] if i[1]["BPI"] >= 50 else C["rust"] for i in items],
        text=[f"{i[1]['BPI']:.1f}" for i in items], textposition="outside"))
    fig.add_vline(x=50, line_dash="dash", line_color="#999", annotation_text="Neutral")
    fig.update_layout(title="Bargaining Power Index Comparison", xaxis_title="BPI (0-100)",
        xaxis=dict(range=[0, 100]), height=max(300, len(items) * 55), **LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 Economic Partnership Agreements Bargaining Engine")
    st.caption("Sequential bargaining engine for African trade negotiators.")
    st.markdown("---")

    st.markdown("### Country & Peers")
    country = st.selectbox("Primary Country", COUNTRIES, index=0)
    peers = st.multiselect("Comparison Countries",
        [c for c in COUNTRIES if c != country], default=["Kenya", "Nigeria"])

    st.markdown("---")
    st.markdown("### Game Parameters")

    with st.expander("Discount Factors (Patience)"):
        d_af = st.slider("African State δ", 0.50, 0.99, 0.85, 0.01,
            help="Higher = more patient. Default 0.85 reflects moderate time pressure.")
        d_eu = st.slider("EU δ", 0.50, 0.99, 0.92, 0.01,
            help="EU is typically more patient (institutional continuity). Default 0.92.")
        d_ac = st.slider("AfCFTA Council δ", 0.50, 0.99, 0.80, 0.01)

    with st.expander("Behavioural Perturbations"):
        sq_bias = st.slider("Status-Quo Bias", 0.0, 0.50, 0.15, 0.01,
            help="Samuelson & Zeckhauser (1988). Range 0.10-0.30.")
        loss_av = st.slider("Loss Aversion (λ)", 1.0, 4.0, 2.25, 0.05,
            help="Kahneman & Tversky (1979). Literature: 1.5-2.5.")
        ambig = st.slider("Ambiguity Premium", 0.0, 0.30, 0.10, 0.01,
            help="Ellsberg (1961). Discount for uncertain AfCFTA outcomes.")

    with st.expander("EPA Lock-in Costs"):
        epa_sunk = st.slider("EPA Sunk Cost", 0.0, 0.50, 0.20, 0.01)
        epa_mfn = st.slider("MFN Clause Penalty", 0.0, 0.30, 0.12, 0.01,
            help="Cost of triggering EPA MFN clause by offering better AfCFTA terms.")
        epa_still = st.slider("Standstill Constraint", 0.0, 0.20, 0.08, 0.01)

    with st.expander("Great-Power Shadow"):
        cn_infra = st.slider("Chinese Infra Offer", 0.0, 0.50, 0.20, 0.01)
        cn_debt = st.slider("Chinese Debt Constraint", 0.0, 0.30, 0.10, 0.01)
        us_agoa = st.slider("AGOA Withdrawal Risk", 0.0, 0.20, 0.08, 0.01)
        eu_cond = st.slider("EU Aid Conditionality", 0.0, 0.30, 0.12, 0.01)

    with st.expander("AfCFTA Opportunity"):
        ac_mkt = st.slider("Market Access Gain", 0.10, 0.60, 0.35, 0.01)
        ac_ind = st.slider("Industrialisation Bonus", 0.0, 0.30, 0.15, 0.01)

    depth = st.slider("Game Depth (rounds)", 1, 3, 2,
        help="Alternating-move rounds. Depth 2 ≈ 750 nodes. Depth 3 is slower.")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Section", [
        "Dashboard", "Game Tree & Equilibrium", "Threat Points",
        "Sensitivity Analysis", "Concession Sequencing",
        "Comparative Analysis", "Data Explorer", "Methodology",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.68rem;opacity:0.55;">'
        'Data: UN Comtrade · EU Access2Markets · AfCFTA e-Tariff Book · '
        'World Bank WGI · AidData<br>v1.0 · Mar 2026</p>',
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
    st.markdown(f"# {country}: Bargaining Asymmetry Dashboard")
    st.caption("Pre-negotiation intelligence for AfCFTA-EPA strategy. Adjust parameters in the sidebar.")

    bp = bpi(td, wgi, gp, par)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Bargaining Power Index", f"{bp['BPI']:.0f} / 100")
    with c2: metric_card("GDP (USD bn)", f"${PROFILES[country]['gdp']:.0f}B")
    with c3: metric_card("EU Trade Dependence", f"{(td['x_eu']+td['m_eu'])/2:.1f}%")
    with c4: metric_card("Negotiation Capacity", f"{wgi['nci']:.2f}")

    st.markdown("---")
    l, r = st.columns(2)
    with l: st.plotly_chart(viz_radar(bp, country), use_container_width=True)
    with r: st.plotly_chart(viz_sunburst(td, country), use_container_width=True)

    edf = epa_schedule(country)
    if edf["pct_liberalised"].sum() > 0:
        st.plotly_chart(viz_epa(edf, country), use_container_width=True)
    else:
        st.info(f"{country} has no active EPA liberalisation schedule.")

    st.plotly_chart(viz_gp(gp, country), use_container_width=True)

    with st.expander("Full Country Profile"):
        rows = [
            ("Economy", "GDP (USD bn)", PROFILES[country]["gdp"]),
            ("Economy", "Population (mn)", PROFILES[country]["pop"]),
            ("Economy", "Trade Openness", f"{td['openness']:.0%}"),
            ("Trade", "Exports to EU (%)", td["x_eu"]),
            ("Trade", "Imports from EU (%)", td["m_eu"]),
            ("Trade", "Exports to China (%)", td["x_cn"]),
            ("Trade", "Imports from China (%)", td["m_cn"]),
            ("Trade", "Exports to Africa (%)", td["x_af"]),
            ("Governance", "Govt Effectiveness (WGI)", wgi["ge"]),
            ("Governance", "Regulatory Quality (WGI)", wgi["rq"]),
            ("Governance", "Neg. Capacity Index", wgi["nci"]),
            ("Great Powers", "Chinese Loans (USD bn)", gp["cn_loan"]),
            ("Great Powers", "Debt to China (% GDP)", gp["cn_debt_gdp"]),
            ("Great Powers", "EU Dev. Aid (USD mn)", gp["eu_aid"]),
            ("EPA", "Arrangement", PROFILES[country]["epa"]),
            ("AfCFTA", "Schedule Submitted", "Yes" if PROFILES[country]["afcfta_sched"] else "No"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Category", "Indicator", "Value"]),
                     use_container_width=True, hide_index=True)

    st.markdown('<p class="source-note">Sources: UN Comtrade 2022 · EU Access2Markets · '
                'World Bank WGI 2022 · AidData · USTR AGOA</p>', unsafe_allow_html=True)


# ──────────── GAME TREE ────────────
elif page == "Game Tree & Equilibrium":
    st.markdown(f"# {country}: Sequential Game Tree")
    st.caption("Extensive-form game solved via backward induction (Subgame Perfect Equilibrium).")

    with st.spinner("Building & solving game tree…"):
        tree = build_tree(par, td, wgi, depth)
        eq_path, eq_pay = solve(tree)

    st.markdown("### Equilibrium Outcome")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Africa Payoff", f"{eq_pay['africa']:.1f}")
    with c2: metric_card("EU Payoff", f"{eq_pay['eu']:.1f}")
    with c3: metric_card("AfCFTA Payoff", f"{eq_pay['afcfta']:.1f}")

    st.markdown("### Equilibrium Path")
    eq_rows = []
    for nid in eq_path:
        n = tree[nid]
        if n.action != "START":
            eq_rows.append(dict(Round=n.rnd, Player=n.player, Action=LABELS.get(n.action, n.action)))
    if eq_rows:
        st.dataframe(pd.DataFrame(eq_rows), use_container_width=True, hide_index=True)

    st.markdown("### Interactive Game Tree")
    st.caption("Teal = Africa nodes · Rust = EU nodes · Diamonds = terminal · Bold path = equilibrium.")
    st.plotly_chart(viz_game_tree(tree), use_container_width=True)

    with st.expander("Game Statistics"):
        total = len(tree); terms = sum(1 for n in tree.values() if n.terminal)
        st.markdown(f"- Total nodes: **{total}**  \n- Terminal: **{terms}**  \n- Depth: **{depth}** rounds")

    st.markdown('<p class="source-note">Model: Selten (1965) backward induction; '
                'Kahneman & Tversky (1979) prospect theory; '
                'Samuelson & Zeckhauser (1988) status-quo bias.</p>', unsafe_allow_html=True)


# ──────────── THREAT POINTS ────────────
elif page == "Threat Points":
    st.markdown(f"# {country}: Credible Threat Analysis")
    st.caption("A threat is credible if executing it beats backing down and worsens the counterparty.")

    tdf = threat_points(par, td, wgi)
    l, r = st.columns(2)
    with l: st.plotly_chart(viz_threats_bar(tdf), use_container_width=True)
    with r: st.plotly_chart(viz_threats_bubble(tdf), use_container_width=True)

    st.markdown("### Threat Detail")
    st.dataframe(tdf[["Threat", "Africa", "EU", "Credible", "Score", "Implication"]],
                 use_container_width=True, hide_index=True)

    st.markdown("### Red-Line Identification")
    for _, row in tdf[tdf["Credible"].isin(["Yes", "Partially"])].iterrows():
        icon = "🟢" if row["Credible"] == "Yes" else "🟡"
        st.markdown(f"{icon} **{row['Threat']}** — {row['Implication']}")
    for _, row in tdf[tdf["Credible"] == "No"].iterrows():
        st.markdown(f"🔴 **{row['Threat']}** — {row['Implication']}")

    # CSV export
    csv = tdf.to_csv(index=False).encode()
    st.download_button("Download threat analysis (CSV)", csv,
                       f"threats_{country.replace(' ','_')}_{datetime.now():%Y%m%d}.csv", "text/csv")

    st.markdown('<p class="source-note">Framework: Schelling (1960) credible commitment; '
                'Nash (1953) outside-option bargaining.</p>', unsafe_allow_html=True)


# ──────────── SENSITIVITY ────────────
elif page == "Sensitivity Analysis":
    st.markdown(f"# {country}: Sensitivity Analysis")
    st.caption("How equilibrium outcomes shift as parameters change.")

    PMAP = {
        "Chinese Infra Offer": ("cn_infra", 0.0, 0.50),
        "Status-Quo Bias": ("sq_bias", 0.0, 0.50),
        "Loss Aversion (λ)": ("loss_av", 1.0, 4.0),
        "MFN Clause Penalty": ("epa_mfn", 0.0, 0.30),
        "AfCFTA Market Gain": ("ac_mkt", 0.10, 0.60),
        "EU Aid Conditionality": ("eu_cond", 0.0, 0.30),
        "Africa δ": ("d_af", 0.50, 0.99),
        "Ambiguity Premium": ("ambig", 0.0, 0.30),
    }

    st.markdown("### Single-Parameter Sweep")
    sel = st.selectbox("Parameter", list(PMAP.keys()))
    attr, lo, hi = PMAP[sel]
    with st.spinner("Sweeping…"):
        sdf = sensitivity_1d(par, td, wgi, attr, np.linspace(lo, hi, 20))
    st.plotly_chart(viz_sens_line(sdf, sel), use_container_width=True)

    st.markdown("---")
    st.markdown("### Two-Parameter Heatmap")
    cl, cr = st.columns(2)
    with cl: p1n = st.selectbox("X-axis", list(PMAP.keys()), 0, key="px")
    with cr: p2n = st.selectbox("Y-axis", [p for p in PMAP if p != p1n], 0, key="py")
    a1, lo1, hi1 = PMAP[p1n]; a2, lo2, hi2 = PMAP[p2n]

    with st.spinner("Computing surface…"):
        mdf = sensitivity_2d(par, td, wgi, a1, np.linspace(lo1, hi1, 10), a2, np.linspace(lo2, hi2, 10))
    st.plotly_chart(viz_sens_heat(mdf, a1, a2), use_container_width=True)

    st.markdown('<p class="source-note">Methodology: Saltelli et al. (2008) global sensitivity analysis.</p>',
                unsafe_allow_html=True)


# ──────────── CONCESSION SEQUENCING ────────────
elif page == "Concession Sequencing":
    st.markdown(f"# {country}: Optimal Concession Sequence")
    st.caption("Optimal order of AfCFTA tariff concessions, balancing gains vs EPA constraints.")

    sdf = optimal_sequence(par, td, wgi)
    st.plotly_chart(viz_sequence(sdf, country), use_container_width=True)

    st.markdown("### Sequencing Detail")
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.markdown("### AfCFTA Tariff Categories")
    ac = _AFCFTA_CAT.get(country, dict(a=90, b=7, c=3, mfn=12))
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Cat A (Non-sensitive)", f"{ac['a']:.1f}%")
    with c2: metric_card("Cat B (Sensitive)", f"{ac['b']:.1f}%")
    with c3: metric_card("Cat C (Exclusion)", f"{ac['c']:.1f}%")
    with c4: metric_card("Avg MFN Base Rate", f"{ac['mfn']:.1f}%")

    csv = sdf.to_csv(index=False).encode()
    st.download_button("Download sequence (CSV)", csv,
                       f"sequence_{country.replace(' ','_')}_{datetime.now():%Y%m%d}.csv", "text/csv")

    st.markdown('<p class="source-note">AfCFTA categories: Art. 7 Protocol on Trade in Goods — '
                '90 % non-sensitive (5/10 yr), 7 % sensitive (10/13 yr), 3 % exclusion. '
                'Source: AfCFTA e-Tariff Book.</p>', unsafe_allow_html=True)


# ──────────── COMPARATIVE ────────────
elif page == "Comparative Analysis":
    st.markdown("# Cross-Country Comparative Analysis")
    st.caption("Benchmark bargaining positions to identify coalition partners.")

    all_c = [country] + peers
    bd = {}
    for c in all_c:
        cp = Params(**{k: getattr(par, k) for k in par.__dataclass_fields__})
        cp.cap = _WGI[c]["nci"] / 0.5
        bd[c] = bpi(_TRADE[c], _WGI[c], _GP[c], cp)

    st.plotly_chart(viz_bpi_bar(bd), use_container_width=True)
    st.plotly_chart(viz_radar_compare(bd), use_container_width=True)

    st.markdown("### Comparison Table")
    rows = []
    for c in all_c:
        p = PROFILES[c]; t = _TRADE[c]; w = _WGI[c]; g = _GP[c]
        rows.append(dict(Country=c, GDP=p["gdp"], EU_Export=t["x_eu"],
            China_Import=t["m_cn"], Africa_Export=t["x_af"],
            Neg_Capacity=w["nci"], CN_Debt_GDP=g["cn_debt_gdp"],
            BPI=bd[c]["BPI"], EPA=p["epa"]))
    cdf = pd.DataFrame(rows).sort_values("BPI", ascending=False)
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.markdown("### Coalition Identification")
    primary = bd[country]["BPI"]
    close = sorted([(c, abs(bd[c]["BPI"] - primary)) for c in all_c if c != country], key=lambda x: x[1])
    if close:
        st.markdown(f"Closest coalition partner: **{close[0][0]}** (BPI gap: {close[0][1]:.1f})")

    csv = cdf.to_csv(index=False).encode()
    st.download_button("Download comparison (CSV)", csv,
                       f"comparison_{datetime.now():%Y%m%d}.csv", "text/csv")


# ──────────── DATA EXPLORER ────────────
elif page == "Data Explorer":
    st.markdown("# Data Explorer")
    st.caption("Browse and export all underlying datasets.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Profiles", "Trade Dependence", "WGI Governance", "Great Powers", "Sectors"])

    with tab1:
        df = pd.DataFrame([dict(Country=c, **PROFILES[c]) for c in COUNTRIES])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        df = pd.DataFrame([dict(Country=c, **_TRADE[c]) for c in COUNTRIES])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        df = pd.DataFrame([dict(Country=c, **_WGI[c]) for c in COUNTRIES])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab4:
        df = pd.DataFrame([dict(Country=c, **_GP[c]) for c in COUNTRIES])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab5:
        df = pd.DataFrame([dict(Sector=k, **v) for k, v in SECTORS.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Export")
    choice = st.selectbox("Dataset", ["Profiles", "Trade", "WGI", "Great Powers", "EPA Schedule"])
    if st.button("Generate CSV"):
        if choice == "Profiles":
            out = pd.DataFrame([dict(Country=c, **PROFILES[c]) for c in COUNTRIES])
        elif choice == "Trade":
            out = pd.DataFrame([dict(Country=c, **_TRADE[c]) for c in COUNTRIES])
        elif choice == "WGI":
            out = pd.DataFrame([dict(Country=c, **_WGI[c]) for c in COUNTRIES])
        elif choice == "Great Powers":
            out = pd.DataFrame([dict(Country=c, **_GP[c]) for c in COUNTRIES])
        else:
            out = epa_schedule(country)
        st.download_button("Download", out.to_csv(index=False).encode(),
                           f"afcfta_{choice.lower().replace(' ','_')}_{datetime.now():%Y%m%d}.csv", "text/csv")


# ──────────── METHODOLOGY ────────────
elif page == "Methodology":
    st.markdown("# Methodology & Literature Grounding")

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
