"""
Visualization Layer: Charts, game trees, sensitivity plots, comparative dashboards.
Uses Plotly for interactive visualizations within Streamlit.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# BRAND PALETTE
# ─────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#20808D",       # Muted teal
    "secondary": "#A84B2F",     # Terra/rust
    "dark_teal": "#1B474D",
    "light_cyan": "#BCE2E7",
    "mauve": "#944454",
    "gold": "#FFC553",
    "olive": "#848456",
    "brown": "#6E522B",
    "bg": "#FCFAF6",
    "paper": "#F3F3EE",
    "text": "#13343B",
    "muted": "#2E565D",
}

CHART_SEQUENCE = [
    COLORS["primary"], COLORS["secondary"], COLORS["dark_teal"],
    COLORS["light_cyan"], COLORS["mauve"], COLORS["gold"],
    COLORS["olive"], COLORS["brown"]
]

PLAYER_COLORS = {
    "Africa": "#20808D",
    "EU": "#A84B2F",
    "AfCFTA": "#FFC553",
    "Nature": "#848456",
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["paper"],
    margin=dict(l=40, r=40, t=60, b=40),
)


# ─────────────────────────────────────────────────────────────
# 1. GAME TREE VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_game_tree(viz_nodes, viz_edges, title="Extensive-Form Game Tree"):
    """
    Render game tree as an interactive network diagram using Plotly.
    Highlights equilibrium path in bold.
    """
    if not viz_nodes:
        return go.Figure().update_layout(title="No game tree data available")

    # Assign positions using a layered layout
    positions = _compute_tree_layout(viz_nodes, viz_edges)

    # Edge traces
    edge_x, edge_y = [], []
    eq_edge_x, eq_edge_y = [], []

    for edge in viz_edges:
        src = positions.get(edge["source"], (0, 0))
        tgt = positions.get(edge["target"], (0, 0))
        if edge.get("is_equilibrium"):
            eq_edge_x += [src[0], tgt[0], None]
            eq_edge_y += [src[1], tgt[1], None]
        else:
            edge_x += [src[0], tgt[0], None]
            edge_y += [src[1], tgt[1], None]

    traces = []

    # Regular edges
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#C0C0C0"),
        hoverinfo="none", showlegend=False
    ))

    # Equilibrium path edges
    if eq_edge_x:
        traces.append(go.Scatter(
            x=eq_edge_x, y=eq_edge_y, mode="lines",
            line=dict(width=3, color=COLORS["secondary"]),
            hoverinfo="none", name="Equilibrium Path",
            showlegend=True
        ))

    # Node traces by player
    for player in ["Africa", "EU", "AfCFTA", "Nature"]:
        player_nodes = [n for n in viz_nodes if n["player"] == player]
        if not player_nodes:
            continue

        node_x = [positions.get(n["id"], (0, 0))[0] for n in player_nodes]
        node_y = [positions.get(n["id"], (0, 0))[1] for n in player_nodes]
        node_text = [n["label"] for n in player_nodes]
        node_size = [14 if n.get("is_equilibrium") else 8 for n in player_nodes]
        node_symbol = ["diamond" if n.get("is_terminal") else "circle" for n in player_nodes]

        hover_text = []
        for n in player_nodes:
            ht = f"<b>{n['label']}</b><br>Player: {n['player']}<br>Round: {n['round']}"
            if n.get("is_terminal"):
                ht += f"<br>Africa: {n['payoff_africa']:.1f}<br>EU: {n['payoff_eu']:.1f}<br>AfCFTA: {n['payoff_afcfta']:.1f}"
            if n.get("is_equilibrium"):
                ht += "<br><b>★ Equilibrium</b>"
            hover_text.append(ht)

        traces.append(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=node_size,
                color=PLAYER_COLORS.get(player, "#888"),
                symbol=node_symbol,
                line=dict(width=1, color="#fff")
            ),
            text=hover_text, hoverinfo="text",
            name=player, showlegend=True
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def _compute_tree_layout(viz_nodes, viz_edges):
    """Simple layered tree layout: x = breadth position, y = -depth."""
    # Build adjacency
    children_map = {}
    parent_map = {}
    for edge in viz_edges:
        children_map.setdefault(edge["source"], []).append(edge["target"])
        parent_map[edge["target"]] = edge["source"]

    # Find root
    all_targets = {e["target"] for e in viz_edges}
    all_sources = {e["source"] for e in viz_edges}
    roots = all_sources - all_targets
    if not roots:
        roots = {viz_nodes[0]["id"]} if viz_nodes else set()

    positions = {}
    level_counts = {}

    def assign_pos(node_id, depth):
        if node_id in positions:
            return
        count = level_counts.get(depth, 0)
        level_counts[depth] = count + 1
        positions[node_id] = (count, -depth)
        for child in children_map.get(node_id, []):
            assign_pos(child, depth + 1)

    for root in roots:
        assign_pos(root, 0)

    # Normalize x positions per level
    max_per_level = {}
    for nid, (x, y) in positions.items():
        level = -y
        max_per_level[level] = max(max_per_level.get(level, 0), x)

    for nid in positions:
        x, y = positions[nid]
        level = -y
        max_x = max_per_level.get(level, 1) or 1
        positions[nid] = (x / max_x if max_x > 0 else 0.5, y)

    return positions


# ─────────────────────────────────────────────────────────────
# 2. BARGAINING POWER RADAR CHART
# ─────────────────────────────────────────────────────────────

def plot_bargaining_power_radar(bpi_data: dict, country: str):
    """Spider/radar chart showing BPI components."""
    categories = ["Trade Diversification", "Governance Capacity", "Outside Options",
                   "EPA Lock-in Penalty", "Economic Weight"]
    # Invert lock-in (higher = worse, so show as policy space)
    values = [
        bpi_data.get("Trade Diversification", 50),
        bpi_data.get("Governance Capacity", 50),
        bpi_data.get("Outside Options", 50),
        100 - bpi_data.get("EPA Lock-in Penalty", 20),
        bpi_data.get("Economic Weight", 10),
    ]
    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed,
        fill="toself", fillcolor=f"rgba(32, 128, 141, 0.25)",
        line=dict(color=COLORS["primary"], width=2),
        name=country
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            bgcolor=COLORS["paper"],
        ),
        title=dict(text=f"Bargaining Power Profile: {country}", font=dict(size=16)),
        height=420,
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_comparative_radar(bpi_dict: dict):
    """Overlay radar charts for multiple countries."""
    categories = ["Trade Diversification", "Governance Capacity", "Outside Options",
                   "Policy Space (inv. lock-in)", "Economic Weight"]
    fig = go.Figure()

    for i, (country, bpi) in enumerate(bpi_dict.items()):
        values = [
            bpi.get("Trade Diversification", 50),
            bpi.get("Governance Capacity", 50),
            bpi.get("Outside Options", 50),
            100 - bpi.get("EPA Lock-in Penalty", 20),
            bpi.get("Economic Weight", 10),
        ]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        color = CHART_SEQUENCE[i % len(CHART_SEQUENCE)]
        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            name=country,
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            bgcolor=COLORS["paper"],
        ),
        title=dict(text="Comparative Bargaining Power Profiles", font=dict(size=16)),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 3. SENSITIVITY ANALYSIS CHARTS
# ─────────────────────────────────────────────────────────────

def plot_sensitivity_line(df: pd.DataFrame, param_name: str):
    """Line chart: payoffs vs parameter value."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Parameter Value"], y=df["Africa Payoff"],
        mode="lines+markers", name="Africa",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["Parameter Value"], y=df["EU Payoff"],
        mode="lines+markers", name="EU",
        line=dict(color=COLORS["secondary"], width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["Parameter Value"], y=df["AfCFTA Payoff"],
        mode="lines+markers", name="AfCFTA Council",
        line=dict(color=COLORS["gold"], width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title=dict(text=f"Sensitivity: Payoffs vs {param_name}", font=dict(size=16)),
        xaxis_title=param_name,
        yaxis_title="Equilibrium Payoff",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_sensitivity_heatmap(df: pd.DataFrame, param1: str, param2: str):
    """Heatmap: Africa payoff across two parameter dimensions."""
    pivot = df.pivot_table(index=param2, columns=param1, values="Africa Payoff", aggfunc="mean")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{v:.2f}" for v in pivot.columns],
        y=[f"{v:.2f}" for v in pivot.index],
        colorscale=[[0, COLORS["secondary"]], [0.5, COLORS["paper"]], [1, COLORS["primary"]]],
        colorbar=dict(title="Africa Payoff"),
        hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>Payoff: %{{z:.1f}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Africa Payoff: {param1} vs {param2}", font=dict(size=16)),
        xaxis_title=param1,
        yaxis_title=param2,
        height=450,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 4. THREAT POINT VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_threat_analysis(threats_df: pd.DataFrame):
    """Grouped bar chart comparing threat payoffs to status quo."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=threats_df["Threat"], y=threats_df["Africa Payoff"],
        name="Africa Payoff", marker_color=COLORS["primary"],
    ))
    fig.add_trace(go.Bar(
        x=threats_df["Threat"], y=threats_df["EU Payoff"],
        name="EU Payoff", marker_color=COLORS["secondary"],
    ))

    # Status quo lines
    sq_a = threats_df["Status Quo (Africa)"].iloc[0]
    sq_eu = threats_df["Status Quo (EU)"].iloc[0]
    fig.add_hline(y=sq_a, line_dash="dash", line_color=COLORS["primary"],
                  annotation_text=f"Africa SQ: {sq_a:.0f}", annotation_position="top left")
    fig.add_hline(y=sq_eu, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text=f"EU SQ: {sq_eu:.0f}", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text="Threat Point Analysis: Payoff Comparison", font=dict(size=16)),
        xaxis_title="Threat Strategy",
        yaxis_title="Payoff",
        barmode="group",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_threat_credibility(threats_df: pd.DataFrame):
    """Bubble chart: credibility score vs payoff improvement."""
    threats_df = threats_df.copy()
    threats_df["Payoff Improvement"] = threats_df["Africa Payoff"] - threats_df["Status Quo (Africa)"]
    threats_df["EU Impact"] = threats_df["Status Quo (EU)"] - threats_df["EU Payoff"]

    color_map = {"Yes": COLORS["primary"], "No": COLORS["secondary"], "Partially": COLORS["gold"]}
    colors = [color_map.get(c, COLORS["muted"]) for c in threats_df["Credible"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threats_df["Payoff Improvement"],
        y=threats_df["EU Impact"],
        mode="markers+text",
        marker=dict(
            size=threats_df["Credibility Score"] * 50 + 10,
            color=colors,
            line=dict(width=1, color="#fff"),
            opacity=0.8,
        ),
        text=threats_df["Threat"],
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate="<b>%{text}</b><br>Africa Improvement: %{x:.1f}<br>EU Impact: %{y:.1f}<extra></extra>",
        showlegend=False,
    ))

    fig.add_vline(x=0, line_dash="dot", line_color="#999")
    fig.add_hline(y=0, line_dash="dot", line_color="#999")

    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                       text="Credible & Effective", showarrow=False,
                       font=dict(color=COLORS["primary"], size=11))
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                       text="Not Credible", showarrow=False,
                       font=dict(color=COLORS["secondary"], size=11))

    fig.update_layout(
        title=dict(text="Threat Credibility Map", font=dict(size=16)),
        xaxis_title="Africa's Payoff Improvement over Status Quo",
        yaxis_title="EU's Payoff Worsening (higher = more pressure)",
        height=420,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 5. EPA SCHEDULE TIMELINE
# ─────────────────────────────────────────────────────────────

def plot_epa_timeline(epa_df: pd.DataFrame, country: str):
    """Dual-axis: liberalisation % and revenue loss over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=epa_df["years"], y=epa_df["pct_liberalised"],
        name="Tariff Lines Liberalised (%)",
        marker_color=COLORS["primary"],
        opacity=0.7,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=epa_df["years"], y=epa_df["tariff_revenue_loss_usd_mn"],
        name="Revenue Loss (USD mn)",
        line=dict(color=COLORS["secondary"], width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text=f"{country}: EPA Liberalisation Schedule & Revenue Impact", font=dict(size=16)),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text="Tariff Lines Liberalised (%)", secondary_y=False)
    fig.update_yaxes(title_text="Revenue Loss (USD mn)", secondary_y=True)

    return fig


# ─────────────────────────────────────────────────────────────
# 6. TRADE DEPENDENCE SUNBURST
# ─────────────────────────────────────────────────────────────

def plot_trade_dependence(trade_dep: dict, country: str):
    """Sunburst chart of trade partner dependence."""
    labels = ["Total Trade", "Exports", "Imports",
              "EU (Exp)", "China (Exp)", "US (Exp)", "Africa (Exp)", "Other (Exp)",
              "EU (Imp)", "China (Imp)", "US (Imp)", "Africa (Imp)", "Other (Imp)"]

    exp_eu = trade_dep.get("export_to_eu_pct", 20)
    exp_cn = trade_dep.get("export_to_china_pct", 8)
    exp_us = trade_dep.get("export_to_us_pct", 5)
    exp_af = trade_dep.get("export_to_africa_pct", 15)
    exp_other = max(0, 100 - exp_eu - exp_cn - exp_us - exp_af)

    imp_eu = trade_dep.get("import_from_eu_pct", 20)
    imp_cn = trade_dep.get("import_from_china_pct", 18)
    imp_us = trade_dep.get("import_from_us_pct", 5)
    imp_af = trade_dep.get("import_from_africa_pct", 8)
    imp_other = max(0, 100 - imp_eu - imp_cn - imp_us - imp_af)

    parents = ["", "Total Trade", "Total Trade",
               "Exports", "Exports", "Exports", "Exports", "Exports",
               "Imports", "Imports", "Imports", "Imports", "Imports"]
    values = [0, 50, 50,
              exp_eu, exp_cn, exp_us, exp_af, exp_other,
              imp_eu, imp_cn, imp_us, imp_af, imp_other]

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=[
            COLORS["paper"], COLORS["primary"], COLORS["secondary"],
            COLORS["dark_teal"], COLORS["mauve"], COLORS["gold"], COLORS["primary"], COLORS["olive"],
            COLORS["dark_teal"], COLORS["mauve"], COLORS["gold"], COLORS["primary"], COLORS["olive"],
        ]),
        hovertemplate="<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{country}: Trade Partner Dependence (%)", font=dict(size=16)),
        height=450,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 7. CONCESSION SEQUENCE CHART
# ─────────────────────────────────────────────────────────────

def plot_concession_sequence(seq_df: pd.DataFrame, country: str):
    """Horizontal bar chart showing optimal sector sequencing."""
    color_map = {
        "Phase 1 (Immediate)": COLORS["primary"],
        "Phase 2 (Medium-term)": COLORS["gold"],
        "Phase 3 (Deferred)": COLORS["secondary"],
    }
    colors = [color_map.get(p, COLORS["muted"]) for p in seq_df["Priority"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=seq_df["Sector"],
        x=seq_df["Net Benefit"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in seq_df["Net Benefit"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Net Benefit: %{x:.1f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=f"{country}: Optimal Sector Concession Sequence", font=dict(size=16)),
        xaxis_title="Net Benefit Score",
        yaxis_title="",
        height=380,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 8. COMPARATIVE BPI BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_bpi_comparison(bpi_dict: dict):
    """Bar chart comparing BPI across countries."""
    countries = list(bpi_dict.keys())
    bpi_values = [bpi_dict[c]["BPI"] for c in countries]

    # Sort by BPI
    sorted_pairs = sorted(zip(countries, bpi_values), key=lambda x: x[1])
    countries = [p[0] for p in sorted_pairs]
    bpi_values = [p[1] for p in sorted_pairs]

    colors = [COLORS["primary"] if v >= 50 else COLORS["secondary"] for v in bpi_values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=countries,
        x=bpi_values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in bpi_values],
        textposition="outside",
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="#999",
                  annotation_text="Neutral", annotation_position="top")

    fig.update_layout(
        title=dict(text="Bargaining Power Index: Cross-Country Comparison", font=dict(size=16)),
        xaxis_title="BPI (0-100)",
        xaxis=dict(range=[0, 100]),
        yaxis_title="",
        height=max(300, len(countries) * 50),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 9. GREAT POWER INFLUENCE CHART
# ─────────────────────────────────────────────────────────────

def plot_great_power_influence(gp: dict, country: str):
    """Grouped bar chart of great power influence vectors."""
    categories = ["Infrastructure\nLoans (bn)", "FDI Stock\n(bn)", "Dev Aid\n(EU, mn)", "AGOA Exports\n(mn)"]
    china_vals = [gp.get("chinese_loans_usd_bn", 0), gp.get("chinese_fdi_stock_usd_bn", 0), 0, 0]
    eu_vals = [0, 0, gp.get("eu_development_aid_usd_mn", 0) / 100, 0]
    us_vals = [0, 0, 0, gp.get("agoa_exports_usd_mn", 0) / 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="China", x=categories, y=china_vals, marker_color=COLORS["mauve"]))
    fig.add_trace(go.Bar(name="EU", x=categories, y=eu_vals, marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(name="US", x=categories, y=us_vals, marker_color=COLORS["gold"]))

    fig.update_layout(
        title=dict(text=f"{country}: Great Power Economic Footprint", font=dict(size=16)),
        yaxis_title="USD (billions / hundreds of millions)",
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **LAYOUT_DEFAULTS,
    )
    return fig
