"""
AfCFTA–EPA Bargaining Asymmetry Simulator
Sequential Game Engine for African States

An interactive sequential-game simulator that quantifies bargaining power
asymmetries when negotiating AfCFTA protocols while locked into EPA schedules,
modeling great-power (China/US/EU) shadow influence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

from data_layer import (
    COUNTRY_PROFILES, SECTOR_DATA,
    get_all_countries, get_country_summary,
    get_epa_tariff_schedule, get_afcfta_tariff_categories,
    get_trade_dependence, get_wgi_scores, get_great_power_data,
)
from game_engine import (
    GameParameters, build_game_tree, backward_induction,
    compute_threat_points, run_sensitivity_analysis,
    run_multi_sensitivity, compute_optimal_sequence,
    compute_bargaining_power_index, flatten_tree_for_viz,
    ACTION_LABELS,
)
from visualisations import (
    plot_game_tree, plot_bargaining_power_radar, plot_comparative_radar,
    plot_sensitivity_line, plot_sensitivity_heatmap,
    plot_threat_analysis, plot_threat_credibility,
    plot_epa_timeline, plot_trade_dependence,
    plot_concession_sequence, plot_bpi_comparison,
    plot_great_power_influence, COLORS,
)


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AfCFTA–EPA Bargaining Asymmetry Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #13343B;
    }
    [data-testid="stSidebar"] * {
        color: #F3F3EE !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #BCE2E7 !important;
        font-weight: 500;
    }
    .stMetric {
        background-color: #F3F3EE;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid #20808D;
    }
    h1, h2, h3 {
        color: #13343B !important;
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #E5E3D4;
        border-radius: 8px;
    }
    .source-note {
        font-size: 0.78rem;
        color: #2E565D;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AfCFTA–EPA Simulator")
    st.markdown("Sequential bargaining game engine for African trade negotiators.")
    st.markdown("---")

    # Country selection
    st.markdown("### Country & Scenario")
    selected_country = st.selectbox(
        "Primary Country",
        get_all_countries(),
        index=0,
        help="Select the African state to analyse as the primary negotiator."
    )

    comparison_countries = st.multiselect(
        "Comparison Countries",
        [c for c in get_all_countries() if c != selected_country],
        default=["Kenya", "Nigeria"],
        help="Select peer countries for comparative analysis."
    )

    st.markdown("---")
    st.markdown("### Game Parameters")

    with st.expander("Discount Factors (Patience)", expanded=False):
        delta_africa = st.slider(
            "African State δ", 0.50, 0.99, 0.85, 0.01,
            help="Higher = more patient. Default 0.85 reflects moderate time pressure."
        )
        delta_eu = st.slider(
            "EU δ", 0.50, 0.99, 0.92, 0.01,
            help="EU is typically more patient due to institutional continuity. Default 0.92."
        )
        delta_afcfta = st.slider(
            "AfCFTA Council δ", 0.50, 0.99, 0.80, 0.01,
            help="AfCFTA faces implementation pressure. Default 0.80."
        )

    with st.expander("Behavioral Perturbations", expanded=False):
        status_quo_bias = st.slider(
            "Status-Quo Bias", 0.0, 0.50, 0.15, 0.01,
            help="Kahneman (1991): default bias inflating EPA lock-in costs. Literature range: 0.10-0.30."
        )
        loss_aversion = st.slider(
            "Loss Aversion (λ)", 1.0, 4.0, 2.25, 0.05,
            help="Tversky-Kahneman lambda: losses loom larger than gains. Literature: 1.5-2.5."
        )
        ambiguity_premium = st.slider(
            "Ambiguity Premium", 0.0, 0.30, 0.10, 0.01,
            help="Additional discount for uncertain AfCFTA outcomes. Ellsberg (1961)."
        )

    with st.expander("EPA Lock-in Costs", expanded=False):
        epa_sunk_cost = st.slider(
            "EPA Sunk Cost", 0.0, 0.50, 0.20, 0.01,
            help="Fraction of tariff revenue already foregone under EPA implementation."
        )
        epa_mfn_penalty = st.slider(
            "MFN Clause Penalty", 0.0, 0.30, 0.12, 0.01,
            help="Cost of triggering EPA's MFN clause by offering better AfCFTA terms."
        )
        epa_standstill = st.slider(
            "Standstill Constraint", 0.0, 0.20, 0.08, 0.01,
            help="Cost of inability to raise tariffs under EPA standstill provision."
        )

    with st.expander("Great-Power Shadow Effects", expanded=False):
        china_infra = st.slider(
            "Chinese Infrastructure Offer", 0.0, 0.50, 0.20, 0.01,
            help="Value of Chinese BRI infrastructure as outside option in bargaining."
        )
        china_debt = st.slider(
            "Chinese Debt Constraint", 0.0, 0.30, 0.10, 0.01,
            help="Constraining effect of existing Chinese debt on policy autonomy."
        )
        us_agoa = st.slider(
            "AGOA Withdrawal Risk", 0.0, 0.20, 0.08, 0.01,
            help="Risk of US withdrawing AGOA eligibility."
        )
        eu_conditionality = st.slider(
            "EU Aid Conditionality", 0.0, 0.30, 0.12, 0.01,
            help="EU development aid conditioned on EPA compliance."
        )

    with st.expander("AfCFTA Opportunity", expanded=False):
        afcfta_market = st.slider(
            "Market Access Gain", 0.10, 0.60, 0.35, 0.01,
            help="Potential gain from continental market access under AfCFTA."
        )
        afcfta_industry = st.slider(
            "Industrialisation Bonus", 0.0, 0.30, 0.15, 0.01,
            help="Value-chain development multiplier from AfCFTA integration."
        )

    game_depth = st.slider(
        "Game Depth (rounds)", 1, 3, 2,
        help="Number of alternating-move rounds. Depth 2 = 4 plies. Higher = slower computation."
    )

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Section",
        ["Dashboard", "Game Tree & Equilibrium", "Threat Points",
         "Sensitivity Analysis", "Concession Sequencing",
         "Comparative Analysis", "Data Explorer", "Methodology"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem; opacity:0.6;">'
        'Data: UN Comtrade, EU Access2Markets, AfCFTA e-Tariff Book, '
        'World Bank WGI, AidData. All publicly available.<br>'
        'Model: Extensive-form game with behavioral perturbations.<br>'
        'v1.0 · March 2026</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# BUILD PARAMETERS & DATA
# ─────────────────────────────────────────────────────────────

wgi = get_wgi_scores(selected_country)
trade_dep = get_trade_dependence(selected_country)
gp = get_great_power_data(selected_country)

params = GameParameters(
    delta_africa=delta_africa,
    delta_eu=delta_eu,
    delta_afcfta=delta_afcfta,
    status_quo_bias=status_quo_bias,
    loss_aversion=loss_aversion,
    ambiguity_premium=ambiguity_premium,
    epa_sunk_cost=epa_sunk_cost,
    epa_mfn_clause_penalty=epa_mfn_penalty,
    epa_standstill_cost=epa_standstill,
    china_infrastructure_offer=china_infra,
    china_debt_constraint=china_debt,
    us_agoa_threat=us_agoa,
    eu_aid_conditionality=eu_conditionality,
    afcfta_market_access_gain=afcfta_market,
    afcfta_industrialisation_bonus=afcfta_industry,
    n_rounds=game_depth,
    capacity_modifier=wgi.get("negotiation_capacity_index", 0.5) / 0.5,
)


# ─────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.markdown(f"# {selected_country}: Bargaining Asymmetry Dashboard")
    st.markdown(
        "Pre-negotiation intelligence for AfCFTA-EPA strategy. "
        "Adjust parameters in the sidebar to explore scenarios."
    )

    # BPI computation
    bpi = compute_bargaining_power_index(trade_dep, wgi, gp, params)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bargaining Power Index", f"{bpi['BPI']:.0f}/100",
                   help="Composite index: trade diversification, governance, outside options, EPA lock-in")
    with col2:
        profile = COUNTRY_PROFILES[selected_country]
        st.metric("GDP", f"${profile['gdp_usd_bn']:.0f}B")
    with col3:
        st.metric("EU Trade Dep.", f"{(trade_dep['export_to_eu_pct'] + trade_dep['import_from_eu_pct'])/2:.1f}%")
    with col4:
        st.metric("Negotiation Capacity", f"{wgi['negotiation_capacity_index']:.2f}",
                   help="WGI-derived composite (Govt Effectiveness + Regulatory Quality + Rule of Law)")

    st.markdown("---")

    # Two-column layout
    left, right = st.columns(2)

    with left:
        st.plotly_chart(plot_bargaining_power_radar(bpi, selected_country), use_container_width=True)

    with right:
        st.plotly_chart(plot_trade_dependence(trade_dep, selected_country), use_container_width=True)

    # EPA timeline
    epa_df = get_epa_tariff_schedule(selected_country)
    if epa_df["pct_liberalised"].sum() > 0:
        st.plotly_chart(plot_epa_timeline(epa_df, selected_country), use_container_width=True)
    else:
        st.info(f"{selected_country} does not have an active EPA liberalisation schedule.")

    # Great power influence
    st.plotly_chart(plot_great_power_influence(gp, selected_country), use_container_width=True)

    # Country profile table
    with st.expander("Full Country Profile Data"):
        st.dataframe(get_country_summary(selected_country), use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="source-note">'
        'Sources: UN Comtrade (2022), EU Access2Markets, World Bank WGI (2022), '
        'AidData Chinese Finance dataset. All figures calibrated to publicly available data.'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: GAME TREE & EQUILIBRIUM
# ─────────────────────────────────────────────────────────────

elif page == "Game Tree & Equilibrium":
    st.markdown(f"# {selected_country}: Sequential Game Tree")
    st.markdown(
        "Extensive-form game with alternating moves. Africa acts first, "
        "EU responds, then the cycle repeats. Solved via backward induction "
        "(Subgame Perfect Equilibrium)."
    )

    with st.spinner("Building and solving game tree..."):
        tree = build_game_tree(params, trade_dep, wgi, max_depth=game_depth)
        eq_path, eq_payoffs = backward_induction(tree)
        viz_nodes, viz_edges = flatten_tree_for_viz(tree, max_nodes=150)

    # Equilibrium summary
    st.markdown("### Equilibrium Outcome")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Africa Payoff", f"{eq_payoffs.get('africa', 0):.1f}")
    with col2:
        st.metric("EU Payoff", f"{eq_payoffs.get('eu', 0):.1f}")
    with col3:
        st.metric("AfCFTA Payoff", f"{eq_payoffs.get('afcfta', 0):.1f}")

    # Equilibrium path
    st.markdown("### Equilibrium Path (Optimal Strategy Sequence)")
    eq_actions = []
    for nid in eq_path:
        node = tree[nid]
        if node.action != "START":
            eq_actions.append({
                "Round": node.round_num,
                "Player": node.player,
                "Action": ACTION_LABELS.get(node.action, node.action),
                "Description": node.description,
            })
    if eq_actions:
        st.dataframe(pd.DataFrame(eq_actions), use_container_width=True, hide_index=True)

    # Game tree visualization
    st.markdown("### Interactive Game Tree")
    st.markdown(
        "Nodes coloured by player. Diamonds are terminal nodes with payoffs. "
        "Bold red path is the equilibrium. Hover for details."
    )
    fig_tree = plot_game_tree(viz_nodes, viz_edges, f"{selected_country} Bargaining Game Tree")
    st.plotly_chart(fig_tree, use_container_width=True)

    # Game statistics
    with st.expander("Game Tree Statistics"):
        total_nodes = len(tree)
        terminal_nodes = sum(1 for n in tree.values() if n.is_terminal)
        eq_nodes = sum(1 for n in tree.values() if n.is_equilibrium)
        st.markdown(f"- Total nodes: **{total_nodes}**")
        st.markdown(f"- Terminal nodes: **{terminal_nodes}**")
        st.markdown(f"- Equilibrium path nodes: **{eq_nodes}**")
        st.markdown(f"- Game depth: **{game_depth}** rounds (alternating moves)")
        st.markdown(f"- Africa actions: **{len([a for a in ACTION_LABELS if 'EU' not in ACTION_LABELS[a]])}**")
        st.markdown(f"- EU responses: **4**")

    st.markdown(
        '<p class="source-note">'
        'Model: Extensive-form game solved by backward induction (Selten, 1965). '
        'Behavioral perturbations follow Kahneman & Tversky (1979) prospect theory and '
        'Samuelson & Zeckhauser (1988) status-quo bias.'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: THREAT POINTS
# ─────────────────────────────────────────────────────────────

elif page == "Threat Points":
    st.markdown(f"# {selected_country}: Credible Threat Analysis")
    st.markdown(
        "Identifies viable threat points for negotiators. A threat is credible "
        "if the threatening party prefers executing it to backing down, "
        "and it worsens the counterparty's position."
    )

    threats_df = compute_threat_points(params, trade_dep, wgi)

    # Threat analysis charts
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_threat_analysis(threats_df), use_container_width=True)
    with right:
        st.plotly_chart(plot_threat_credibility(threats_df), use_container_width=True)

    # Threat detail table
    st.markdown("### Threat Point Detail")
    display_cols = ["Threat", "Africa Payoff", "EU Payoff", "Credible",
                    "Credibility Score", "Strategic Implication"]
    st.dataframe(
        threats_df[display_cols].style.format({
            "Africa Payoff": "{:.1f}",
            "EU Payoff": "{:.1f}",
            "Credibility Score": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Red lines
    st.markdown("### Red-Line Identification")
    st.markdown(
        "Based on the threat analysis, the following red lines are identified "
        "for negotiation briefing purposes:"
    )

    credible = threats_df[threats_df["Credible"].isin(["Yes", "Partially"])]
    if len(credible) > 0:
        for _, row in credible.iterrows():
            marker = "🟢" if row["Credible"] == "Yes" else "🟡"
            st.markdown(f"{marker} **{row['Threat']}** — {row['Strategic Implication']}")
    else:
        st.warning("No credible threats identified under current parameters. Consider adjusting game parameters.")

    not_credible = threats_df[threats_df["Credible"] == "No"]
    if len(not_credible) > 0:
        st.markdown("**Non-credible threats (avoid deploying):**")
        for _, row in not_credible.iterrows():
            st.markdown(f"🔴 **{row['Threat']}** — {row['Strategic Implication']}")

    st.markdown(
        '<p class="source-note">'
        'Threat credibility framework: Schelling (1960) "The Strategy of Conflict"; '
        'Nash (1953) bargaining solution with outside options.'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "Sensitivity Analysis":
    st.markdown(f"# {selected_country}: Sensitivity Analysis")
    st.markdown(
        "Explore how equilibrium outcomes shift as parameters change. "
        "Essential for identifying tipping points and robust strategies."
    )

    param_options = {
        "Chinese Infrastructure Offer": ("china_infrastructure_offer", 0.0, 0.50),
        "Status-Quo Bias": ("status_quo_bias", 0.0, 0.50),
        "Loss Aversion (λ)": ("loss_aversion", 1.0, 4.0),
        "EPA MFN Clause Penalty": ("epa_mfn_clause_penalty", 0.0, 0.30),
        "AfCFTA Market Access Gain": ("afcfta_market_access_gain", 0.10, 0.60),
        "EU Aid Conditionality": ("eu_aid_conditionality", 0.0, 0.30),
        "Africa Discount Factor (δ)": ("delta_africa", 0.50, 0.99),
        "Ambiguity Premium": ("ambiguity_premium", 0.0, 0.30),
    }

    st.markdown("### Single-Parameter Sweep")
    selected_param = st.selectbox("Parameter to sweep", list(param_options.keys()))
    pname, pmin, pmax = param_options[selected_param]
    sweep_range = np.linspace(pmin, pmax, 20)

    with st.spinner("Running sensitivity analysis..."):
        sens_df = run_sensitivity_analysis(params, trade_dep, wgi, pname, sweep_range)

    st.plotly_chart(plot_sensitivity_line(sens_df, selected_param), use_container_width=True)

    # Show strategy changes
    with st.expander("Equilibrium Strategy by Parameter Value"):
        st.dataframe(
            sens_df[["Parameter Value", "Africa Payoff", "EU Payoff", "Equilibrium Strategy"]].style.format({
                "Parameter Value": "{:.3f}",
                "Africa Payoff": "{:.1f}",
                "EU Payoff": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Two-Parameter Heatmap")
    col1, col2 = st.columns(2)
    with col1:
        param1_name = st.selectbox("X-axis parameter", list(param_options.keys()), index=0, key="p1")
    with col2:
        param2_name = st.selectbox(
            "Y-axis parameter",
            [p for p in param_options.keys() if p != param1_name],
            index=0,
            key="p2"
        )

    p1_key, p1_min, p1_max = param_options[param1_name]
    p2_key, p2_min, p2_max = param_options[param2_name]

    with st.spinner("Computing 2D sensitivity surface..."):
        multi_df = run_multi_sensitivity(
            params, trade_dep, wgi,
            p1_key, np.linspace(p1_min, p1_max, 12),
            p2_key, np.linspace(p2_min, p2_max, 12),
        )

    st.plotly_chart(plot_sensitivity_heatmap(multi_df, p1_key, p2_key), use_container_width=True)

    st.markdown(
        '<p class="source-note">'
        'Sensitivity tables follow Saltelli et al. (2008) "Global Sensitivity Analysis" methodology. '
        'Parameters grounded in behavioral economics literature (citations in Methodology tab).'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: CONCESSION SEQUENCING
# ─────────────────────────────────────────────────────────────

elif page == "Concession Sequencing":
    st.markdown(f"# {selected_country}: Optimal Concession Sequence")
    st.markdown(
        "Determines the optimal order of AfCFTA tariff concessions across sectors, "
        "balancing gains against EPA constraints and employment risk."
    )

    seq_df = compute_optimal_sequence(params, trade_dep, wgi, SECTOR_DATA)

    st.plotly_chart(plot_concession_sequence(seq_df, selected_country), use_container_width=True)

    st.markdown("### Detailed Sequencing Recommendation")
    st.dataframe(
        seq_df.style.format({
            "AfCFTA Gain": "{:.1f}",
            "EPA Cost": "{:.1f}",
            "Employment Risk": "{:.1f}",
            "Net Benefit": "{:.1f}",
            "Risk Score": "{:.2f}",
        }).apply(
            lambda x: ["background-color: rgba(32,128,141,0.15)" if "Phase 1" in str(v)
                       else "background-color: rgba(255,197,83,0.15)" if "Phase 2" in str(v)
                       else "" for v in x],
            subset=["Priority"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # AfCFTA category breakdown
    st.markdown("### AfCFTA Tariff Category Breakdown")
    afcfta_cats = get_afcfta_tariff_categories(selected_country)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cat A (Non-sensitive)", f"{afcfta_cats['cat_a_pct']:.1f}%",
                   help="Liberalised in 5 years (non-LDC) / 10 years (LDC)")
    with col2:
        st.metric("Cat B (Sensitive)", f"{afcfta_cats['cat_b_pct']:.1f}%",
                   help="Liberalised in 10 years / 13 years")
    with col3:
        st.metric("Cat C (Exclusion)", f"{afcfta_cats['cat_c_pct']:.1f}%",
                   help="Excluded from tariff liberalisation")
    with col4:
        st.metric("Base MFN Average", f"{afcfta_cats['base_mfn_avg']:.1f}%",
                   help="Average MFN applied tariff (May 2019 base rate)")

    st.markdown(
        '<p class="source-note">'
        'AfCFTA categories per Art. 7 Protocol on Trade in Goods: 90% non-sensitive (5/10yr), '
        '7% sensitive (10/13yr), 3% exclusion. Source: AfCFTA e-Tariff Book, AU documentation.'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "Comparative Analysis":
    st.markdown("# Cross-Country Comparative Analysis")
    st.markdown(
        "Compare bargaining positions across African states to identify "
        "coalition partners and peer-group strategies."
    )

    all_compare = [selected_country] + comparison_countries
    bpi_dict = {}
    for c in all_compare:
        c_trade = get_trade_dependence(c)
        c_wgi = get_wgi_scores(c)
        c_gp = get_great_power_data(c)
        # Use same behavioral params, just different country data
        c_params = GameParameters(**{k: v for k, v in params.__dict__.items()})
        c_params.capacity_modifier = c_wgi.get("negotiation_capacity_index", 0.5) / 0.5
        bpi_dict[c] = compute_bargaining_power_index(c_trade, c_wgi, c_gp, c_params)

    # BPI comparison
    st.plotly_chart(plot_bpi_comparison(bpi_dict), use_container_width=True)

    # Radar overlay
    st.plotly_chart(plot_comparative_radar(bpi_dict), use_container_width=True)

    # Comparative table
    st.markdown("### Detailed Comparison")
    comp_rows = []
    for c in all_compare:
        profile = COUNTRY_PROFILES.get(c, {})
        c_trade = get_trade_dependence(c)
        c_wgi = get_wgi_scores(c)
        c_gp = get_great_power_data(c)
        comp_rows.append({
            "Country": c,
            "GDP (USD bn)": profile.get("gdp_usd_bn", "N/A"),
            "EU Export Dep. (%)": c_trade.get("export_to_eu_pct", 0),
            "China Import Dep. (%)": c_trade.get("import_from_china_pct", 0),
            "Intra-Africa Export (%)": c_trade.get("export_to_africa_pct", 0),
            "Negotiation Capacity": c_wgi.get("negotiation_capacity_index", 0),
            "Chinese Debt (% GDP)": c_gp.get("debt_to_china_pct_gdp", 0),
            "BPI": bpi_dict[c]["BPI"],
            "EPA Status": profile.get("epa_status", "N/A"),
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("BPI", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Coalition identification
    st.markdown("### Coalition Identification")
    st.markdown(
        "Countries with similar BPI profiles and complementary trade structures "
        "are natural negotiation coalition partners."
    )
    primary_bpi = bpi_dict[selected_country]["BPI"]
    similar = [(c, abs(bpi_dict[c]["BPI"] - primary_bpi)) for c in all_compare if c != selected_country]
    similar.sort(key=lambda x: x[1])
    if similar:
        st.markdown(f"**Closest coalition partner for {selected_country}:** {similar[0][0]} "
                    f"(BPI difference: {similar[0][1]:.1f})")

    st.markdown(
        '<p class="source-note">'
        'BPI methodology combines UNCTAD trade-policy-space indicators with '
        'World Bank WGI governance metrics and AidData external influence measures.'
        '</p>', unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────

elif page == "Data Explorer":
    st.markdown("# Data Explorer")
    st.markdown("Browse and export all underlying datasets.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Country Profiles", "Trade Dependence", "WGI Governance",
        "Great Power Influence", "Sector Data"
    ])

    with tab1:
        rows = []
        for c in get_all_countries():
            p = COUNTRY_PROFILES[c]
            rows.append({"Country": c, **p})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        rows = []
        for c in get_all_countries():
            td = get_trade_dependence(c)
            rows.append({"Country": c, **td})
        trade_df = pd.DataFrame(rows)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

    with tab3:
        rows = []
        for c in get_all_countries():
            w = get_wgi_scores(c)
            rows.append({"Country": c, **w})
        wgi_df = pd.DataFrame(rows)
        st.dataframe(wgi_df, use_container_width=True, hide_index=True)

    with tab4:
        rows = []
        for c in get_all_countries():
            g = get_great_power_data(c)
            rows.append({"Country": c, **g})
        gp_df = pd.DataFrame(rows)
        st.dataframe(gp_df, use_container_width=True, hide_index=True)

    with tab5:
        st.dataframe(pd.DataFrame(SECTOR_DATA).T.reset_index().rename(columns={"index": "Sector"}),
                     use_container_width=True, hide_index=True)

    # Export
    st.markdown("### Export Data")
    export_option = st.selectbox("Select dataset to export", [
        "Country Profiles", "Trade Dependence", "WGI Governance",
        "Great Power Influence", "EPA Schedule", "Full Summary"
    ])

    if st.button("Generate CSV"):
        if export_option == "Country Profiles":
            export_data = pd.DataFrame([{"Country": c, **COUNTRY_PROFILES[c]} for c in get_all_countries()])
        elif export_option == "Trade Dependence":
            export_data = pd.DataFrame([{"Country": c, **get_trade_dependence(c)} for c in get_all_countries()])
        elif export_option == "WGI Governance":
            export_data = pd.DataFrame([{"Country": c, **get_wgi_scores(c)} for c in get_all_countries()])
        elif export_option == "Great Power Influence":
            export_data = pd.DataFrame([{"Country": c, **get_great_power_data(c)} for c in get_all_countries()])
        elif export_option == "EPA Schedule":
            export_data = get_epa_tariff_schedule(selected_country)
        else:
            export_data = get_country_summary(selected_country)

        csv = export_data.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"afcfta_epa_{export_option.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )


# ─────────────────────────────────────────────────────────────
# PAGE: METHODOLOGY
# ─────────────────────────────────────────────────────────────

elif page == "Methodology":
    st.markdown("# Methodology & Literature Grounding")

    st.markdown("## Model Overview")
    st.markdown("""
    This simulator implements an **extensive-form sequential bargaining game** between three
    strategic actors:

    1. **African State** (e.g., Ghana): The primary decision-maker choosing concession strategies
    2. **EU / DG Trade**: Responds to African moves within EPA framework constraints
    3. **AfCFTA Council**: Background player whose preferences shape continental dynamics

    Great-power influence (China, US) enters as **exogenous parameters** affecting payoff functions,
    not as strategic players — consistent with their indirect role in AfCFTA/EPA negotiations.
    """)

    st.markdown("## Game Structure")
    st.markdown("""
    **Game type:** Finite extensive-form game with perfect information

    **Solution concept:** Subgame Perfect Equilibrium (SPE) via backward induction (Selten, 1965)

    **Move sequence per round:**
    1. Africa chooses from 6 strategic actions (selective liberalisation, full liberalisation,
       accept EPA deepening, threaten withdrawal, delay, leverage Chinese offer)
    2. EU responds from 4 options (accept, counter with conditionality, threaten MFN enforcement,
       offer adjustment support)
    3. Payoffs computed; if game continues, Africa moves again

    **Behavioral perturbations** (applied to payoff functions):
    - **Status-quo bias** (Samuelson & Zeckhauser, 1988): Inflates perceived cost of departing
      from EPA status quo
    - **Loss aversion** (Kahneman & Tversky, 1979): Losses weighted by λ ≈ 2.25 relative to gains
    - **Ambiguity premium** (Ellsberg, 1961): Additional discount for uncertain AfCFTA outcomes
    """)

    st.markdown("## Bargaining Power Index (BPI)")
    st.markdown("""
    The composite BPI (0-100) captures five dimensions of negotiation strength:

    | Component (Weight) | Measurement | Source |
    |---|---|---|
    | Trade Diversification (25%) | Inverse of EU export/import dependence | UN Comtrade |
    | Governance Capacity (25%) | WGI composite (Govt Effectiveness + Regulatory Quality + Rule of Law) | World Bank WGI |
    | Outside Options (20%) | Chinese infrastructure + intra-African trade | AidData, Comtrade |
    | Policy Space (20%) | Inverse of EPA lock-in costs | EU Access2Markets |
    | Economic Weight (10%) | AGOA exports, GDP | USTR, World Bank |
    """)

    st.markdown("## Data Sources")
    st.markdown("""
    All data is drawn from publicly available sources:

    | Dataset | Source | Reference Period |
    |---|---|---|
    | Tariff schedules (AfCFTA) | AfCFTA e-Tariff Book (etariff.au-afcfta.org) | 2023-2024 |
    | EPA liberalisation schedules | EU Access2Markets, ActionAid Ghana | 2016-2029 projected |
    | Trade flows / dependence | UN Comtrade (comtrade.un.org) | 2022 |
    | Governance indicators | World Bank WGI (info.worldbank.org/governance/wgi) | 2022 |
    | Chinese lending / FDI | AidData (aiddata.org), World Bank debtor reporting | 2000-2023 |
    | AGOA data | USTR / agoa.info | 2022 |
    | Sector parameters | UNCTAD, World Bank, national statistical offices | Various |
    """)

    st.markdown("## Parameter Justification")
    st.markdown("""
    | Parameter | Default | Literature Range | Reference |
    |---|---|---|---|
    | Status-quo bias | 0.15 | 0.10-0.30 | Samuelson & Zeckhauser (1988) |
    | Loss aversion (λ) | 2.25 | 1.5-2.5 | Tversky & Kahneman (1991), Novemsky & Kahneman (2005) |
    | Ambiguity premium | 0.10 | 0.05-0.20 | Ellsberg (1961), Camerer & Weber (1992) |
    | Africa discount (δ) | 0.85 | 0.70-0.95 | Rubinstein (1982) bargaining model |
    | EU discount (δ) | 0.92 | 0.85-0.98 | Institutional patience literature |
    | MFN clause penalty | 0.12 | 0.05-0.20 | EPA Art. 35 analysis, Ravenhill (2011) |
    """)

    st.markdown("## Limitations & Caveats")
    st.markdown("""
    - **Parameter sensitivity:** Results are highly sensitive to parameter choices. The sensitivity
      analysis tab allows full exploration; users should report findings as ranges, not point estimates.
    - **Political framing:** The "asymmetry" framing is analytical, not normative. The tool is
      positioned as neutral capacity-building for all parties.
    - **Data currency:** Embedded data reflects 2022-2024 reference periods. Users should cross-check
      against the most recent national statistics.
    - **Simplified game:** The 3-player extensive form abstracts from intra-African coordination
      problems, domestic political economy, and time-varying shock processes.
    - **Behavioral parameters:** While grounded in laboratory experiments, field estimates of
      status-quo bias and loss aversion in trade negotiations are scarce.
    """)

    st.markdown("## Key References")
    st.markdown("""
    1. Selten, R. (1965). "Spieltheoretische Behandlung eines Oligopolmodells mit Nachfrageträgheit." *Zeitschrift für die gesamte Staatswissenschaft*.
    2. Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." *Econometrica* 47(2).
    3. Samuelson, W. & Zeckhauser, R. (1988). "Status Quo Bias in Decision Making." *Journal of Risk and Uncertainty* 1(1).
    4. Rubinstein, A. (1982). "Perfect Equilibrium in a Bargaining Model." *Econometrica* 50(1).
    5. Nash, J. (1953). "Two-Person Cooperative Games." *Econometrica* 21(1).
    6. Schelling, T. (1960). *The Strategy of Conflict*. Harvard University Press.
    7. Ravenhill, J. (2011). "The Political Economy of the EPA." *Review of African Political Economy*.
    8. UNCTAD (2023). *Economic Development in Africa Report*.
    9. AfCFTA Secretariat (2023). *e-Tariff Book User Guide*.
    10. AidData (2021). *Banking on the Belt and Road*. Williamsburg, VA: AidData.
    """)

    st.markdown(
        '<p class="source-note">'
        'This tool is designed for pre-negotiation scenario analysis. Outputs should be '
        'interpreted as indicative scenario ranges, not precise predictions. '
        'Position as neutral capacity-building for all stakeholders.'
        '</p>', unsafe_allow_html=True
    )
