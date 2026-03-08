"""
Game Engine: Sequential extensive-form game for AfCFTA–EPA bargaining asymmetry.

Models a multi-stage negotiation between an African state (Player A) and the
EU/AfCFTA Council (Player B), with Chinese influence as an exogenous shock parameter.

Key features:
- Extensive-form game tree with backward induction
- Behavioral perturbations (status-quo bias, loss aversion, time discounting)
- Credible threat-point identification
- Sensitivity analysis across parameter space
- Optimal concession sequencing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from itertools import product as itertools_product


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class GameParameters:
    """All tunable parameters for the bargaining game."""
    # Discount factors (patience)
    delta_africa: float = 0.85          # African state's discount factor
    delta_eu: float = 0.92              # EU's discount factor (more patient)
    delta_afcfta: float = 0.80          # AfCFTA Council's discount factor

    # Behavioral perturbations
    status_quo_bias: float = 0.15       # Inflates cost of changing from EPA status quo
    loss_aversion: float = 2.25         # Kahneman-Tversky lambda for losses vs gains
    ambiguity_premium: float = 0.10     # Additional discount for uncertain AfCFTA outcomes

    # EPA lock-in parameters
    epa_sunk_cost: float = 0.20         # Fraction of tariff revenue already foregone
    epa_mfn_clause_penalty: float = 0.12 # Cost of triggering MFN clause via AfCFTA offers
    epa_standstill_cost: float = 0.08   # Cost of inability to raise tariffs

    # AfCFTA opportunity parameters
    afcfta_market_access_gain: float = 0.35  # Potential gain from continental market
    afcfta_industrialisation_bonus: float = 0.15  # Value-chain development multiplier
    afcfta_rules_of_origin_cost: float = 0.05  # Compliance cost

    # Great power shadow effects
    china_infrastructure_offer: float = 0.20  # Value of Chinese infrastructure as outside option
    china_debt_constraint: float = 0.10       # Constraining effect of existing Chinese debt
    us_agoa_threat: float = 0.08              # AGOA withdrawal risk
    eu_aid_conditionality: float = 0.12       # EU development aid tied to EPA compliance

    # Game rounds
    n_rounds: int = 4

    # Negotiation capacity modifier (from WGI)
    capacity_modifier: float = 1.0


@dataclass
class GameNode:
    """A node in the extensive-form game tree."""
    node_id: str
    player: str               # "Africa", "EU", "AfCFTA", "Nature"
    round_num: int
    action: str               # The action that led to this node
    parent_id: Optional[str]
    payoff_africa: float = 0.0
    payoff_eu: float = 0.0
    payoff_afcfta: float = 0.0
    is_terminal: bool = False
    is_equilibrium: bool = False
    children: List[str] = field(default_factory=list)
    threat_credibility: float = 0.0
    description: str = ""


# ─────────────────────────────────────────────────────────────
# PAYOFF FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_africa_payoff(
    action_sequence: List[str],
    params: GameParameters,
    trade_dep: dict,
    wgi: dict,
    round_num: int
) -> float:
    """
    Compute African state's payoff given an action sequence.

    Payoff = (AfCFTA gains - EPA costs - behavioral penalties) * capacity * discount
    """
    base_payoff = 50.0  # Normalised baseline (status quo value)

    for i, action in enumerate(action_sequence):
        discount = params.delta_africa ** i

        if action == "ACCEPT_EPA_DEEPENING":
            # Gain: continued EU market access; Cost: reduced policy space
            eu_gain = trade_dep.get("export_to_eu_pct", 20) * 0.8
            policy_cost = params.epa_standstill_cost * 100
            base_payoff += (eu_gain - policy_cost) * discount

        elif action == "SELECTIVE_AFCFTA_LIBERALISATION":
            # Gain: AfCFTA market access for selected sectors
            afcfta_gain = params.afcfta_market_access_gain * 100 * 0.6
            roi_cost = params.afcfta_rules_of_origin_cost * 100
            mfn_risk = params.epa_mfn_clause_penalty * 100 * 0.3  # Partial trigger
            base_payoff += (afcfta_gain - roi_cost - mfn_risk) * discount

        elif action == "FULL_AFCFTA_LIBERALISATION":
            # Gain: full continental market; Risk: MFN clause trigger
            afcfta_gain = params.afcfta_market_access_gain * 100
            ind_bonus = params.afcfta_industrialisation_bonus * 100
            mfn_penalty = params.epa_mfn_clause_penalty * 100
            base_payoff += (afcfta_gain + ind_bonus - mfn_penalty) * discount

        elif action == "REJECT_AND_THREATEN_WITHDRAWAL":
            # Threat point: walk away from EPA
            eu_loss = trade_dep.get("export_to_eu_pct", 20) * 0.5
            china_option = params.china_infrastructure_offer * 100
            credibility = wgi.get("negotiation_capacity_index", 0.5)
            base_payoff += (-eu_loss + china_option * credibility) * discount

        elif action == "STATUS_QUO":
            # No change: apply status quo bias
            sq_cost = params.status_quo_bias * 10 * (i + 1)  # Increases over time
            base_payoff -= sq_cost * discount

        elif action == "DELAY_CONCESSION":
            # Stall: time costs apply
            time_cost = 5.0 * (i + 1)
            patience_benefit = (1 - params.delta_eu) * 30 if params.delta_eu < params.delta_africa else 0
            base_payoff += (patience_benefit - time_cost) * discount

        elif action == "LEVERAGE_CHINESE_OFFER":
            # Use Chinese infrastructure offer as outside option
            china_val = params.china_infrastructure_offer * 100
            debt_penalty = params.china_debt_constraint * 100
            eu_reaction = params.eu_aid_conditionality * 50
            base_payoff += (china_val - debt_penalty - eu_reaction) * discount

    # Apply behavioral perturbations
    # Loss aversion: losses weighted more heavily
    if base_payoff < 50.0:
        base_payoff = 50.0 - (50.0 - base_payoff) * params.loss_aversion

    # Ambiguity premium for non-EPA paths
    non_epa_actions = [a for a in action_sequence if "AFCFTA" in a or "CHINESE" in a]
    if non_epa_actions:
        base_payoff *= (1 - params.ambiguity_premium * len(non_epa_actions) / len(action_sequence))

    # Capacity modifier
    base_payoff *= params.capacity_modifier

    return round(base_payoff, 2)


def compute_eu_payoff(
    action_sequence: List[str],
    params: GameParameters,
    trade_dep: dict,
    round_num: int
) -> float:
    """Compute EU's payoff from the action sequence."""
    base_payoff = 50.0

    for i, action in enumerate(action_sequence):
        discount = params.delta_eu ** i

        if action == "ACCEPT_EPA_DEEPENING":
            base_payoff += 15.0 * discount  # EU gains from deeper integration

        elif action == "SELECTIVE_AFCFTA_LIBERALISATION":
            # EU mildly negative: partial market share erosion
            base_payoff -= 5.0 * discount

        elif action == "FULL_AFCFTA_LIBERALISATION":
            # EU negative: significant market share loss in Africa
            import_share_loss = trade_dep.get("import_from_eu_pct", 20) * 0.3
            base_payoff -= import_share_loss * discount

        elif action == "REJECT_AND_THREATEN_WITHDRAWAL":
            # EU loses if credible: losing a signatory
            base_payoff -= 20.0 * discount

        elif action == "STATUS_QUO":
            base_payoff += 8.0 * discount  # EU benefits from current arrangement

        elif action == "DELAY_CONCESSION":
            base_payoff -= 3.0 * discount

        elif action == "LEVERAGE_CHINESE_OFFER":
            # EU strongly dislikes: geopolitical loss
            base_payoff -= 18.0 * discount

    return round(base_payoff, 2)


def compute_afcfta_payoff(
    action_sequence: List[str],
    params: GameParameters,
    round_num: int
) -> float:
    """Compute AfCFTA Council's payoff."""
    base_payoff = 50.0

    for i, action in enumerate(action_sequence):
        discount = params.delta_afcfta ** i

        if action == "ACCEPT_EPA_DEEPENING":
            base_payoff -= 10.0 * discount  # Against continental integration

        elif action == "SELECTIVE_AFCFTA_LIBERALISATION":
            base_payoff += 12.0 * discount

        elif action == "FULL_AFCFTA_LIBERALISATION":
            base_payoff += 25.0 * discount

        elif action == "REJECT_AND_THREATEN_WITHDRAWAL":
            base_payoff += 5.0 * discount  # Supports reduced EPA dependence

        elif action == "STATUS_QUO":
            base_payoff -= 8.0 * discount

        elif action == "LEVERAGE_CHINESE_OFFER":
            base_payoff -= 5.0 * discount  # Mixed: reduces EU but increases China

    return round(base_payoff, 2)


# ─────────────────────────────────────────────────────────────
# GAME TREE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

# Actions available to each player
AFRICA_ACTIONS = [
    "SELECTIVE_AFCFTA_LIBERALISATION",
    "FULL_AFCFTA_LIBERALISATION",
    "ACCEPT_EPA_DEEPENING",
    "REJECT_AND_THREATEN_WITHDRAWAL",
    "DELAY_CONCESSION",
    "LEVERAGE_CHINESE_OFFER",
]

EU_RESPONSES = [
    "ACCEPT_CONCESSION",
    "COUNTER_WITH_CONDITIONALITY",
    "THREATEN_MFN_ENFORCEMENT",
    "OFFER_ADJUSTMENT_SUPPORT",
]

ACTION_LABELS = {
    "SELECTIVE_AFCFTA_LIBERALISATION": "Selective AfCFTA Lib.",
    "FULL_AFCFTA_LIBERALISATION": "Full AfCFTA Lib.",
    "ACCEPT_EPA_DEEPENING": "Accept EPA Deepening",
    "REJECT_AND_THREATEN_WITHDRAWAL": "Reject & Threaten",
    "DELAY_CONCESSION": "Delay Concession",
    "LEVERAGE_CHINESE_OFFER": "Leverage China Offer",
    "ACCEPT_CONCESSION": "EU: Accept",
    "COUNTER_WITH_CONDITIONALITY": "EU: Counter w/ Conditions",
    "THREATEN_MFN_ENFORCEMENT": "EU: Threaten MFN",
    "OFFER_ADJUSTMENT_SUPPORT": "EU: Offer Adj. Support",
    "STATUS_QUO": "Status Quo",
}


def build_game_tree(
    params: GameParameters,
    trade_dep: dict,
    wgi: dict,
    max_depth: int = 2
) -> Dict[str, GameNode]:
    """
    Build extensive-form game tree with alternating moves.
    Round structure: Africa moves -> EU responds -> Terminal / next round
    """
    nodes = {}
    node_counter = [0]

    def make_id():
        node_counter[0] += 1
        return f"N{node_counter[0]:03d}"

    # Root node
    root_id = make_id()
    root = GameNode(
        node_id=root_id,
        player="Africa",
        round_num=1,
        action="START",
        parent_id=None,
        description="Game begins. African state chooses initial negotiation strategy."
    )
    nodes[root_id] = root

    def build_subtree(parent_id: str, round_num: int, action_history: List[str]):
        if round_num > max_depth:
            # Terminal node
            parent = nodes[parent_id]
            parent.is_terminal = True
            parent.payoff_africa = compute_africa_payoff(action_history, params, trade_dep, wgi, round_num)
            parent.payoff_eu = compute_eu_payoff(action_history, params, trade_dep, round_num)
            parent.payoff_afcfta = compute_afcfta_payoff(action_history, params, round_num)
            return

        parent = nodes[parent_id]

        if parent.player == "Africa":
            for action in AFRICA_ACTIONS:
                child_id = make_id()
                child = GameNode(
                    node_id=child_id,
                    player="EU",
                    round_num=round_num,
                    action=action,
                    parent_id=parent_id,
                    description=f"Round {round_num}: Africa plays '{ACTION_LABELS.get(action, action)}'"
                )
                nodes[child_id] = child
                parent.children.append(child_id)
                build_subtree(child_id, round_num, action_history + [action])

        elif parent.player == "EU":
            for response in EU_RESPONSES:
                child_id = make_id()
                child = GameNode(
                    node_id=child_id,
                    player="Africa",
                    round_num=round_num + 1,
                    action=response,
                    parent_id=parent_id,
                    description=f"Round {round_num}: EU responds '{ACTION_LABELS.get(response, response)}'"
                )
                nodes[child_id] = child
                parent.children.append(child_id)

                # Map EU responses to Africa-facing actions for payoff computation
                mapped_action = _map_eu_response(response, action_history[-1] if action_history else "STATUS_QUO")
                build_subtree(child_id, round_num + 1, action_history + [mapped_action])

    build_subtree(root_id, 1, [])
    return nodes


def _map_eu_response(eu_response: str, africa_action: str) -> str:
    """Map EU response to an effective action for payoff computation."""
    mapping = {
        "ACCEPT_CONCESSION": africa_action,  # EU accepts Africa's move
        "COUNTER_WITH_CONDITIONALITY": "ACCEPT_EPA_DEEPENING",  # Pushes toward EPA
        "THREATEN_MFN_ENFORCEMENT": "STATUS_QUO",
        "OFFER_ADJUSTMENT_SUPPORT": "SELECTIVE_AFCFTA_LIBERALISATION",
    }
    return mapping.get(eu_response, "STATUS_QUO")


# ─────────────────────────────────────────────────────────────
# BACKWARD INDUCTION SOLVER
# ─────────────────────────────────────────────────────────────

def backward_induction(nodes: Dict[str, GameNode]) -> Tuple[List[str], Dict[str, float]]:
    """
    Solve the game tree via backward induction (Subgame Perfect Equilibrium).
    Returns the equilibrium path and payoffs.
    """
    # Find root
    root_id = None
    for nid, node in nodes.items():
        if node.parent_id is None:
            root_id = nid
            break

    if root_id is None:
        return [], {}

    def solve(node_id: str) -> Tuple[float, float, float]:
        node = nodes[node_id]

        if node.is_terminal:
            return node.payoff_africa, node.payoff_eu, node.payoff_afcfta

        # Solve all children
        child_results = {}
        for child_id in node.children:
            child_results[child_id] = solve(child_id)

        # Current player picks best action
        if node.player == "Africa":
            best_child = max(child_results.keys(), key=lambda c: child_results[c][0])
        elif node.player == "EU":
            best_child = max(child_results.keys(), key=lambda c: child_results[c][1])
        else:  # AfCFTA
            best_child = max(child_results.keys(), key=lambda c: child_results[c][2])

        # Mark equilibrium
        nodes[best_child].is_equilibrium = True
        node.payoff_africa, node.payoff_eu, node.payoff_afcfta = child_results[best_child]

        return child_results[best_child]

    solve(root_id)

    # Extract equilibrium path
    eq_path = []
    current = root_id
    while True:
        node = nodes[current]
        eq_path.append(current)
        eq_children = [c for c in node.children if nodes[c].is_equilibrium]
        if not eq_children:
            break
        current = eq_children[0]

    return eq_path, {
        "africa": nodes[eq_path[-1]].payoff_africa,
        "eu": nodes[eq_path[-1]].payoff_eu,
        "afcfta": nodes[eq_path[-1]].payoff_afcfta,
    }


# ─────────────────────────────────────────────────────────────
# THREAT POINT ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_threat_points(
    params: GameParameters,
    trade_dep: dict,
    wgi: dict
) -> pd.DataFrame:
    """
    Identify credible threat points and their payoff implications.
    A threat is credible if: (1) the threatening party prefers executing it
    to backing down, and (2) it worsens the other party's payoff.
    """
    threats = []

    # Threat 1: Selective AfCFTA liberalisation
    africa_payoff = compute_africa_payoff(["SELECTIVE_AFCFTA_LIBERALISATION"], params, trade_dep, wgi, 1)
    eu_payoff = compute_eu_payoff(["SELECTIVE_AFCFTA_LIBERALISATION"], params, trade_dep, 1)
    status_quo_a = compute_africa_payoff(["STATUS_QUO"], params, trade_dep, wgi, 1)
    status_quo_eu = compute_eu_payoff(["STATUS_QUO"], params, trade_dep, 1)

    credible = africa_payoff > status_quo_a and eu_payoff < status_quo_eu
    threats.append({
        "Threat": "Selective AfCFTA Liberalisation",
        "Africa Payoff": africa_payoff,
        "EU Payoff": eu_payoff,
        "Status Quo (Africa)": status_quo_a,
        "Status Quo (EU)": status_quo_eu,
        "Credible": "Yes" if credible else "No",
        "Credibility Score": min(1.0, max(0, (africa_payoff - status_quo_a) / 20) * max(0, (status_quo_eu - eu_payoff) / 20)),
        "Strategic Implication": "Pressures EU by demonstrating continental commitment without full MFN trigger"
    })

    # Threat 2: Full AfCFTA liberalisation (MFN trigger)
    africa_payoff2 = compute_africa_payoff(["FULL_AFCFTA_LIBERALISATION"], params, trade_dep, wgi, 1)
    eu_payoff2 = compute_eu_payoff(["FULL_AFCFTA_LIBERALISATION"], params, trade_dep, 1)
    credible2 = africa_payoff2 > status_quo_a
    threats.append({
        "Threat": "Full AfCFTA Liberalisation",
        "Africa Payoff": africa_payoff2,
        "EU Payoff": eu_payoff2,
        "Status Quo (Africa)": status_quo_a,
        "Status Quo (EU)": status_quo_eu,
        "Credible": "Yes" if credible2 else "No",
        "Credibility Score": min(1.0, max(0, (africa_payoff2 - status_quo_a) / 20)),
        "Strategic Implication": "Nuclear option: triggers EPA MFN clause but opens continental market"
    })

    # Threat 3: Leverage Chinese infrastructure offer
    africa_payoff3 = compute_africa_payoff(["LEVERAGE_CHINESE_OFFER"], params, trade_dep, wgi, 1)
    eu_payoff3 = compute_eu_payoff(["LEVERAGE_CHINESE_OFFER"], params, trade_dep, 1)
    credible3 = africa_payoff3 > status_quo_a and eu_payoff3 < status_quo_eu
    threats.append({
        "Threat": "Leverage Chinese Infrastructure Offer",
        "Africa Payoff": africa_payoff3,
        "EU Payoff": eu_payoff3,
        "Status Quo (Africa)": status_quo_a,
        "Status Quo (EU)": status_quo_eu,
        "Credible": "Yes" if credible3 else "No",
        "Credibility Score": min(1.0, max(0, (africa_payoff3 - status_quo_a) / 15) * max(0, (status_quo_eu - eu_payoff3) / 15)),
        "Strategic Implication": "Geopolitical leverage: signals alternative partnerships to extract EU concessions"
    })

    # Threat 4: EPA withdrawal threat
    africa_payoff4 = compute_africa_payoff(["REJECT_AND_THREATEN_WITHDRAWAL"], params, trade_dep, wgi, 1)
    eu_payoff4 = compute_eu_payoff(["REJECT_AND_THREATEN_WITHDRAWAL"], params, trade_dep, 1)
    credible4 = africa_payoff4 > (status_quo_a * 0.7)  # Lower threshold: bluffing is common
    threats.append({
        "Threat": "EPA Withdrawal Threat",
        "Africa Payoff": africa_payoff4,
        "EU Payoff": eu_payoff4,
        "Status Quo (Africa)": status_quo_a,
        "Status Quo (EU)": status_quo_eu,
        "Credible": "Partially" if credible4 else "No",
        "Credibility Score": min(1.0, max(0, wgi.get("negotiation_capacity_index", 0.5) * 1.2)),
        "Strategic Implication": "High-risk threat: credibility depends on governance capacity and outside options"
    })

    # Threat 5: Delay / stall concessions
    africa_payoff5 = compute_africa_payoff(["DELAY_CONCESSION", "DELAY_CONCESSION"], params, trade_dep, wgi, 2)
    eu_payoff5 = compute_eu_payoff(["DELAY_CONCESSION", "DELAY_CONCESSION"], params, trade_dep, 2)
    credible5 = params.delta_africa > params.delta_eu
    threats.append({
        "Threat": "Delay / Stall Concessions",
        "Africa Payoff": africa_payoff5,
        "EU Payoff": eu_payoff5,
        "Status Quo (Africa)": status_quo_a,
        "Status Quo (EU)": status_quo_eu,
        "Credible": "Yes" if credible5 else "No",
        "Credibility Score": min(1.0, max(0, params.delta_africa - params.delta_eu + 0.5)),
        "Strategic Implication": "Patience play: effective only if Africa is more patient than EU (unlikely by default)"
    })

    return pd.DataFrame(threats)


# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    base_params: GameParameters,
    trade_dep: dict,
    wgi: dict,
    param_name: str,
    param_range: np.ndarray
) -> pd.DataFrame:
    """
    Sweep one parameter and track equilibrium payoffs.
    """
    results = []

    for val in param_range:
        p = GameParameters(**{k: v for k, v in base_params.__dict__.items()})
        setattr(p, param_name, val)
        p.capacity_modifier = wgi.get("negotiation_capacity_index", 0.5) / 0.5

        tree = build_game_tree(p, trade_dep, wgi, max_depth=2)
        eq_path, payoffs = backward_induction(tree)

        # Extract equilibrium strategy
        eq_actions = []
        for nid in eq_path:
            node = tree[nid]
            if node.action not in ("START",):
                eq_actions.append(ACTION_LABELS.get(node.action, node.action))

        results.append({
            "Parameter Value": val,
            "Africa Payoff": payoffs.get("africa", 0),
            "EU Payoff": payoffs.get("eu", 0),
            "AfCFTA Payoff": payoffs.get("afcfta", 0),
            "Equilibrium Strategy": " → ".join(eq_actions[:4]) if eq_actions else "N/A",
        })

    return pd.DataFrame(results)


def run_multi_sensitivity(
    base_params: GameParameters,
    trade_dep: dict,
    wgi: dict,
    param1: str,
    param1_range: np.ndarray,
    param2: str,
    param2_range: np.ndarray
) -> pd.DataFrame:
    """
    Two-parameter sensitivity analysis. Returns a matrix of Africa payoffs.
    """
    results = []

    for v1 in param1_range:
        for v2 in param2_range:
            p = GameParameters(**{k: v for k, v in base_params.__dict__.items()})
            setattr(p, param1, v1)
            setattr(p, param2, v2)
            p.capacity_modifier = wgi.get("negotiation_capacity_index", 0.5) / 0.5

            tree = build_game_tree(p, trade_dep, wgi, max_depth=2)
            _, payoffs = backward_induction(tree)

            results.append({
                param1: v1,
                param2: v2,
                "Africa Payoff": payoffs.get("africa", 0),
                "EU Payoff": payoffs.get("eu", 0),
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# OPTIMAL CONCESSION SEQUENCING
# ─────────────────────────────────────────────────────────────

def compute_optimal_sequence(
    params: GameParameters,
    trade_dep: dict,
    wgi: dict,
    sectors: dict
) -> pd.DataFrame:
    """
    Determine the optimal order of AfCFTA tariff concessions across sectors,
    given EPA constraints and great-power dynamics.
    """
    results = []

    for sector_name, sector_data in sectors.items():
        # Compute net benefit of liberalising this sector under AfCFTA
        afcfta_gain = sector_data["afcfta_opportunity"] * params.afcfta_market_access_gain * 100
        epa_cost = sector_data["epa_exposure"] * params.epa_mfn_clause_penalty * 100
        employment_risk = sector_data["employment_share"] * params.loss_aversion * 10
        china_competitive_pressure = params.china_infrastructure_offer * sector_data.get("epa_exposure", 0.5) * 20

        net_benefit = afcfta_gain - epa_cost - employment_risk + china_competitive_pressure * 0.3
        net_benefit *= params.capacity_modifier

        # Risk score
        risk = (sector_data["epa_exposure"] * 0.4 +
                (1 - sector_data["afcfta_opportunity"]) * 0.3 +
                sector_data["employment_share"] * 0.3)

        results.append({
            "Sector": sector_name,
            "AfCFTA Gain": round(afcfta_gain, 1),
            "EPA Cost": round(epa_cost, 1),
            "Employment Risk": round(employment_risk, 1),
            "Net Benefit": round(net_benefit, 1),
            "Risk Score": round(risk, 2),
            "Priority": "",
            "Recommended Timing": "",
            "Sensitivity": sector_data["sensitivity"],
        })

    df = pd.DataFrame(results).sort_values("Net Benefit", ascending=False).reset_index(drop=True)

    # Assign priorities and timing
    n = len(df)
    for i in range(n):
        if i < n * 0.33:
            df.at[i, "Priority"] = "Phase 1 (Immediate)"
            df.at[i, "Recommended Timing"] = "Year 1-2"
        elif i < n * 0.66:
            df.at[i, "Priority"] = "Phase 2 (Medium-term)"
            df.at[i, "Recommended Timing"] = "Year 3-5"
        else:
            df.at[i, "Priority"] = "Phase 3 (Deferred)"
            df.at[i, "Recommended Timing"] = "Year 5-10"

    return df


# ─────────────────────────────────────────────────────────────
# BARGAINING POWER INDEX
# ─────────────────────────────────────────────────────────────

def compute_bargaining_power_index(
    trade_dep: dict,
    wgi: dict,
    gp: dict,
    params: GameParameters
) -> dict:
    """
    Composite Bargaining Power Index (BPI) for the African state.
    BPI = f(trade diversification, governance capacity, outside options, EPA lock-in)
    Scale: 0 (no power) to 100 (maximum power)
    """
    # Component 1: Trade Diversification (less dependent = more power)
    eu_dep = (trade_dep.get("export_to_eu_pct", 20) + trade_dep.get("import_from_eu_pct", 20)) / 2
    diversification_score = max(0, 100 - eu_dep * 2)

    # Component 2: Governance / Negotiation Capacity
    capacity_score = wgi.get("negotiation_capacity_index", 0.5) * 100

    # Component 3: Outside Options (Chinese alternative + AGOA + intra-Africa)
    china_option = min(30, gp.get("chinese_loans_usd_bn", 0) * 3)
    africa_trade = trade_dep.get("export_to_africa_pct", 15)
    outside_options_score = min(100, china_option + africa_trade * 2)

    # Component 4: EPA Lock-in (more locked in = less power)
    epa_lockout = params.epa_sunk_cost * 100 + params.epa_standstill_cost * 100
    lock_in_penalty = min(40, epa_lockout)

    # Component 5: Economic weight
    weight_score = min(20, gp.get("agoa_exports_usd_mn", 0) / 100)

    # Weighted composite
    bpi = (
        diversification_score * 0.25 +
        capacity_score * 0.25 +
        outside_options_score * 0.20 +
        (100 - lock_in_penalty) * 0.20 +
        weight_score * 0.10
    )

    return {
        "BPI": round(bpi, 1),
        "Trade Diversification": round(diversification_score, 1),
        "Governance Capacity": round(capacity_score, 1),
        "Outside Options": round(outside_options_score, 1),
        "EPA Lock-in Penalty": round(lock_in_penalty, 1),
        "Economic Weight": round(weight_score, 1),
    }


# ─────────────────────────────────────────────────────────────
# GAME TREE FLATTENING FOR VISUALIZATION
# ─────────────────────────────────────────────────────────────

def flatten_tree_for_viz(
    nodes: Dict[str, GameNode],
    max_nodes: int = 120
) -> Tuple[List[dict], List[dict]]:
    """
    Flatten game tree into nodes and edges for Plotly treemap / network viz.
    Limits output to prevent browser overload.
    """
    viz_nodes = []
    viz_edges = []
    count = 0

    # Find root
    root_id = None
    for nid, node in nodes.items():
        if node.parent_id is None:
            root_id = nid
            break

    # BFS traversal with limit
    queue = [root_id]
    visited = set()

    while queue and count < max_nodes:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        node = nodes[current_id]
        count += 1

        label = ACTION_LABELS.get(node.action, node.action)
        if node.is_terminal:
            label += f"\n(A:{node.payoff_africa:.0f}, EU:{node.payoff_eu:.0f})"

        viz_nodes.append({
            "id": node.node_id,
            "label": label,
            "player": node.player,
            "round": node.round_num,
            "is_terminal": node.is_terminal,
            "is_equilibrium": node.is_equilibrium,
            "payoff_africa": node.payoff_africa,
            "payoff_eu": node.payoff_eu,
            "payoff_afcfta": node.payoff_afcfta,
        })

        for child_id in node.children:
            if count < max_nodes:
                viz_edges.append({
                    "source": node.node_id,
                    "target": child_id,
                    "is_equilibrium": nodes[child_id].is_equilibrium,
                })
                queue.append(child_id)

    return viz_nodes, viz_edges
