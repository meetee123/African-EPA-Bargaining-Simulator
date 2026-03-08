# AfCFTA–EPA Bargaining Asymmetry Simulator

**Sequential Game Engine for African States**

An interactive Streamlit application that quantifies bargaining power asymmetries when African states negotiate AfCFTA protocols while locked into EU Economic Partnership Agreement (EPA) schedules. Models great-power (China / US / EU) shadow influence via an extensive-form game engine with behavioural perturbations.

---

## What It Does

| Feature | Description |
|---|---|
| **Dashboard** | Country-level KPIs, BPI radar chart, trade dependence sunburst, EPA timeline, great-power footprint |
| **Game Tree & Equilibrium** | Interactive extensive-form game solved by backward induction (Subgame Perfect Equilibrium) |
| **Threat Points** | Credible-threat identification with payoff comparison, credibility map, and red-line briefing |
| **Sensitivity Analysis** | Single-parameter sweeps and two-parameter heatmaps across 8 tunable behavioural/structural parameters |
| **Concession Sequencing** | Optimal sector-level AfCFTA tariff liberalisation ordering given EPA constraints |
| **Comparative Analysis** | Cross-country BPI benchmarking and coalition identification |
| **Data Explorer** | Browse / export all embedded datasets as CSV |
| **Methodology** | Full literature grounding, parameter justification, data-source table, limitations |

## Countries Covered

Ghana (primary archetype), Côte d'Ivoire, Kenya, Nigeria, Senegal, South Africa, Ethiopia, Tanzania.

---

## Model Specification

- **Game type:** Finite extensive-form, perfect information, 3 players (Africa, EU, AfCFTA Council)
- **Solution:** Subgame Perfect Equilibrium via backward induction (Selten 1965)
- **Africa actions (6):** Selective AfCFTA lib., full AfCFTA lib., accept EPA deepening, threaten withdrawal, delay, leverage China
- **EU responses (4):** Accept, counter with conditionality, threaten MFN enforcement, offer adjustment support
- **Behavioural perturbations:** Status-quo bias (Samuelson & Zeckhauser 1988), loss aversion λ (Kahneman & Tversky 1979), ambiguity premium (Ellsberg 1961)
- **BPI components:** Trade diversification (25%), governance capacity (25%), outside options (20%), policy space (20%), economic weight (10%)

---

## Data Sources

All data is embedded directly in the code (no external API calls required). Values are calibrated to the following publicly available datasets:

| Dataset | Source | Period |
|---|---|---|
| AfCFTA tariff schedules | AfCFTA e-Tariff Book (etariff.au-afcfta.org) | 2023-2024 |
| EPA liberalisation schedules | EU Access2Markets; ActionAid Ghana | 2016-2029 |
| Trade flows & dependence | UN Comtrade via WITS (comtrade.un.org) | 2022 |
| Governance indicators | World Bank WGI (info.worldbank.org/governance/wgi) | 2022 |
| Chinese finance / FDI | AidData (aiddata.org); World Bank IDS | 2000-2023 |
| AGOA trade | USTR / agoa.info | 2022 |
| Sector parameters | UNCTAD; World Bank | Various |

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires **Python 3.10-3.12** (some dependencies may lack 3.13 wheels).

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository and configure:
   - **Main file:** `app.py`
   - **Python version:** `3.12` (set in Advanced Settings — Cloud defaults to 3.13)
4. Click **Deploy**. No secrets required.

---

## Project Structure

```
afcfta-simulator/
├── app.py                 # Single-file application (~1200 lines)
├── requirements.txt       # 5 exact-pinned dependencies
├── README.md              # This file
└── .streamlit/
    └── config.toml        # Theme configuration
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.41.1 | Web application framework |
| pandas | 2.2.3 | Data handling |
| numpy | 1.26.4 | Numerical computation |
| plotly | 5.24.1 | Interactive visualisations |
| scipy | 1.14.1 | Scientific computing |

---

## Limitations

- **Parameter sensitivity:** Results are parameter-dependent. Use the Sensitivity tab; report as ranges, not point estimates.
- **Framing:** "Asymmetry" is analytical, not normative. Positioned as neutral capacity-building.
- **Data currency:** 2022-2024 reference period. Cross-check with latest national statistics.
- **Simplification:** The 3-player model abstracts from intra-African coordination problems and domestic political economy.
- **Behavioural parameters:** Grounded in laboratory experiments; field estimates for trade negotiations are scarce.

---

## License

For research and capacity-building purposes. Not for commercial redistribution.
