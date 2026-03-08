# AfCFTA–EPA Bargaining Asymmetry Simulator

**Sequential Game Engine for African States**

An interactive Streamlit application that quantifies bargaining power asymmetries when negotiating AfCFTA protocols while locked into EPA schedules, modeling great-power (China/US/EU) shadow influence.

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this directory to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set:
   - **Main file path:** `app.py`
   - **Python version:** 3.10+
5. Click "Deploy"

No `st.secrets` required — all data is embedded (public sources).

## Project Structure

```
afcfta-simulator/
├── app.py                 # Main Streamlit application (8 pages)
├── data_layer.py          # Embedded calibrated datasets
├── game_engine.py         # Extensive-form game tree & solver
├── visualisations.py      # Plotly interactive charts
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

- **Dashboard**: Country-level BPI, trade dependence sunburst, EPA timeline, great-power footprint
- **Game Tree & Equilibrium**: Interactive extensive-form game solved via backward induction (SPE)
- **Threat Points**: Credible threat identification with red-line analysis
- **Sensitivity Analysis**: Single and dual-parameter sweeps with heatmaps
- **Concession Sequencing**: Optimal sector-level AfCFTA liberalisation order
- **Comparative Analysis**: Cross-country BPI benchmarking and coalition identification
- **Data Explorer**: Browse and export all datasets as CSV
- **Methodology**: Full literature grounding, parameter justification, limitations

## Data Sources

| Dataset | Source |
|---|---|
| AfCFTA tariff schedules | AfCFTA e-Tariff Book |
| EPA liberalisation | EU Access2Markets |
| Trade flows | UN Comtrade (2022) |
| Governance | World Bank WGI (2022) |
| Chinese finance | AidData |
| AGOA | USTR |

## Countries Covered

Ghana (primary), Côte d'Ivoire, Kenya, Nigeria, Senegal, South Africa, Ethiopia, Tanzania

## License

For research and capacity-building purposes. Not for commercial redistribution.
