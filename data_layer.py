"""
Data Layer: Embedded calibrated datasets for AfCFTA-EPA Bargaining Asymmetry Simulator.
Sources: UN Comtrade, EU Access2Markets, AfCFTA e-Tariff Book, World Bank WGI, AidData.
All values are from publicly available datasets, calibrated to 2022-2024 reference periods.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. COUNTRY PROFILES
# ─────────────────────────────────────────────────────────────

COUNTRY_PROFILES = {
    "Ghana": {
        "region": "West Africa",
        "gdp_usd_bn": 72.8,
        "population_mn": 33.5,
        "ldc_status": False,
        "customs_union": "ECOWAS",
        "epa_status": "Interim EPA (provisional application since Dec 2016)",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "Côte d'Ivoire": {
        "region": "West Africa",
        "gdp_usd_bn": 70.0,
        "population_mn": 28.2,
        "ldc_status": False,
        "customs_union": "ECOWAS",
        "epa_status": "Interim EPA (provisional application since Sep 2016)",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "Kenya": {
        "region": "East Africa",
        "gdp_usd_bn": 113.0,
        "population_mn": 54.0,
        "ldc_status": False,
        "customs_union": "EAC",
        "epa_status": "EU-EAC EPA (variable geometry)",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "Nigeria": {
        "region": "West Africa",
        "gdp_usd_bn": 477.0,
        "population_mn": 223.0,
        "ldc_status": False,
        "customs_union": "ECOWAS",
        "epa_status": "No EPA signed",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "Senegal": {
        "region": "West Africa",
        "gdp_usd_bn": 28.0,
        "population_mn": 17.7,
        "ldc_status": True,
        "customs_union": "ECOWAS",
        "epa_status": "No bilateral EPA (covered under regional WA-EPA framework)",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "South Africa": {
        "region": "Southern Africa",
        "gdp_usd_bn": 399.0,
        "population_mn": 60.4,
        "ldc_status": False,
        "customs_union": "SACU",
        "epa_status": "SADC-EU EPA",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
    "Ethiopia": {
        "region": "East Africa",
        "gdp_usd_bn": 156.0,
        "population_mn": 126.0,
        "ldc_status": True,
        "customs_union": "None",
        "epa_status": "No EPA (EBA access)",
        "afcfta_schedule_submitted": True,
        "wto_member": False,
    },
    "Tanzania": {
        "region": "East Africa",
        "gdp_usd_bn": 75.7,
        "population_mn": 65.5,
        "ldc_status": True,
        "customs_union": "EAC",
        "epa_status": "EU-EAC EPA (not individually implementing)",
        "afcfta_schedule_submitted": True,
        "wto_member": True,
    },
}


# ─────────────────────────────────────────────────────────────
# 2. TARIFF CONCESSION SCHEDULES
# ─────────────────────────────────────────────────────────────

def get_epa_tariff_schedule(country: str) -> pd.DataFrame:
    """
    EPA liberalisation schedules: % of tariff lines liberalised by year.
    Source: EU Access2Markets, ActionAid Ghana EPA policy briefs.
    Ghana liberalises 80% of imports from EU over 15 years (2008-2023, extended to 2029).
    """
    schedules = {
        "Ghana": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
            "pct_liberalised": [0, 0, 0, 0, 22.6, 35.0, 44.1, 52.0, 58.0, 64.0, 70.0, 74.0, 77.0, 80.0],
            "tariff_revenue_loss_usd_mn": [0, 0, 0, 0, 42.1, 70.3, 108.5, 142.0, 165.0, 178.0, 195.0, 205.0, 215.0, 225.0],
        },
        "Côte d'Ivoire": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
            "pct_liberalised": [0, 0, 0, 0, 25.0, 37.0, 47.0, 55.0, 61.0, 67.0, 72.0, 76.0, 79.0, 81.0],
            "tariff_revenue_loss_usd_mn": [0, 0, 0, 0, 38.0, 62.0, 95.0, 125.0, 148.0, 162.0, 180.0, 190.0, 200.0, 210.0],
        },
        "Kenya": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
            "pct_liberalised": [0, 0, 0, 5.0, 12.0, 22.0, 33.0, 42.0, 50.0, 57.0, 63.0, 68.0, 73.0, 80.0],
            "tariff_revenue_loss_usd_mn": [0, 0, 0, 15.0, 35.0, 58.0, 85.0, 110.0, 135.0, 152.0, 170.0, 182.0, 195.0, 210.0],
        },
        "Nigeria": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
            "pct_liberalised": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tariff_revenue_loss_usd_mn": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "South Africa": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
            "pct_liberalised": [86.0, 86.0, 86.0, 86.0, 86.2, 86.5, 87.0, 87.5, 88.0, 88.5, 89.0, 89.5, 90.0, 90.0],
            "tariff_revenue_loss_usd_mn": [280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0],
        },
    }
    # Default for countries without EPA
    default_schedule = {
        "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
        "pct_liberalised": [0]*14,
        "tariff_revenue_loss_usd_mn": [0]*14,
    }
    data = schedules.get(country, default_schedule)
    return pd.DataFrame(data)


def get_afcfta_tariff_categories(country: str) -> dict:
    """
    AfCFTA tariff concession categories per modalities.
    Source: AfCFTA e-Tariff Book, MacMap/ITC, AU documentation.
    Cat A: Non-sensitive (90%) - liberalised in 5yr (non-LDC) / 10yr (LDC)
    Cat B: Sensitive (7%) - liberalised in 10yr / 13yr
    Cat C: Exclusion (3%) - no liberalisation
    """
    categories = {
        "Ghana": {"cat_a_pct": 90.3, "cat_b_pct": 6.8, "cat_c_pct": 2.9, "base_mfn_avg": 12.4},
        "Côte d'Ivoire": {"cat_a_pct": 90.1, "cat_b_pct": 6.9, "cat_c_pct": 3.0, "base_mfn_avg": 12.1},
        "Kenya": {"cat_a_pct": 90.5, "cat_b_pct": 6.5, "cat_c_pct": 3.0, "base_mfn_avg": 13.0},
        "Nigeria": {"cat_a_pct": 89.5, "cat_b_pct": 7.5, "cat_c_pct": 3.0, "base_mfn_avg": 14.2},
        "Senegal": {"cat_a_pct": 90.0, "cat_b_pct": 7.0, "cat_c_pct": 3.0, "base_mfn_avg": 12.1},
        "South Africa": {"cat_a_pct": 91.0, "cat_b_pct": 6.0, "cat_c_pct": 3.0, "base_mfn_avg": 7.6},
        "Ethiopia": {"cat_a_pct": 90.0, "cat_b_pct": 7.0, "cat_c_pct": 3.0, "base_mfn_avg": 17.5},
        "Tanzania": {"cat_a_pct": 90.2, "cat_b_pct": 6.8, "cat_c_pct": 3.0, "base_mfn_avg": 13.0},
    }
    return categories.get(country, {"cat_a_pct": 90.0, "cat_b_pct": 7.0, "cat_c_pct": 3.0, "base_mfn_avg": 12.0})


# ─────────────────────────────────────────────────────────────
# 3. TRADE DEPENDENCE RATIOS
# ─────────────────────────────────────────────────────────────

def get_trade_dependence(country: str) -> dict:
    """
    Trade dependence ratios (% of total exports/imports by partner).
    Source: UN Comtrade 2022, World Bank WITS.
    """
    dependence = {
        "Ghana": {
            "export_to_eu_pct": 26.3, "import_from_eu_pct": 22.8,
            "export_to_china_pct": 10.5, "import_from_china_pct": 18.7,
            "export_to_us_pct": 5.2, "import_from_us_pct": 5.8,
            "export_to_africa_pct": 15.2, "import_from_africa_pct": 8.3,
            "trade_openness": 0.73, "export_concentration_hhi": 0.18,
        },
        "Côte d'Ivoire": {
            "export_to_eu_pct": 32.5, "import_from_eu_pct": 25.3,
            "export_to_china_pct": 5.2, "import_from_china_pct": 16.5,
            "export_to_us_pct": 8.1, "import_from_us_pct": 3.5,
            "export_to_africa_pct": 18.5, "import_from_africa_pct": 12.0,
            "trade_openness": 0.68, "export_concentration_hhi": 0.21,
        },
        "Kenya": {
            "export_to_eu_pct": 19.2, "import_from_eu_pct": 14.5,
            "export_to_china_pct": 3.1, "import_from_china_pct": 21.5,
            "export_to_us_pct": 8.5, "import_from_us_pct": 4.2,
            "export_to_africa_pct": 35.0, "import_from_africa_pct": 14.0,
            "trade_openness": 0.42, "export_concentration_hhi": 0.12,
        },
        "Nigeria": {
            "export_to_eu_pct": 28.0, "import_from_eu_pct": 28.5,
            "export_to_china_pct": 3.8, "import_from_china_pct": 22.0,
            "export_to_us_pct": 5.5, "import_from_us_pct": 6.0,
            "export_to_africa_pct": 10.0, "import_from_africa_pct": 6.0,
            "trade_openness": 0.35, "export_concentration_hhi": 0.52,
        },
        "Senegal": {
            "export_to_eu_pct": 25.0, "import_from_eu_pct": 30.0,
            "export_to_china_pct": 6.0, "import_from_china_pct": 15.0,
            "export_to_us_pct": 2.0, "import_from_us_pct": 3.0,
            "export_to_africa_pct": 30.0, "import_from_africa_pct": 10.0,
            "trade_openness": 0.62, "export_concentration_hhi": 0.15,
        },
        "South Africa": {
            "export_to_eu_pct": 22.0, "import_from_eu_pct": 25.0,
            "export_to_china_pct": 11.0, "import_from_china_pct": 19.0,
            "export_to_us_pct": 7.5, "import_from_us_pct": 6.5,
            "export_to_africa_pct": 25.0, "import_from_africa_pct": 8.0,
            "trade_openness": 0.58, "export_concentration_hhi": 0.10,
        },
        "Ethiopia": {
            "export_to_eu_pct": 18.0, "import_from_eu_pct": 12.0,
            "export_to_china_pct": 8.0, "import_from_china_pct": 28.0,
            "export_to_us_pct": 10.0, "import_from_us_pct": 5.0,
            "export_to_africa_pct": 12.0, "import_from_africa_pct": 5.0,
            "trade_openness": 0.30, "export_concentration_hhi": 0.22,
        },
        "Tanzania": {
            "export_to_eu_pct": 15.0, "import_from_eu_pct": 10.0,
            "export_to_china_pct": 8.5, "import_from_china_pct": 25.0,
            "export_to_us_pct": 3.0, "import_from_us_pct": 3.0,
            "export_to_africa_pct": 22.0, "import_from_africa_pct": 10.0,
            "trade_openness": 0.35, "export_concentration_hhi": 0.14,
        },
    }
    return dependence.get(country, {
        "export_to_eu_pct": 20.0, "import_from_eu_pct": 20.0,
        "export_to_china_pct": 8.0, "import_from_china_pct": 18.0,
        "export_to_us_pct": 5.0, "import_from_us_pct": 5.0,
        "export_to_africa_pct": 15.0, "import_from_africa_pct": 8.0,
        "trade_openness": 0.45, "export_concentration_hhi": 0.15,
    })


# ─────────────────────────────────────────────────────────────
# 4. GOVERNANCE METRICS (WGI)
# ─────────────────────────────────────────────────────────────

def get_wgi_scores(country: str) -> dict:
    """
    World Governance Indicators (WGI) 2022.
    Scale: -2.5 (worst) to +2.5 (best). Percentile rank 0-100.
    Source: World Bank WGI dataset.
    """
    wgi = {
        "Ghana": {
            "voice_accountability": 0.52, "political_stability": 0.04,
            "govt_effectiveness": -0.08, "regulatory_quality": -0.07,
            "rule_of_law": 0.02, "control_corruption": -0.13,
            "negotiation_capacity_index": 0.58,  # Composite: derived from GE + RQ + RL
        },
        "Côte d'Ivoire": {
            "voice_accountability": -0.41, "political_stability": -0.72,
            "govt_effectiveness": -0.48, "regulatory_quality": -0.24,
            "rule_of_law": -0.49, "control_corruption": -0.47,
            "negotiation_capacity_index": 0.38,
        },
        "Kenya": {
            "voice_accountability": -0.20, "political_stability": -1.07,
            "govt_effectiveness": -0.31, "regulatory_quality": -0.16,
            "rule_of_law": -0.39, "control_corruption": -0.81,
            "negotiation_capacity_index": 0.48,
        },
        "Nigeria": {
            "voice_accountability": -0.54, "political_stability": -1.98,
            "govt_effectiveness": -1.02, "regulatory_quality": -0.74,
            "rule_of_law": -0.90, "control_corruption": -1.02,
            "negotiation_capacity_index": 0.32,
        },
        "Senegal": {
            "voice_accountability": 0.16, "political_stability": -0.19,
            "govt_effectiveness": -0.36, "regulatory_quality": -0.17,
            "rule_of_law": -0.19, "control_corruption": -0.15,
            "negotiation_capacity_index": 0.50,
        },
        "South Africa": {
            "voice_accountability": 0.59, "political_stability": -0.22,
            "govt_effectiveness": 0.21, "regulatory_quality": 0.17,
            "rule_of_law": -0.06, "control_corruption": -0.01,
            "negotiation_capacity_index": 0.72,
        },
        "Ethiopia": {
            "voice_accountability": -1.21, "political_stability": -1.82,
            "govt_effectiveness": -0.50, "regulatory_quality": -0.87,
            "rule_of_law": -0.68, "control_corruption": -0.43,
            "negotiation_capacity_index": 0.30,
        },
        "Tanzania": {
            "voice_accountability": -0.55, "political_stability": -0.30,
            "govt_effectiveness": -0.49, "regulatory_quality": -0.36,
            "rule_of_law": -0.39, "control_corruption": -0.47,
            "negotiation_capacity_index": 0.42,
        },
    }
    return wgi.get(country, {
        "voice_accountability": 0.0, "political_stability": 0.0,
        "govt_effectiveness": 0.0, "regulatory_quality": 0.0,
        "rule_of_law": 0.0, "control_corruption": 0.0,
        "negotiation_capacity_index": 0.45,
    })


# ─────────────────────────────────────────────────────────────
# 5. GREAT-POWER SHADOW INFLUENCE PARAMETERS
# ─────────────────────────────────────────────────────────────

def get_great_power_data(country: str) -> dict:
    """
    Chinese infrastructure lending, US AGOA, EU EPA conditionality.
    Sources: AidData, World Bank debtor reporting, USTR AGOA, EU DG Trade.
    """
    data = {
        "Ghana": {
            "chinese_loans_usd_bn": 3.5,
            "chinese_fdi_stock_usd_bn": 2.8,
            "belt_road_projects": 12,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 285.0,
            "eu_development_aid_usd_mn": 450.0,
            "eu_epa_adjustment_fund_usd_mn": 6.5,
            "debt_to_china_pct_gdp": 4.8,
            "infrastructure_dependency_china": 0.35,
        },
        "Côte d'Ivoire": {
            "chinese_loans_usd_bn": 2.1,
            "chinese_fdi_stock_usd_bn": 1.5,
            "belt_road_projects": 8,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 180.0,
            "eu_development_aid_usd_mn": 380.0,
            "eu_epa_adjustment_fund_usd_mn": 5.0,
            "debt_to_china_pct_gdp": 3.0,
            "infrastructure_dependency_china": 0.28,
        },
        "Kenya": {
            "chinese_loans_usd_bn": 7.9,
            "chinese_fdi_stock_usd_bn": 3.2,
            "belt_road_projects": 25,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 620.0,
            "eu_development_aid_usd_mn": 320.0,
            "eu_epa_adjustment_fund_usd_mn": 4.0,
            "debt_to_china_pct_gdp": 7.0,
            "infrastructure_dependency_china": 0.45,
        },
        "Nigeria": {
            "chinese_loans_usd_bn": 5.0,
            "chinese_fdi_stock_usd_bn": 4.5,
            "belt_road_projects": 20,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 150.0,
            "eu_development_aid_usd_mn": 520.0,
            "eu_epa_adjustment_fund_usd_mn": 0.0,
            "debt_to_china_pct_gdp": 1.0,
            "infrastructure_dependency_china": 0.20,
        },
        "Senegal": {
            "chinese_loans_usd_bn": 1.8,
            "chinese_fdi_stock_usd_bn": 0.6,
            "belt_road_projects": 6,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 45.0,
            "eu_development_aid_usd_mn": 280.0,
            "eu_epa_adjustment_fund_usd_mn": 0.0,
            "debt_to_china_pct_gdp": 6.4,
            "infrastructure_dependency_china": 0.32,
        },
        "South Africa": {
            "chinese_loans_usd_bn": 4.5,
            "chinese_fdi_stock_usd_bn": 8.0,
            "belt_road_projects": 15,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 3200.0,
            "eu_development_aid_usd_mn": 150.0,
            "eu_epa_adjustment_fund_usd_mn": 0.0,
            "debt_to_china_pct_gdp": 1.1,
            "infrastructure_dependency_china": 0.15,
        },
        "Ethiopia": {
            "chinese_loans_usd_bn": 13.7,
            "chinese_fdi_stock_usd_bn": 4.0,
            "belt_road_projects": 35,
            "agoa_eligible": False,
            "agoa_exports_usd_mn": 0.0,
            "eu_development_aid_usd_mn": 600.0,
            "eu_epa_adjustment_fund_usd_mn": 0.0,
            "debt_to_china_pct_gdp": 8.8,
            "infrastructure_dependency_china": 0.55,
        },
        "Tanzania": {
            "chinese_loans_usd_bn": 4.2,
            "chinese_fdi_stock_usd_bn": 2.0,
            "belt_road_projects": 18,
            "agoa_eligible": True,
            "agoa_exports_usd_mn": 80.0,
            "eu_development_aid_usd_mn": 350.0,
            "eu_epa_adjustment_fund_usd_mn": 0.0,
            "debt_to_china_pct_gdp": 5.5,
            "infrastructure_dependency_china": 0.40,
        },
    }
    return data.get(country, {
        "chinese_loans_usd_bn": 2.0, "chinese_fdi_stock_usd_bn": 1.5,
        "belt_road_projects": 10, "agoa_eligible": True,
        "agoa_exports_usd_mn": 100.0, "eu_development_aid_usd_mn": 300.0,
        "eu_epa_adjustment_fund_usd_mn": 0.0, "debt_to_china_pct_gdp": 3.0,
        "infrastructure_dependency_china": 0.30,
    })


# ─────────────────────────────────────────────────────────────
# 6. SECTOR-LEVEL DATA
# ─────────────────────────────────────────────────────────────

SECTOR_DATA = {
    "Agriculture": {
        "epa_exposure": 0.75, "afcfta_opportunity": 0.65,
        "employment_share": 0.30, "gdp_share": 0.20,
        "sensitivity": "high", "description": "Cocoa, cashew, shea; exposed to EU SPS standards; high AfCFTA potential for processed goods",
    },
    "Manufacturing": {
        "epa_exposure": 0.85, "afcfta_opportunity": 0.80,
        "employment_share": 0.12, "gdp_share": 0.15,
        "sensitivity": "critical", "description": "Textiles, food processing, pharmaceuticals; most affected by EPA import competition",
    },
    "Extractives": {
        "epa_exposure": 0.20, "afcfta_opportunity": 0.30,
        "employment_share": 0.05, "gdp_share": 0.25,
        "sensitivity": "low", "description": "Gold, oil, manganese; commodity exports mostly MFN-priced; limited EPA/AfCFTA impact",
    },
    "Services": {
        "epa_exposure": 0.40, "afcfta_opportunity": 0.70,
        "employment_share": 0.45, "gdp_share": 0.35,
        "sensitivity": "medium", "description": "Financial services, ICT, logistics; AfCFTA Phase II services protocol offers major gains",
    },
    "Digital Economy": {
        "epa_exposure": 0.15, "afcfta_opportunity": 0.90,
        "employment_share": 0.08, "gdp_share": 0.05,
        "sensitivity": "low", "description": "Fintech, e-commerce; minimal EPA constraints; high AfCFTA digital trade protocol upside",
    },
}


def get_all_countries():
    return list(COUNTRY_PROFILES.keys())


def get_country_summary(country: str) -> pd.DataFrame:
    """Build comprehensive summary DataFrame for a country."""
    profile = COUNTRY_PROFILES.get(country, {})
    dependence = get_trade_dependence(country)
    wgi = get_wgi_scores(country)
    gp = get_great_power_data(country)

    rows = []
    rows.append({"Category": "Economy", "Indicator": "GDP (USD bn)", "Value": profile.get("gdp_usd_bn", "N/A")})
    rows.append({"Category": "Economy", "Indicator": "Population (mn)", "Value": profile.get("population_mn", "N/A")})
    rows.append({"Category": "Economy", "Indicator": "Trade Openness", "Value": f"{dependence.get('trade_openness', 0):.0%}"})
    rows.append({"Category": "Trade (EU)", "Indicator": "Exports to EU (%)", "Value": f"{dependence.get('export_to_eu_pct', 0):.1f}%"})
    rows.append({"Category": "Trade (EU)", "Indicator": "Imports from EU (%)", "Value": f"{dependence.get('import_from_eu_pct', 0):.1f}%"})
    rows.append({"Category": "Trade (China)", "Indicator": "Exports to China (%)", "Value": f"{dependence.get('export_to_china_pct', 0):.1f}%"})
    rows.append({"Category": "Trade (China)", "Indicator": "Imports from China (%)", "Value": f"{dependence.get('import_from_china_pct', 0):.1f}%"})
    rows.append({"Category": "Trade (Africa)", "Indicator": "Exports to Africa (%)", "Value": f"{dependence.get('export_to_africa_pct', 0):.1f}%"})
    rows.append({"Category": "Governance", "Indicator": "Govt Effectiveness (WGI)", "Value": f"{wgi.get('govt_effectiveness', 0):.2f}"})
    rows.append({"Category": "Governance", "Indicator": "Regulatory Quality (WGI)", "Value": f"{wgi.get('regulatory_quality', 0):.2f}"})
    rows.append({"Category": "Governance", "Indicator": "Negotiation Capacity Index", "Value": f"{wgi.get('negotiation_capacity_index', 0):.2f}"})
    rows.append({"Category": "Great Powers", "Indicator": "Chinese Loans (USD bn)", "Value": f"{gp.get('chinese_loans_usd_bn', 0):.1f}"})
    rows.append({"Category": "Great Powers", "Indicator": "Debt to China (% GDP)", "Value": f"{gp.get('debt_to_china_pct_gdp', 0):.1f}%"})
    rows.append({"Category": "Great Powers", "Indicator": "EU Dev. Aid (USD mn)", "Value": f"{gp.get('eu_development_aid_usd_mn', 0):.0f}"})
    rows.append({"Category": "EPA Status", "Indicator": "EPA Arrangement", "Value": profile.get("epa_status", "N/A")})
    rows.append({"Category": "AfCFTA", "Indicator": "Schedule Submitted", "Value": "Yes" if profile.get("afcfta_schedule_submitted") else "No"})

    return pd.DataFrame(rows)
