"""
P2-ETF-LINGAM-Engine Configuration
==================================
Central configuration for the quantitative trading engine.
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"

# ==============================================================================
# HUGGINGFACE DATASET CONFIGURATION
# ==============================================================================
HF_CONFIG = {
    "api_key": os.environ.get("HF_API_KEY", ""),
    "model": os.environ.get("HF_MODEL", "default-model")
}

INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_DATASET = "P2SAMAPA/p2-etf-lingam-results"

# ==============================================================================
# ETF UNIVERSE DEFINITIONS
# ==============================================================================

# FI/Commodity Module - 7 assets + Benchmark
FI_COMMODITY_ASSETS = [
    "GLD",   # SPDR Gold Shares
    "TLT",   # iShares 20+ Year Treasury Bond
    "VCIT",  # Vanguard Intermediate-Term Corporate Bond
    "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond
    "HYG",   # iShares iBoxx $ High Yield Corporate Bond
    "VNQ",   # Vanguard Real Estate ETF
    "SLV",   # iShares Silver Trust
]

FI_COMMODITY_BENCHMARK = "AGG"  # iShares Core US Aggregate Bond

# Equity Module - 14 assets + Benchmark
EQUITY_ASSETS = [
    "QQQ",   # Invesco QQQ Trust (Tech/Nasdaq)
    "XLK",   # Technology Select Sector SPDR
    "XLF",   # Financial Select Sector SPDR
    "XLE",   # Energy Select Sector SPDR
    "XLV",   # Health Care Select Sector SPDR
    "XLI",   # Industrial Select Sector SPDR
    "XLY",   # Consumer Discretionary Select Sector SPDR
    "XLP",   # Consumer Staples Select Sector SPDR
    "XLU",   # Utilities Select Sector SPDR
    "XME",   # Metallurgical Mining Company
    "IWM",   # iShares Russell 2000 ETF (Small Cap)
    "XLB",   # Materials Select Sector SPDR
    "XLRE",  # Real Estate Select Sector SPDR
    "GDX",   # VanEck Gold Miners
]

EQUITY_BENCHMARK = "SPY"  # SPDR S&P 500 ETF Trust

# Combined ETF universe (all assets + both benchmarks) used by data.loader
ETF_UNIVERSE = list(set(FI_COMMODITY_ASSETS + EQUITY_ASSETS + [FI_COMMODITY_BENCHMARK, EQUITY_BENCHMARK]))

# ==============================================================================
# MACRO VARIABLES (used as features in causal discovery)
# ==============================================================================
MACRO_VARIABLES = [
    "VIX",        # CBOE Volatility Index
    "DXY",        # US Dollar Index
    "T10Y2Y",     # 10-Year minus 2-Year Treasury Yield Spread
    "TBILL_3M",   # 3-Month Treasury Bill Rate
    "IG_SPREAD",  # Investment Grade Credit Spread
    "HY_SPREAD",  # High Yield Credit Spread
]

# ==============================================================================
# LINGAM CONFIGURATION (single measure: pwling, 100 bootstraps)
# ==============================================================================
LINGAM_CONFIG = {
    "measure": "pwling",
    "bootstrap": True,
    "n_samplings": 500,
    "significance_level": 0.05,
    "subset_variable_names": None,
}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
ROLLING_WINDOW_DAYS = 252

# Keep all original shrinking windows (2008 to 2024 inclusive, every year)
SHRINKING_WINDOW_YEARS = list(range(2008, 2025))   # 17 windows

MIN_CAUSAL_THRESHOLD = 0.3

# ==============================================================================
# CONSENSUS SCORING WEIGHTS (Shrinking Window)
# ==============================================================================
CONSENSUS_WEIGHTS = {
    "annualized_return": 0.60,
    "sharpe_ratio": 0.20,
    "max_drawdown": 0.20,
}
NEGATIVE_RETURN_ZERO_WEIGHT = True

# ==============================================================================
# STREAMLIT DISPLAY CONFIGURATION
# ==============================================================================
DISPLAY_CONFIG = {
    "primary_color": "#6B21A8",
    "positive_color": "#10B981",
    "negative_color": "#EF4444",
    "neutral_color": "#6B7280",
    "background_color": "#FFFFFF",
    "border_radius": "8px",
    "font_family": "Inter, system-ui, sans-serif",
}

# ==============================================================================
# DATE CONFIGURATION
# ==============================================================================
DATA_START_DATE = "2008-01-01"
DATA_END_DATE = "2026-04-11"
PREDICTION_DATE = "2026-04-11"

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================
OUTPUT_COLUMNS = [
    "date",
    "universe",
    "predicted_leader_etf",
    "predicted_return",
    "causal_confidence",
    "top_3_picks",
    "followers",
    "macro_context",
    "dag_edges",
    "dag_strengths",
    "model_version",
    "training_mode",
    "window_start",
    "window_end",
]

METRICS_TO_CALCULATE = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "best_day",
    "worst_day",
    "volatility",
    "sortino_ratio",
]

MODEL_VERSION = "v1.0.0"
ENGINE_NAME = "P2-ETF-LINGAM-Engine"
