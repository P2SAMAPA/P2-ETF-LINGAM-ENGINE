# P2-ETF-LINGAM-Engine: LiNGAM-Powered Quantitative Trading Engine

## 1. Project Overview

**Project Name**: P2-ETF-LINGAM-Engine
**Core Value**: Causal discovery-driven quantitative trading engine using LiNGAM + DirectLiNGAM algorithms to identify leader-follower relationships between ETFs and generate predictions based on direct causal strength scores.

**Target Users**: Quantitative traders, portfolio managers, and researchers seeking causal insights over correlation-based strategies.

---

## 2. Data Specification

### 2.1 Input Dataset
- **Source**: HuggingFace `P2SAMAPA/fi-etf-macro-signal-master-data`
- **Time Range**: 2008-01-01 to 2026-04-11 (4,769 rows)
- **Macro Variables**: VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD

### 2.2 ETF Universes

#### FI/Commodity Module
| Ticker | Name | Category |
|--------|------|----------|
| GLD | SPDR Gold Shares | Precious Metals |
| TLT | iShares 20+ Year Treasury Bond | Treasury |
| VCIT | Vanguard Intermediate-Term Corporate Bond | Corporate Bonds |
| LQD | iShares iBoxx $ Investment Grade Corporate Bond | IG Corporate |
| HYG | iShares iBoxx $ High Yield Corporate Bond | HY Corporate |
| VNQ | Vanguard Real Estate ETF | Real Estate |
| SLV | iShares Silver Trust | Silver |
| **Benchmark** | AGG (iShares Core US Aggregate Bond) | - |

#### Equity Module
| Ticker | Name | Sector |
|--------|------|--------|
| QQQ | Invesco QQQ Trust | Tech/Nasdaq |
| XLK | Technology Select Sector SPDR | Technology |
| XLF | Financial Select Sector SPDR | Financials |
| XLE | Energy Select Sector SPDR | Energy |
| XLV | Health Care Select Sector SPDR | Healthcare |
| XLI | Industrial Select Sector SPDR | Industrials |
| XLY | Consumer Discretionary Select Sector SPDR | Consumer Disc |
| XLP | Consumer Staples Select Sector SPDR | Consumer Staples |
| XLU | Utilities Select Sector SPDR | Utilities |
| XME | Metallurgical Mining Company | Metals/Mining |
| IWM | iShares Russell 2000 ETF | Small Cap |
| XLB | Materials Select Sector SPDR | Materials |
| XLRE | Real Estate Select Sector SPDR | Real Estate |
| GDX | VanEck Gold Miners | Gold Miners |
| **Benchmark** | SPY (SPDR S&P 500 ETF Trust) | - |

### 2.3 Output Dataset
- **Destination**: HuggingFace `P2SAMAPA/p2-etf-lingam-results`
- **Content**: Predictions, metrics, causal DAGs, signal history

---

## 3. Algorithm Specification

### 3.1 LiNGAM + DirectLiNGAM
- **Purpose**: Discover causal DAGs from observational ETF return data
- **Library**: `lingam` package
- **Approach**:
  1. Build causal DAG using LiNGAM (Linear Non-Gaussian Acyclic Model)
  2. Orient causal directions using non-Gaussianity
  3. Calculate direct causal effects between variables using DirectLiNGAM

### 3.2 Causal Discovery Parameters
```python
LINGAM_CONFIG = {
    "measure": "pwling",           # Pairwise likelihood LiNGAM
    "bootstrap": True,
    "n_samplings": 100,            # Bootstrap samples
    "significance_level": 0.05,    # Significance threshold
}
```

---

## 4. Training Methodology

### 4.1 Data Split: 80/10/10
| Split | Purpose | Percentage |
|-------|---------|------------|
| **Train** | Model learning | 80% |
| **Validation** | Hyperparameter tuning | 10% |
| **Test (OOS)** | Out-of-sample performance | 10% |

### 4.2 Training Modes

#### Mode 1: Fixed Full Data
```
Training Window: 2008-01-01 → 2026-04-11 (YTD)
Split: 80/10/10 within this window
```

#### Mode 2: Shrinking Windows
| Window | Start Date | End Date |
|--------|------------|----------|
| Window 1 | 2008-01-01 | 2026-04-11 |
| Window 2 | 2009-01-01 | 2026-04-11 |
| Window 3 | 2010-01-01 | 2026-04-11 |
| ... | ... | ... |
| Window N | 2024-01-01 | 2026-04-11 |

### 4.3 Shrinking Window Consensus
For each ETF, calculate **Weighted Consensus Score** across all windows:

```
Consensus_Score = Σ (Window_Score × Window_Weight)
```

| Metric | Weight | Rule |
|--------|--------|------|
| **Annualized Return** | 60% | If return ≤ 0 → Weight = 0 |
| **Sharpe Ratio** | 20% | Standard normalization |
| **Max Drawdown** | 20% | Inverted: lower = better |

**Final Prediction**: ETF with highest Weighted Consensus Score

---

## 5. Display Specification

### 5.1 Navigation
- **Tab 1**: "Option A — Fixed Income / Alts" (FI/Commodity Module)
- **Tab 2**: "Option B — Equity Sectors" (Equity Module)

### 5.2 Hero Card Component
**Location**: Top of each tab

**Content**:
- **Leader ETF**: Predicted top ticker (e.g., "GLD")
- **Conviction Score**: Confidence percentage (e.g., "99.9%")
- **Top 3 Picks**: Leader, 2nd, 3rd with convictions
- **Prediction Date**: Target trading date
- **Model Methodology**: Fixed Split or Shrinking Window
- **Training Date Range**: Window and OOS periods

### 5.3 Metrics Dashboard

#### KPI Boxes (per strategy)
| Metric | Description |
|--------|-------------|
| Total Return | Overall strategy return % |
| Sharpe Ratio | Risk-adjusted return |
| PEAK→TROUGH | Maximum drawdown |
| Win Rate | % winning trades |
| Best Day | Best single-day return |

#### Performance Chart
- **Line Chart**: Strategy vs Benchmark
- **Y-axis**: Growth of $1
- **X-axis**: Time
- **Purple line**: Strategy
- **Grey dashed**: Benchmark (AGG/SPY)

### 5.4 Signal History Table
| Column | Description |
|--------|-------------|
| Date | Trading date |
| Pick | Selected ETF ticker |
| Conviction | Model confidence % |
| Actual Return | Realized return % |
| Hit | Success indicator |

### 5.5 Design System
- **Primary Color**: Purple (#6B21A8)
- **Background**: White
- **Positive**: Green (#10B981)
- **Negative**: Red (#EF4444)
- **Typography**: Sans-serif (Inter/system)
- **Border Radius**: Rounded corners (8px)
- **Spacing**: Generous whitespace

---

## 6. Technical Stack

### 6.1 Python Dependencies
```
lingam>=1.8.0
networkx>=3.2
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
streamlit>=1.28.0
datasets>=2.14.0
huggingface_hub>=0.19.0
matplotlib>=3.7.0
```

### 6.2 Project Structure
```
P2-ETF-LINGAM-ENGINE/
├── SPEC.md
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── main.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── preprocessing.py
├── core/
│   ├── __init__.py
│   ├── lingam_engine.py
│   ├── causal_analyzer.py
│   └── metrics.py
├── modules/
│   ├── fi_commodity/
│   │   ├── causal_discovery.py
│   │   ├── leader_identifier.py
│   │   └── signal_generator.py
│   └── equity/
│       ├── causal_discovery.py
│       ├── leader_identifier.py
│       └── signal_generator.py
├── streamlit_app/
│   ├── app.py
│   ├── components/
│   │   ├── hero_card.py
│   │   ├── dag_visualizer.py
│   │   └── metrics_display.py
│   └── pages/
│       ├── fi_commodity.py
│       └── equity.py
└── output/
    ├── predictions.py
    └── hf_uploader.py
```

---

## 7. Deployment

### 7.1 Environment Variables
```bash
HF_TOKEN=your_huggingface_token
```

### 7.2 Streamlit Cloud
- **Requirements**: `requirements.txt` + `streamlit_app/app.py`
- **Secrets**: `HF_TOKEN` for dataset upload

---

## 8. Success Metrics

### 8.1 Model Performance
- Leader prediction accuracy
- Causal relationship stability
- Return correlation (predicted vs actual)

### 8.2 Business KPIs
- Prediction coverage (% valid days)
- Strategy Sharpe ratio
- Dashboard engagement
