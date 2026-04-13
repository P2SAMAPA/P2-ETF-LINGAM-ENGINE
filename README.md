# P2-ETF-LINGAM-ENGINE

A causal discovery-driven quantitative trading engine using LiNGAM (Linear Non-Gaussian Acyclic Model) for ETF universe analysis.

## Features

- **Dual Module Architecture**: Separate engines for FI/Commodity and Equity ETFs
- **Causal Discovery**: LiNGAM + DirectLiNGAM for discovering causal relationships
- **Leader Identification**: Identifies leading ETFs that drive other assets
- **Shrinking Window Consensus**: Weighted scoring across multiple training windows
- **Streamlit Dashboard**: SAMBA-style visualization with hero cards and performance metrics
- [ Tried kernel but took more than 6 hours so stuck to pwling, similarly for Shrinking Windows, no bootstrapping to reduce training hours]

## Universe Configuration

### FI/Commodity Module
- **Assets**: GLD, TLT, VCIT, LQD, HYG, VNQ, SLV
- **Benchmark**: AGG

### Equity Module
- **Assets**: QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, IWM, XLB, XLRE, GDX
- **Benchmark**: SPY

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Training
```bash
# Run full pipeline for both universes
python main.py --mode both

# Run specific universe
python main.py --universe fi_commodity --mode fixed

# Run with shrinking windows
python main.py --universe equity --mode shrinking --bootstrap
```

### Run Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

## Configuration

Edit `config.py` to modify:
- ETF universes
- LiNGAM parameters
- Training configuration
- Consensus scoring weights

## HuggingFace Integration

- **Input Dataset**: `P2SAMAPA/fi-etf-macro-signal-master-data`
- **Output Dataset**: `P2SAMAPA/p2-etf-lingam-results`

Set `HF_TOKEN` environment variable for upload functionality.

## License

MIT
