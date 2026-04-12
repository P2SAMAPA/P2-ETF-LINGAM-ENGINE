"""
P2-ETF-LINGAM-Engine Streamlit Dashboard
========================================
Main Streamlit application for displaying causal ETF predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import requests
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from streamlit_app.utils import (
    apply_custom_css,
    get_etf_display_name,
    calculate_next_trading_day,
    create_sample_data
)
from streamlit_app.components.hero_card import (
    render_hero_card,
    render_comparison_card
)
from streamlit_app.components.metrics_display import (
    render_kpi_boxes,
    render_performance_chart,
    render_signal_history_table
)

# HuggingFace dataset configuration
HF_DATASET_REPO = "P2SAMAPA/p2-etf-lingam-results"
PREDICTIONS_FILE = "predictions.parquet"  # Change if your file has a different name

@st.cache_data(ttl=3600)
def load_predictions_from_hf() -> pd.DataFrame:
    """
    Load predictions parquet file from HuggingFace dataset.
    Returns empty DataFrame if loading fails.
    """
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{PREDICTIONS_FILE}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        return df
    except Exception as e:
        st.warning(f"Could not load predictions from HuggingFace: {e}")
        return pd.DataFrame()

def get_latest_prediction(df: pd.DataFrame, universe: str) -> dict:
    """
    Extract the most recent prediction for a given universe from the parquet data.
    Assumes flat columns: top_3_picks_tickers, top_3_picks_scores, metrics_*
    """
    if df.empty:
        return create_sample_data(universe)

    # Ensure date column is datetime
    if 'date' not in df.columns:
        return create_sample_data(universe)
    df['date'] = pd.to_datetime(df['date'])

    # Filter by universe column
    if 'universe' not in df.columns:
        return create_sample_data(universe)
    uni_df = df[df['universe'].str.lower() == universe.lower()].copy()
    if uni_df.empty:
        return create_sample_data(universe)

    # Get most recent prediction
    latest = uni_df.sort_values('date', ascending=False).iloc[0]

    # Build top_3_picks from flat columns
    tickers_str = latest.get('top_3_picks_tickers', '')
    scores_str = latest.get('top_3_picks_scores', '')
    top_3_picks = []
    if tickers_str and scores_str and not pd.isna(tickers_str) and not pd.isna(scores_str):
        tickers = [t.strip() for t in str(tickers_str).split(',')]
        scores = [float(s.strip()) for s in str(scores_str).split(',')]
        for t, s in zip(tickers[:3], scores[:3]):
            top_3_picks.append({'ticker': t, 'score': s})
    # Fallback if missing
    while len(top_3_picks) < 3:
        top_3_picks.append({'ticker': 'N/A', 'score': 0.0})

    # Build metrics dict from flat columns
    metrics = {
        'total_return': latest.get('metrics_total_return', 0.0),
        'sharpe_ratio': latest.get('metrics_sharpe_ratio', 0.0),
        'max_drawdown': latest.get('metrics_max_drawdown', 0.0),
        'win_rate': latest.get('metrics_win_rate', 0.0),
        'best_day': latest.get('metrics_best_day', 0.0),
    }
    # Convert any NaNs to 0
    for k in metrics:
        if pd.isna(metrics[k]):
            metrics[k] = 0.0

    leader = latest.get('predicted_leader_etf', 'N/A')
    if pd.isna(leader) or leader == '' or leader == 'N/A':
        # No valid leader in this row, fallback to sample data
        return create_sample_data(universe)

    # Determine benchmark
    if universe == 'fi_commodity':
        benchmark = config.FI_COMMODITY_BENCHMARK
    else:
        benchmark = config.EQUITY_BENCHMARK

    return {
        'leader': leader,
        'leader_name': get_etf_display_name(leader),
        'conviction': latest.get('causal_confidence', 0.5),
        'top_3_picks': top_3_picks,
        'prediction_date': latest['date'].strftime('%Y-%m-%d'),
        'training_mode': latest.get('training_mode', 'fixed'),
        'metrics': metrics,
        'benchmark': benchmark,
    }

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="P2 — ETF LINGAM Engine",
        page_icon="📊",
        layout="wide"
    )

    apply_custom_css()

    st.title("📊 P2 — ETF LINGAM Engine")

    # Sidebar for debugging
    with st.sidebar:
        st.markdown("### 🔍 Debug Info")
        if st.button("Refresh data from HuggingFace"):
            st.cache_data.clear()
            st.rerun()
        predictions_df = load_predictions_from_hf()
        if not predictions_df.empty:
            st.success(f"Loaded {len(predictions_df)} predictions")
            st.write("Columns:", list(predictions_df.columns))
            st.write("Sample:", predictions_df.head(2).to_dict())
        else:
            st.warning("No predictions loaded. Using sample data.")

    tabs = st.tabs(["Fixed Income / Alts", "Equity Sectors"])

    with tabs[0]:
        render_fi_commodity_tab(predictions_df)

    with tabs[1]:
        render_equity_tab(predictions_df)

def render_fi_commodity_tab(predictions_df: pd.DataFrame):
    st.markdown("### Fixed Income / Alts Module")
    data = get_latest_prediction(predictions_df, 'fi_commodity')
    _render_universe_tab(data)

def render_equity_tab(predictions_df: pd.DataFrame):
    st.markdown("### Equity Sectors Module")
    data = get_latest_prediction(predictions_df, 'equity')
    _render_universe_tab(data)

def _render_universe_tab(data: dict):
    """Render the prediction display for a given universe."""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Hero card HTML
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
            border-radius: 16px;
            padding: 32px;
            border: 1px solid #d8b4fe;
            margin-bottom: 24px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 style="font-size: 64px; font-weight: 700; color: #6B21A8; margin: 0; line-height: 1;">
                        {data['leader']}
                    </h1>
                    <p style="font-size: 18px; color: #6b7280; margin-top: 8px;">
                        {data['leader_name']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin: 0;">
                        Conviction
                    </p>
                    <p style="font-size: 36px; font-weight: 600; color: #6B21A8; margin: 0;">
                        {data['conviction'] * 100:.1f}%
                    </p>
                </div>
            </div>

            <div style="margin-top: 24px; padding-top: 24px; border-top: 1px solid #d8b4fe;">
                <p style="font-size: 14px; color: #6b7280; margin-bottom: 8px;">2nd & 3rd Picks</p>
                <p style="font-size: 16px; color: #4b5563; margin: 8px 0;">
                    2nd: <span style="font-weight: 600; color: #6B21A8;">{data['top_3_picks'][1]['ticker']}</span>
                    {data['top_3_picks'][1].get('score', 0) * 100:.1f}%
                </p>
                <p style="font-size: 16px; color: #4b5563; margin: 8px 0;">
                    3rd: <span style="font-weight: 600; color: #6B21A8;">{data['top_3_picks'][2]['ticker']}</span>
                    {data['top_3_picks'][2].get('score', 0) * 100:.1f}%
                </p>
            </div>

            <div style="margin-top: 24px; display: flex; gap: 12px;">
                <span style="
                    background: #6B21A8;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 500;
                ">
                    Signal for: {data['prediction_date']}
                </span>
                <span style="
                    background: #f3e8ff;
                    color: #6B21A8;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 500;
                ">
                    {data['training_mode']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Performance Metrics")
        render_kpi_boxes(data['metrics'])

    st.markdown("---")
    st.markdown("#### Strategy Performance")

    # For now, we still use simulated returns for the chart
    # (You can replace with actual stored backtest data if available)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    strategy_returns = pd.Series(np.random.randn(300) * 0.01 + 0.0003, index=dates)
    benchmark_returns = pd.Series(np.random.randn(300) * 0.008 + 0.0002, index=dates)

    render_comparison_card(
        f"SAMBA {data['leader']}",
        data['benchmark'],
        data['metrics']
    )

    render_performance_chart(
        strategy_returns,
        benchmark_returns,
        data['benchmark'],
        f"SAMBA {data['leader']} vs {data['benchmark']}"
    )

    st.markdown("---")
    st.markdown("#### Signal History")
    # Optional: load actual history if stored
    sample_signals = [
        {'date': '2024-04-10', 'ticker': 'GLD', 'conviction': 0.999, 'actual_return': -0.0018, 'is_hit': False},
        {'date': '2024-04-03', 'ticker': 'TLT', 'conviction': 0.75, 'actual_return': 0.023, 'is_hit': True},
        {'date': '2024-03-27', 'ticker': 'GLD', 'conviction': 0.82, 'actual_return': 0.015, 'is_hit': True},
        {'date': '2024-03-20', 'ticker': 'HYG', 'conviction': 0.68, 'actual_return': -0.008, 'is_hit': False},
        {'date': '2024-03-13', 'ticker': 'VNQ', 'conviction': 0.71, 'actual_return': 0.012, 'is_hit': True},
    ]
    render_signal_history_table(sample_signals)

if __name__ == "__main__":
    main()
