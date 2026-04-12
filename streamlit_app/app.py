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
    render_header,
    get_etf_display_name,
    calculate_next_trading_day,
    create_sample_data  # Keep as fallback
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
PREDICTIONS_FILE = "predictions.parquet"  # Adjust if filename differs

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_from_hf() -> pd.DataFrame:
    """
    Load predictions parquet file from HuggingFace dataset.
    Returns empty DataFrame if loading fails.
    """
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{PREDICTIONS_FILE}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        return df
    except Exception as e:
        st.warning(f"Could not load predictions from HuggingFace: {e}")
        return pd.DataFrame()

def get_latest_prediction(df: pd.DataFrame, universe: str) -> dict:
    """
    Extract the most recent prediction for a given universe.
    Returns a dictionary with the same structure as create_sample_data().
    """
    if df.empty:
        return create_sample_data(universe)  # fallback to sample

    # Filter by universe (column may be 'universe' or derived from 'predicted_leader_etf' mapping)
    if 'universe' in df.columns:
        uni_df = df[df['universe'].str.lower() == universe.lower()]
    else:
        # Fallback: if no universe column, try to infer from ticker list
        fi_assets = set(config.FI_COMMODITY_ASSETS + [config.FI_COMMODITY_BENCHMARK])
        eq_assets = set(config.EQUITY_ASSETS + [config.EQUITY_BENCHMARK])
        if universe == 'fi_commodity':
            uni_df = df[df['predicted_leader_etf'].isin(fi_assets)]
        else:
            uni_df = df[df['predicted_leader_etf'].isin(eq_assets)]

    if uni_df.empty:
        return create_sample_data(universe)

    # Get most recent prediction by date
    latest = uni_df.sort_values('date', ascending=False).iloc[0]

    # Build display dictionary similar to create_sample_data
    top_3_picks = latest.get('top_3_picks', [])
    if isinstance(top_3_picks, str):
        import json
        top_3_picks = json.loads(top_3_picks)  # if stored as JSON string

    # Ensure top_3_picks has at least 3 entries (pad if needed)
    while len(top_3_picks) < 3:
        top_3_picks.append({'ticker': 'N/A', 'score': 0.0})

    metrics = latest.get('metrics', {})
    if isinstance(metrics, str):
        import json
        metrics = json.loads(metrics)

    # Get ETF display name
    ticker = latest['predicted_leader_etf']
    leader_name = get_etf_display_name(ticker)

    # Determine benchmark based on universe
    if universe == 'fi_commodity':
        benchmark = config.FI_COMMODITY_BENCHMARK
    else:
        benchmark = config.EQUITY_BENCHMARK

    return {
        'leader': ticker,
        'leader_name': leader_name,
        'conviction': latest.get('causal_confidence', 0.5),
        'top_3_picks': top_3_picks[:3],
        'prediction_date': latest.get('date', datetime.now().strftime('%Y-%m-%d')),
        'training_mode': latest.get('training_mode', 'fixed'),
        'metrics': {
            'total_return': metrics.get('total_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'best_day': metrics.get('best_day', 0),
        },
        'benchmark': benchmark,
    }

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title="P2 — ETF LINGAM Engine",
        page_icon="📊",
        layout="wide"
    )

    # Apply custom CSS
    apply_custom_css()

    # Header
    st.title("📊 P2 — ETF LINGAM Engine")

    # Load predictions from HuggingFace
    predictions_df = load_predictions_from_hf()

    # Create tabs
    tab1, tab2 = st.tabs([
        "Option A — Fixed Income / Alts",
        "Option B — Equity Sectors"
    ])

    # Fixed Income / Alts Tab
    with tab1:
        render_fi_commodity_tab(predictions_df)

    # Equity Tab
    with tab2:
        render_equity_tab(predictions_df)


def render_fi_commodity_tab(predictions_df: pd.DataFrame):
    """Render FI/Commodity universe tab using real predictions."""
    st.markdown("### Fixed Income / Alts Module")

    # Get latest prediction for FI/Commodity
    data = get_latest_prediction(predictions_df, 'fi_commodity')

    # Hero Card
    st.markdown("#### Predicted Leader ETF")

    col1, col2 = st.columns([2, 1])

    with col1:
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

    # Comparison section
    st.markdown("---")
    st.markdown("#### Strategy Performance")

    # For demo, we still use sample returns; you could replace with actual strategy returns if stored
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

    # Signal history - could also be loaded from HF if you have a history file
    st.markdown("---")
    st.markdown("#### Signal History")

    # Try to load a history file (optional)
    history_df = load_history_from_hf()
    if not history_df.empty and 'fi_commodity' in history_df['universe'].values:
        history = history_df[history_df['universe'] == 'fi_commodity'].to_dict('records')
        render_signal_history_table(history)
    else:
        # Fallback sample history
        sample_signals = [
            {'date': '2024-04-10', 'ticker': 'GLD', 'conviction': 0.999, 'actual_return': -0.0018, 'is_hit': False},
            {'date': '2024-04-03', 'ticker': 'TLT', 'conviction': 0.75, 'actual_return': 0.023, 'is_hit': True},
            {'date': '2024-03-27', 'ticker': 'GLD', 'conviction': 0.82, 'actual_return': 0.015, 'is_hit': True},
            {'date': '2024-03-20', 'ticker': 'HYG', 'conviction': 0.68, 'actual_return': -0.008, 'is_hit': False},
            {'date': '2024-03-13', 'ticker': 'VNQ', 'conviction': 0.71, 'actual_return': 0.012, 'is_hit': True},
        ]
        render_signal_history_table(sample_signals)


def render_equity_tab(predictions_df: pd.DataFrame):
    """Render Equity universe tab using real predictions."""
    st.markdown("### Equity Sectors Module")

    data = get_latest_prediction(predictions_df, 'equity')

    # Hero Card (same structure as FI)
    col1, col2 = st.columns([2, 1])

    with col1:
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

    # Comparison section
    st.markdown("---")
    st.markdown("#### Strategy Performance")

    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    strategy_returns = pd.Series(np.random.randn(300) * 0.012 + 0.0004, index=dates)
    benchmark_returns = pd.Series(np.random.randn(300) * 0.01 + 0.0003, index=dates)

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

    # Signal history
    st.markdown("---")
    st.markdown("#### Signal History")

    history_df = load_history_from_hf()
    if not history_df.empty and 'equity' in history_df['universe'].values:
        history = history_df[history_df['universe'] == 'equity'].to_dict('records')
        render_signal_history_table(history)
    else:
        sample_signals = [
            {'date': '2024-04-10', 'ticker': 'QQQ', 'conviction': 0.72, 'actual_return': 0.021, 'is_hit': True},
            {'date': '2024-04-03', 'ticker': 'XLK', 'conviction': 0.65, 'actual_return': -0.012, 'is_hit': False},
            {'date': '2024-03-27', 'ticker': 'QQQ', 'conviction': 0.78, 'actual_return': 0.031, 'is_hit': True},
            {'date': '2024-03-20', 'ticker': 'XLE', 'conviction': 0.58, 'actual_return': 0.018, 'is_hit': True},
            {'date': '2024-03-13', 'ticker': 'QQQ', 'conviction': 0.69, 'actual_return': -0.005, 'is_hit': False},
        ]
        render_signal_history_table(sample_signals)


@st.cache_data(ttl=3600)
def load_history_from_hf() -> pd.DataFrame:
    """Optional: load historical signals if stored as separate file."""
    try:
        url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/signal_history.parquet"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_parquet(BytesIO(response.content))
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
