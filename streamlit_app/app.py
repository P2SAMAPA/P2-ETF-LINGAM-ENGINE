"""
P2-ETF-LINGAM-Engine Streamlit Dashboard
========================================
Displays fixed-split and consensus (shrinking window) predictions per universe.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import ast
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from streamlit_app.utils import apply_custom_css, get_etf_display_name
from streamlit_app.components.metrics_display import (
    render_kpi_boxes,
    render_comparison_card,
    render_performance_chart,
    render_signal_history_table
)

HF_DATASET_REPO = "P2SAMAPA/p2-etf-lingam-results"
PREDICTIONS_FILE = "predictions.parquet"

@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{PREDICTIONS_FILE}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_parquet(BytesIO(resp.content))
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        return pd.DataFrame()

def parse_followers(followers_str):
    if pd.isna(followers_str) or not followers_str:
        return []
    try:
        parsed = ast.literal_eval(str(followers_str))
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    try:
        parsed = json.loads(str(followers_str))
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    return []

def extract_prediction(row):
    """Convert a single row (Series) into a prediction dict."""
    followers = parse_followers(row.get('followers', '[]'))
    top_3_picks = []
    for item in followers[:3]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            top_3_picks.append({'ticker': str(item[0]), 'score': float(item[1])})
    while len(top_3_picks) < 3:
        top_3_picks.append({'ticker': 'N/A', 'score': 0.0})

    leader = row.get('predicted_leader_etf', 'N/A')
    if pd.isna(leader) or str(leader).strip() in ('', 'N/A'):
        leader = top_3_picks[0]['ticker'] if top_3_picks[0]['ticker'] != 'N/A' else 'N/A'

    metrics = {
        'total_return': row.get('metrics_total_return', 0.0),
        'sharpe_ratio': row.get('metrics_sharpe_ratio', 0.0),
        'max_drawdown': row.get('metrics_max_drawdown', 0.0),
        'win_rate': row.get('metrics_win_rate', 0.0),
        'best_day': row.get('metrics_best_day', 0.0),
    }
    for k in metrics:
        if pd.isna(metrics[k]):
            metrics[k] = 0.0

    return {
        'leader': leader,
        'leader_name': get_etf_display_name(leader),
        'conviction': row.get('causal_confidence', 0.0),
        'top_3_picks': top_3_picks,
        'prediction_date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
        'training_mode': row.get('training_mode', 'fixed'),
        'metrics': metrics,
        'benchmark': config.FI_COMMODITY_BENCHMARK if row.get('universe') == 'fi_commodity' else config.EQUITY_BENCHMARK,
    }

def render_prediction_card(data, title):
    """Render a single prediction card."""
    if data is None:
        st.info(f"No {title} prediction available. Run training with --upload.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
                    border-radius: 16px; padding: 24px; border: 1px solid #d8b4fe; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <h2 style="font-size: 48px; color: #6B21A8;">{data['leader']}</h2>
                    <p>{data['leader_name']}</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 12px; color: #6b7280;">Conviction</p>
                    <p style="font-size: 28px; font-weight: 600; color: #6B21A8;">{data['conviction']*100:.1f}%</p>
                </div>
            </div>
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #d8b4fe;">
                <p><strong>2nd:</strong> {data['top_3_picks'][1]['ticker']} ({data['top_3_picks'][1]['score']*100:.1f}%)</p>
                <p><strong>3rd:</strong> {data['top_3_picks'][2]['ticker']} ({data['top_3_picks'][2]['score']*100:.1f}%)</p>
            </div>
            <div style="margin-top: 16px; display: flex; gap: 8px;">
                <span style="background:#6B21A8; color:white; padding:4px 12px; border-radius:20px;">{data['prediction_date']}</span>
                <span style="background:#f3e8ff; color:#6B21A8; padding:4px 12px; border-radius:20px;">{data['training_mode']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        render_kpi_boxes(data['metrics'])

def main():
    st.set_page_config(page_title="P2 — ETF LINGAM Engine", layout="wide")
    apply_custom_css()
    st.title("📊 P2 — ETF LINGAM Engine")

    with st.sidebar:
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        df = load_predictions()
        if not df.empty:
            st.success(f"Loaded {len(df)} predictions")
            # Show summary of available combos
            summary = df.groupby(['universe', 'training_mode']).size().reset_index(name='count')
            st.write("Available predictions:", summary)
        else:
            st.error("No data loaded. Run training with --upload.")

    if df.empty:
        st.stop()

    tabs = st.tabs(["Fixed Income / Alts", "Equity Sectors"])

    for tab, universe in zip(tabs, ['fi_commodity', 'equity']):
        with tab:
            st.markdown(f"### {universe.replace('_', ' ').title()} Module")
            # Get fixed split row (most recent)
            fixed_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'fixed')]
            fixed_data = None
            if not fixed_rows.empty:
                latest_fixed = fixed_rows.sort_values('date', ascending=False).iloc[0]
                fixed_data = extract_prediction(latest_fixed)

            # Get shrinking window row (most recent)
            shrink_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'shrinking')]
            shrink_data = None
            if not shrink_rows.empty:
                latest_shrink = shrink_rows.sort_values('date', ascending=False).iloc[0]
                shrink_data = extract_prediction(latest_shrink)

            # Display two columns: Fixed Split | Shrinking Window
            col_fixed, col_shrink = st.columns(2)
            with col_fixed:
                st.markdown("#### 🔹 Fixed Split Training")
                render_prediction_card(fixed_data, "fixed split")
            with col_shrink:
                st.markdown("#### 🔸 Shrinking Window (Consensus)")
                render_prediction_card(shrink_data, "shrinking window")

            st.markdown("---")
            # Optional: placeholders for charts/history (can be customized later)
            st.markdown("#### Strategy Performance (example)")
            dates = pd.date_range('2023-01-01', periods=300, freq='D')
            np.random.seed(42)
            strategy_returns = pd.Series(np.random.randn(300)*0.01+0.0003, index=dates)
            benchmark_returns = pd.Series(np.random.randn(300)*0.008+0.0002, index=dates)
            # Use fixed data for comparison if available, else placeholder
            bench = fixed_data['benchmark'] if fixed_data else (shrink_data['benchmark'] if shrink_data else "SPY")
            render_comparison_card("SAMBA Strategy", bench, fixed_data['metrics'] if fixed_data else {})
            render_performance_chart(strategy_returns, benchmark_returns, bench, "Strategy vs Benchmark")

            st.markdown("#### Signal History (placeholder)")
            sample_signals = [
                {'date': '2024-04-10', 'ticker': 'GLD', 'conviction': 0.999, 'actual_return': -0.0018, 'is_hit': False},
                {'date': '2024-04-03', 'ticker': 'TLT', 'conviction': 0.75, 'actual_return': 0.023, 'is_hit': True},
            ]
            render_signal_history_table(sample_signals)

if __name__ == "__main__":
    main()
