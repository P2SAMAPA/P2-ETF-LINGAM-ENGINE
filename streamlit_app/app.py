"""
P2-ETF-LINGAM-Engine Streamlit Dashboard
========================================
Displays fixed-split and consensus predictions.
"""
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import ast
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from streamlit_app.utils import (
    apply_custom_css,
    get_etf_display_name,
    calculate_next_trading_day,
)

# ==============================================================================
# Constants
# ==============================================================================

HF_DATASET_REPO = "P2SAMAPA/p2-etf-lingam-results"
PREDICTIONS_FILE = "predictions.parquet"


# ==============================================================================
# Data loading & parsing
# ==============================================================================

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
    """Convert a row into a prediction dict using existing columns."""
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
        'annualized_return': row.get('metrics_annualized_return', 0.0),
        'sharpe_ratio': row.get('metrics_sharpe_ratio', 0.0),
        'max_drawdown': row.get('metrics_max_drawdown', 0.0),
        'win_rate': row.get('metrics_win_rate', 0.0),
        'best_day': row.get('metrics_best_day', 0.0),
    }
    for k in metrics:
        if pd.isna(metrics[k]):
            metrics[k] = 0.0

    # Use the 'return' field if present and non‑zero; otherwise fallback to annualized test return
    predicted_return = row.get('return', 0.0)
    if pd.isna(predicted_return) or predicted_return == 0.0:
        predicted_return = metrics['annualized_return']

    return {
        'leader': leader,
        'leader_name': get_etf_display_name(leader),
        'predicted_return': predicted_return,
        'top_3_picks': top_3_picks,
        'prediction_date': (
            row['date'].strftime('%Y-%m-%d')
            if hasattr(row['date'], 'strftime')
            else str(row['date'])
        ),
        'training_mode': row.get('training_mode', 'fixed'),
        'metrics': metrics,
        'benchmark': (
            config.FI_COMMODITY_BENCHMARK
            if row.get('universe') == 'fi_commodity'
            else config.EQUITY_BENCHMARK
        ),
    }


# ==============================================================================
# Rendering functions
# ==============================================================================

def render_kpi_boxes(metrics: dict):
    """Render 5 KPI boxes with equal height and centered content."""
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    kpis = [
        (
            col1,
            "Ann. Return",
            f"{metrics.get('annualized_return', 0)*100:.1f}%",
            "#10B981" if metrics.get('annualized_return', 0) >= 0 else "#EF4444",
        ),
        (
            col2,
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            "#6B7280",
        ),
        (
            col3,
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0)*100:.1f}%",
            "#EF4444",
        ),
        (
            col4,
            "Win Rate",
            f"{metrics.get('win_rate', 0)*100:.1f}%",
            "#10B981",
        ),
        (
            col5,
            "Best Day",
            f"{metrics.get('best_day', 0)*100:.1f}%",
            "#10B981",
        ),
    ]
    for col, label, value, color in kpis:
        with col:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 600; color: {color};">{value}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_card(data):
    """Render a single prediction card with hero section and KPI boxes."""
    if data is None:
        st.info("No prediction available. Run training with --upload.")
        return

    next_trading_day = calculate_next_trading_day()

    st.markdown(
        f"""
        <div style="background-color: #F8F9FA; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; color: #6B7280; margin-bottom: 0.25rem;">Signal for {next_trading_day}</div>
            <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                <span style="font-size: 3rem; font-weight: 700;">{data['leader']}</span>
                <span style="font-size: 1.2rem; color: #6B7280;">{data['leader_name']}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #059669; margin-top: 0.5rem;">
                Predicted Return: {data['predicted_return']*100:.2f}%
            </div>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <div><span style="color: #6B7280;">2nd:</span> {data['top_3_picks'][1]['ticker']} ({data['top_3_picks'][1]['score']*100:.1f}%)</div>
                <div><span style="color: #6B7280;">3rd:</span> {data['top_3_picks'][2]['ticker']} ({data['top_3_picks'][2]['score']*100:.1f}%)</div>
            </div>
            <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                <div><span style="color: #6B7280;">🗓️ Next Trading Day:</span> {next_trading_day}</div>
                <div><span style="background-color: #E5E7EB; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem;">{data['training_mode']}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_boxes(data['metrics'])


# ==============================================================================
# Main app
# ==============================================================================

def main():
    st.set_page_config(page_title="P2 — ETF LINGAM Engine", layout="wide")
    apply_custom_css()

    st.title("🔮 P2 — ETF LINGAM Engine")

    with st.sidebar:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    df = load_predictions()
    if df.empty:
        st.error("No prediction data available. Please run training with --upload.")
        st.stop()

    tabs = st.tabs(["Fixed Income / Alts", "Equity Sectors"])
    for tab, universe in zip(tabs, ['fi_commodity', 'equity']):
        with tab:
            st.markdown(f"### {universe.replace('_', ' ').title()} Module")

            fixed_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'fixed')]
            fixed_data = None
            if not fixed_rows.empty:
                latest_fixed = fixed_rows.sort_values('date', ascending=False).iloc[0]
                fixed_data = extract_prediction(latest_fixed)

            shrink_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'shrinking')]
            shrink_data = None
            if not shrink_rows.empty:
                latest_shrink = shrink_rows.sort_values('date', ascending=False).iloc[0]
                shrink_data = extract_prediction(latest_shrink)

            col_fixed, col_shrink = st.columns(2)
            with col_fixed:
                st.markdown("#### Fixed Split Training")
                render_prediction_card(fixed_data)
            with col_shrink:
                st.markdown("#### Shrinking Window (Consensus)")
                render_prediction_card(shrink_data)


if __name__ == "__main__":
    main()
