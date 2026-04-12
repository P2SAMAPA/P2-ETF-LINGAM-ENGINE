"""
P2-ETF-LINGAM-Engine Streamlit Dashboard
========================================
Displays fixed-split and consensus predictions using existing metrics.
No duplicate metrics, no "SAMBA" label.
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
from streamlit_app.utils import apply_custom_css, get_etf_display_name

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

    # Metrics directly from flat columns (these are the OOS metrics)
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

# ==============================================================================
# Rendering functions
# ==============================================================================
def render_kpi_boxes(metrics: dict):
    """Render 5 KPI boxes using metrics from the prediction."""
    cols = st.columns(5)
    kpis = [
        ("Total Return", f"{metrics.get('total_return', 0)*100:.1f}%", "#10B981" if metrics.get('total_return',0)>=0 else "#EF4444"),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}", "#6B7280"),
        ("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%", "#EF4444"),
        ("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%", "#10B981"),
        ("Best Day", f"{metrics.get('best_day', 0)*100:.1f}%", "#10B981"),
    ]
    for col, (label, value, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 28px; font-weight: 700; color: {color};">{value}</div>
                <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; margin-top: 8px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def render_signal_history_table(signals):
    """Placeholder for signal history (can be replaced later)."""
    if not signals:
        st.info("Historical signals not yet stored. Extend training script to save `signal_history`.")
        return
    df = pd.DataFrame(signals)
    st.dataframe(df, use_container_width=True)

def render_prediction_card(data):
    """Render a single prediction card with leader, conviction, top picks, and KPI boxes."""
    if data is None:
        st.info("No prediction available. Run training with --upload.")
        return

    # Hero section: leader, conviction, top picks, date, mode
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

    # KPI boxes (metrics) directly below the hero card
    render_kpi_boxes(data['metrics'])

# ==============================================================================
# Main app
# ==============================================================================
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

            # Fixed split
            fixed_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'fixed')]
            fixed_data = None
            if not fixed_rows.empty:
                latest_fixed = fixed_rows.sort_values('date', ascending=False).iloc[0]
                fixed_data = extract_prediction(latest_fixed)

            # Shrinking window
            shrink_rows = df[(df['universe'] == universe) & (df['training_mode'] == 'shrinking')]
            shrink_data = None
            if not shrink_rows.empty:
                latest_shrink = shrink_rows.sort_values('date', ascending=False).iloc[0]
                shrink_data = extract_prediction(latest_shrink)

            col_fixed, col_shrink = st.columns(2)
            with col_fixed:
                st.markdown("#### 🔹 Fixed Split Training")
                render_prediction_card(fixed_data)
            with col_shrink:
                st.markdown("#### 🔸 Shrinking Window (Consensus)")
                render_prediction_card(shrink_data)

            st.markdown("---")
            st.markdown("#### Signal History")
            # Placeholder – replace with real data when available
            render_signal_history_table([])

if __name__ == "__main__":
    main()
