"""
Streamlit Utility Functions
===========================
Helper functions for the Streamlit app.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config


def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=config.ENGINE_NAME,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    /* Main layout */
    .stApp {
        background-color: #f9fafb;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1f2937;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
    }

    /* Cards */
    .stCard {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e5e7eb;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
    }

    /* Dividers */
    hr {
        margin: 24px 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }

    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #f3e8ff;
        color: #6B21A8;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(universe: str = "fi_commodity"):
    """
    Render the page header.

    Args:
        universe: Current universe ('fi_commodity' or 'equity')
    """
    display_name = config.ETF_UNIVERSE[universe]['display_name']

    st.title(f"📊 P2 — ETF LINGAM Engine")
    st.markdown(f"**{display_name} Module** | Causal discovery-driven predictions")


def render_tab_bar() -> str:
    """
    Render the tab navigation bar.

    Returns:
        Selected tab name
    """
    tab1_name = config.ETF_UNIVERSE['fi_commodity']['tab_name']
    tab2_name = config.ETF_UNIVERSE['equity']['tab_name']

    tabs = st.tabs([tab1_name, tab2_name])

    return tabs


def get_etf_display_name(ticker: str) -> str:
    """
    Get the full display name for an ETF ticker.

    Args:
        ticker: ETF ticker symbol

    Returns:
        Full ETF name
    """
    return config.ETF_METADATA.get(ticker, {}).get('name', ticker)


def format_return(value: float, as_pct: bool = True) -> str:
    """
    Format a return value for display.

    Args:
        value: Return value
        as_pct: Whether to format as percentage

    Returns:
        Formatted string
    """
    if as_pct:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value * 100:.2f}%"
    else:
        return f"{value:.4f}"


def format_date(date_str: str) -> str:
    """
    Format a date string for display.

    Args:
        date_str: Date string

    Returns:
        Formatted date
    """
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        return date_str


def calculate_next_trading_day() -> str:
    """
    Calculate the next US trading day.

    Returns:
        Date string for next trading day
    """
    today = datetime.now()
    next_day = today + timedelta(days=1)

    # Skip weekends
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)

    return next_day.strftime('%Y-%m-%d')


def render_info_box(message: str, box_type: str = "info"):
    """
    Render an info box.

    Args:
        message: Message to display
        box_type: Type of box ('info', 'warning', 'error', 'success')
    """
    colors = {
        'info': ('#3B82F6', '#EFF6FF'),
        'warning': ('#F59E0B', '#FFFBEB'),
        'error': ('#EF4444', '#FEF2F2'),
        'success': ('#10B981', '#ECFDF5')
    }

    color, bg = colors.get(box_type, colors['info'])

    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 4px solid {color};
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
    ">
        {message}
    </div>
    """, unsafe_allow_html=True)


def create_sample_data(universe: str) -> Dict:
    """
    Create sample prediction data for demo.

    Args:
        universe: Universe name

    Returns:
        Dictionary with sample prediction data
    """
    if universe == 'fi_commodity':
        leader = 'GLD'
        top_picks = [
            {'ticker': 'GLD', 'score': 0.85},
            {'ticker': 'TLT', 'score': 0.12},
            {'ticker': 'SLV', 'score': 0.03}
        ]
        benchmark = 'AGG'
    else:
        leader = 'QQQ'
        top_picks = [
            {'ticker': 'QQQ', 'score': 0.72},
            {'ticker': 'XLK', 'score': 0.18},
            {'ticker': 'XLE', 'score': 0.10}
        ]
        benchmark = 'SPY'

    return {
        'leader': leader,
        'leader_name': get_etf_display_name(leader),
        'conviction': 0.999,
        'top_3_picks': top_picks,
        'prediction_date': calculate_next_trading_day(),
        'training_mode': 'Fixed Split (70/15/15)',
        'window_start': '2008-01-01',
        'window_end': datetime.now().strftime('%Y-%m-%d'),
        'benchmark': benchmark,
        'metrics': {
            'total_return': 0.117,
            'sharpe_ratio': 2.79,
            'max_drawdown': -0.107,
            'win_rate': 0.55,
            'best_day': 0.045
        }
    }