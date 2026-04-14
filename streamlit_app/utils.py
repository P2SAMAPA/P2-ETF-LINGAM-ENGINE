"""
Streamlit Utility Functions
===========================
Helper functions for the Streamlit app.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config

# ==============================================================================
# Local ETF metadata (fallback when config.ETF_METADATA is missing)
# ==============================================================================
LOCAL_ETF_METADATA = {
    # FI/Commodity assets
    'GLD': {'name': 'SPDR Gold Trust', 'sector': 'Commodity'},
    'TLT': {'name': 'iShares 20+ Year Treasury Bond ETF', 'sector': 'Fixed Income'},
    'VCIT': {'name': 'Vanguard Intermediate-Term Corporate Bond ETF', 'sector': 'Fixed Income'},
    'LQD': {'name': 'iShares iBoxx $ Investment Grade Corporate Bond ETF', 'sector': 'Fixed Income'},
    'HYG': {'name': 'iShares iBoxx $ High Yield Corporate Bond ETF', 'sector': 'Fixed Income'},
    'VNQ': {'name': 'Vanguard Real Estate ETF', 'sector': 'Real Estate'},
    'SLV': {'name': 'iShares Silver Trust', 'sector': 'Commodity'},
    'AGG': {'name': 'iShares Core U.S. Aggregate Bond ETF', 'sector': 'Fixed Income'},
    # Equity assets
    'QQQ': {'name': 'Invesco QQQ Trust', 'sector': 'Technology'},
    'XLK': {'name': 'Technology Select Sector SPDR Fund', 'sector': 'Technology'},
    'XLF': {'name': 'Financial Select Sector SPDR Fund', 'sector': 'Financials'},
    'XLE': {'name': 'Energy Select Sector SPDR Fund', 'sector': 'Energy'},
    'XLV': {'name': 'Health Care Select Sector SPDR Fund', 'sector': 'Healthcare'},
    'XLI': {'name': 'Industrial Select Sector SPDR Fund', 'sector': 'Industrials'},
    'XLY': {'name': 'Consumer Discretionary Select Sector SPDR Fund', 'sector': 'Consumer Discretionary'},
    'XLP': {'name': 'Consumer Staples Select Sector SPDR Fund', 'sector': 'Consumer Staples'},
    'XLU': {'name': 'Utilities Select Sector SPDR Fund', 'sector': 'Utilities'},
    'XME': {'name': 'SPDR S&P Metals & Mining ETF', 'sector': 'Materials'},
    'IWM': {'name': 'iShares Russell 2000 ETF', 'sector': 'Small Cap'},
    'XLB': {'name': 'Materials Select Sector SPDR Fund', 'sector': 'Materials'},
    'XLRE': {'name': 'Real Estate Select Sector SPDR Fund', 'sector': 'Real Estate'},
    'GDX': {'name': 'VanEck Gold Miners ETF', 'sector': 'Commodity'},
    'SPY': {'name': 'SPDR S&P 500 ETF Trust', 'sector': 'Equity'},
    # Fallback
    'N/A': {'name': 'Not Available', 'sector': 'Unknown'},
}

# Try to import pandas trading calendar support
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    HAS_TRADING_CALENDAR = True
except ImportError:
    HAS_TRADING_CALENDAR = False

# Try to import pytz for timezone support
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False


def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=config.ENGINE_NAME if hasattr(config, 'ENGINE_NAME') else "P2-ETF-LINGAM-Engine",
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
    st.title(f"📊 P2 — ETF LINGAM Engine")
    # Display name from config if available, otherwise generic
    if hasattr(config, 'ETF_UNIVERSE') and universe in config.ETF_UNIVERSE:
        display_name = config.ETF_UNIVERSE[universe].get('display_name', universe.title())
    else:
        display_name = "Fixed Income / Commodity" if universe == 'fi_commodity' else "Equity Sectors"
    st.markdown(f"**{display_name} Module** | Causal discovery-driven predictions")


def render_tab_bar():
    """
    Render the tab navigation bar.

    Returns:
        Selected tab object
    """
    # Try to get tab names from config, fallback to defaults
    if hasattr(config, 'ETF_UNIVERSE'):
        fi_tab = config.ETF_UNIVERSE.get('fi_commodity', {}).get('tab_name', 'Fixed Income / Alts')
        eq_tab = config.ETF_UNIVERSE.get('equity', {}).get('tab_name', 'Equity Sectors')
    else:
        fi_tab = "Fixed Income / Alts"
        eq_tab = "Equity Sectors"

    tabs = st.tabs([fi_tab, eq_tab])
    return tabs


def get_etf_display_name(ticker: str) -> str:
    """
    Get the full display name for an ETF ticker.

    Args:
        ticker: ETF ticker symbol

    Returns:
        Full ETF name
    """
    # First try config.ETF_METADATA if it exists
    if hasattr(config, 'ETF_METADATA') and ticker in config.ETF_METADATA:
        return config.ETF_METADATA[ticker].get('name', ticker)
    # Otherwise use local fallback
    return LOCAL_ETF_METADATA.get(ticker, {}).get('name', ticker)


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


def calculate_next_trading_day(as_of_date=None):
    """
    Calculate the correct US stock market trading day for predictions.
    
    Logic:
    - If before market open (9:30 AM ET): Show today (if trading day)
    - If during market hours: Show today
    - If after market close (4:00 PM ET): Show next trading day
    - If weekend/holiday: Show next trading day
    
    Args:
        as_of_date: Optional datetime to calculate from. Defaults to current time.
    
    Returns:
        Date string for the appropriate trading day in YYYY-MM-DD format
    """
    # Get current time in Eastern Time if pytz is available
    if HAS_PYTZ:
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        current_date = now_ny.date()
        current_time = now_ny.time()
    else:
        # Fallback to system time (assume Eastern Time for simplicity)
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
    
    # Market hours: 9:30 AM to 4:00 PM ET
    market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
    market_close = datetime.strptime("16:00:00", "%H:%M:%S").time()
    
    # Determine if we're before market open
    is_before_market_open = current_time < market_open
    is_after_market_close = current_time > market_close
    
    # Use enhanced trading calendar if available
    if HAS_TRADING_CALENDAR:
        # Normalize to date only for pandas
        base_date = pd.Timestamp(current_date)
        
        # NYSE holiday calendar (includes major US market holidays)
        nyse_calendar = USFederalHolidayCalendar()
        trading_day = CustomBusinessDay(calendar=nyse_calendar)
        
        # Check if today is a trading day
        is_trading_day = base_date in pd.date_range(start=base_date, periods=1, freq=trading_day)
        
        # Apply market hours logic
        if is_before_market_open and is_trading_day:
            # Before market open on a trading day -> show today
            result_date = base_date
        elif is_after_market_close or not is_trading_day:
            # After market close OR not a trading day -> next trading day
            result_date = base_date + trading_day
        else:
            # During market hours -> show today
            result_date = base_date
        
        return result_date.strftime('%Y-%m-%d')
    else:
        # Fallback to simple weekend skipping
        result_date = datetime(current_date.year, current_date.month, current_date.day)
        
        if is_before_market_open:
            # Before market open - check if today is weekday
            if result_date.weekday() < 5:
                return result_date.strftime('%Y-%m-%d')
            else:
                # Weekend - find next weekday
                while result_date.weekday() >= 5:
                    result_date += timedelta(days=1)
                return result_date.strftime('%Y-%m-%d')
        else:
            # After market close - get tomorrow and skip weekends
            result_date += timedelta(days=1)
            while result_date.weekday() >= 5:
                result_date += timedelta(days=1)
            return result_date.strftime('%Y-%m-%d')


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
        # Try to get benchmark from config if available
        if hasattr(config, 'FI_COMMODITY_BENCHMARK'):
            benchmark = config.FI_COMMODITY_BENCHMARK
    else:
        leader = 'QQQ'
        top_picks = [
            {'ticker': 'QQQ', 'score': 0.72},
            {'ticker': 'XLK', 'score': 0.18},
            {'ticker': 'XLE', 'score': 0.10}
        ]
        benchmark = 'SPY'
        if hasattr(config, 'EQUITY_BENCHMARK'):
            benchmark = config.EQUITY_BENCHMARK

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
