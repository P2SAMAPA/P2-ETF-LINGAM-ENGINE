"""
Hero Card Component
===================
Displays the main prediction with conviction score.
"""

import streamlit as st
import datetime


def render_hero_card(
    ticker: str,
    ticker_name: str,
    conviction: float,
    top_3_picks: list,
    prediction_date: str,
    training_mode: str,
    window_start: str,
    window_end: str,
    primary_color: str = "#6B21A8"
):
    """
    Render the hero card component.

    Args:
        ticker: Leader ETF ticker
        ticker_name: Full name of ticker
        conviction: Confidence score (0-1)
        top_3_picks: List of top 3 picks with scores
        prediction_date: Date for prediction
        training_mode: 'fixed' or 'shrinking'
        window_start: Start date of training window
        window_end: End date of training window
        primary_color: Primary accent color
    """
    conviction_pct = conviction * 100 if conviction <= 1 else conviction

    # Hero card container
    st.markdown(f"""
    <style>
    .hero-card {{
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        border-radius: 16px;
        padding: 32px;
        border: 1px solid #d8b4fe;
        box-shadow: 0 4px 20px rgba(107, 33, 168, 0.1);
        margin-bottom: 24px;
    }}
    .hero-ticker {{
        font-size: 64px;
        font-weight: 700;
        color: {primary_color};
        margin: 0;
        line-height: 1;
    }}
    .hero-name {{
        font-size: 18px;
        color: #6b7280;
        margin-top: 8px;
    }}
    .hero-conviction {{
        font-size: 36px;
        font-weight: 600;
        color: {primary_color};
    }}
    .hero-conviction-label {{
        font-size: 14px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .secondary-picks {{
        margin-top: 24px;
        padding-top: 24px;
        border-top: 1px solid #d8b4fe;
    }}
    .pick-item {{
        font-size: 16px;
        color: #4b5563;
        margin: 8px 0;
    }}
    .pick-ticker {{
        font-weight: 600;
        color: {primary_color};
    }}
    .meta-tags {{
        margin-top: 24px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .meta-tag {{
        background: {primary_color};
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }}
    .meta-tag.secondary {{
        background: #f3e8ff;
        color: {primary_color};
    }}
    </style>

    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="hero-ticker">{ticker}</h1>
                <p class="hero-name">{ticker_name}</p>
            </div>
            <div style="text-align: right;">
                <p class="hero-conviction-label">Conviction</p>
                <p class="hero-conviction">{conviction_pct:.1f}%</p>
            </div>
        </div>

        <div class="secondary-picks">
            <p style="font-size: 14px; color: #6b7280; margin-bottom: 8px;">2nd & 3rd Picks</p>
    """)

    # Add 2nd and 3rd picks
    for i, pick in enumerate(top_3_picks[1:3], start=2):
        score_pct = pick.get('score', 0) * 100 if pick.get('score', 0) <= 1 else pick.get('score', 0)
        st.markdown(f"""
            <p class="pick-item">{i}nd: <span class="pick-ticker">{pick.get('ticker', 'N/A')}</span> {score_pct:.1f}%</p>
        """)

    st.markdown("""
        </div>

        <div class="meta-tags">
            <span class="meta-tag">Signal for: """ + prediction_date + """</span>
            <span class="meta-tag secondary">Mode: """ + training_mode + """</span>
        </div>
    </div>
    """)


def render_signal_card(
    date: str,
    ticker: str,
    conviction: float,
    actual_return: float = None,
    is_hit: bool = None
):
    """
    Render a signal history card.

    Args:
        date: Signal date
        ticker: Selected ticker
        conviction: Confidence score
        actual_return: Actual return if available
        is_hit: Whether signal was correct
    """
    conviction_pct = conviction * 100 if conviction <= 1 else conviction
    return_str = f"{actual_return * 100:.2f}%" if actual_return is not None else "N/A"
    hit_indicator = "✓" if is_hit else "✗"
    hit_color = "#10B981" if is_hit else "#EF4444" if is_hit is not None else "#6B7280"

    st.markdown(f"""
    <style>
    .signal-row {{
        display: flex;
        align-items: center;
        padding: 12px;
        background: white;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #e5e7eb;
    }}
    .signal-date {{
        flex: 1;
        font-weight: 500;
        color: #374151;
    }}
    .signal-ticker {{
        flex: 1;
        font-weight: 600;
        color: #6B21A8;
    }}
    .signal-conviction {{
        flex: 1;
        color: #6b7280;
    }}
    .signal-return {{
        flex: 1;
        color: #374151;
    }}
    .signal-hit {{
        width: 40px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: """ + hit_color + """;
    }}
    </style>

    <div class="signal-row">
        <span class="signal-date">{date}</span>
        <span class="signal-ticker">{ticker}</span>
        <span class="signal-conviction">{conviction_pct:.1f}%</span>
        <span class="signal-return">{return_str}</span>
        <span class="signal-hit">{hit_indicator}</span>
    </div>
    """)


def render_comparison_card(
    strategy_name: str,
    benchmark_name: str,
    metrics: dict,
    primary_color: str = "#6B21A8"
):
    """
    Render a strategy vs benchmark comparison card.

    Args:
        strategy_name: Name of strategy
        benchmark_name: Name of benchmark
        metrics: Dictionary with comparison metrics
        primary_color: Accent color
    """
    # Calculate colors based on values
    return_color = "#10B981" if metrics.get('return', 0) > 0 else "#EF4444"
    sharpe_color = "#10B981" if metrics.get('sharpe', 0) > 1 else "#EF4444" if metrics.get('sharpe', 0) < 0 else "#6B7280"
    dd_color = "#EF4444"  # Always red for drawdown

    st.markdown(f"""
    <style>
    .comparison-card {{
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }}
    .comparison-header {{
        font-size: 18px;
        font-weight: 600;
        color: {primary_color};
        margin-bottom: 16px;
    }}
    .comparison-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
    }}
    .metric-box {{
        text-align: center;
        padding: 12px;
        background: #f9fafb;
        border-radius: 8px;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 600;
    }}
    .metric-label {{
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        margin-top: 4px;
    }}
    </style>

    <div class="comparison-card">
        <div class="comparison-header">
            {strategy_name} vs {benchmark_name}
        </div>
        <div class="comparison-grid">
            <div class="metric-box">
                <div class="metric-value" style="color: {return_color}">
                    {metrics.get('return', 0)*100:.1f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: {sharpe_color}">
                    {metrics.get('sharpe', 0):.2f}
                </div>
                <div class="metric-label">Sharpe</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: {dd_color}">
                    {metrics.get('max_dd', 0)*100:.1f}%
                </div>
                <div class="metric-label">PEAK→TROUGH</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">
                    {metrics.get('win_rate', 0)*100:.0f}%
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: {return_color}">
                    {metrics.get('best_day', 0)*100:.1f}%
                </div>
                <div class="metric-label">Best Day</div>
            </div>
        </div>
    </div>
    """)
