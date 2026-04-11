"""
Metrics Display Component
=========================
Displays performance metrics in various formats.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List


def render_kpi_boxes(metrics: Dict, columns: int = 5):
    """
    Render KPI metric boxes.

    Args:
        metrics: Dictionary of metric name -> value
        columns: Number of columns for grid
    """
    cols = st.columns(columns)

    metric_labels = {
        'total_return': ('Total Return', '%', False),
        'sharpe_ratio': ('Sharpe Ratio', '', False),
        'max_drawdown': ('PEAK→TROUGH', '%', True),
        'win_rate': ('Win Rate', '%', False),
        'best_day': ('Best Day', '%', False),
    }

    for i, (key, (label, suffix, is_negative)) in enumerate(metric_labels.items()):
        if key not in metrics:
            continue

        value = metrics[key]
        if suffix == '%':
            display_value = f"{value * 100:.1f}%"
        else:
            display_value = f"{value:.2f}"

        # Determine color
        if key == 'max_drawdown':
            color = "#EF4444"  # Always red for drawdown
        elif value > 0:
            color = "#10B981"  # Green for positive
        elif value < 0:
            color = "#EF4444"  # Red for negative
        else:
            color = "#6B7280"  # Gray for neutral

        with cols[i % columns]:
            st.markdown(f"""
            <style>
            .kpi-box {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .kpi-value {{
                font-size: 28px;
                font-weight: 700;
                color: {color};
            }}
            .kpi-label {{
                font-size: 12px;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 8px;
            }}
            </style>
            <div class="kpi-box">
                <div class="kpi-value">{display_value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """)


def render_performance_chart(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    benchmark_name: str = "Benchmark",
    title: str = "Cumulative Performance"
):
    """
    Render cumulative performance chart.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series (optional)
        benchmark_name: Name of benchmark
        title: Chart title
    """
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    df = pd.DataFrame({
        'Date': strategy_cumulative.index,
        'Strategy': strategy_cumulative.values
    })

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        df['Benchmark'] = benchmark_cumulative.values

    # Create Plotly figure
    fig = go.Figure()

    # Strategy line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Strategy'],
        mode='lines',
        name='SAMBA',
        line=dict(color='#6B21A8', width=2)
    ))

    # Benchmark line
    if benchmark_returns is not None:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Benchmark'],
            mode='lines',
            name=benchmark_name,
            line=dict(color='#9CA3AF', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        template='plotly_white',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_signal_history_table(signals: List[Dict]):
    """
    Render signal history table.

    Args:
        signals: List of signal dictionaries
    """
    if not signals:
        st.info("No signal history available")
        return

    # Create DataFrame
    rows = []
    for signal in signals:
        rows.append({
            'Date': signal.get('date', 'N/A'),
            'Pick': signal.get('ticker', 'N/A'),
            'Conviction': f"{signal.get('conviction', 0) * 100:.1f}%",
            'Actual Return': f"{signal.get('actual_return', 0) * 100:.2f}%",
            'Hit': '✓' if signal.get('is_hit') else '✗'
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Calculate hit rate
    if signals:
        hits = sum(1 for s in signals if s.get('is_hit'))
        total = len(signals)
        st.caption(f"Hit rate: {hits}/{total} ({hits/total*100:.1f}%)" if total > 0 else "")


def render_causal_dag_visualization(edges: List, title: str = "Causal DAG"):
    """
    Render causal DAG visualization.

    Args:
        edges: List of (source, target, strength) tuples
        title: Chart title
    """
    if not edges:
        st.info("No causal edges to display")
        return

    # Create edge data
    edge_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Strength'])
    edge_df['Strength'] = edge_df['Strength'].abs()

    # Create sankey-like diagram
    all_nodes = list(set(edge_df['Source'].tolist() + edge_df['Target'].tolist()))
    node_dict = {node: i for i, node in enumerate(all_nodes)}

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_nodes,
            color='#6B21A8'
        ),
        link=dict(
            source=[node_dict[s] for s in edge_df['Source']],
            target=[node_dict[t] for t in edge_df['Target']],
            value=edge_df['Strength'].tolist(),
            color='rgba(107, 33, 168, 0.3)'
        )
    )])

    fig.update_layout(
        title=title,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metric_comparison(
    strategy_metrics: Dict,
    benchmark_metrics: Dict = None,
    metric_names: List[str] = None
):
    """
    Render comparison of strategy vs benchmark metrics.

    Args:
        strategy_metrics: Strategy metrics dictionary
        benchmark_metrics: Benchmark metrics dictionary
        metric_names: List of metric names to display
    """
    if metric_names is None:
        metric_names = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

    labels = {
        'total_return': 'Total Return',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'win_rate': 'Win Rate'
    }

    # Create comparison data
    comparison_data = []
    for metric in metric_names:
        if metric in strategy_metrics:
            comparison_data.append({
                'Metric': labels.get(metric, metric),
                'Strategy': strategy_metrics[metric],
                'Benchmark': benchmark_metrics.get(metric) if benchmark_metrics else None
            })

    df = pd.DataFrame(comparison_data)

    # Melt for grouped bar chart
    df_melted = df.melt(id_vars='Metric', var_name='Type', value_name='Value')

    # Format values
    df_melted['Value'] = df_melted['Value'].apply(
        lambda x: x * 100 if isinstance(x, (int, float)) and abs(x) < 10 else x
    )

    fig = px.bar(
        df_melted,
        x='Metric',
        y='Value',
        color='Type',
        barmode='group',
        color_discrete_map={
            'Strategy': '#6B21A8',
            'Benchmark': '#9CA3AF'
        }
    )

    fig.update_layout(
        yaxis_title='Value',
        height=300,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)