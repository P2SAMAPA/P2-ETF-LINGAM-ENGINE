"""
FI/Commodity Module - Signal Generator
======================================
Generates trading signals for FI/Commodity ETFs based on causal analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import config


class FISignalGenerator:
    """
    Generates trading signals for FI/Commodity ETF universe.
    """

    def __init__(self):
        """Initialize FI/Commodity signal generator."""
        self.assets = config.FI_COMMODITY_ASSETS
        self.benchmark = config.FI_COMMODITY_BENCHMARK

    def generate_signals(
        self,
        leader_report: Dict,
        current_returns: pd.DataFrame,
        prediction_date: str
    ) -> Dict:
        """
        Generate trading signals based on leader report.

        Args:
            leader_report: Report from leader identifier
            current_returns: Current period returns
            prediction_date: Date for prediction

        Returns:
            Dictionary with signals and metadata
        """
        signals = {
            'date': prediction_date,
            'universe': 'fi_commodity',
            'primary_signal': None,
            'secondary_signals': [],
            'all_signals': [],
            'confidence': 0.0,
        }

        # Get top picks
        top_3 = leader_report.get('top_3_picks', [])
        if not top_3:
            return signals

        # Primary signal
        primary = top_3[0]
        signals['primary_signal'] = {
            'ticker': primary['ticker'],
            'conviction': leader_report.get('consensus_conviction', 0),
            'score': primary.get('score', 0),
            'ann_return': primary.get('ann_return', 0),
            'sharpe': primary.get('sharpe', 0),
            'max_dd': primary.get('max_dd', 0),
        }
        signals['confidence'] = leader_report.get('consensus_conviction', 0) * 100

        # Secondary signals
        for pick in top_3[1:3]:
            signals['secondary_signals'].append({
                'rank': len(signals['secondary_signals']) + 2,
                'ticker': pick['ticker'],
                'score': pick.get('score', 0),
            })

        # All signals with rankings
        signals['all_signals'] = [
            {
                'rank': i + 1,
                'ticker': p['ticker'],
                'score': p.get('score', 0),
                'conviction': p.get('score', 0) / top_3[0]['score'] if top_3 and top_3[0]['score'] > 0 else 0
            }
            for i, p in enumerate(top_3)
        ]

        return signals

    def backtest_signal(
        self,
        signals: Dict,
        historical_returns: pd.DataFrame
    ) -> List[Dict]:
        """
        Backtest a signal against historical data.

        Args:
            signals: Signal dictionary
            historical_returns: Historical returns data

        Returns:
            List of backtest results
        """
        results = []
        ticker = signals['primary_signal']['ticker'] if signals['primary_signal'] else None

        if not ticker or ticker not in historical_returns.columns:
            return results

        returns = historical_returns[ticker].dropna()

        for i in range(len(returns)):
            date = returns.index[i]
            ret = returns.iloc[i]

            # Check if this was a signal date (weekly)
            is_signal_date = (i % 5 == 0) if i >= 5 else False

            results.append({
                'date': date,
                'ticker': ticker,
                'return': ret,
                'is_signal': is_signal_date,
                'hit': ret > 0,  # Simple hit definition
            })

        return results

    def calculate_signal_metrics(
        self,
        backtest_results: List[Dict]
    ) -> Dict:
        """
        Calculate metrics for signal performance.

        Args:
            backtest_results: Results from backtest

        Returns:
            Dictionary of signal metrics
        """
        if not backtest_results:
            return {}

        signal_results = [r for r in backtest_results if r.get('is_signal', False)]

        if not signal_results:
            return {'n_signals': 0}

        returns = [r['return'] for r in signal_results]

        total_return = (1 + np.array(returns)).prod() - 1
        hit_rate = sum(1 for r in signal_results if r['hit']) / len(signal_results)
        avg_return = np.mean(returns)

        return {
            'n_signals': len(signal_results),
            'total_return': total_return,
            'hit_rate': hit_rate,
            'avg_return': avg_return,
            'avg_return_pct': avg_return * 100,
        }
