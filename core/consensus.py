"""
Consensus Scoring Module
========================
Weighted consensus calculation for shrinking window ETF selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from core.metrics import (
    calculate_all_metrics,
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)
import config


class ConsensusScorer:
    """
    Calculates weighted consensus scores across shrinking windows.

    Scoring Weights:
    - 60%: Annualized Return (negatives get zero weight)
    - 20%: Sharpe Ratio
    - 20%: Max Drawdown (inverted - lower is better)
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        exclude_negative_returns: bool = True
    ):
        """
        Initialize consensus scorer.

        Args:
            weights: Dictionary with metric weights
            exclude_negative_returns: If True, negative return windows get 0 weight
        """
        self.weights = weights or config.CONSENSUS_WEIGHTS
        self.exclude_negative_returns = exclude_negative_returns

    def calculate_window_score(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate consensus score for a single window.

        Args:
            returns: Returns series for the window
            periods_per_year: Trading periods per year

        Returns:
            Window score (0-1 scale)
        """
        # Calculate metrics
        metrics = calculate_all_metrics(returns, periods_per_year)
        ann_return = metrics['annualized_return']
        sharpe = metrics['sharpe_ratio']
        max_dd = metrics['max_drawdown']

        # Zero weight for negative returns
        if self.exclude_negative_returns and ann_return <= 0:
            return 0.0

        # Normalize scores
        scores = {}

        # Annualized Return (60%): Normalize 0-50% -> 0-1
        scores['annualized_return'] = min(max(ann_return / 0.5, 0), 1)

        # Sharpe Ratio (20%): Normalize -2 to 4 -> 0 to 1
        scores['sharpe_ratio'] = min(max((sharpe + 2) / 6, 0), 1)

        # Max Drawdown (20%): Invert, normalize 0 to -50% -> 1 to 0
        abs_dd = abs(max_dd)
        scores['max_drawdown'] = 1 - min(abs_dd / 0.5, 1)

        # Calculate weighted score
        weighted_score = (
            scores['annualized_return'] * self.weights['annualized_return'] +
            scores['sharpe_ratio'] * self.weights['sharpe_ratio'] +
            scores['max_drawdown'] * self.weights['max_drawdown']
        )

        return weighted_score

    def calculate_consensus_scores(
        self,
        window_results: List[Dict],
        assets: List[str]
    ) -> pd.DataFrame:
        """
        Calculate consensus scores for all assets across all windows.

        Args:
            window_results: List of window analysis results. Each dict must have
                           'window_start' and 'returns' keys.
            assets: List of asset tickers

        Returns:
            DataFrame with consensus scores per asset
        """
        results = []

        for window in window_results:
            # Skip windows that don't have the required 'returns' key
            if 'returns' not in window:
                print(f"Warning: Window {window.get('window_start', 'unknown')} missing 'returns' key. Skipping.")
                continue

            window_start = window['window_start']
            window_returns = window['returns']

            for ticker in assets:
                if ticker not in window_returns.columns:
                    continue

                ticker_returns = window_returns[ticker].dropna()
                if len(ticker_returns) < 20:  # Minimum samples
                    continue

                score = self.calculate_window_score(ticker_returns)

                results.append({
                    'window_start': window_start,
                    'ticker': ticker,
                    'score': score,
                    'annualized_return': calculate_annualized_return(ticker_returns),
                    'sharpe_ratio': calculate_sharpe_ratio(ticker_returns),
                    'max_drawdown': calculate_max_drawdown(ticker_returns),
                })

        return pd.DataFrame(results)

    def get_final_leader(
        self,
        consensus_df: pd.DataFrame,
        min_windows: int = 5
    ) -> Tuple[str, float, List[Dict]]:
        """
        Determine final leader from consensus scores.

        Args:
            consensus_df: DataFrame from calculate_consensus_scores
            min_windows: Minimum windows for valid consensus

        Returns:
            Tuple of (leader_ticker, conviction, top_3_picks)
        """
        if consensus_df.empty:
            return "", 0.0, []

        # Filter to positive scores only
        positive_df = consensus_df[consensus_df['score'] > 0]

        if positive_df.empty:
            return "", 0.0, []

        # Group by ticker
        ticker_stats = positive_df.groupby('ticker').agg({
            'score': ['mean', 'sum', 'count'],
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).reset_index()

        ticker_stats.columns = [
            'ticker', 'avg_score', 'total_score', 'n_windows',
            'avg_ann_return', 'avg_sharpe', 'avg_max_dd'
        ]

        # Filter by minimum windows
        ticker_stats = ticker_stats[ticker_stats['n_windows'] >= min_windows]

        if ticker_stats.empty:
            return "", 0.0, []

        # Rank by total score
        ticker_stats = ticker_stats.sort_values('total_score', ascending=False)

        # Get leader
        leader_row = ticker_stats.iloc[0]
        leader = leader_row['ticker']

        # Calculate conviction (leader score / total score)
        total_all = ticker_stats['total_score'].sum()
        conviction = leader_row['total_score'] / total_all if total_all > 0 else 0.0

        # Get top 3 picks
        top_3 = []
        for i, row in ticker_stats.head(3).iterrows():
            top_3.append({
                'ticker': row['ticker'],
                'score': row['avg_score'],
                'ann_return': row['avg_ann_return'],
                'sharpe': row['avg_sharpe'],
                'max_dd': row['avg_max_dd']
            })

        return leader, conviction, top_3

    def generate_shrinking_window_results(
        self,
        returns: pd.DataFrame,
        assets: List[str],
        start_years: List[int],
        end_date: str
    ) -> List[Dict]:
        """
        Generate shrinking window results for all windows.

        Args:
            returns: DataFrame with return data
            assets: List of asset tickers
            start_years: List of window start years
            end_date: End date for all windows

        Returns:
            List of window results
        """
        window_results = []

        for start_year in start_years:
            start_date = f"{start_year}-01-01"

            # Filter data for this window
            window_returns = returns[
                (returns.index >= start_date) &
                (returns.index <= end_date)
            ]

            if len(window_returns) < 50:  # Minimum samples
                continue

            window_results.append({
                'window_start': start_date,
                'window_end': end_date,
                'n_samples': len(window_returns),
                'returns': window_returns
            })

        return window_results


def main():
    """Test consensus scoring."""
    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    sample_data = pd.DataFrame({
        'GLD': np.random.randn(n) * 0.01 + 0.0003,
        'TLT': np.random.randn(n) * 0.008 + 0.0002,
        'SLV': np.random.randn(n) * 0.015 + 0.0001,
    }, index=dates)

    scorer = ConsensusScorer()
    windows = scorer.generate_shrinking_window_results(
        sample_data,
        ['GLD', 'TLT', 'SLV'],
        [2020, 2021, 2022],
        '2025-12-31'
    )

    print(f"Generated {len(windows)} windows")

    for window in windows:
        print(f"\nWindow: {window['window_start']} to {window['window_end']}")
        for ticker in ['GLD', 'TLT', 'SLV']:
            if ticker in window['returns'].columns:
                score = scorer.calculate_window_score(window['returns'][ticker].dropna())
                print(f"  {ticker}: score = {score:.4f}")


if __name__ == "__main__":
    main()
