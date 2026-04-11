"""
Equity Module - Leader Identifier
=================================
Identifies leader ETFs for Equity universe across different training windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from core.metrics import calculate_all_metrics, calculate_consensus_score
import config


class EquityLeaderIdentifier:
    """
    Identifies and ranks leader ETFs for Equity universe.
    """

    def __init__(self):
        """Initialize Equity leader identifier."""
        self.assets = config.EQUITY_ASSETS
        self.benchmark = config.EQUITY_BENCHMARK

    def rank_by_causal_strength(
        self,
        causal_predictions: List[Dict]
    ) -> List[Dict]:
        """
        Rank ETFs by causal strength.

        Args:
            causal_predictions: List from causal discovery

        Returns:
            Ranked list of ETFs
        """
        ranked = []

        for pred in causal_predictions:
            ranked.append({
                'ticker': pred['ticker'],
                'causal_score': pred['causal_influence'],
                'n_followers': pred['n_followers'],
                'confidence': pred['confidence'],
            })

        ranked.sort(key=lambda x: x['causal_score'], reverse=True)
        return ranked

    def evaluate_window_performance(
        self,
        returns: pd.DataFrame,
        ticker: str,
        benchmark_ticker: str
    ) -> Dict:
        """
        Evaluate performance of a ticker in a specific window.

        Args:
            returns: DataFrame with returns
            ticker: Ticker to evaluate
            benchmark_ticker: Benchmark ticker

        Returns:
            Performance metrics
        """
        if ticker not in returns.columns:
            return {}

        ticker_returns = returns[ticker].dropna()
        benchmark_returns = returns[benchmark_ticker].dropna() if benchmark_ticker in returns.columns else ticker_returns

        # Calculate metrics
        ticker_metrics = calculate_all_metrics(ticker_returns)

        # Calculate excess return over benchmark
        if benchmark_ticker in returns.columns:
            ticker_metrics['excess_return'] = ticker_metrics['annualized_return'] - calculate_all_metrics(benchmark_returns)['annualized_return']
        else:
            ticker_metrics['excess_return'] = 0.0

        return ticker_metrics

    def calculate_window_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate weighted consensus score for a window.

        Args:
            metrics: Performance metrics

        Returns:
            Consensus score (0-1)
        """
        return calculate_consensus_score(
            metrics,
            weights=config.CONSENSUS_WEIGHTS
        )

    def identify_consensus_leader(
        self,
        window_results: List[Dict]
    ) -> Tuple[str, float, List[Dict]]:
        """
        Identify consensus leader across all shrinking windows.

        Args:
            window_results: List of window analysis results

        Returns:
            Tuple of (leader_ticker, conviction, top_3_picks)
        """
        # Aggregate scores by ticker
        ticker_scores = {}

        for window in window_results:
            ticker = window['leader_ticker']
            score = window['consensus_score']

            if ticker not in ticker_scores:
                ticker_scores[ticker] = []

            ticker_scores[ticker].append(score)

        # Calculate average consensus score for each ticker
        ticker_avg_scores = {}
        for ticker, scores in ticker_scores.items():
            # Only consider windows with positive scores
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                ticker_avg_scores[ticker] = np.mean(valid_scores)
            else:
                ticker_avg_scores[ticker] = 0.0

        # Sort by average score
        ranked = sorted(
            ticker_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not ranked or ranked[0][1] == 0:
            return self.assets[0], 0.0, []

        # Get top leader
        leader = ranked[0][0]
        total_score = sum(s for _, s in ranked)
        conviction = ranked[0][1] / total_score if total_score > 0 else 0.0

        # Get top 3 picks
        top_3 = [
            {'ticker': t, 'score': s}
            for t, s in ranked[:3]
        ]

        return leader, conviction, top_3

    def generate_leader_report(
        self,
        causal_predictions: List[Dict],
        window_results: List[Dict],
        returns: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive leader identification report.

        Args:
            causal_predictions: Causal discovery results
            window_results: Window analysis results
            returns: Historical returns

        Returns:
            Leader report dictionary
        """
        # Get ranked by causal strength
        causal_ranked = self.rank_by_causal_strength(causal_predictions)

        # Get consensus leader
        consensus_leader, conviction, top_3 = self.identify_consensus_leader(window_results)

        # Get current leader from most recent causal analysis
        current_leader = causal_ranked[0]['ticker'] if causal_ranked else None

        # Evaluate recent performance
        recent_returns = returns.iloc[-252:] if len(returns) >= 252 else returns
        recent_metrics = {}
        for ticker in self.assets[:5]:  # Top 5 from causal
            if ticker in recent_returns.columns:
                recent_metrics[ticker] = calculate_all_metrics(recent_returns[ticker].dropna())

        return {
            'consensus_leader': consensus_leader,
            'consensus_conviction': conviction,
            'top_3_picks': top_3,
            'current_leader': current_leader,
            'causal_ranking': causal_ranked[:5],
            'recent_performance': recent_metrics,
        }

    def get_sector_rotation_signals(
        self,
        causal_predictions: List[Dict]
    ) -> Dict[str, str]:
        """
        Generate sector rotation signals based on causal leadership.

        Args:
            causal_predictions: Causal discovery results

        Returns:
            Dictionary of sector -> signal ticker
        """
        sector_signals = {}

        # Map sectors to relevant ETFs
        sector_etfs = {
            'Technology': ['QQQ', 'XLK'],
            'Financials': ['XLF'],
            'Energy': ['XLE'],
            'Healthcare': ['XLV'],
            'Industrials': ['XLI'],
            'Consumer_Discretionary': ['XLY'],
            'Consumer_Staples': ['XLP'],
            'Utilities': ['XLU'],
            'Materials': ['XME', 'XLB'],
            'Real_Estate': ['XLRE'],
            'Small_Cap': ['IWM'],
            'Gold_Miners': ['GDX'],
        }

        for pred in causal_predictions:
            ticker = pred['ticker']
            score = pred['causal_influence']

            for sector, etfs in sector_etfs.items():
                if ticker in etfs and sector not in sector_signals:
                    sector_signals[sector] = ticker

        return sector_signals