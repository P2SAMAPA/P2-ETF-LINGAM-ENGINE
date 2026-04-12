"""
Output Module
=============
Formats predictions and manages HuggingFace dataset output.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import config


class PredictionFormatter:
    """
    Formats prediction results for output and display.
    """

    def __init__(self):
        """Initialize prediction formatter."""
        self.version = config.MODEL_VERSION

    def format_prediction(
        self,
        signals: Dict,
        metrics: Dict,
        causal_info: Dict,
        training_mode: str = 'fixed'
    ) -> Dict:
        """
        Format a complete prediction for output.

        Args:
            signals: Signal dictionary
            metrics: Performance metrics
            causal_info: Causal discovery info
            training_mode: 'fixed' or 'shrinking'

        Returns:
            Formatted prediction dictionary
        """
        primary = signals.get('primary_signal', {}) or {}

        prediction = {
            'date': signals.get('date', datetime.now().strftime('%Y-%m-%d')),
            'universe': signals.get('universe', 'unknown'),
            'predicted_leader_etf': primary.get('ticker', 'N/A'),
            'predicted_return': primary.get('ann_return', 0),
            'causal_confidence': signals.get('confidence', 0) / 100,
            'top_3_picks': [
                {'ticker': s.get('ticker'), 'score': s.get('score', 0)}
                for s in signals.get('all_signals', [])[:3]
            ],
            'followers': causal_info.get('followers', []),
            'macro_context': self._format_macro_context(causal_info),
            'dag_edges': self._format_dag_edges(causal_info),
            'dag_strengths': self._format_dag_strengths(causal_info),
            'model_version': self.version,
            'training_mode': training_mode,
            'window_start': causal_info.get('window_start', 'N/A'),
            'window_end': causal_info.get('window_end', 'N/A'),
            'metrics': {
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'best_day': metrics.get('best_day', 0),
            }
        }

        return prediction

    def _format_macro_context(self, causal_info: Dict) -> Dict:
        """Format macro context for output."""
        return {
            'vix': causal_info.get('vix', 0),
            'dxy': causal_info.get('dxy', 0),
            't10y2y': causal_info.get('t10y2y', 0),
            'tbill_3m': causal_info.get('tbill_3m', 0),
        }

    def _format_dag_edges(self, causal_info: Dict) -> List:
        """Format DAG edges for output."""
        edges = causal_info.get('causal_edges', [])
        return [[e[0], e[1]] for e in edges[:20]]  # Top 20 edges

    def _format_dag_strengths(self, causal_info: Dict) -> List:
        """Format DAG strengths for output."""
        edges = causal_info.get('causal_edges', [])
        return [[e[2]] for e in edges[:20]]  # Top 20 strengths

    def create_summary_dataframe(
        self,
        predictions: List[Dict]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Summary DataFrame
        """
        if not predictions:
            return pd.DataFrame()

        summary_rows = []

        for pred in predictions:
            row = {
                'date': pred.get('date'),
                'universe': pred.get('universe'),
                'leader': pred.get('predicted_leader_etf'),
                'confidence': pred.get('causal_confidence'),
                'return': pred.get('predicted_return'),
                'sharpe': pred.get('metrics', {}).get('sharpe_ratio'),
                'max_dd': pred.get('metrics', {}).get('max_drawdown'),
                'win_rate': pred.get('metrics', {}).get('win_rate'),
                'training_mode': pred.get('training_mode'),
            }
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def format_for_streamlit(self, prediction: Dict) -> Dict:
        """
        Format prediction specifically for Streamlit display.

        Args:
            prediction: Prediction dictionary

        Returns:
            Streamlit-ready dictionary
        """
        primary = prediction.get('primary_signal', {}) or {}
        metrics = prediction.get('metrics', {})

        return {
            'leader': prediction.get('predicted_leader_etf', 'N/A'),
            'confidence': prediction.get('causal_confidence', 0),
            'conviction_pct': f"{prediction.get('causal_confidence', 0) * 100:.1f}%",
            'ann_return': f"{prediction.get('predicted_return', 0) * 100:.2f}%",
            'sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'max_dd': f"{metrics.get('max_drawdown', 0) * 100:.1f}%",
            'win_rate': f"{metrics.get('win_rate', 0) * 100:.1f}%",
            'best_day': f"{metrics.get('best_day', 0) * 100:.1f}%",
            'training_mode': prediction.get('training_mode', 'N/A'),
            'window': f"{prediction.get('window_start', 'N/A')} - {prediction.get('window_end', 'N/A')}",
            'top_picks': [
                f"{i+1}: {p['ticker']}"
                for i, p in enumerate(prediction.get('top_3_picks', [])[:3])
            ],
        }
