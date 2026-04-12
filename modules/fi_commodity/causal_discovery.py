"""
FI/Commodity Module - Causal Discovery
======================================
Causal discovery for Fixed Income and Commodity ETFs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(str(__file__).replace('/modules/fi_commodity/causal_discovery.py', ''))

import config
from data.loader import get_universe_data, split_data
from data.preprocessing import handle_missing_values, remove_outliers, normalize_features
from core.lingam_engine import LingamEngine
from core.causal_analyzer import CausalAnalyzer


class FICausalDiscovery:
    """
    Causal discovery for FI/Commodity ETF universe.
    """

    def __init__(self):
        """Initialize FI/Commodity causal discovery."""
        self.assets = config.FI_COMMODITY_ASSETS
        self.benchmark = config.FI_COMMODITY_BENCHMARK
        self.lingam = LingamEngine()
        self.analyzer = CausalAnalyzer()
        self.causal_edges = []
        self.leader = None
        self.followers = []

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare data for causal discovery.

        Args:
            train_df: Training data
            val_df: Optional validation data

        Returns:
            Prepared DataFrame
        """
        # Combine train and val for larger sample
        if val_df is not None:
            data = pd.concat([train_df, val_df])
        else:
            data = train_df.copy()

        # Select relevant columns
        available = [a for a in self.assets + [self.benchmark] if a in data.columns]
        data = data[available].copy()

        # Include macro variables
        for macro in config.MACRO_VARIABLES:
            if macro in data.columns:
                data[macro] = data[macro]

        # Handle missing values
        data = handle_missing_values(data)

        # Remove outliers
        data = remove_outliers(data, n_std=5.0)

        return data

    def discover_causal_structure(
        self,
        data: pd.DataFrame,
        use_bootstrap: bool = True
    ) -> Dict:
        """
        Discover causal structure using LiNGAM.

        Args:
            data: Prepared DataFrame
            use_bootstrap: Whether to use bootstrap for confidence

        Returns:
            Dictionary with causal discovery results
        """
        # Fit LiNGAM model
        if use_bootstrap:
            self.lingam.fit_with_bootstrap(data)
        else:
            self.lingam.fit(data)

        # Get causal edges
        self.causal_edges = self.lingam.get_causal_edges()

        # Build DAG
        self.analyzer.build_dag(self.causal_edges, list(data.columns))

        # Identify leader
        self.leader, leader_score, self.followers = self.analyzer.identify_leader_variable(
            benchmark=self.benchmark
        )

        return {
            'causal_edges': self.causal_edges,
            'leader': self.leader,
            'leader_score': leader_score,
            'followers': self.followers,
            'causal_matrix': self.lingam.get_direct_effects(),
            'causal_order': self.lingam.get_causal_order(),
        }

    def get_leader_predictions(self) -> List[Dict]:
        """
        Get predictions for all potential leaders.

        Returns:
            List of leader predictions with scores
        """
        predictions = []
        variable_names = self.lingam.model.variable_names if self.lingam.model else []

        for var in self.assets:
            if var not in variable_names:
                continue

            # Get followers and their strengths
            followers = self.lingam.identify_followers(var)

            # Calculate aggregate causal influence
            total_influence = sum(abs(s) for _, s in followers)

            # Get bootstrap confidence
            confidence = 0.0
            if self.lingam.bootstrap_results:
                for target, _ in followers[:3]:  # Top 3 followers
                    confidence += self.lingam.get_bootstrap_confidence(var, target)
                if followers:
                    confidence /= len(followers[:3])

            predictions.append({
                'ticker': var,
                'causal_influence': total_influence,
                'n_followers': len(followers),
                'followers': followers[:5],  # Top 5
                'confidence': confidence,
            })

        # Sort by causal influence
        predictions.sort(key=lambda x: x['causal_influence'], reverse=True)

        return predictions
