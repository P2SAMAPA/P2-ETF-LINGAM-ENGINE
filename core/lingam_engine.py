"""
LiNGAM Causal Discovery Engine
==============================
Implements LiNGAM and DirectLiNGAM algorithms for causal discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from lingam import DirectLiNGAM
import config as config_module


class LingamEngine:
    """
    LiNGAM-based causal discovery engine for ETF universe analysis.
    """

    def __init__(self, engine_config: dict = None):
        if engine_config is None:
            self.config = config_module.LINGAM_CONFIG.copy()
        else:
            self.config = engine_config.copy()
        self.model = None
        self.variable_names = None
        self.causal_order = None
        self.adjacency_matrix = None
        self.bootstrap_results = None

    def fit(self, data: pd.DataFrame, measure: str = None) -> 'LingamEngine':
        """
        Fit the LiNGAM model to the data.

        Args:
            data: DataFrame with ETF returns (rows=samples, columns=variables)
            measure: Causal measure to use (e.g., "pwling", "kernel").
                     If None, uses config value.
        Returns:
            self
        """
        self.variable_names = list(data.columns)
        measure = measure or self.config.get('measure', 'pwling')
        self.model = DirectLiNGAM(measure=measure)
        self.model.fit(data.values)
        self.causal_order = self.model.causal_order_
        self.adjacency_matrix = self.model.adjacency_matrix_
        return self

    def fit_with_bootstrap(self, data: pd.DataFrame, n_samplings: int = None, measure: str = None) -> 'LingamEngine':
        self.fit(data, measure=measure)
        n_samplings = n_samplings or self.config.get('n_samplings', 500)
        bootstrap_result = self.model.bootstrap(data.values, n_sampling=n_samplings)
        self.bootstrap_results = {
            'result': bootstrap_result,
            'n_samplings': n_samplings,
        }
        return self

    def get_causal_matrix(self) -> np.ndarray:
        return self.adjacency_matrix

    def get_causal_order(self) -> List[str]:
        if self.causal_order is None or self.variable_names is None:
            return []
        return [self.variable_names[i] for i in self.causal_order]

    def get_direct_effects(self) -> pd.DataFrame:
        if self.adjacency_matrix is None or self.variable_names is None:
            return pd.DataFrame()
        return pd.DataFrame(self.adjacency_matrix, index=self.variable_names, columns=self.variable_names)

    def get_causal_edges(self, threshold: float = config_module.MIN_CAUSAL_THRESHOLD) -> List[Tuple[str, str, float]]:
        if self.adjacency_matrix is None or self.variable_names is None:
            return []
        edges = []
        n_vars = len(self.variable_names)
        for i in range(n_vars):
            cause = self.variable_names[i]
            for j in range(n_vars):
                effect = self.variable_names[j]
                strength = self.adjacency_matrix[j, i]
                if abs(strength) >= threshold and i != j:
                    edges.append((cause, effect, strength))
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        return edges

    def get_bootstrap_confidence(self, cause: str, effect: str) -> float:
        if self.bootstrap_results is None or self.variable_names is None:
            return 0.0
        try:
            cause_idx = self.variable_names.index(cause)
            effect_idx = self.variable_names.index(effect)
            direction_counts = self.bootstrap_results['result'].get_causal_direction_counts(
                min_causal_effect=config_module.MIN_CAUSAL_THRESHOLD
            )
            prob = 0.0
            for i in range(len(direction_counts['from'])):
                if direction_counts['from'][i] == cause_idx and direction_counts['to'][i] == effect_idx:
                    prob = direction_counts['count'][i] / self.bootstrap_results['n_samplings']
                    break
            return prob
        except (ValueError, IndexError, KeyError):
            return 0.0

    def predict_effect(self, data: pd.DataFrame, cause: str, effect: str) -> np.ndarray:
        if self.adjacency_matrix is None or self.variable_names is None:
            return np.array([])
        cause_idx = self.variable_names.index(cause)
        effect_idx = self.variable_names.index(effect)
        causal_strength = self.adjacency_matrix[effect_idx, cause_idx]
        return data[cause].values * causal_strength

    def identify_leaders(self, threshold: float = config_module.MIN_CAUSAL_THRESHOLD) -> Dict[str, float]:
        if self.adjacency_matrix is None or self.variable_names is None:
            return {}
        leader_scores = {}
        for idx, var in enumerate(self.variable_names):
            outgoing = np.sum(np.abs(self.adjacency_matrix[:, idx]))
            leader_scores[var] = float(outgoing)
        return leader_scores

    def identify_followers(self, leader: str, threshold: float = config_module.MIN_CAUSAL_THRESHOLD) -> List[Tuple[str, float]]:
        if self.adjacency_matrix is None or self.variable_names is None:
            return []
        leader_idx = self.variable_names.index(leader)
        followers = []
        for i, var in enumerate(self.variable_names):
            if i != leader_idx:
                strength = self.adjacency_matrix[i, leader_idx]
                if abs(strength) >= threshold:
                    followers.append((var, float(strength)))
        followers.sort(key=lambda x: abs(x[1]), reverse=True)
        return followers
