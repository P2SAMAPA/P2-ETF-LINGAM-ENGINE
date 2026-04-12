"""
LiNGAM Causal Discovery Engine
==============================
Implements LiNGAM and DirectLiNGAM algorithms for causal discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from lingam import DirectLiNGAM
import config


class LingamEngine:
    """
    LiNGAM-based causal discovery engine for ETF universe analysis.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the LiNGAM engine.

        Args:
            config: Dictionary with LiNGAM configuration parameters
        """
        self.config = config or config.LINGAM_CONFIG.copy()
        self.model = None
        self.causal_order = None
        self.adjacency_matrix = None
        self.bootstrap_results = None

    def fit(self, data: pd.DataFrame) -> 'LingamEngine':
        """
        Fit the LiNGAM model to the data.

        Args:
            data: DataFrame with ETF returns (rows=samples, columns=variables)

        Returns:
            self
        """
        # Use DirectLiNGAM for more robust causal discovery
        self.model = DirectLiNGAM(
            measure=self.config.get('measure', 'pwling')
        )

        # Fit the model
        self.model.fit(data.values)

        # Store results
        self.causal_order = self.model.causal_order_
        self.adjacency_matrix = self.model.adjacency_matrix_

        return self

    def fit_with_bootstrap(
        self,
        data: pd.DataFrame,
        n_samplings: int = None
    ) -> 'LingamEngine':
        """
        Fit LiNGAM with bootstrap for confidence estimation.

        Args:
            data: DataFrame with ETF returns
            n_samplings: Number of bootstrap samples

        Returns:
            self
        """
        n_samplings = n_samplings or self.config.get('n_samplings', 100)

        # Fit base model
        self.fit(data)

        # Perform bootstrap using the model's bootstrap method
        bootstrap_result = self.model.bootstrap(data.values, n_sampling=n_samplings)

        self.bootstrap_results = {
            'result': bootstrap_result,
            'n_samplings': n_samplings,
        }

        return self

    def get_causal_matrix(self) -> np.ndarray:
        """
        Get the causal adjacency matrix.

        Returns:
            Adjacency matrix where element [i,j] is causal effect from j to i
        """
        return self.adjacency_matrix

    def get_causal_order(self) -> List[str]:
        """
        Get the causal order of variables.

        Returns:
            List of variable names in causal order
        """
        if self.causal_order is None:
            return []
        return [self.model.variable_names[i] for i in self.causal_order]

    def get_direct_effects(self) -> pd.DataFrame:
        """
        Get matrix of direct causal effects between all pairs.

        Returns:
            DataFrame with direct effects (row=effect, col=cause)
        """
        if self.adjacency_matrix is None:
            return pd.DataFrame()

        df = pd.DataFrame(
            self.adjacency_matrix,
            index=self.model.variable_names,
            columns=self.model.variable_names
        )
        return df

    def get_causal_edges(
        self,
        threshold: float = config.MIN_CAUSAL_THRESHOLD
    ) -> List[Tuple[str, str, float]]:
        """
        Get significant causal edges.

        Args:
            threshold: Minimum absolute effect size

        Returns:
            List of (cause, effect, strength) tuples
        """
        edges = []
        for i, cause in enumerate(self.model.variable_names):
            for j, effect in enumerate(self.model.variable_names):
                strength = self.adjacency_matrix[j, i]
                if abs(strength) >= threshold and i != j:
                    edges.append((cause, effect, strength))

        # Sort by absolute strength
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        return edges

    def get_bootstrap_confidence(
        self,
        cause: str,
        effect: str
    ) -> float:
        """
        Get bootstrap confidence for a causal relationship.

        Args:
            cause: Source variable name
            effect: Target variable name

        Returns:
            Confidence level (0-1)
        """
        if self.bootstrap_results is None:
            return 0.0

        try:
            cause_idx = self.model.variable_names.index(cause)
            effect_idx = self.model.variable_names.index(effect)

            # Get causal direction counts from bootstrap results
            direction_counts = self.bootstrap_results['result'].get_causal_direction_counts(
                min_causal_effect=config.MIN_CAUSAL_THRESHOLD
            )

            # Find the probability for the specific direction
            prob = 0.0
            for i in range(len(direction_counts['from'])):
                if (direction_counts['from'][i] == cause_idx and 
                    direction_counts['to'][i] == effect_idx):
                    prob = direction_counts['count'][i] / self.bootstrap_results['n_samplings']
                    break

            return prob
        except (ValueError, IndexError, KeyError):
            return 0.0

    def predict_effect(
        self,
        data: pd.DataFrame,
        cause: str,
        effect: str
    ) -> np.ndarray:
        """
        Predict the effect of cause on effect variable.

        Args:
            data: DataFrame with current values
            cause: Cause variable name
            effect: Effect variable name

        Returns:
            Predicted effect values
        """
        cause_idx = self.model.variable_names.index(cause)
        effect_idx = self.model.variable_names.index(effect)

        causal_strength = self.adjacency_matrix[effect_idx, cause_idx]
        predicted_effect = data[cause].values * causal_strength

        return predicted_effect

    def identify_leaders(
        self,
        threshold: float = config.MIN_CAUSAL_THRESHOLD
    ) -> Dict[str, float]:
        """
        Identify leader variables (high out-degree).

        Args:
            threshold: Minimum causal effect threshold

        Returns:
            Dictionary of {variable: leader_score}
        """
        leader_scores = {}

        for var in self.model.variable_names:
            var_idx = self.model.variable_names.index(var)

            # Sum of outgoing causal effects
            outgoing = np.sum(np.abs(self.adjacency_matrix[:, var_idx]))
            outgoing = np.sum(np.abs(self.adjacency_matrix[var_idx, :]))

            leader_scores[var] = float(outgoing)

        return leader_scores

    def identify_followers(
        self,
        leader: str,
        threshold: float = config.MIN_CAUSAL_THRESHOLD
    ) -> List[Tuple[str, float]]:
        """
        Identify followers of a leader variable.

        Args:
            leader: Leader variable name
            threshold: Minimum causal effect threshold

        Returns:
            List of (follower, strength) tuples sorted by strength
        """
        leader_idx = self.model.variable_names.index(leader)
        followers = []

        for i, var in enumerate(self.model.variable_names):
            if i != leader_idx:
                strength = self.adjacency_matrix[i, leader_idx]
                if abs(strength) >= threshold:
                    followers.append((var, float(strength)))

        # Sort by strength
        followers.sort(key=lambda x: abs(x[1]), reverse=True)
        return followers
