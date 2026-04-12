"""
Causal Analyzer Module
======================
Utilities for analyzing and interpreting causal DAGs.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import config


class CausalAnalyzer:
    """
    Analyzer for causal discovery results.
    """

    def __init__(self):
        """Initialize the causal analyzer."""
        self.graph = None

    def build_dag(
        self,
        edges: List[Tuple[str, str, float]],
        variable_names: List[str]
    ) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from causal edges.

        Args:
            edges: List of (cause, effect, strength) tuples
            variable_names: List of all variable names

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add all nodes
        for var in variable_names:
            G.add_node(var)

        # Add edges
        for cause, effect, strength in edges:
            G.add_edge(cause, effect, weight=strength)

        self.graph = G
        return G

    def get_graph_metrics(self) -> Dict:
        """
        Calculate graph-level metrics.

        Returns:
            Dictionary of graph metrics
        """
        if self.graph is None:
            return {}

        metrics = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph),
        }

        return metrics

    def get_node_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for each node.

        Returns:
            DataFrame with node metrics
        """
        if self.graph is None:
            return pd.DataFrame()

        metrics = []

        for node in self.graph.nodes():
            # In-degree (how many variables affect this one)
            in_degree = self.graph.in_degree(node, weight='weight')

            # Out-degree (how many variables this one affects)
            out_degree = self.graph.out_degree(node, weight='weight')

            # Total causal influence
            total_influence = sum(
                d['weight'] for _, _, d in self.graph.out_edges(node, data=True)
            )
            total_received = sum(
                d['weight'] for _, _, d in self.graph.in_edges(node, data=True)
            )

            metrics.append({
                'variable': node,
                'in_degree': self.graph.in_degree(node),
                'out_degree': self.graph.out_degree(node),
                'total_influence': abs(total_influence),
                'total_received': abs(total_received),
                'net_causal_flow': total_influence - total_received,
            })

        return pd.DataFrame(metrics)

    def identify_leader_variable(
        self,
        benchmark: str,
        exclude_vars: List[str] = None
    ) -> Tuple[str, float, List[str]]:
        """
        Identify the leader variable with highest causal influence.

        Args:
            benchmark: Benchmark variable to exclude from leaders
            exclude_vars: Additional variables to exclude

        Returns:
            Tuple of (leader_name, leader_score, followers_list)
        """
        if self.graph is None:
            return "", 0.0, []

        exclude = [benchmark] + (exclude_vars or []) + config.MACRO_VARIABLES

        candidates = [
            n for n in self.graph.nodes()
            if n not in exclude and self.graph.out_degree(n) > 0
        ]

        if not candidates:
            return "", 0.0, []

        # Score candidates by total outgoing influence
        scores = {}
        for node in candidates:
            outgoing = sum(
                abs(d['weight'])
                for _, _, d in self.graph.out_edges(node, data=True)
            )
            scores[node] = outgoing

        # Get top leader
        leader = max(scores, key=scores.get)
        leader_score = scores[leader]

        # Get followers
        followers = [
            (target, data['weight'])
            for _, target, data in self.graph.out_edges(leader, data=True)
        ]
        followers.sort(key=lambda x: abs(x[1]), reverse=True)

        return leader, leader_score, followers

    def calculate_causal_strength_matrix(
        self,
        variables: List[str]
    ) -> pd.DataFrame:
        """
        Calculate a matrix of causal strengths between all variable pairs.

        Args:
            variables: List of variable names

        Returns:
            Matrix of causal strengths
        """
        if self.graph is None:
            return pd.DataFrame(index=variables, columns=variables)

        matrix = np.zeros((len(variables), len(variables)))

        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if self.graph.has_edge(cause, effect):
                    matrix[j, i] = self.graph[cause][effect]['weight']

        return pd.DataFrame(
            matrix,
            index=variables,
            columns=variables
        )

    def get_shortest_causal_paths(
        self,
        source: str,
        target: str
    ) -> List[List[str]]:
        """
        Find shortest causal paths between two variables.

        Args:
            source: Source variable
            target: Target variable

        Returns:
            List of paths (each path is a list of variable names)
        """
        if self.graph is None:
            return []

        try:
            paths = list(nx.all_shortest_paths(self.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []

    def visualize_as_dict(self) -> Dict:
        """
        Export graph as dictionary for visualization.

        Returns:
            Dictionary with nodes and edges for visualization
        """
        if self.graph is None:
            return {'nodes': [], 'edges': []}

        nodes = [
            {
                'id': n,
                'label': n,
                'in_degree': self.graph.in_degree(n),
                'out_degree': self.graph.out_degree(n),
            }
            for n in self.graph.nodes()
        ]

        edges = [
            {
                'source': u,
                'target': v,
                'strength': d['weight'],
            }
            for u, v, d in self.graph.edges(data=True)
        ]

        return {'nodes': nodes, 'edges': edges}
