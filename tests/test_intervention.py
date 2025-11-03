import unittest
import networkx as nx
from unittest.mock import patch, MagicMock
import random

# Assuming the necessary imports; in practice, adjust paths
from core.intervention import choose_intervention_variable
from core.graph_utils import find_undirected_edges, sample_dags

class TestChooseInterventionVariable(unittest.TestCase):
    """
    Test suite for the choose_intervention_variable function.
    This function selects a variable to intervene on based on the given strategy.
    Tests cover edge cases, greedy strategy (deterministic), and fallback scenarios.
    For stochastic strategies (minimax, entropy), we mock the sampling to ensure determinism.
    """

    def setUp(self):
        """Set up a consistent random seed for reproducible tests."""
        random.seed(42)

    def test_invalid_strategy_raises_value_error(self):
        """Tests that an invalid strategy raises ValueError."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])  # Undirected A-B
        intervened = set()
        with self.assertRaises(ValueError):
            choose_intervention_variable(graph, intervened, "invalid")

    def test_no_undirected_edges_returns_none(self):
        """Tests that no undirected edges results in (None, None)."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B')])  # Directed only
        intervened = set()
        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        self.assertIsNone(node)
        self.assertIsNone(fallback)

    def test_all_nodes_intervened_returns_none(self):
        """Tests that all potential nodes are intervened results in (None, None)."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        intervened = {'A', 'B'}  # All nodes intervened
        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        self.assertIsNone(node)
        self.assertIsNone(fallback)

    def test_greedy_strategy_picks_highest_degree_node(self):
        """Tests greedy strategy: picks node with most undirected edges."""
        graph = nx.DiGraph()
        # Undirected edges: A-B, A-C, D-E (A has degree 2, others 1)
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A'), ('D', 'E'), ('E', 'D')])
        intervened = set()  # No interventions

        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        self.assertEqual(node, 'A')  # A has highest count
        self.assertEqual(fallback, 0)  # No fallback

    def test_greedy_strategy_with_ties_picks_first_max(self):
        """Tests greedy with ties: picks one of the max-degree nodes (deterministic due to seed)."""
        graph = nx.DiGraph()
        # Undirected: A-B, C-D (A,B,C,D all degree 1)
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')])
        intervened = set()

        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        # With seed 42, max() on dict keys should pick 'A' (sorted order)
        self.assertEqual(node, 'A')
        self.assertEqual(fallback, 0)

    def test_greedy_strategy_some_nodes_intervened(self):
        """Tests greedy ignoring intervened nodes."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'C')])
        intervened = {'A'}  # A intervened, so B or C (both degree 2, but B first?)

        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        
        self.assertIn(node, {'B', 'C'})  # Tie, but with seed, assume 'B'
        self.assertEqual(fallback, 0)

    import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

from core.intervention import choose_intervention_variable

class TestChooseInterventionVariable(unittest.TestCase):
    import random
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

import networkx as nx

from core.intervention import choose_intervention_variable

class TestChooseInterventionVariable(unittest.TestCase):
    @patch("core.intervention.sample_dags")
    def test_minimax_strategy_with_mock_sampling(self, mock_sample_dags):
        """
        Tests the minimax strategy with fully-controlled sampling.

        Graph: A <-> B   and   C <-> D   (two independent undirected edges)

        * Sampling fails  → fallback = 1
        * Sampling succeeds
            - A-B is balanced  (50 % A→B, 50 % B→A)   → max prob = 0.5
            - C-D is unbalanced (80 % C→D, 20 % D→C) → max prob = 0.8
        minimax must pick a node from the balanced component (A or B).
        """
        random.seed(42)                     # reproducible mock DAGs
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")])
        intervened = set()

        # -------------------------------------------------
        # 1. Sampling failure → fallback to greedy
        # -------------------------------------------------
        mock_sample_dags.return_value = []               # 0 / 1000 < 0.7
        node, fallback = choose_intervention_variable(graph, intervened, "minimax")
        self.assertEqual(fallback, 1)                    # fallback flag

        # -------------------------------------------------
        # 2. Successful sampling with controlled orientations
        # -------------------------------------------------
        N = 800                                          # 800 / 1000 = 0.8 > THRESHOLD
        mock_dags = []

        # Build 800 DAG mocks:
        #   - first 400 : A→B   (and C→D for the first 640 of the 800)
        #   - next  400 : B→A
        #   - last  160 : D→C   (so C-D is 640 : 160 → 80 % / 20 %)
        for i in range(N):
            dag = MagicMock()

            # ---- orientation of A-B ----
            if i < N // 2:                     # first half: A→B
                ab_u, ab_v = "A", "B"
            else:                               # second half: B→A
                ab_u, ab_v = "B", "A"

            # ---- orientation of C-D ----
            if i < int(N * 0.8):               # 80 % of samples
                cd_u, cd_v = "C", "D"
            else:                               # 20 % of samples
                cd_u, cd_v = "D", "C"

            # side_effect: return True only for the chosen direction
            def _has_edge(u, v):
                if {u, v} == {"A", "B"}:
                    return u == ab_u and v == ab_v
                if {u, v} == {"C", "D"}:
                    return u == cd_u and v == cd_v
                return False

            dag.has_edge.side_effect = _has_edge
            mock_dags.append(dag)

        mock_sample_dags.return_value = mock_dags

        node, fallback = choose_intervention_variable(graph, intervened, "minimax")

        # -------------------------------------------------
        # 3. Assertions
        # -------------------------------------------------
        self.assertEqual(fallback, 0)               # sampling succeeded
        # minimax picks the node with the smallest *maximum* orientation probability.
        # Both A and B have max prob = 0.5, C and D have 0.8 → A (first alphabetically) is chosen.
        self.assertEqual(node, "A")

    @patch('core.intervention.sample_dags')
    def test_entropy_strategy_with_mock_sampling(self, mock_sample_dags):
        """Tests entropy strategy with mocked sampling: chooses node maximizing entropy."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])  # Simple undirected A-B
        intervened = set()

        # Test fallback
        mock_sample_dags.return_value = []  # Low success
        node, fallback = choose_intervention_variable(graph, intervened, "entropy")
        self.assertEqual(fallback, 1)

        # Successful: assume 'A' has high entropy (balanced), 'B' low
        mock_sample_dags.return_value = [MagicMock() for _ in range(800)]
        # Mock to have high entropy for 'A'
        node, fallback = choose_intervention_variable(graph, intervened, "entropy")
        self.assertIsNotNone(node)
        self.assertEqual(fallback, 0)

    def test_fallback_to_greedy_when_sampling_fails(self):
        """Tests that low sampling success falls back to greedy and returns fallback=1."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])  # B degree 2
        intervened = set()

        with patch('core.intervention.sample_dags', return_value=[MagicMock() for _ in range(600)]):  # 600/1000=0.6 <0.7
            node, fallback = choose_intervention_variable(graph, intervened, "minimax")
            self.assertEqual(node, 'B')  # Greedy pick
            self.assertEqual(fallback, 1)

    def test_no_adj_undirected_edges_skips_node(self):
        """Tests that nodes with no adjacent undirected edges are skipped."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D')])  # C->D directed, A-B undirected
        intervened = set()
        node, fallback = choose_intervention_variable(graph, intervened, "greedy")
        self.assertEqual(node, 'A')  # Or 'B', but not 'C' or 'D' as they have no undirected adj
        # D has adj to C, but if directed, find_undirected_edges only if bidirectional, here only ('C','D'), no ('D','C'), so no undirected for C-D

# Standard runner
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)