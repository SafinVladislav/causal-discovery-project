import unittest
from collections import defaultdict
from math import log
from unittest import mock

# Import both the function and the module so we can reassign names inside the module under test.
from core.intervention import choose_intervention_variable
import core.intervention as intervention


# A tiny FakeDAG we can control from tests. It implements only has_edge(u, v).
class FakeDAG:
    def __init__(self, edges):
        # edges is an iterable of directed-edge tuples (u, v)
        self._edges = set(edges)

    def has_edge(self, u, v):
        return (u, v) in self._edges


class TestChooseInterventionVariable(unittest.TestCase):

    def setUp(self):
        # Save originals so we can restore them in tearDown
        self._orig_find = getattr(intervention, "find_undirected_edges", None)
        self._orig_sample = getattr(intervention, "sample_dags", None)

        # Default stubs (tests will override as needed) placed on the intervention module
        def default_find_undirected_edges(graph):
            return []

        def default_sample_dags(graph, n_samples=1000):
            return []

        intervention.find_undirected_edges = default_find_undirected_edges
        intervention.sample_dags = default_sample_dags

    def tearDown(self):
        # Restore originals to avoid cross-test pollution
        if self._orig_find is None:
            del intervention.find_undirected_edges
        else:
            intervention.find_undirected_edges = self._orig_find

        if self._orig_sample is None:
            del intervention.sample_dags
        else:
            intervention.sample_dags = self._orig_sample

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            choose_intervention_variable(None, set(), strategy="not-a-strategy")

    def test_no_undirected_edges_returns_none(self):
        # find_undirected_edges returns empty -> None
        def f(graph):
            return []

        intervention.find_undirected_edges = f
        res = choose_intervention_variable(object(), set())
        self.assertIsNone(res)

    def test_all_nodes_intervened_returns_none(self):
        # If all nodes that appear in undirected edges are in intervened, return None
        def f(graph):
            return [("A", "B"), ("B", "A")]

        intervention.find_undirected_edges = f
        res = choose_intervention_variable(object(), intervened={"A", "B"})
        self.assertIsNone(res)

    def test_greedy_strategy_selects_node_with_highest_first_position_count(self):
        # Greedy counts only the first element of each undirected edge tuple.
        def f(graph):
            return [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]

        intervention.find_undirected_edges = f
        res = choose_intervention_variable(object(), intervened=set(), strategy="greedy")
        # counts: A:2 (from edges where it's first), B:1 => choose 'A'
        self.assertEqual(res, "A")

    def test_sample_dags_empty_fallbacks_to_greedy(self):
        # If sample_dags returns empty, function should fallback to greedy.
        def f(graph):
            return [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]

        intervention.find_undirected_edges = f

        def empty_sampler(graph, n_samples=1000):
            return []

        intervention.sample_dags = empty_sampler

        # greedy would pick 'A' (first element appears twice)
        res = choose_intervention_variable(object(), intervened=set(), strategy="entropy")
        self.assertEqual(res, "A")

    def test_choose_intervention_variable_three_strategies_different_results(self):
        """
        Construct a small "graph" and a deterministic sample_dags that
        make each strategy pick a different node:

          - greedy -> "G"  (highest count as first element in undirected edge list)
          - minimax -> "M" (M's largest orientation class size is 3, smaller than E's 4)
          - entropy -> "E" (E's orientation distribution is more entropic than M's)

        We place the stubs directly on the intervention module so the function under test uses them.
        """

        # Mock find_undirected_edges to return both directions for each pair.
        def f(graph):
            return [
                # Put G-as-first-element multiple times so greedy picks G.
                ("G", "A"), ("G", "B"), ("G", "C"),
                ("A", "G"), ("B", "G"), ("C", "G"),

                # M connected to H and J (two neighbors)
                ("M", "H"), ("H", "M"), ("M", "J"), ("J", "M"),

                # E connected to I and K (two neighbors)
                ("E", "I"), ("I", "E"), ("E", "K"), ("K", "E"),
            ]

        intervention.find_undirected_edges = f

        # Build 6 DAG samples (n_samples = 6). We'll craft orientations so that:
        # - For node M (neighbors H,J): we produce 3 samples with (out,in) and 3 with (in,out)
        # - For node E (neighbors I,K): we produce 4 samples (out,out), 1 (out,in), 1 (in,out)
        dag1 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),  # G out to all
            ("M", "H"), ("J", "M"),               # M: (out,in)
            ("E", "I"), ("E", "K"),               # E: (out,out)
        })

        dag2 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),
            ("H", "M"), ("M", "J"),               # M: (in,out)
            ("E", "I"), ("E", "K"),               # E: (out,out)
        })

        dag3 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),
            ("M", "H"), ("J", "M"),               # M: (out,in)
            ("E", "I"), ("E", "K"),               # E: (out,out)
        })

        dag4 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),
            ("H", "M"), ("M", "J"),               # M: (in,out)
            ("E", "I"), ("E", "K"),               # E: (out,out)
        })

        dag5 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),
            ("H", "M"), ("M", "J"),               # M: (in,out)
            ("E", "I"), ("K", "E"),               # E: (out,in)
        })

        dag6 = FakeDAG({
            ("G", "A"), ("G", "B"), ("G", "C"),
            ("M", "H"), ("J", "M"),               # M: (out,in)
            ("I", "E"), ("E", "K"),               # E: (in,out)
        })

        intervention.sample_dags = lambda graph, n_samples=6: [dag1, dag2, dag3, dag4, dag5, dag6]

        # Now assert picks for each strategy:
        res_greedy = choose_intervention_variable(object(), intervened=set(), n_samples=6, strategy="greedy")
        self.assertEqual(res_greedy, "G", "greedy should pick the high-count-first-element node 'G'")

        res_minimax = choose_intervention_variable(object(), intervened=set(), n_samples=6, strategy="minimax")
        self.assertIn(res_minimax, ["H", "J", "M"], "minimax should prefer 'H', 'J', or 'M' as they have the same minimal orientation class size")

        res_entropy = choose_intervention_variable(object(), intervened=set(), n_samples=6, strategy="entropy")
        self.assertEqual(res_entropy, "E", "entropy should prefer 'E' which has a higher entropy orientation distribution")


if __name__ == '__main__':
    unittest.main()