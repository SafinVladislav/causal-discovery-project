import unittest
import networkx as nx
import random

# Import the function you are testing from your project's core module
# You will need to ensure the parent directory is in your Python path
# For Colab, you would do this with:
# import sys
# sys.path.append('..')
from core.graph_utils import find_undirected_edges, has_directed_cycle, \
has_v_structure, is_bad_graph, propagate_orientations, get_chain_components, \
generate_dag_from_cpdag, sample_dags, check_if_estimated_correctly

class TestFindUndirectedEdges(unittest.TestCase):
    """
    A test suite for the find_undirected_edges function.
    Each test method checks a specific scenario.
    """

    def test_mixed_graph_with_undirected_and_directed(self):
        """
        Tests a typical scenario with a mix of directed and undirected edges.
        The function should correctly identify only the undirected pairs.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Undirected edge
            ('A', 'C'),              # Directed edge
            ('C', 'D'), ('D', 'C'),  # Undirected edge
            ('E', 'F')               # Another directed edge
        ])
        
        expected_undirected = {('A', 'B'), ('C', 'D')}
        # The function returns a list, so we convert the results to a set of tuples
        # to handle any ordering inconsistencies and ensure a robust comparison.
        found_undirected = {tuple(sorted(edge)) for edge in find_undirected_edges(graph)}
        
        self.assertEqual(set(found_undirected), set(expected_undirected))

    def test_graph_with_only_directed_edges(self):
        """
        Tests a graph containing only directed edges. The function should return
        an empty list, as there are no reciprocal edges.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D')
        ])
        
        # We expect the function to return an empty list
        self.assertEqual(find_undirected_edges(graph), [])

    def test_graph_with_only_undirected_edges(self):
        """
        Tests a graph where all edges are undirected.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('X', 'Y'), ('Y', 'X'),
            ('Y', 'Z'), ('Z', 'Y')
        ])
        
        expected_undirected = {('X', 'Y'), ('Y', 'Z')}
        found_undirected = {tuple(sorted(edge)) for edge in find_undirected_edges(graph)}
        
        self.assertEqual(set(found_undirected), set(expected_undirected))
        
    def test_empty_graph(self):
        """
        Tests an empty graph with no nodes or edges.
        The function should return an empty list.
        """
        graph = nx.DiGraph()
        
        self.assertEqual(find_undirected_edges(graph), [])
        
    def test_graph_with_single_node(self):
        """
        Tests a graph with a single isolated node.
        There should be no edges, and the function should return an empty list.
        """
        graph = nx.DiGraph()
        graph.add_node('A')
        
        self.assertEqual(find_undirected_edges(graph), [])

    def test_disconnected_components(self):
        """
        Tests a graph composed of multiple, separate components.
        The function should correctly find undirected edges in each component.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Component 1
            ('C', 'D'),              # Component 2 (directed)
            ('E', 'F'), ('F', 'E')   # Component 3
        ])
        
        expected_undirected = {('A', 'B'), ('E', 'F')}
        found_undirected = {tuple(sorted(edge)) for edge in find_undirected_edges(graph)}
        
        self.assertEqual(set(found_undirected), set(expected_undirected))

class TestHasDirectedCycle(unittest.TestCase):
    """
    A test suite for the graph utility functions.
    """
    def test_has_directed_cycle_with_cycle(self):
        """
        Tests a graph that contains a clear directed cycle.
        The function should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        
        original_edges = list(graph.edges())
        self.assertTrue(has_directed_cycle(graph, ('C', 'A')))
        # Crucial assertion: The function must not permanently alter the graph.
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_with_no_cycle(self):
        """
        Tests a directed acyclic graph (DAG).
        The function should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        original_edges = list(graph.edges())
        self.assertFalse(has_directed_cycle(graph, ('A', 'C')))
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_with_cycle_and_undirected_edges(self):
        """
        Tests a graph with a directed cycle that is "hidden" by
        reciprocal, undirected edges. The function should correctly
        identify the cycle after temporarily removing the undirected edges.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'), # Undirected edge (A <-> B)
            ('B', 'C'),
            ('C', 'D'),
            ('D', 'B')  # This creates a cycle: B -> C -> D -> B
        ])
        
        original_edges = list(graph.edges())
        self.assertTrue(has_directed_cycle(graph, ('D', 'B')))
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_with_only_undirected_edges(self):
        """
        Tests a graph composed solely of undirected edges.
        There are no directed cycles, so the function should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B'),
            ('C', 'A')
        ])
        
        original_edges = list(graph.edges())
        self.assertFalse(has_directed_cycle(graph, ('C', 'A')))
        self.assertEqual(set(graph.edges()), set(original_edges))
  
class TestVStructure(unittest.TestCase):
    """
    Test suite for the has_v_structure function.
    """

    def test_has_simple_v_structure(self):
        """Tests a basic case where a v-structure exists."""
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v')])
        self.assertTrue(has_v_structure(graph, ('z', 'v')))

    def test_no_v_structure_due_to_u_z_edge(self):
        """Tests a case where a v-structure does not exist due to a 'u'->'z' edge."""
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('u', 'z')])
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_no_v_structure_due_to_z_u_edge(self):
        """Tests a case where a v-structure does not exist due to a 'z'->'u' edge."""
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('z', 'u')])
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_no_v_structure_due_to_reverse_edge(self):
        """Tests a case where a v-structure does not exist due to a reverse edge 'v'->'u'."""
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('v', 'u'), ('z', 'v')])
        self.assertFalse(has_v_structure(graph, ('z', 'v')))

    def test_complex_graph_with_v_structure(self):
        """Tests a more complex graph containing a v-structure."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('a', 'b'), ('c', 'b'), ('b', 'd'),  # The v-structure is 'a'->'b' and 'c'->'b'
            ('d', 'e')
        ])
        self.assertTrue(has_v_structure(graph, ('a', 'b')))

    def test_complex_graph_without_v_structure(self):
        """Tests a complex graph that has no v-structure."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')
        ])
        self.assertFalse(has_v_structure(graph, ('d', 'a')))

    def test_single_edge_graph(self):
        """Tests a graph with only one edge."""
        graph = nx.DiGraph()
        graph.add_edge('a', 'b')
        self.assertFalse(has_v_structure(graph, ('a', 'b')))

    def test_graph_with_collider_but_no_v_structure(self):
        """
        Tests a graph with a collider (two incoming edges) but no v-structure
        because the incoming nodes are connected.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('a', 'c'), ('b', 'c'), ('a', 'b')])
        self.assertFalse(has_v_structure(graph, ('a', 'c')))

    def test_multiple_v_structures(self):
        """Tests a graph containing multiple v-structures."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('a', 'b'), ('c', 'b'), # V-structure 1
            ('d', 'e'), ('f', 'e')  # V-structure 2
        ])
        self.assertTrue(has_v_structure(graph, ('d', 'e')))

class TestBadGraph(unittest.TestCase):
    """
    Test suite for the is_bad_graph function.
    """
    def test_bad_graph_with_cycle(self):
        """Tests a graph that is bad due to a cycle."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        self.assertTrue(is_bad_graph(graph, ('A', 'B')))

    def test_bad_graph_with_v_structure(self):
        """Tests a graph that is bad due to a v-structure."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('B', 'C')])
        self.assertTrue(is_bad_graph(graph, ('B', 'C')))

    def test_good_graph(self):
        """Tests a graph that is not bad (no cycles, no v-structures)."""
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        self.assertFalse(is_bad_graph(graph, ('B', 'C')))
    
    def test_bad_graph_with_both(self):
        """Tests a graph that contains both a cycle and a v-structure."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Cycle
            ('C', 'D'), ('E', 'D'),  # V-structure
        ])
        self.assertTrue(is_bad_graph(graph, ('E', 'D')))

class TestPropagateOrientations(unittest.TestCase):
    """
    Test suite for the propagate_orientations function.
    """
    def setUp(self):
        """Set up a new graph for each test."""
        self.graph = nx.DiGraph()

    def test_orient_to_avoid_cycle(self):
        """
        Tests a case where an undirected edge is oriented to prevent a cycle.
        Graph: A-B, B->C, C->A. Expected orientation: A-B becomes B->A.
        """
        self.graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'A')])
        expected_orientations = {('B', 'A')}
        result = propagate_orientations(self.graph)
        self.assertSetEqual(result, expected_orientations)

    def test_orient_to_avoid_v_structure(self):
        """
        Tests a case where an undirected edge is oriented to prevent a v-structure.
        Graph: A-C, B->C. Expected orientation: A-C becomes C->A.
        """
        self.graph.add_edges_from([('A', 'C'), ('C', 'A'), ('B', 'C')])
        expected_orientations = {('C', 'A')}
        result = propagate_orientations(self.graph)
        self.assertSetEqual(result, expected_orientations)

    def test_no_orientation_possible(self):
        """
        Tests a simple chain graph where no rules apply and no edges are oriented.
        Graph: A-B-C. Expected orientation: empty set.
        """
        self.graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        expected_orientations = set()
        result = propagate_orientations(self.graph)
        self.assertSetEqual(result, expected_orientations)
    
    def test_already_directed_graph(self):
        """
        Tests a graph with no undirected edges.
        Graph: A->B, B->C. Expected orientation: empty set.
        """
        self.graph.add_edges_from([('A', 'B'), ('B', 'C')])
        expected_orientations = set()
        result = propagate_orientations(self.graph)
        self.assertSetEqual(result, expected_orientations)

    def test_complex_graph_with_multiple_orientations(self):
        """
        Tests a graph where multiple orientations are possible.
        Graph: A-C, B->C and C->D. A-C becomes C->A.
        Then A->E, D->E is not a v-structure because it is connected A->D.
        """
        self.graph.add_edges_from([
            ('A', 'C'), ('C', 'A'),  # Undirected edge
            ('B', 'C'),              # Directed edge to form v-structure
            ('C', 'D'), ('D', 'C'),  # Undirected edge
            ('A', 'D'),
        ])
        expected_orientations = {('C', 'A'), ('C', 'D')}
        result = propagate_orientations(self.graph)
        self.assertSetEqual(result, expected_orientations)

class TestChainComponents(unittest.TestCase):
    """
    Test suite for the get_chain_components function.
    """
    def test_single_undirected_chain(self):
        """
        Tests a simple graph with a single undirected chain component.
        Graph: A-B-C (where '-' denotes an undirected edge, i.e., A->B, B->A).
        Expected: A single subgraph with nodes {A, B, C} and the original edges.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        # Verify the nodes and edges of the single component
        nodes = components[0].nodes()
        self.assertCountEqual(nodes, ['A', 'B', 'C'])
        edges = components[0].edges()
        self.assertCountEqual(edges, [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])

    def test_multiple_disconnected_chains(self):
        """
        Tests a graph with two completely separate undirected chains.
        Graph: A-B, C-D.
        Expected: Two subgraphs, one for {A, B} and one for {C, D}.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 2)
        
        nodes1 = components[0].nodes()
        nodes2 = components[1].nodes()
        self.assertCountEqual(nodes1, ['A', 'B'])
        self.assertCountEqual(nodes2, ['C', 'D'])

    def test_mixed_graph_with_directed_edges(self):
        """
        Tests a graph with both undirected and directed edges.
        Graph: A-B, C->D.
        Expected: Only the undirected component {A, B} should be returned.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        nodes = components[0].nodes()
        self.assertCountEqual(nodes, ['A', 'B'])
        edges = components[0].edges()
        self.assertCountEqual(edges, [('A', 'B'), ('B', 'A')])

    def test_graph_with_no_undirected_edges(self):
        """
        Tests a graph that is purely directed.
        Graph: A->B->C.
        Expected: An empty list, as there are no undirected components.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 0)

    def test_isolated_nodes(self):
        """
        Tests a graph containing isolated nodes.
        Graph: A-B, C, D->E.
        Expected: A single subgraph for {A, B}. Isolated nodes and directed edges are ignored.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('D', 'E')])
        graph.add_node('C')
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        nodes = components[0].nodes()
        self.assertCountEqual(nodes, ['A', 'B'])

def check_generated_dag(obj, graph, dag):
    obj.assertIsNotNone(dag)
    obj.assertEqual(len(find_undirected_edges(dag)), 0)
    graph_skeleton = {tuple(sorted(e)) for e in graph.edges()}
    dag_skeleton = {tuple(sorted(e)) for e in dag.edges()}
    obj.assertSetEqual(graph_skeleton, dag_skeleton, "DAG skeleton does not match graph skeleton")

    bad = False
    for edge in dag.edges():
        if is_bad_graph(dag, edge):
            bad = True
            break
    obj.assertEqual(bad, False)

class TestDagGeneration(unittest.TestCase):
    """
    Test suite for the DAG generation and sampling functions.
    """
    def setUp(self):
        """Set a consistent random seed for reproducible tests."""
        random.seed(42)

    def test_generates_valid_dag_simple_case(self):
        """
        Tests that the function can generate a valid DAG from a simple CPDAG.
        Graph: A-B-C. Expected: A valid DAG (e.g., A->B->C or C->B->A etc).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        dag = generate_dag_from_cpdag(graph)
        #dag = nx.DiGraph()
        #dag.add_edges_from([('A', 'B'), ('C', 'B')])
        check_generated_dag(self, graph, dag)

    def test_generates_valid_dag_complex_case(self):
        """
        Tests that the function can generate a valid DAG from a more complex CPDAG.
        Graph: A-B, B-C, C-A. A-D. D-B.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'A'), ('A', 'C'), ('A', 'D'), ('D', 'A'), ('D', 'B'), ('B', 'D')])
        dag = generate_dag_from_cpdag(graph)
        
        check_generated_dag(self, graph, dag)

    def test_sample_dags_count(self):
        """
        Tests that sample_dags returns the correct number of valid DAGs.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        n_samples = 5
        dags = sample_dags(graph, n_samples=n_samples)
        self.assertEqual(len(dags), n_samples)
        for dag in dags:
            check_generated_dag(self, graph, dag)

    def test_sample_dags_with_zero_samples(self):
        """
        Tests that sample_dags returns an empty list when n_samples is 0.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        dags = sample_dags(graph, n_samples=0)
        self.assertEqual(len(dags), 0)

class TestCheckIfEstimatedCorrectly(unittest.TestCase):
    """
    Test suite for the check_if_estimated_correctly function.
    """

    def test_exact_match(self):
        """Test when the estimated graph exactly matches the true graph."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'C')])
        
        self.assertTrue(check_if_estimated_correctly(estimated, true_graph))

    def test_undirected_match_no_directed_edges(self):
        """Test when skeletons match and estimated has no directed edges (all undirected)."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        
        self.assertTrue(check_if_estimated_correctly(estimated, true_graph))

    def test_extra_directed_edge_in_estimated(self):
        """Test when estimated has an extra directed edge not in true graph."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])  # Extra edge C->D
        
        self.assertFalse(check_if_estimated_correctly(estimated, true_graph))

    def test_missing_undirected_edge(self):
        """Test when estimated is missing an undirected edge present in true graph."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'C')])  # Missing C-D edge
        
        self.assertFalse(check_if_estimated_correctly(estimated, true_graph))

    def test_correct_directed_subset(self):
        """Test when estimated has a subset of directed edges from true graph."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'C')])  # Correct skeleton, directed subset
        
        self.assertTrue(check_if_estimated_correctly(estimated, true_graph))

    def test_wrong_direction_in_estimated(self):
        """Test when estimated has a directed edge in the wrong direction."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('C', 'B')])  # Wrong direction for B-C edge
        
        self.assertFalse(check_if_estimated_correctly(estimated, true_graph))

    def test_extra_undirected_edge(self):
        """Test when estimated has an extra undirected edge not in true graph."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        estimated = nx.DiGraph()
        estimated.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'C')])  # Extra C-D edge
        
        self.assertFalse(check_if_estimated_correctly(estimated, true_graph))

    def test_complex_correct_case(self):
        """Test a complex case where estimated is correct."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')])
        
        estimated = nx.DiGraph()
        # Correct skeleton with some edges undirected
        estimated.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # A-B undirected
            ('B', 'C'), ('C', 'B'),  # B-C undirected
            ('C', 'D'),              # C->D directed
            ('A', 'D')               # A->D directed
        ])
        
        self.assertTrue(check_if_estimated_correctly(estimated, true_graph))

    def test_complex_incorrect_case(self):
        """Test a complex case where estimated has incorrect directed edges."""
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')])
        
        estimated = nx.DiGraph()
        # Correct skeleton but wrong direction for A-D
        estimated.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # A-B undirected
            ('B', 'C'), ('C', 'B'),  # B-C undirected
            ('C', 'D'),              # C->D directed
            ('D', 'A')               # Wrong direction for A-D edge
        ])
        
        self.assertFalse(check_if_estimated_correctly(estimated, true_graph))
        
# This block is standard practice and allows you to run the tests
# directly from the command line or a Colab cell.
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
