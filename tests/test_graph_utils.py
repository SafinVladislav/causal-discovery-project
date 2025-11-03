import unittest
import networkx as nx
import random
import itertools

from core.graph_utils import find_undirected_edges, has_directed_cycle, \
has_v_structure, is_bad_graph, propagate_orientations, get_chain_components, \
sample_dags, check_if_estimated_correctly, \
to_undirected_with_v_structures

#==================================================
class TestToUndirectedWithVStructures(unittest.TestCase):

    def test_simple_chain(self):
        """Test a simple chain A -> B -> C with no v-structures."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C'})
        expected_edges = {('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')}
        self.assertEqual(set(result.edges()), expected_edges)

    def test_collider(self):
        """Test a collider A -> C <- B, which is a v-structure."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'C'), ('B', 'C')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C'})
        expected_edges = {('A', 'C'), ('B', 'C')}
        self.assertEqual(set(result.edges()), expected_edges)
        # Ensure no reverse edges for directed parts
        self.assertFalse(result.has_edge('C', 'A'))
        self.assertFalse(result.has_edge('C', 'B'))

    def test_common_cause(self):
        """Test common cause A -> B, A -> C with no v-structures."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'B'), ('A', 'C')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C'})
        expected_edges = {('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A')}
        self.assertEqual(set(result.edges()), expected_edges)

    def test_diamond(self):
        """Test diamond A -> B -> D, A -> C -> D with v-structure at D."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C', 'D'})
        expected_edges = {('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A'), ('B', 'D'), ('C', 'D')}
        self.assertEqual(set(result.edges()), expected_edges)
        # Ensure no reverse for v-structure edges
        self.assertFalse(result.has_edge('D', 'B'))
        self.assertFalse(result.has_edge('D', 'C'))

    def test_shielded_collider(self):
        """Test shielded collider A -> C <- B with A - B edge, no unshielded v-structure."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'C'), ('B', 'C'), ('A', 'B')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C'})
        expected_edges = {('A', 'C'), ('C', 'A'), ('B', 'C'), ('C', 'B'), ('A', 'B'), ('B', 'A')}
        self.assertEqual(set(result.edges()), expected_edges)

    def test_empty_graph(self):
        """Test empty graph."""
        directed_graph = nx.DiGraph()
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(len(result.nodes()), 0)
        self.assertEqual(len(result.edges()), 0)

    def test_single_node(self):
        """Test single node graph."""
        directed_graph = nx.DiGraph()
        directed_graph.add_node('A')
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A'})
        self.assertEqual(len(result.edges()), 0)

    def test_disconnected_components(self):
        """Test disconnected: A -> B and C -> D with v-structure in one."""
        directed_graph = nx.DiGraph()
        directed_graph.add_edges_from([('A', 'B'), ('C', 'E'), ('D', 'E')])
        
        result = to_undirected_with_v_structures(directed_graph)
        
        self.assertEqual(set(result.nodes()), {'A', 'B', 'C', 'D', 'E'})
        expected_edges = {('A', 'B'), ('B', 'A'), ('C', 'E'), ('D', 'E')}
        self.assertEqual(set(result.edges()), expected_edges)

#==================================================
class TestFindUndirectedEdges(unittest.TestCase):
    def test_fully_undirected_edges(self):
        """Test graph with bidirectional edges only (fully undirected)."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B'),
            ('C', 'D'), ('D', 'C')
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'C')]
        self.assertEqual(set(undirected), set(expected))

    def test_fully_directed_edges(self):
        """Test graph with unidirectional edges only (no undirected edges)."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'),  # A -> B
            ('B', 'C'),  # B -> C
            ('C', 'D')   # C -> D
        ])
        
        undirected = find_undirected_edges(graph)
        self.assertEqual(undirected, [])

    def test_mixed_directions(self):
        """Test graph with both directed and undirected edges."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # undirected A-B
            ('B', 'C'),              # directed B->C
            ('C', 'D'), ('D', 'C'),  # undirected C-D
            ('D', 'E')               # directed D->E
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')]
        self.assertEqual(set(undirected), set(expected))

    def test_single_undirected_edge(self):
        """Test graph with exactly one undirected edge."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A')  # only undirected edge
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A')]
        self.assertEqual(set(undirected), set(expected))

    def test_self_loop(self):
        """Test that self-loops are handled correctly (should not be undirected)."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'A'),  # self-loop
            ('A', 'B'), ('B', 'A')  # undirected edge
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A')]
        self.assertEqual(set(undirected), set(expected))

    def test_empty_graph(self):
        """Test empty graph returns empty list."""
        graph = nx.DiGraph()
        undirected = find_undirected_edges(graph)
        self.assertEqual(undirected, [])

    def test_single_node_no_edges(self):
        """Test single node with no edges."""
        graph = nx.DiGraph()
        graph.add_node('A')
        undirected = find_undirected_edges(graph)
        self.assertEqual(undirected, [])

    def test_disconnected_components(self):
        """Test graph with disconnected components having different edge types."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            # Component 1: undirected
            ('A', 'B'), ('B', 'A'),
            # Component 2: directed
            ('C', 'D'),
            # Component 3: mixed
            ('E', 'F'), ('F', 'E'), ('F', 'G')
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A'), ('E', 'F'), ('F', 'E')]
        self.assertEqual(set(undirected), set(expected))

    def test_complex_essential_graph(self):
        """Test a realistic essential graph from the paper's example (Figure 2)."""
        # Essential graph for [G] from Figure 2: V2->V5, V3->V5 directed, others undirected
        graph = nx.DiGraph()
        graph.add_nodes_from(['V1', 'V2', 'V3', 'V4', 'V5'])
        graph.add_edges_from([
            # Directed edges (V2->V5, V3->V5)
            ('V2', 'V5'),
            ('V3', 'V5'),
            # Undirected edges (both directions)
            ('V1', 'V2'), ('V2', 'V1'),
            ('V1', 'V3'), ('V3', 'V1'),
            ('V2', 'V4'), ('V4', 'V2'),
            ('V3', 'V4'), ('V4', 'V3')
        ])
        
        undirected = find_undirected_edges(graph)
        expected_undirected = [
            ('V1', 'V2'), ('V2', 'V1'),
            ('V1', 'V3'), ('V3', 'V1'),
            ('V2', 'V4'), ('V4', 'V2'),
            ('V3', 'V4'), ('V4', 'V3')
        ]
        self.assertEqual(set(undirected), set(expected_undirected))
        
        # Verify directed edges are NOT returned
        self.assertNotIn(('V2', 'V5'), undirected)
        self.assertNotIn(('V5', 'V2'), undirected)
        self.assertNotIn(('V3', 'V5'), undirected)
        self.assertNotIn(('V5', 'V3'), undirected)

    def test_no_duplicate_edges(self):
        """Test that adding duplicate edges doesn't affect results."""
        graph = nx.DiGraph()
        # Add edges multiple times (networkx ignores duplicates)
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # undirected
            ('A', 'B'), ('B', 'A'),  # duplicates
            ('B', 'C')               # directed
        ])
        
        undirected = find_undirected_edges(graph)
        expected = [('A', 'B'), ('B', 'A')]
        self.assertEqual(set(undirected), set(expected))

#==================================================
class TestHasDirectedCycle(unittest.TestCase):
    """
    Test suite for the has_directed_cycle function.
    Note: In actual usage, before calling this function for a new orientation u -> v,
    the opposite edge v -> u is temporarily removed from the graph to simulate the orientation.
    These tests simulate that by removing the opposite edge before the call and restoring it after.
    """

    def test_has_directed_cycle_creating_cycle(self):
        """
        Tests a case where orienting the new edge would create a directed cycle.
        Graph: A <-> B (undirected), B -> C, C -> A (directed).
        Orienting A -> B: should detect cycle A -> B -> C -> A.
        (Check path from B to A after removing B -> A: B -> C -> A)
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Undirected A <-> B
            ('B', 'C'),              # Directed B -> C
            ('C', 'A')               # Directed C -> A
        ])
        
        u, v = 'A', 'B'  # New orientation A -> B
        opposite_edge = (v, u)  # B -> A
        
        original_edges = list(graph.edges())
        
        # Simulate orientation: remove opposite edge
        graph.remove_edge(*opposite_edge)
        
        self.assertTrue(has_directed_cycle(graph, (u, v)))
        
        # Restore the graph
        graph.add_edge(*opposite_edge)
        
        # Ensure graph is unchanged
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_no_cycle_created(self):
        """
        Tests a case where orienting the new edge does not create a directed cycle.
        Graph: A <-> B (undirected), B -> C, C -> A (directed).
        Orienting B -> A: should not detect cycle (check path from A to B: no path).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Undirected A <-> B
            ('B', 'C'),              # Directed B -> C
            ('C', 'A')               # Directed C -> A
        ])
        
        u, v = 'B', 'A'  # New orientation B -> A
        opposite_edge = (v, u)  # A -> B
        
        original_edges = list(graph.edges())
        
        # Simulate orientation: remove opposite edge
        graph.remove_edge(*opposite_edge)
        
        self.assertFalse(has_directed_cycle(graph, (u, v)))
        
        # Restore the graph
        graph.add_edge(*opposite_edge)
        
        # Ensure graph is unchanged
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_existing_cycle_irrelevant(self):
        """
        Tests a case with an existing directed cycle, but the new orientation does not close a new cycle.
        Graph: A -> B -> C -> A (existing cycle), D <-> E (undirected).
        Orienting D -> E: no path from E to D, so False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'A'),  # Existing cycle
            ('D', 'E'), ('E', 'D')               # Undirected D <-> E
        ])
        
        u, v = 'D', 'E'  # New orientation D -> E
        opposite_edge = (v, u)  # E -> D
        
        original_edges = list(graph.edges())
        
        # Simulate orientation: remove opposite edge
        graph.remove_edge(*opposite_edge)
        
        self.assertFalse(has_directed_cycle(graph, (u, v)))
        
        # Restore the graph
        graph.add_edge(*opposite_edge)
        
        # Ensure graph is unchanged
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_with_only_undirected_no_cycle(self):
        """
        Tests a graph with only undirected edges; orienting one should not create a cycle.
        Graph: A <-> B <-> C (undirected chain).
        Orienting A -> B: remove B -> A, check path from B to A (B -> C, but no path back).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B')
        ])
        
        u, v = 'A', 'B'  # New orientation A -> B
        opposite_edge = (v, u)  # B -> A
        
        original_edges = list(graph.edges())
        
        # Simulate orientation: remove opposite edge
        graph.remove_edge(*opposite_edge)
        
        self.assertFalse(has_directed_cycle(graph, (u, v)))
        
        # Restore the graph
        graph.add_edge(*opposite_edge)
        
        # Ensure graph is unchanged
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_self_loop_equivalent(self):
        """
        Tests a case where the new orientation would create a self-loop equivalent (u -> v with v -> u path of length 0, but impossible).
        But since new edge is between different nodes, and if u == v, but function assumes u != v.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'A')])  # Self-loop
        
        u, v = 'A', 'A'  # Invalid, but test handling (though function assumes u != v)
        
        original_edges = list(graph.edges())
        
        self.assertFalse(has_directed_cycle(graph, (u, v)))
        
        # Ensure graph is unchanged
        self.assertEqual(set(graph.edges()), set(original_edges))

    def test_has_directed_cycle_empty_graph(self):
        """
        Tests an empty graph or isolated nodes; no cycle.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(['A', 'B'])
        graph.add_edges_from([('A', 'B'), ('B', 'A')])  # Undirected A <-> B
        
        u, v = 'A', 'B'
        opposite_edge = (v, u)
        
        original_edges = list(graph.edges())
        
        graph.remove_edge(*opposite_edge)
        
        self.assertFalse(has_directed_cycle(graph, (u, v)))
        
        graph.add_edge(*opposite_edge)
        
        self.assertEqual(set(graph.edges()), set(original_edges))

#==================================================
class TestHasVStructure(unittest.TestCase):
    """
    Test suite for the has_v_structure function.
    The function checks whether orienting u -> v would create a v-structure at v
    with any other parent z of v (i.e., z -> v and no edge between u and z).
    """

    def test_simple_v_structure(self):
        """
        Basic v-structure: u -> v, z -> v, no edge between u and z.
        Should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v')])
        
        self.assertTrue(has_v_structure(graph, ('u', 'v')))

    def test_v_structure_with_multiple_parents(self):
        """
        v has multiple parents, one forms v-structure with u.
        Should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z1', 'v'), ('z2', 'v')])
        
        self.assertTrue(has_v_structure(graph, ('u', 'v')))

    def test_no_v_structure_due_to_u_z_edge(self):
        """
        v-structure blocked by u <-> z (undirected edge).
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('u', 'z'), ('z', 'u')])
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_no_v_structure_due_to_directed_u_to_z(self):
        """
        v-structure blocked by u -> z.
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('u', 'z')])
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_no_v_structure_due_to_z_to_u(self):
        """
        v-structure blocked by z -> u.
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('z', 'u')])
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_reverse_edge_v_to_u(self):
        """
        Edge v -> u exists (undirected u <-> v), but u -> v is being tested.
        Should not create v-structure with z.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('v', 'u'), ('z', 'v')])
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_no_other_parents(self):
        """
        v has no other parents besides u.
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edge('u', 'v')
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_parent_is_child(self):
        """
        A parent z is also a child of v (z -> v and v -> z), so excluded.
        Should not count z as a parent for v-structure.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('v', 'z')])
        
        # z is a successor, so not in parents for v-structure check
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_complex_graph_with_v_structure(self):
        """
        Larger graph: v has parents u and z, no edge between u and z.
        Also has other edges.
        Should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('a', 'b'), ('b', 'c'),
            ('u', 'v'), ('z', 'v'),
            ('v', 'd'), ('d', 'e')
        ])
        
        self.assertTrue(has_v_structure(graph, ('u', 'v')))

    def test_complex_graph_without_v_structure(self):
        """
        v has parents u and z, but u <-> z connected.
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('u', 'v'), ('z', 'v'),
            ('u', 'z'), ('z', 'u')
        ])
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_self_loop_irrelevant(self):
        """
        Self-loop on v or u should not affect v-structure check.
        Should behave normally.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('z', 'v'), ('v', 'v')])
        
        self.assertTrue(has_v_structure(graph, ('u', 'v')))

    def test_isolated_new_edge(self):
        """
        Only the edge u -> v exists, no other parents.
        Should return False.
        """
        graph = nx.DiGraph()
        graph.add_edge('u', 'v')
        
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

    def test_multiple_potential_v_structures_one_valid(self):
        """
        v has three parents: u, z1, z2.
        u-z1 connected → blocked
        u-z2 not connected → valid v-structure
        Should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('u', 'v'), ('z1', 'v'), ('z2', 'v'),
            ('u', 'z1'), ('z1', 'u')  # Blocks u-z1
            # No edge u-z2 → allows v-structure
        ])
        
        self.assertTrue(has_v_structure(graph, ('u', 'v')))

    def test_undirected_edge_to_v_but_not_parent(self):
        """
        Edge w <-> v exists, but w is not a parent (no w -> v).
        Should not be considered a parent.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('u', 'v'), ('w', 'v'), ('v', 'w')])  # w <-> v
        
        # Only u is a parent (w is successor)
        self.assertFalse(has_v_structure(graph, ('u', 'v')))

#==================================================
class TestIsBadGraph(unittest.TestCase):
    """
    Test suite for the is_bad_graph function.
    This function checks whether a graph (or a proposed orientation) would
    violate DAG constraints: no directed cycles and no illegal v-structures.
    It is used both globally and per-edge during orientation propagation.
    """

    def test_bad_due_to_directed_cycle(self):
        """
        Graph contains a directed cycle → should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        self.assertTrue(is_bad_graph(graph))

    def test_bad_due_to_v_structure(self):
        """
        Graph contains an unshielded collider (v-structure) → should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('B', 'C')])  # A -> C <- B, no A-B edge
        self.assertTrue(is_bad_graph(graph))

    def test_good_graph_no_cycle_no_v_structure(self):
        """
        Valid DAG: chain A -> B -> C → should return False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        self.assertFalse(is_bad_graph(graph))

    def test_bad_with_both_cycle_and_v_structure(self):
        """
        Graph has both a cycle and a v-structure → should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'A'),        # Cycle
            ('C', 'D'), ('E', 'D')         # V-structure C -> D <- E
        ])
        self.assertTrue(is_bad_graph(graph))

    def test_undirected_edges_do_not_trigger_v_structure(self):
        """
        Undirected edges (bidirectional in DiGraph) should not trigger v-structure check.
        A <-> C, B <-> C → no v-structure (since edges are bidirectional).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('C', 'A'), ('B', 'C'), ('C', 'B')])
        self.assertFalse(is_bad_graph(graph))

    def test_v_structure_blocked_by_undirected_edge(self):
        """
        A -> C <- B, but A <-> B exists → v-structure is shielded → should be False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('B', 'C'), ('A', 'B'), ('B', 'A')])
        self.assertFalse(is_bad_graph(graph))

    def test_new_edge_creates_cycle(self):
        """
        Test is_bad_graph(graph, new_oriented_edge): orienting A -> B creates cycle.
        Existing: B -> C -> A, and A <-> B.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        new_edge = ('A', 'B')
        self.assertTrue(is_bad_graph(graph, new_edge))

    def test_new_edge_creates_v_structure(self):
        """
        Orienting U -> V creates v-structure: U -> V <- Z, no U-Z edge.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('U', 'V'), ('Z', 'V')])  # U <-> V, Z -> V
        new_edge = ('U', 'V')
        self.assertTrue(is_bad_graph(graph, new_edge))

    def test_new_edge_creates_no_problem(self):
        """
        Orienting U -> V: no cycle, no v-structure created.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('U', 'V'), ('V', 'U'), ('V', 'W')])
        new_edge = ('U', 'V')
        self.assertFalse(is_bad_graph(graph, new_edge))

    def test_no_undirected_edges_pure_dag_good(self):
        """
        Pure DAG with no undirected edges → check only for cycles and v-structures.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])  # Diamond, no v-structure
        self.assertFalse(is_bad_graph(graph))

    def test_no_undirected_edges_pure_dag_bad_cycle(self):
        """
        Pure DAG with cycle → should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        self.assertTrue(is_bad_graph(graph))

    def test_no_undirected_edges_pure_dag_bad_v_structure(self):
        """
        Pure DAG with v-structure → should return True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('B', 'C')])
        self.assertTrue(is_bad_graph(graph))

    def test_complex_mixed_graph_good(self):
        """
        Complex graph with undirected and directed edges, no violations.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),  # Undirected
            ('B', 'C'), ('C', 'B'),  # Undirected
            ('C', 'D')              # Directed
        ])
        # No cycle, no v-structure
        self.assertFalse(is_bad_graph(graph))

    def test_complex_mixed_graph_bad_cycle_via_new_edge(self):
        """
        Adding an edge would create a cycle.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'A')
        ])
        new_edge = ('A', 'B')
        self.assertTrue(is_bad_graph(graph, new_edge))

    def test_complex_mixed_graph_bad_v_structure_via_new_edge(self):
        """
        Adding U -> V creates v-structure with Z -> V.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('U', 'V'),
            ('Z', 'V'),
            ('V', 'W')
        ])
        new_edge = ('U', 'V')
        self.assertTrue(is_bad_graph(graph, new_edge))

    def test_is_bad_graph_called_without_edge_checks_all(self):
        """
        When called without new_edge, should scan all existing directed edges.
        If any causes cycle or v-structure → True.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'A'),  # Cycle
            ('X', 'Y'), ('Z', 'Y')              # V-structure
        ])
        self.assertTrue(is_bad_graph(graph))

    def test_is_bad_graph_called_without_edge_good_graph(self):
        """
        No violations in any edge → False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'C'),
            ('D', 'E'), ('E', 'F')
        ])
        self.assertFalse(is_bad_graph(graph))

    def test_empty_graph(self):
        """
        Empty graph or only nodes → should return False.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(['A', 'B', 'C'])
        self.assertFalse(is_bad_graph(graph))

    def test_single_undirected_edge(self):
        """
        One undirected edge → no cycle, no v-structure → False.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        self.assertFalse(is_bad_graph(graph))
#==================================================
class TestPropagateOrientations(unittest.TestCase):
    """
    Test suite for the propagate_orientations function.
    This function iteratively orients undirected edges in a chain graph to avoid cycles and v-structures,
    modifying the input graph in place by removing reverse edges, and returns the set of oriented edges.
    """

    def test_orient_to_avoid_cycle(self):
        """
        Tests orientation to prevent a cycle.
        Initial: A <-> B, B -> C, C -> A.
        Expects: Orient B -> A (to avoid A -> B -> C -> A cycle).
        Final graph: B -> A (no reverse), B -> C, C -> A.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'A')])
        
        original_undirected = set(find_undirected_edges(graph))
        expected_oriented = {('B', 'A')}
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(len(find_undirected_edges(graph)), 0)
        self.assertTrue(graph.has_edge('B', 'A'))
        self.assertFalse(graph.has_edge('A', 'B'))
        # Ensure no new bad structures created
        self.assertFalse(is_bad_graph(graph))

    def test_orient_to_avoid_v_structure(self):
        """
        Tests orientation to prevent a v-structure.
        Initial: A <-> C, B -> C.
        Expects: Orient C -> A (to avoid A -> C <- B v-structure).
        Final graph: C -> A (no reverse), B -> C.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'C'), ('C', 'A'), ('B', 'C')])
        
        original_undirected = set(find_undirected_edges(graph))
        expected_oriented = {('C', 'A')}
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(len(find_undirected_edges(graph)), 0)
        self.assertTrue(graph.has_edge('C', 'A'))
        self.assertFalse(graph.has_edge('A', 'C'))
        self.assertFalse(is_bad_graph(graph))

    def test_no_orientation_possible_simple_chain(self):
        """
        Simple undirected chain A <-> B <-> C: no forcing rules apply.
        Expects: No orientations, graph unchanged.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        
        original_edges = set(graph.edges())
        expected_oriented = set()
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(set(graph.edges()), original_edges)
        self.assertEqual(len(find_undirected_edges(graph)), 4)

    def test_already_directed_graph(self):
        """
        Fully directed DAG: A -> B -> C.
        Expects: No changes, empty set returned.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        
        original_edges = set(graph.edges())
        expected_oriented = set()
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(set(graph.edges()), original_edges)
        self.assertEqual(len(find_undirected_edges(graph)), 0)

    def test_multiple_orientations_propagation(self):
        """
        Complex case with propagation: Initial A <-> C, B -> C, C <-> D, A -> D.
        First: Orient C -> A (avoid v-structure at C).
        Then: Orient C -> D (now A -> D and C -> D, but check if forces; actually in code, it should orient based on bad checks).
        Expects: Both A-C and C-D oriented as C -> A and C -> D.
        Note: A -> D exists, so orienting A -> C would create cycle? Wait, no direct cycle, but v-structure forces C -> A.
        For C-D: After C -> A, now check C <-> D: orienting C -> D vs D -> C.
        D -> C would create v-structure if other parents, but here A -> D, no direct.
        In this setup, orienting D -> C: check if bad: parents of C would be B, D; B and D not connected? Assume no edge B-D, so v-structure B -> C <- D.
        Yes, so bad_v_u (D->C) true, bad_u_v (C->D) false → orient C -> D.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'C'), ('C', 'A'),  # Undirected A-C
            ('B', 'C'),              # B -> C
            ('C', 'D'), ('D', 'C'),  # Undirected C-D
            ('A', 'D')               # A -> D
        ])
        
        # Assume no B-D edge
        expected_oriented = {('C', 'A'), ('C', 'D')}
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(len(find_undirected_edges(graph)), 0)
        self.assertTrue(graph.has_edge('C', 'A'))
        self.assertFalse(graph.has_edge('A', 'C'))
        self.assertTrue(graph.has_edge('C', 'D'))
        self.assertFalse(graph.has_edge('D', 'C'))
        self.assertFalse(is_bad_graph(graph))

    def test_empty_or_no_undirected(self):
        """
        Graph with no undirected edges or empty.
        Expects: Empty set, no changes.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(['A', 'B'])
        # No edges
        
        original_edges = set(graph.edges())
        expected_oriented = set()
        
        result = propagate_orientations(graph)
        
        self.assertSetEqual(result, expected_oriented)
        self.assertEqual(set(graph.edges()), original_edges)
#==================================================
class TestGetChainComponents(unittest.TestCase):
    """
    Test suite for the get_chain_components function.
    This function extracts connected components from undirected edges in the graph,
    returning a list of DiGraph subgraphs (with bidirectional edges) for each component.
    Isolated nodes and directed-only parts are ignored.
    """

    def test_single_undirected_chain(self):
        """
        Tests a simple graph with a single undirected chain component.
        Graph: A <-> B <-> C.
        Expected: One component with nodes {A, B, C} and bidirectional edges.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        comp = components[0]
        self.assertCountEqual(comp.nodes(), ['A', 'B', 'C'])
        self.assertCountEqual(comp.edges(), [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])

    def test_multiple_disconnected_chains(self):
        """
        Tests a graph with two separate undirected chains.
        Graph: A <-> B, C <-> D.
        Expected: Two components: {A, B} and {C, D}.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 2)
        
        nodes_set = {frozenset(comp.nodes()) for comp in components}
        self.assertEqual(nodes_set, {frozenset(['A', 'B']), frozenset(['C', 'D'])})

    def test_mixed_graph_with_directed_edges(self):
        """
        Tests a graph with undirected and directed edges.
        Graph: A <-> B, C -> D.
        Expected: One component {A, B} (directed C->D ignored).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('C', 'D')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        comp = components[0]
        self.assertCountEqual(comp.nodes(), ['A', 'B'])
        self.assertCountEqual(comp.edges(), [('A', 'B'), ('B', 'A')])

    def test_graph_with_no_undirected_edges(self):
        """
        Tests a purely directed graph.
        Graph: A -> B -> C.
        Expected: Empty list (no undirected components).
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        components = get_chain_components(graph)
        self.assertEqual(len(components), 0)

    def test_isolated_nodes_and_mixed(self):
        """
        Tests a graph with isolated nodes and mixed edges.
        Graph: A <-> B, isolated C, D -> E.
        Expected: One component {A, B}.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('D', 'E')])
        graph.add_node('C')  # Isolated
        components = get_chain_components(graph)
        self.assertEqual(len(components), 1)
        
        comp = components[0]
        self.assertCountEqual(comp.nodes(), ['A', 'B'])
        self.assertCountEqual(comp.edges(), [('A', 'B'), ('B', 'A')])
#==================================================
from core.graph_utils import orient_random_restarts
from unittest.mock import patch, Mock

class TestOrientRandomRestarts(unittest.TestCase):
    """
    Test suite for the orient_random_restarts function.
    This function attempts up to MAX_ATTEMPTS random orientations of undirected edges
    until finding a valid DAG (no cycles, no illegal v-structures), returning the graph or None.
    Tests focus on validity, skeleton preservation, and behavior on simple cases.
    Due to randomness, some tests run multiple trials or use seeding.
    """

    def setUp(self):
        """Set a fixed seed for reproducibility in random tests."""
        random.seed(42)

    def test_returns_valid_dag_simple_chain(self):
        """
        Simple undirected chain A <-> B <-> C.
        Should return a valid DAG (e.g., A->B->C or C->B->A) quickly.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        
        dag = orient_random_restarts(graph)
        
        self.assertIsNotNone(dag)
        # Check skeleton preserved
        orig_skeleton = {tuple(sorted(e)) for e in graph.edges()}
        dag_skeleton = {tuple(sorted(e)) for e in dag.edges()}
        self.assertSetEqual(orig_skeleton, dag_skeleton)
        # No undirected edges left
        self.assertEqual(len(find_undirected_edges(dag)), 0)
        # Valid DAG: no bad graph
        self.assertFalse(is_bad_graph(dag))

    def test_returns_valid_dag_complex_chain(self):
        """
        More complex CPDAG with multiple undirected edges.
        Graph: A <-> B <-> C <-> D, with some directed constraints if needed.
        Should still find a valid topological order.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B'),
            ('C', 'D'), ('D', 'C')
        ])
        
        dag = orient_random_restarts(graph)
        
        self.assertIsNotNone(dag)
        orig_skeleton = {tuple(sorted(e)) for e in graph.edges()}
        dag_skeleton = {tuple(sorted(e)) for e in dag.edges()}
        self.assertSetEqual(orig_skeleton, dag_skeleton)
        self.assertEqual(len(find_undirected_edges(dag)), 0)
        self.assertFalse(is_bad_graph(dag))

    def test_returns_none_if_all_attempts_fail(self):
        """
        Hypothetical graph where no valid DAG exists (e.g., inherent cycle in skeleton).
        But since essential graphs should allow DAGs, mock is_bad_graph to always return True.
        Should exhaust attempts and return None.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])  # Simple, but mock bad
        
        with patch('core.graph_utils.is_bad_graph') as mock_bad:
            mock_bad.return_value = True  # Force all to be bad
            dag = orient_random_restarts(graph)
        
        self.assertIsNone(dag)

    def test_preserve_skeleton_and_no_undirected(self):
        """
        General check: If returns non-None, skeleton matches, no undirected left, valid.
        Run on simple graph with seed.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        
        dag = orient_random_restarts(graph)
        self.assertIsNotNone(dag)  # With seed, should succeed
        
        # Skeleton
        orig_skeleton = {tuple(sorted(e)) for e in graph.edges()}
        dag_skeleton = {tuple(sorted(e)) for e in dag.edges()}
        self.assertSetEqual(orig_skeleton, dag_skeleton)
        
        # No undirected
        self.assertEqual(len(find_undirected_edges(dag)), 0)
        
        # Valid
        self.assertFalse(is_bad_graph(dag))

    @patch('random.random')
    def test_deterministic_orientation_via_mock(self, mock_random):
        """
        Mock random to force specific orientations and check if valid.
        For simple A <-> B: mock to always choose A->B.
        Assume it's valid.
        """
        mock_random.side_effect = [0.4] * 100  # <0.5 → first option (a,b)
        
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        
        dag = orient_random_restarts(graph)
        self.assertIsNotNone(dag)
        self.assertTrue(dag.has_edge('A', 'B'))
        self.assertFalse(dag.has_edge('B', 'A'))
        self.assertFalse(is_bad_graph(dag))

class TestSampleDags(unittest.TestCase):
    """
    Test suite for the sample_dags function.
    This uses Parallel to generate n_samples DAGs via orient_random_restarts,
    filtering non-None. Tests validity and count.
    """

    def setUp(self):
        random.seed(42)

    def test_samples_correct_number_valid_dags(self):
        """
        Request 5 samples from simple chain.
        Should return list of len <=5, all valid DAGs.
        With seed, expect full 5.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')])
        n_samples = 5
        
        dags = sample_dags(graph, n_samples)
        
        self.assertEqual(len(dags), n_samples)  # With seed, all succeed
        for dag in dags:
            self.assertIsNotNone(dag)
            self.assertEqual(len(find_undirected_edges(dag)), 0)
            self.assertFalse(is_bad_graph(dag))
            # All share same skeleton
            orig_skeleton = {tuple(sorted(e)) for e in graph.edges()}
            dag_skeleton = {tuple(sorted(e)) for e in dag.edges()}
            self.assertSetEqual(orig_skeleton, dag_skeleton)

    def test_zero_samples_returns_empty(self):
        """
        n_samples=0 → empty list.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        
        dags = sample_dags(graph, 0)
        self.assertEqual(len(dags), 0)

    import networkx as nx
    from unittest.mock import patch, Mock
    from joblib import Parallel, delayed

    def test_some_failures_filters_none(self):
        """
        Mock orient_random_restarts to succeed only on even indices (0, 2, 4).
        We must patch Parallel to use 'threading' to ensure the side_effect's
        'nonlocal' counter is shared correctly.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'A')])
        n_samples = 5
        
        # This state logic is now correct, *if* we use threads
        success_calls = [True, False, True, False, True] 
        call_counter = 0

        def mock_orient_side_effect(g):
            nonlocal call_counter
            # Use a lock to make this thread-safe (good practice)
            # (Though for this specific test, it might work without it,
            # it's better to be explicit)
            try:
                is_successful = success_calls[call_counter]
            except IndexError:
                is_successful = False
            
            call_counter += 1
            
            if is_successful:
                mock_dag = Mock()
                mock_dag.edges.return_value = [('A', 'B')] 
                return mock_dag
            else:
                return None

        # --- THIS IS THE FIX ---
        
        # 1. Get the original Parallel class so we can wrap it
        from joblib import Parallel
        OriginalParallel = Parallel
        
        # 2. Define our wrapper that forces the threading backend
        #    This will be used *in place* of the real Parallel
        #    by the code inside sample_dags.
        def ThreadedParallelWrapper(*args, **kwargs):
            # Force the backend to 'threading'
            kwargs['backend'] = 'threading'
            # Call the *real* Parallel class with the modified kwargs
            return OriginalParallel(*args, **kwargs)
        # --- END FIX ---

        # Patch *both* the function we are testing AND Parallel
        # Note: 'core.graph_utils.Parallel' is where Parallel is *used*
        with patch('core.graph_utils.orient_random_restarts') as mock_orient, \
            patch('core.graph_utils.Parallel', new=ThreadedParallelWrapper): 
            
            mock_orient.side_effect = mock_orient_side_effect
                
            dags = sample_dags(graph, n_samples)
        
        # Assertions will now pass
        self.assertEqual(len(dags), 3) # The filter will find 2 Nones
        self.assertEqual(call_counter, n_samples) # The counter will be 5
        self.assertEqual(dags[0].edges(), [('A', 'B')])

    def test_parallel_works_on_complex(self):
        """
        Simple test: n=3 on longer chain, all valid.
        """
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B'),
            ('C', 'D'), ('D', 'C')
        ])
        n_samples = 3
        
        dags = sample_dags(graph, n_samples)
        
        self.assertEqual(len(dags), n_samples)
        for dag in dags:
            self.assertFalse(is_bad_graph(dag))
            self.assertEqual(len(find_undirected_edges(dag)), 0)
#==================================================
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
