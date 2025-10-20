import unittest
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pandas as pd

# Import your modules
from core.orienting_alg import orient_with_logic_and_experiments
from core.intervention import quasi_experiment, choose_intervention_variable
from core.graph_utils import find_undirected_edges
from core.intervention import silent_simulate
# You may need to uncomment if you use this library
# from torch import empty_strided 

class TestCausalDiscovery(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)  # For reproducible tests
    
    def create_simple_chain_model(self):
        """Create a simple V1 -> V2 -> V3 chain model."""
        model = BayesianNetwork([('V1', 'V2'), ('V2', 'V3')])
        cpds = [
            TabularCPD(variable='V1', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V2', variable_card=2, 
                       values=[[0.8, 0.3], [0.2, 0.7]],
                       evidence=['V1'], evidence_card=[2]),
            TabularCPD(variable='V3', variable_card=2,
                       values=[[0.9, 0.2], [0.1, 0.8]],
                       evidence=['V2'], evidence_card=[2])
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model
    
    def create_collider_model(self):
        """Create a collider structure: V1 -> V2 <- V3."""
        model = BayesianNetwork([('V1', 'V2'), ('V3', 'V2')])
        cpds = [
            TabularCPD(variable='V1', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V3', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V2', variable_card=2,
                       values=[[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]],
                       evidence=['V1', 'V3'], evidence_card=[2, 2])
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model
    
    def create_diamond_model(self):
        """Create a diamond structure: V1 -> V2 -> V4, V1 -> V3 -> V4."""
        model = BayesianNetwork([('V1', 'V2'), ('V1', 'V3'), ('V2', 'V4'), ('V3', 'V4')])
        cpds = [
            TabularCPD(variable='V1', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V2', variable_card=2,
                       values=[[0.8, 0.2], [0.2, 0.8]],
                       evidence=['V1'], evidence_card=[2]),
            TabularCPD(variable='V3', variable_card=2,
                       values=[[0.7, 0.3], [0.3, 0.7]],
                       evidence=['V1'], evidence_card=[2]),
            TabularCPD(variable='V4', variable_card=2,
                       values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
                       evidence=['V2', 'V3'], evidence_card=[2, 2])
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model

    def create_hub_model(self):
        """Create a hub structure (common cause): A -> B, A -> C, A -> D."""
        model = BayesianNetwork([('A', 'B'), ('A', 'C'), ('A', 'D')])
        cpds = [
            TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='B', variable_card=2, 
                       values=[[0.8, 0.2], [0.2, 0.8]], evidence=['A'], evidence_card=[2]),
            TabularCPD(variable='C', variable_card=2,
                       values=[[0.7, 0.3], [0.3, 0.7]], evidence=['A'], evidence_card=[2]),
            TabularCPD(variable='D', variable_card=2,
                       values=[[0.6, 0.4], [0.4, 0.6]], evidence=['A'], evidence_card=[2]),
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model

    def create_m_structure(self):
        """Create an M-structure: V1 -> V2 <- V3 -> V4."""
        model = BayesianNetwork([('V1', 'V2'), ('V3', 'V2'), ('V3', 'V4')])
        cpds = [
            TabularCPD(variable='V1', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V3', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='V2', variable_card=2,
                       values=[[0.9, 0.7, 0.3, 0.1], [0.1, 0.3, 0.7, 0.9]],
                       evidence=['V1', 'V3'], evidence_card=[2, 2]),
            TabularCPD(variable='V4', variable_card=2,
                       values=[[0.8, 0.2], [0.2, 0.8]], evidence=['V3'], evidence_card=[2]),
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model
    
    def create_mediated_common_cause_model(self):
        """Create A -> B -> C and A -> C."""
        model = BayesianNetwork([('A', 'B'), ('B', 'C'), ('A', 'C')])
        cpds = [
            TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='B', variable_card=2, values=[[0.2, 0.7], [0.8, 0.3]], 
                       evidence=['A'], evidence_card=[2]),
            TabularCPD(variable='C', variable_card=2,
                       values=[[0.1, 0.8, 0.6, 0.9], [0.9, 0.2, 0.4, 0.1]],
                       evidence=['B', 'A'], evidence_card=[2, 2])
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model

    def create_complex_intertwined_graph(self):
        """Create a more complex graph: A->B, B->C, A->D, D->E, C->E."""
        model = BayesianNetwork([('A', 'B'), ('B', 'C'), ('A', 'D'), ('D', 'E'), ('C', 'E')])
        cpds = [
            TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]]),
            TabularCPD(variable='B', variable_card=2, values=[[0.2, 0.7], [0.8, 0.3]], evidence=['A'], evidence_card=[2]),
            TabularCPD(variable='C', variable_card=2, values=[[0.1, 0.8], [0.9, 0.2]], evidence=['B'], evidence_card=[2]),
            TabularCPD(variable='D', variable_card=2, values=[[0.7, 0.4], [0.3, 0.6]], evidence=['A'], evidence_card=[2]),
            TabularCPD(variable='E', variable_card=2,
                       values=[[0.1, 0.2, 0.3, 0.9], [0.9, 0.8, 0.7, 0.1]],
                       evidence=['C', 'D'], evidence_card=[2, 2])
        ]
        model.add_cpds(*cpds)
        assert model.check_model()
        return model

    def test_orient_with_logic_diamond(self):
        """Test orientation on a diamond structure."""
        model = self.create_diamond_model()
        observational_data = silent_simulate(model, 5000, show_progress=False)
        
        # Essential graph for the diamond, all edges are undirected
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('V1', 'V2'), ('V2', 'V1'), 
            ('V1', 'V3'), ('V3', 'V1'), 
            ('V2', 'V4'),
            ('V3', 'V4')
        ])
        
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )

        self.assertTrue(num_experiments >= 0)
        true_edges = {('V1', 'V2'), ('V1', 'V3')}
        self.assertEqual(oriented_edges, true_edges)

    def test_quasi_experiment_output_shape(self):
        """Test that quasi_experiment returns data with correct shape."""
        model = self.create_simple_chain_model()
        n_samples = 1000
        exp_data = quasi_experiment(model, 'V2', samples=n_samples)
        
        self.assertIsInstance(exp_data, pd.DataFrame)
        self.assertEqual(exp_data.shape, (n_samples, 3))  # V1, V2, V3
        self.assertIn('V1', exp_data.columns)
        self.assertIn('V2', exp_data.columns)
        self.assertIn('V3', exp_data.columns) 
    
    def test_orient_with_logic_simple_chain(self):
        """Test orientation on a simple chain where PC should work well."""
        model = self.create_simple_chain_model()
        
        # Generate observational data
        observational_data = silent_simulate(model, 1000, show_progress=False)
        
        # Essential graph represented as an nx.DiGraph
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([('V1', 'V2'), ('V2', 'V1'), ('V2', 'V3'), ('V3', 'V2')])
        
        # Run the orientation algorithm
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=1000, aI1=0.01, aI2=0.01
        )
        
        self.assertIsInstance(oriented_edges, set)
        self.assertIsInstance(num_experiments, int)
        self.assertGreaterEqual(num_experiments, 0)
        
        true_edges = {('V1', 'V2'), ('V2', 'V3')}
        self.assertEqual(oriented_edges, true_edges)
    
    def test_orient_with_collider(self):
        """Test orientation on a collider that PC should orient."""
        model = self.create_collider_model()
        observational_data = silent_simulate(model, 1000, show_progress=False)
        
        # Essential graph with collider oriented
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([('V1', 'V2'), ('V3', 'V2')])
        
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=1000, aI1=0.01, aI2=0.01
        )
        
        self.assertTrue(num_experiments == 0)
    
    def test_orient_with_logic_different_strategies(self):
        """Test orientation on a diamond structure with different strategies."""
        model = self.create_diamond_model()
        observational_data = silent_simulate(model, 1000, show_progress=False)
        
        # Essential graph for the diamond, all edges are undirected
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('V1', 'V2'), ('V2', 'V1'), 
            ('V1', 'V3'), ('V3', 'V1'), 
            ('V2', 'V4'),
            ('V3', 'V4')
        ])
        
        strategies = ["entropy", "minimax", "greedy"]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
                    undirected_graph, observational_data, model, nI=1000, aI1=0.01, aI2=0.01, strategy=strategy
                )
                
                self.assertTrue(num_experiments >= 0)
                true_edges = {('V1', 'V2'), ('V1', 'V3')}
                self.assertEqual(oriented_edges, true_edges)
    
    def test_orient_with_logic_empty_graph(self):
        """Test orientation on an empty graph."""
        empty_graph = nx.DiGraph()
        model = self.create_simple_chain_model()
        observational_data = silent_simulate(model, 1000, show_progress=False)
        
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            empty_graph, observational_data, model
        )
        
        self.assertEqual(len(oriented_graph.edges()), 0)
        self.assertEqual(num_experiments, 0)
    
    def test_orient_with_logic_fully_directed(self):
        """Test orientation on a graph that's already fully directed."""
        model = self.create_simple_chain_model()
        observational_data = silent_simulate(model, 1000, show_progress=False)
        
        fully_directed_graph = nx.DiGraph(model.edges())
        
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            fully_directed_graph, observational_data, model
        )
        
        self.assertEqual(num_experiments, 0)
    
    def test_orient_with_logic_hub(self):
        """Test orientation on a common cause (hub) structure."""
        model = self.create_hub_model()
        observational_data = silent_simulate(model, 5000, show_progress=False)

        # Essential graph with A->B, A->C, A->D
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('A', 'C'), ('C', 'A'),
            ('A', 'D'), ('D', 'A')
        ])
        
        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )
        
        self.assertTrue(num_experiments >= 0)
        true_edges = {('A', 'B'), ('A', 'C'), ('A', 'D')}
        self.assertEqual(oriented_edges, true_edges)

    def test_orient_with_logic_m_structure(self):
        """Test orientation on an M-structure."""
        model = self.create_m_structure()
        observational_data = silent_simulate(model, 5000, show_progress=False)

        # Essential graph where the collider (V1->V2<-V3) is oriented, but V3->V4 is not
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('V1', 'V2'), 
            ('V3', 'V2'), 
            ('V3', 'V4'), ('V4', 'V3')
        ])

        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )
        
        self.assertTrue(num_experiments >= 0)
        true_edges = {('V3', 'V4')}
        self.assertEqual(oriented_edges, true_edges)

    def test_orient_with_logic_mediated_common_cause(self):
        """Test orientation on a mediated common cause structure (A->B->C, A->C)."""
        model = self.create_mediated_common_cause_model()
        observational_data = silent_simulate(model, 5000, show_progress=False)

        # Essential graph where the edges are ambiguous.
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('A', 'C'), ('C', 'A'),
            ('B', 'C'), ('C', 'B')
        ])

        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )
        
        self.assertTrue(num_experiments >= 0)
        true_edges = {('A', 'B'), ('B', 'C'), ('A', 'C')}
        self.assertEqual(oriented_edges, true_edges)

    def test_orient_with_logic_complex_intertwined_graph(self):
        """Test orientation on a complex, intertwined graph."""
        model = self.create_complex_intertwined_graph()
        observational_data = silent_simulate(model, 5000, show_progress=False)
        
        # Essential graph for this structure, with ambiguous edges
        undirected_graph = nx.DiGraph()
        undirected_graph.add_nodes_from(model.nodes())
        undirected_graph.add_edges_from([
            ('A', 'B'), ('B', 'A'),
            ('B', 'C'), ('C', 'B'),
            ('A', 'D'), ('D', 'A'),
            ('D', 'E'),
            ('C', 'E')
        ])

        oriented_graph, oriented_edges, num_experiments, _, _, _ = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )
        
        self.assertTrue(num_experiments >= 0)
        true_edges = {('A', 'B'), ('B', 'C'), ('A', 'D')}
        self.assertEqual(oriented_edges, true_edges)


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(verbosity=2)