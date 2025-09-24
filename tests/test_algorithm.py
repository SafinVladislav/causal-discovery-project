import unittest
import numpy as np
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pandas as pd

# Import your modules
from core.algorithm import orient_with_logic_and_experiments
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
    
    def create_triangle_model(self):
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
        
        oriented_graph, oriented_edges, num_experiments = orient_with_logic_and_experiments(
            undirected_graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01
        )
        print(oriented_edges)

        self.assertTrue(num_experiments >= 0)
        true_edges = {('V1', 'V2'), ('V1', 'V3')}
        self.assertEqual(oriented_edges, true_edges)
    

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(verbosity=2)