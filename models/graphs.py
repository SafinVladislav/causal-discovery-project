# Third-party library imports
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def create_true_model():
    return create_example_model()

def create_example_model():
    model = BayesianNetwork([
        ('V1', 'V2'), ('V1', 'V3'), ('V2', 'V4'),
        ('V3', 'V2'), ('V3', 'V4'), ('V4', 'V5')
    ])
    cpds = [
        TabularCPD(variable='V1', variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable='V2', variable_card=2,
                   values=[[0.8, 0.7, 0.6, 0.1], [0.2, 0.3, 0.4, 0.9]],
                   evidence=['V1', 'V3'], evidence_card=[2, 2]),
        TabularCPD(variable='V3', variable_card=2,
                   values=[[0.9, 0.4], [0.1, 0.6]],
                   evidence=['V1'], evidence_card=[2]),
        TabularCPD(variable='V4', variable_card=2,
                   values=[[0.95, 0.6, 0.7, 0.05], [0.05, 0.4, 0.3, 0.95]],
                   evidence=['V2', 'V3'], evidence_card=[2, 2]),
        TabularCPD(variable='V5', variable_card=2,
                   values=[[0.99, 0.5], [0.01, 0.5]],
                   evidence=['V4'], evidence_card=[2])
    ]
    model.add_cpds(*cpds)
    assert model.check_model()
    return model