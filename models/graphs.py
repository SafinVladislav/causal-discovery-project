# Standard library imports
import contextlib
import io as std_io  # Use standard io for BytesIO
import logging
import os
import pickle
import warnings
import gzip

# Third-party library imports
import networkx as nx
import bnlearn as bn
import pandas as pd
import requests
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
from pgmpy.readwrite import BIFReader
from tqdm import tqdm
from IPython.utils import io  # For output capture
from pgmpy.estimators import PC

# Local project imports
from core.algorithm import orient_with_logic_and_experiments
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

from pathlib import Path
import os

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent.parent
RELATIVE_LOADED_DIR = Path("loaded_models")

def create_sprinkler_model():
    model_path = PROJECT_ROOT / RELATIVE_LOADED_DIR / 'sprinkler_model.pkl'

    if os.path.exists(model_path):
        print("Loading pre-trained Sprinkler model from file...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Import the 'sprinkler' dataset and DAG from bnlearn
        df = bn.import_example('sprinkler')
        model_bn = bn.import_DAG('sprinkler', verbose=0)
        
        # Extract edges from the adjacency matrix
        adjmat = model_bn['adjmat']  # This is a pandas DataFrame
        edges = [(source, target) for source, row in adjmat.iterrows() for target, value in row.items() if value]
        
        # Create and fit the pgmpy BayesianNetwork model
        pgmpy_model = BayesianNetwork(edges)
        pgmpy_model.fit(df, estimator=BayesianEstimator)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(pgmpy_model, f)

        return pgmpy_model

def create_asia_model():
    model_path = PROJECT_ROOT / RELATIVE_LOADED_DIR / 'asia_model.pkl'

    if os.path.exists(model_path):
        print("Loading pre-trained Asia model from file...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Import the 'asia' dataset and DAG
        df = bn.import_example('asia')
        model_bn = bn.import_DAG('asia', verbose=0)
        
        # Extract edges from the adjacency matrix
        adjmat = model_bn['adjmat']  # This is a pandas DataFrame
        edges = [(source, target) for source, row in adjmat.iterrows() for target, value in row.items() if value]
        
        # Create and fit the pgmpy BayesianNetwork model
        pgmpy_model = BayesianNetwork(edges)
        pgmpy_model.fit(df, estimator=BayesianEstimator)
        
        with open(model_path, 'wb') as f:
            pickle.dump(pgmpy_model, f)

        return pgmpy_model

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

def create_true_model():
    return create_asia_model()
    #return create_sprinkler_model()