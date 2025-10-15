import contextlib
import io as std_io
import logging
import os
import pickle
import warnings
import gzip
import bnlearn as bn

import networkx as nx
import pandas as pd
import requests
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
from pgmpy.readwrite import BIFReader
from tqdm import tqdm
from IPython.utils import io
from pgmpy.estimators import PC

from core.algorithm import orient_with_logic_and_experiments
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

from pathlib import Path

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent.parent
RELATIVE_LOADED_DIR = Path("loaded_models")

from pgmpy.sampling import BayesianModelSampling
from pgmpy.readwrite import BIFReader as BIFReader

BNLEARN_SUPPORTED = {'alarm', 'andes', 'asia', 'sachs', 'sprinkler'}

BIF_BASE_URL = 'https://www.bnlearn.com/bnrepository/discrete-{size}/{name}-medium.bif'

def create_model(model_name: str, n_samples: int = 10000):
    """
    Generalized model loader: Uses bnlearn if supported, else downloads BIF and loads.
    
    Args:
        model_name (str): e.g., 'alarm', 'child', 'insurance'.
        n_samples (int): Samples to simulate/fit (default 1000 for consistency).
    
    Returns:
        pgmpy BayesianNetwork (fitted and pickled).
    """
    model_path = PROJECT_ROOT / RELATIVE_LOADED_DIR / f'{model_name}_model.pkl'

    if os.path.exists(model_path):
        print(f"Loading pre-trained {model_name.capitalize()} model from file...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    if model_name in BNLEARN_SUPPORTED:
        print(f"Loading {model_name} via bnlearn...")
        df = bn.import_example(model_name)
        model_bn = bn.import_DAG(model_name, verbose=0)
        
        adjmat = model_bn['adjmat']
        edges = [(source, target) for source, row in adjmat.iterrows() 
                 for target, value in row.items() if value]
        
        pgmpy_model = BayesianNetwork(edges)
        pgmpy_model.fit(df, estimator=BayesianEstimator)
    
    else:
        size_map = {
            'child': ('https://www.bnlearn.com/bnrepository/child/child.bif.gz', 'child.bif'),
            'insurance': ('https://www.bnlearn.com/bnrepository/insurance/insurance.bif.gz', 'insurance.bif'),
            'hailfinder': ('https://www.bnlearn.com/bnrepository/hailfinder/hailfinder.bif.gz', 'hailfinder.bif'),
            'win95pts': ('https://www.bnlearn.com/bnrepository/win95pts/win95pts.bif.gz', 'win95pts.bif'),
            'pathfinder': ('https://www.bnlearn.com/bnrepository/pathfinder/pathfinder.bif.gz', 'pathfinder.bif')
        }
        if model_name not in size_map:
            raise ValueError(f"Unsupported model '{model_name}'. Add to size_map or use bnlearn-supported.")
        
        bif_url, bif_filename = size_map[model_name]
        bif_path = PROJECT_ROOT / RELATIVE_LOADED_DIR / bif_filename
        if not os.path.exists(bif_path):
            print(f"Loading {model_name} via BIF download...")
            # 1. Download the GZIP content
            response = requests.get(bif_url)
            response.raise_for_status()
            # Define the path for the DECOMPRESSED BIF file
            # 2. Decompress and save the BIF file
            print(f"Decompressing GZIP content to {bif_filename}...")
            # response.content contains the gzip bytes
            decompressed_content = gzip.decompress(response.content)
            
            with open(bif_path, 'wb') as f:
                f.write(decompressed_content) # Write the decompressed bytes
        
        # 3. Read the now plain-text BIF file
        reader = BIFReader(str(bif_path))
        pgmpy_model = reader.get_model()

    with open(model_path, 'wb') as f:
        pickle.dump(pgmpy_model, f)
    print(f"Saved {model_name} model to pickle.")
    return pgmpy_model

#'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'sprinkler', 'child', 'insurance', 'hailfinder', 'win95pts'