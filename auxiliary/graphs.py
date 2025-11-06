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
from pathlib import Path

from core.orienting_alg import orient_with_logic_and_experiments
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

current_script_path = Path(__file__).resolve()
LOADED_MODELS_DIR = current_script_path.parent.parent / 'models'

from pgmpy.sampling import BayesianModelSampling
from pgmpy.readwrite import BIFReader as BIFReader

def create_model(model_name: str):
    model_path = LOADED_MODELS_DIR / f'{model_name}_model.pkl'

    if os.path.exists(model_path):
        print(f"Loading pre-trained {model_name.capitalize()} model from file...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    if model_name == "example":
        pgmpy_model = BayesianNetwork([
            ('V1', 'V2'),
            ('V1', 'V3'),
            ('V2', 'V3'),
            ('V2', 'V4'),
            ('V3', 'V4'),
            ('V4', 'V5')
        ])

        cpd_v1 = TabularCPD('V1', 2, [[0.5], [0.5]])
        cpd_v2 = TabularCPD('V2', 2, [[0.8, 0.2], [0.2, 0.8]], evidence=['V1'], evidence_card=[2])
        cpd_v3 = TabularCPD('V3', 2, [[0.9, 0.7, 0.7, 0.1], [0.1, 0.3, 0.3, 0.9]], evidence=['V1', 'V2'], evidence_card=[2, 2])
        cpd_v4 = TabularCPD('V4', 2, [[0.9, 0.6, 0.6, 0.2], [0.1, 0.4, 0.4, 0.8]], evidence=['V2', 'V3'], evidence_card=[2, 2])
        cpd_v5 = TabularCPD('V5', 2, [[0.7, 0.3], [0.3, 0.7]], evidence=['V4'], evidence_card=[2])

        pgmpy_model.add_cpds(cpd_v1, cpd_v2, cpd_v3, cpd_v4, cpd_v5)
        return pgmpy_model

    elif model_name == "example_2":
        pgmpy_model = BayesianNetwork([
            ('V1', 'V2'),
            ('V1', 'V3'),
            ('V2', 'V3'),
            ('V3', 'V4')
        ])

        cpd_v1 = TabularCPD('V1', 2, [[0.5], [0.5]])
        cpd_v2 = TabularCPD('V2', 2, [[0.8, 0.2], [0.2, 0.8]], evidence=['V1'], evidence_card=[2])
        cpd_v3 = TabularCPD('V3', 2, [[0.9, 0.7, 0.7, 0.1], [0.1, 0.3, 0.3, 0.9]], evidence=['V1', 'V2'], evidence_card=[2, 2])
        cpd_v4 = TabularCPD('V4', 2, [[0.7, 0.3], [0.3, 0.7]], evidence=['V3'], evidence_card=[2])

        pgmpy_model.add_cpds(cpd_v1, cpd_v2, cpd_v3, cpd_v4)
        return pgmpy_model
    
    model_name_map = {
      'munin_subnetwork_1': ('https://www.bnlearn.com/bnrepository/munin4/munin1.bif.gz', 'munin1.bif'), 
      'munin_full_network': ('https://www.bnlearn.com/bnrepository/munin/munin.bif.gz', 'munin.bif'), 
      'munin_subnetwork_2': ('https://www.bnlearn.com/bnrepository/munin4/munin2.bif.gz', 'munin2.bif'),
      'munin_subnetwork_3': ('https://www.bnlearn.com/bnrepository/munin4/munin3.bif.gz', 'munin3.bif'),
      'munin_subnetwork_4': ('https://www.bnlearn.com/bnrepository/munin4/munin4.bif.gz', 'munin4.bif')
    }
    if model_name in model_name_map:
        bif_url, bif_filename = model_name_map[model_name]
    else:
        bif_url = f'https://www.bnlearn.com/bnrepository/{model_name}/{model_name}.bif.gz'
        bif_filename = f'{model_name}.bif'
    if LOADED_MODELS_DIR:
        os.makedirs(LOADED_MODELS_DIR, exist_ok=True)
    bif_path = LOADED_MODELS_DIR / bif_filename

    if not os.path.exists(bif_path):
        print(f"Loading {model_name} via BIF download...")
        response = requests.get(bif_url)
        response.raise_for_status()
        print(f"Decompressing GZIP content to {bif_filename}...")
        decompressed_content = gzip.decompress(response.content)
        with open(bif_path, 'wb') as f:
            f.write(decompressed_content)
    
    reader = BIFReader(str(bif_path))
    pgmpy_model = reader.get_model()

    with open(model_path, 'wb') as f:
        pickle.dump(pgmpy_model, f)
    print(f"Saved {model_name} model to pickle.")
    return pgmpy_model

#'example', 'example_2', 'asia', 'cancer', 'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'hailfinder', 'hepar2', 'win95pts', 'andes', 'diabetes', 'link', 'munin_subnetwork_1', 'pathfinder', 'pigs', 'munin_full_network', 'munin_subnetwork_2', 'munin_subnetwork_3', 'munin_subnetwork_4'