from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
import contextlib
import io
import logging
import warnings
import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, ranksums
import json
import os
from pgmpy.factors.discrete import TabularCPD

from auxiliary.graphs import create_model
from core.orienting_alg import orient_with_logic_and_experiments
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges, to_undirected_with_v_structures
from auxiliary.visualize import visualize_graphs

import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class DataGenerator:
    def __init__(self, model_name: str):
        self.model = create_model(model_name)

    def silent_simulate(self, model, samples: int):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sampler = BayesianModelSampling(model)
            obs_data = sampler.forward_sample(size=samples)
            return obs_data

    def observational(self, samples: int, variables_to_keep=None) -> pd.DataFrame:
        obs_data = self.silent_simulate(self.model, samples)
        if variables_to_keep != None:
            obs_data = obs_data[variables_to_keep]
        return obs_data

    def quasi_experiment(self, var, samples, variables_to_keep=None):
        intervened_model = self.model.copy()

        if variables_to_keep is not None:
            ancestral_graph_structure = intervened_model.get_ancestral_graph(variables_to_keep)
            anc_gr_str_nodes = ancestral_graph_structure.nodes()

            ancestral_model = BayesianNetwork(
                intervened_model.subgraph(anc_gr_str_nodes).edges()
            )

            cpds_to_add = []
            for node in anc_gr_str_nodes:
                cpd = intervened_model.get_cpds(node)
                if cpd is not None:
                    cpds_to_add.append(cpd)

            ancestral_model.add_cpds(*cpds_to_add)
            intervened_model = ancestral_model

        old_cpd = intervened_model.get_cpds(var)
        values = old_cpd.values.reshape(old_cpd.cardinality[0], -1)
        
        #That's how we conduct an experiment
        new_values = np.zeros_like(values)
        max_indices = np.argmin(values, axis=0)
        new_values[max_indices, np.arange(values.shape[1])] = 1.0

        state_names = {var: old_cpd.state_names[var]}
        evidence = old_cpd.variables[1:] if len(old_cpd.variables) > 1 else None
        evidence_card = old_cpd.cardinality[1:] if evidence else None
        if evidence:
            for ev_var in evidence:
                ev_cpd = intervened_model.get_cpds(ev_var)
                if ev_cpd:
                    state_names[ev_var] = ev_cpd.state_names[ev_var]
                else:
                    for node in intervened_model.nodes():
                        node_cpd = intervened_model.get_cpds(node)
                        if node_cpd and ev_var in node_cpd.state_names:
                            state_names[ev_var] = node_cpd.state_names[ev_var]
                            break

        new_cpd = TabularCPD(
            variable=var,
            variable_card=old_cpd.cardinality[0],
            values=new_values,
            evidence=evidence,
            evidence_card=evidence_card,
            state_names=state_names
        )
        
        intervened_model.remove_cpds(old_cpd)
        intervened_model.add_cpds(new_cpd)
        
        df = self.silent_simulate(intervened_model, samples)
        
        if variables_to_keep is not None:
            df = df[variables_to_keep]
        
        return df
    
    def get_essential_graph(self):
        true_graph = nx.DiGraph(self.model)
        true_edges = set(true_graph.edges())
        return to_undirected_with_v_structures(true_graph)

    def recall(self, oriented):
        undirected_edges = find_undirected_edges(self.get_essential_graph())
        return len(oriented) / (len(undirected_edges) / 2) if len(undirected_edges) > 0 else 1.0

    def precision(self, oriented):
        true_graph = nx.DiGraph(self.model)
        true_edges = set(true_graph.edges())
        return len(oriented & true_edges) / len(oriented) if len(oriented) > 0 else 1.0
