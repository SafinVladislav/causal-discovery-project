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

"""
This class separates the orientation algorithm from ground-truth models used 
for data generation. Currently uses bnlearn models; in future, replace with 
simulations of real-world processes.
"""
class DataGenerator:
    """
    Models are uploaded from the Web and stored permanently in 'models' folder.
    List of all acceptable names: 'example_1', 'example_2', 'asia', 'cancer', 
    'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 
    'mildew', 'water', 'hailfinder', 'hepar2', 'win95pts', 'andes', 'diabetes', 
    'link', 'munin_subnetwork_1', 'pathfinder', 'pigs', 'munin_full_network', 
    'munin_subnetwork_2', 'munin_subnetwork_3', 'munin_subnetwork_4'
    """
    def __init__(self, model_name: str):
        self.model = create_model(model_name)

    """
    This function is meant for generating a certain number of rows
    for a dataset based on a model (either original or modified one).
    """
    def silent_simulate(self, model, samples: int):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sampler = BayesianModelSampling(model)
            obs_data = sampler.forward_sample(size=samples)
            return obs_data

    """
    Generating data based on our original model.
    """
    def observational(self, samples: int) -> pd.DataFrame:
        obs_data = self.silent_simulate(self.model, samples)
        return obs_data

    """
    Here we change conditional distribution of some variable
    and generate data based on our modified model.
    """
    def quasi_experiment(self, var, samples):
        intervened_model = self.model.copy()

        old_cpd = intervened_model.get_cpds(var)
        values = old_cpd.values.reshape(old_cpd.cardinality[0], -1)
        
        #Conduct a quasi-experiment by making the CPD deterministic: 
        #set probability 1 to the most likely state from the original CPD, 
        #simulating a hard intervention.
        new_values = np.zeros_like(values)
        max_indices = np.argmax(values, axis=0)
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
        
        return df
    
    """
    Getting a simplified and correct essential graph (faster than PC - good for testing).
    """
    def get_essential_graph(self):
        true_graph = nx.DiGraph(self.model)
        return to_undirected_with_v_structures(true_graph)

    """
    Calculates the fraction of undirected edges 
    (each stored as two directed edges in the graph) 
    that were successfully oriented.
    """
    def recall(self, oriented, essential_graph):
        undirected_edges = find_undirected_edges(essential_graph)
        return len(oriented & set(undirected_edges)) / (len(undirected_edges) / 2) if len(undirected_edges) > 0 else 1.0

    """
    Percent of correctly oriented edges among all oriented.
    """
    def precision(self, oriented):
        true_graph = nx.DiGraph(self.model)
        true_edges = set(true_graph.edges())
        return len(oriented & true_edges) / len(oriented) if len(oriented) > 0 else 1.0

    """
    Visualizing three graphs - original, outputted by PC (essential) 
    and final one.
    """
    def visualize(self, pc_essential_graph, oriented_graph, pic_dir):
        visualize_graphs(nx.DiGraph(self.model), pc_essential_graph, oriented_graph, pic_dir)