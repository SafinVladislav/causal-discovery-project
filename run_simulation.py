import contextlib
import io
import logging
import warnings
import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from IPython.utils import io
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, ranksums
import json
import os

from core.orienting_alg import orient_with_logic_and_experiments
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges, to_undirected_with_v_structures
from auxiliary.visualize import visualize_graphs
from auxiliary.data_generator import DataGenerator

import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent
RELATIVE_VIS_DIR = Path("visualizations")
OBS_SAMPLES = 5000

def run_simulation(data_generator, n_trials, nI, aI1, aI2, strategy):
    simulation_results = []

    print(f"\nStrategy: {strategy}")

    for _ in tqdm(range(n_trials), desc=f"nI={nI}, aI1={aI1}, aI2={aI2}"):
        obs_data = data_generator.observational(OBS_SAMPLES)
        essential_graph = data_generator.get_essential_graph()

        start_orient = time.time()
        oriented_graph, oriented, num_exp, fallback_perc = orient_with_logic_and_experiments(
            essential_graph, obs_data, data_generator, nI, aI1, aI2, strategy
        )
        end_orient = time.time()
        time_orient = end_orient - start_orient

        undirected_edges = find_undirected_edges(essential_graph)

        simulation_results.append({
            'nI': nI, 'aI1': aI1, 'aI2': aI2,
            'recall': data_generator.recall(oriented),
            'undir': len(undirected_edges) // 2,
            'precision': data_generator.precision(oriented),
            'avg_exp': num_exp,
            'time_orient': time_orient,
            'fallback_perc': fallback_perc,
            'oriented_graph': oriented_graph
        })

    return simulation_results

if __name__ == "__main__":
    #'example', 'example_2', 'asia', 'cancer', 'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'hailfinder', 'hepar2', 'win95pts', 'andes', 'diabetes', 'link', 'munin_subnetwork_1', 'pathfinder', 'pigs', 'munin_full_network', 'munin_subnetwork_2', 'munin_subnetwork_3', 'munin_subnetwork_4'
    #"minimax", "greedy", "entropy"
    trials = 1

    BOLD = '\033[1m'
    END = '\033[0m'

    for model_name in ['asia']:#'link', 'munin_subnetwork_1', 'pathfinder', 'pigs']:
        print(f"{BOLD}\n{model_name.upper()}{END}")
        
        data_generator = DataGenerator(model_name)
        for strategy in ['greedy', 'entropy', 'minimax']:
            results = run_simulation(data_generator, trials, 5000, 0.01, 0.01, strategy)

            print("--- Simulation Results ---")
            print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'recall':<8}{'undir':<8}{'prec':<8}{'Avg Exp':<10}{'Avg Time Orient':<15}{'Fallback perc':<15}")
            for res in results:
                print(f"{res['nI']:<8}{res['aI1']:<8.2f}{res['aI2']:<8.2f}{res['recall']:<8.3f}{res['undir']:<8}{res['precision']:<8.3f}{res['avg_exp']:<10.2f}{res['time_orient']:<15.4f}{res['fallback_perc']:<15.4f}")
            
            total_recall = sum(res['recall'] for res in results) / len(results) if results else 0.0
            total_precision = sum(res['precision'] for res in results) / len(results) if results else 0.0
            total_avg_exp = sum(res['avg_exp'] for res in results) / len(results) if results else 0.0
            total_time_orient = sum(res['time_orient'] for res in results) / len(results) if results else 0.0
            total_fallback_perc = sum(res['fallback_perc'] for res in results) / len(results) if results else 0.0
            
            print(f"{BOLD}{'Total':<8}{'-':<8}{'-':<8}{total_recall:<8.2f}{'-':<8}{total_precision:<8.2f}{total_avg_exp:<10.2f}{total_time_orient:<15.4f}{total_fallback_perc:<15.4f}{END}")

            #visualize_graphs(nx.DiGraph(model), results[0]['oriented_graph'], RELATIVE_VIS_DIR / f"{model_name}.png")