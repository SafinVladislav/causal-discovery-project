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
from pgmpy.estimators import PC

from core.orienting_alg import orient_with_logic_and_experiments
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges, to_undirected_with_v_structures
from auxiliary.data_generator import DataGenerator

import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent
RELATIVE_VIS_DIR = Path("visualizations")
OBS_SAMPLES = 20000

def PC_quality(true_essential, pc_essential):
    true_skeleton = true_essential.to_undirected()
    pc_skeleton = pc_essential.to_undirected()
    
    true_undir = {tuple(sorted(edge)) for edge in true_skeleton.edges()}
    pc_undir = {tuple(sorted(edge)) for edge in pc_skeleton.edges()}
    skeleton_additions = len(pc_undir - true_undir)
    skeleton_deletions = len(true_undir - pc_undir)

    true_dir = {edge for edge in true_essential.edges() if not true_essential.has_edge(edge[1], edge[0])}
    pc_dir = {edge for edge in pc_essential.edges() if not pc_essential.has_edge(edge[1], edge[0])}

    additions = len(pc_dir - true_dir)
    deletions = len(true_dir - pc_dir)

    return skeleton_additions + skeleton_deletions + additions + deletions

def tune_pc_parameters(data_generator, n_trials_tune, n_candidates, sl_candidates):
    ess_graph = data_generator.get_essential_graph()

    results = []
    total_combinations = len(n_candidates) * len(sl_candidates)
    with tqdm(total=total_combinations, desc="Tuning PC parameters") as pbar:
        for n in n_candidates:
            for sl in sl_candidates:
                quality_sum = 0
                for _ in range(n_trials_tune):
                    obs_data = data_generator.observational(n)
                    pc_estimator = PC(data=obs_data)
                    with io.capture_output():
                        estimated_graph = nx.DiGraph(pc_estimator.estimate(
                            variant='stable', ci_test='chi_square',
                            significance_level=sl, return_type="cpdag"
                        ))
                    quality = PC_quality(ess_graph, estimated_graph)
                    quality_sum += quality
                avg_quality = quality_sum / n_trials_tune if n_trials_tune > 0 else 0
                results.append({'n': n, 'sl': sl, 'avg_quality': avg_quality})
                pbar.update(1)
    
    sorted_results = sorted(results, key=lambda x: (x['avg_quality'], x['n'], x['sl']))
    
    top_k = 3
    print(f"\nTop {min(top_k, len(sorted_results))} PC parameter sets:")
    print(f"{'Rank':<6}{'N':<8}{'SL':<8}{'Avg Quality':<10}")
    print("-" * 32)
    for i, res in enumerate(sorted_results[:top_k], 1):
        print(f"{i:<6}{res['n']:<8}{res['sl']:<8.4f}{res['avg_quality']:<10.3f}")

    best = sorted_results[0]
    print(f"\nReturning best PC parameters: N={best['n']}, SL={best['sl']} with average quality {best['avg_quality']:.3f}")
    return best['n'], best['sl']

def run_simulation(data_generator, trials, nIs, aI1s, aI2s, strategy, pc_sl):
    simulation_results = []

    print(f"\nStrategy: {strategy}")

    total_combinations = len(nIs) * len(aI1s) * len(aI2s) * trials
    with tqdm(total=total_combinations, desc="Progress") as pbar_outer:
        for nI in nIs:
            for aI1 in aI1s:
                for aI2 in aI2s:
                    perf = {
                        'time_pc': 0.0,
                        'num_correct_essential_graphs': 0,
                        'time_orient': 0.0,
                        'undir': 0.0,
                        'oriented': 0.0,
                        True:  {'experiments': 0.0, 'recall': 0.0, 'precision': 0.0},
                        False: {'experiments': 0.0, 'recall': 0.0, 'precision': 0.0}
                    }
                    some_oriented_graph = None
                    some_essential_graph = None

                    for _ in range(trials):
                        obs_data = data_generator.observational(OBS_SAMPLES)

                        start_pc = time.time()
                        pc_estimator = PC(data=obs_data)
                        with io.capture_output():
                            essential_graph = data_generator.get_essential_graph()
                            """essential_graph = nx.DiGraph(pc_estimator.estimate(
                                variant='stable', ci_test='chi_square',
                                significance_level=pc_sl, return_type="cpdag"
                            ))"""

                        end_pc = time.time()
                        time_pc = end_pc - start_pc
                        some_essential_graph = essential_graph
                        perf['undir'] += len(find_undirected_edges(essential_graph)) / 2
                        perf['time_pc'] += time_pc

                        is_correct = check_if_estimated_correctly(essential_graph, data_generator.get_essential_graph())
                        perf['num_correct_essential_graphs'] += is_correct

                        start_orient = time.time()
                        oriented_graph, oriented, num_exp, fallback_perc = orient_with_logic_and_experiments(
                            essential_graph, obs_data, data_generator, nI, aI1, aI2, strategy
                        )
                        end_orient = time.time()
                        time_orient = end_orient - start_orient
                        some_oriented_graph = oriented_graph
                        perf['time_orient'] += time_orient
                        perf['oriented'] += len(oriented)

                        perf[is_correct]['experiments'] += num_exp
                        perf[is_correct]['recall'] += data_generator.recall(essential_graph, oriented)
                        perf[is_correct]['precision'] += data_generator.precision(oriented)

                        pbar_outer.update(1)

                    corr_perc = perf['num_correct_essential_graphs'] / trials if trials > 0 else float('-inf')
                    undir = perf['undir'] / trials if trials > 0 else float('-inf')
                    oriented = perf['oriented'] / trials if trials > 0 else float('-inf')
                    exp = (perf[0]['experiments'] + perf[1]['experiments']) / trials if trials > 0 else -float('-inf')
                    exp_corr = perf[1]['experiments'] / perf['num_correct_essential_graphs'] if perf['num_correct_essential_graphs'] > 0 else float('-inf')
                    time_pc = perf['time_pc'] / trials if trials > 0 else float('-inf')
                    time_orient = perf['time_orient'] / trials if trials > 0 else float('-inf')
                    recall = (perf[0]['recall'] + perf[1]['recall']) / trials if trials > 0 else float('-inf')
                    recall_corr = perf[1]['recall'] / perf['num_correct_essential_graphs'] if perf['num_correct_essential_graphs'] > 0 else float('-inf')
                    prec = (perf[0]['precision'] + perf[1]['precision']) / trials if trials > 0 else float('-inf')
                    prec_corr = perf[1]['precision'] / perf['num_correct_essential_graphs'] if perf['num_correct_essential_graphs'] > 0 else float('-inf')

                    simulation_results.append({
                        'nI': nI, 'aI1': aI1, 'aI2': aI2,
                        'corr_perc': corr_perc,
                        'undir': undir,
                        'oriented': oriented,
                        'exp': exp,
                        'exp_corr': exp_corr,
                        'time_pc': time_pc,
                        'time_orient': time_orient,
                        'recall': recall,
                        'recall_corr': recall_corr,
                        'prec': prec,
                        'prec_corr': prec_corr,
                        'essential_graph': some_essential_graph,
                        'oriented_graph': some_oriented_graph
                    })

    return simulation_results

if __name__ == "__main__":
    #'example', 'example_2', 'asia', 'cancer', 'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'hailfinder', 'hepar2', 'win95pts', 'andes', 'munin_subnetwork_1'
    #slow: 'diabetes', 'link', 'pathfinder', 'munin_full_network', 'munin_subnetwork_2', 'munin_subnetwork_3', 'munin_subnetwork_4'
    #"minimax", "greedy", "entropy"
    trials = 1

    BOLD = '\033[1m'
    END = '\033[0m'

    for model_name in ['munin_subnetwork_1']:
        print(f"{BOLD}\n{model_name.upper()}{END}")
        
        data_generator = DataGenerator(model_name)

        #tune_pc_parameters(data_generator, 3, [500, 1000, 2000, 5000, 10000, 20000, 50000], [0.001, 0.005, 0.01, 0.05, 0.1, 0.3])
        for strategy in ["greedy"]:#, "entropy", "minimax"]:
            results = run_simulation(data_generator, trials, [3000], [0.01], [0.01], strategy, 0.005)

            print("--- Simulation Results ---")
            print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'corr_perc':<12}{'undir':<8}{'oriented':<12}{'recall':<8}{'recall_corr':<15}"
                  f"{'prec':<8}{'prec_corr':<12}{'exp':<10}{'exp_corr':<12}"
                  f"{'time_pc':<12}{'time_orient':<15}")

            for res in results:
                print(f"{res['nI']:<8}"
                      f"{res['aI1']:<8.2f}"
                      f"{res['aI2']:<8.2f}"
                      f"{res['corr_perc']:<12.2f}"
                      f"{res['undir']:<8.2f}"
                      f"{res['oriented']:<12.2f}"
                      f"{res['recall']:<8.3f}"
                      f"{res['recall_corr']:<15.3f}"
                      f"{res['prec']:<8.3f}"
                      f"{res['prec_corr']:<12.3f}"
                      f"{res['exp']:<10.2f}"
                      f"{res['exp_corr']:<12.2f}"
                      f"{res['time_pc']:<12.4f}"
                      f"{res['time_orient']:<15.4f}"
                      )

            data_generator.visualize(results[0]['essential_graph'], results[0]['oriented_graph'], RELATIVE_VIS_DIR / f"{model_name}")
        