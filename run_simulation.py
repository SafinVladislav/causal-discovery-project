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
import random
import uuid
from pgmpy.estimators import PC
import shutil

from core.orienting_alg import orient_with_logic_and_experiments
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges, to_undirected_with_v_structures
from auxiliary.data_generator import DataGenerator

import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent
RELATIVE_VIS_DIR = PROJECT_ROOT / Path("visualizations")

"""
This function is meant for choosing best PC parameters.
We use simple metrics of difference between two essential graphs.
"""
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

"""
Tunes PC algorithm parameters by measuring skeleton and orientation differences 
(additions/deletions) between true and estimated essential graphs.
"""
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

OBS_SAMPLES = 20000
PC_SIG_LEV = 0.1

"""
Running essential graph generation and orientation for each set of parameters 'trials' 
times.
"""
def run_simulation(data_generator, trials, nIs, aI1s, aI2s, strategy, image=False, video=False):
    simulation_results = []

    print(f"\nStrategy: {strategy}")

    total_combinations = len(nIs) * len(aI1s) * len(aI2s) * trials
    if (image or video) and (total_combinations != 1):
        print("For visualizations only one set of parameters is accepted!")
        return None

    img_video_path = RELATIVE_VIS_DIR / f"{model_name}_{strategy}"
    if img_video_path.exists():
        shutil.rmtree(img_video_path, ignore_errors=True)
    img_video_path.mkdir(parents=True, exist_ok=True)

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
                        'fallback': 0.0,
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
                            """
                            PC algorithm is rather slow, so simplified essential
                            graph may be good for testing.
                            """
                            #essential_graph = data_generator.get_essential_graph()
                            essential_graph = nx.DiGraph(pc_estimator.estimate(
                                variant='stable', ci_test='chi_square',
                                significance_level=PC_SIG_LEV, return_type="cpdag"
                            ))

                        end_pc = time.time()
                        time_pc = end_pc - start_pc
                        some_essential_graph = essential_graph
                        perf['undir'] += len(find_undirected_edges(essential_graph)) / 2
                        perf['time_pc'] += time_pc

                        is_correct = check_if_estimated_correctly(essential_graph, data_generator.get_essential_graph())
                        perf['num_correct_essential_graphs'] += is_correct

                        start_orient = time.time()
                        oriented_graph, oriented, num_exp, fallback_perc = orient_with_logic_and_experiments(
                            essential_graph, obs_data, data_generator, nI, aI1, aI2, strategy, vis_dir=(img_video_path if video else None)
                        )
                        end_orient = time.time()
                        time_orient = end_orient - start_orient
                        some_oriented_graph = oriented_graph
                        perf['time_orient'] += time_orient
                        perf['oriented'] += len(oriented)
                        perf['fallback'] += fallback_perc

                        perf[is_correct]['experiments'] += num_exp
                        perf[is_correct]['recall'] += data_generator.recall(oriented, essential_graph)
                        perf[is_correct]['precision'] += data_generator.precision(oriented)

                        if image:
                            data_generator.visualize(essential_graph, oriented_graph, img_video_path)

                        pbar_outer.update(1)

                    corr_perc = perf['num_correct_essential_graphs'] / trials if trials > 0 else float('-inf')
                    undir = perf['undir'] / trials if trials > 0 else float('-inf')
                    oriented = perf['oriented'] / trials if trials > 0 else float('-inf')
                    exp = (perf[0]['experiments'] + perf[1]['experiments']) / trials if trials > 0 else -float('-inf')
                    exp_corr = perf[1]['experiments'] / perf['num_correct_essential_graphs'] if perf['num_correct_essential_graphs'] > 0 else float('-inf')
                    fallback = perf['fallback'] / trials if trials > 0 else float('-inf')
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
                        'fallback': fallback,
                        'time_pc': time_pc,
                        'time_orient': time_orient,
                        'recall': recall,
                        'recall_corr': recall_corr,
                        'prec': prec,
                        'prec_corr': prec_corr,
                        'F1': 2 * recall * prec / (recall + prec) if (recall + prec) > 0 else 0,
                        'F1_corr': 2 * recall_corr * prec_corr / (recall_corr + prec_corr) if (recall_corr + prec_corr) > 0 else 0,
                        'essential_graph': some_essential_graph,
                        'oriented_graph': some_oriented_graph
                    })

    return simulation_results

if __name__ == "__main__":
    # 'example_1', 'example_2', 'asia', 'cancer', 'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'hailfinder', 'hepar2', 'win95pts', 'andes', 'munin_subnetwork_1'
    # slow: 'diabetes', 'link', 'pathfinder', 'munin_full_network', 'munin_subnetwork_2', 'munin_subnetwork_3', 'munin_subnetwork_4'
    # "greedy", "entropy", "minimax"

    trials = 1

    BOLD = '\033[1m'
    END = '\033[0m'

    for model_name in ['example_1', 'example_2']:
        print(f"{BOLD}\n{model_name.upper()}{END}")
        
        data_generator = DataGenerator(model_name)

        #tune_pc_parameters(data_generator, 3, [500, 1000, 2000, 5000, 10000, 20000, 50000], [0.001, 0.005, 0.01, 0.05, 0.1, 0.3])
        for strategy in ["greedy", "entropy", "minimax"]:
            results = run_simulation(data_generator, trials, [10000], [0.05], [0.2], strategy)#, image=True, video=True)

            print("--- Simulation Results ---")

            for res in results:
                print(f"nI - {res['nI']}; "
                      f"aI1 - {res['aI1']:.3f}; "
                      f"aI2 - {res['aI2']:.3f}; "
                      f"corr_perc - {res['corr_perc']:.3f}; "
                      f"undir - {res['undir']:.3f}; "
                      f"oriented - {res['oriented']:.3f}; "
                      f"recall - {res['recall']:.3f}; "
                      f"recall_corr - {res['recall_corr']:.3f}; "
                      f"prec - {res['prec']:.3f}; "
                      f"prec_corr - {res['prec_corr']:.3f}; "
                      f"F1 - {res['F1']:.3f}; "
                      f"F1_corr - {res['F1_corr']:.3f}; "
                      f"exp - {res['exp']:.2f}; "
                      f"exp_corr - {res['exp_corr']:.2f}; "
                      f"fallback - {res['fallback']:.2f}; "
                      f"time_pc - {res['time_pc']:.2f}; "
                      f"time_orient - {res['time_orient']:.2f}"
                      )
        