# Standard library imports
import contextlib
import io
import logging
import warnings
import time

# Third-party library imports
import networkx as nx
import pandas as pd
from tqdm import tqdm
from IPython.utils import io

# Local project imports
from core.orienting_alg import orient_with_logic_and_experiments
from models.graphs import create_model
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges, to_undirected_with_v_structures

from pathlib import Path

import logging
# Suppress pgmpy warnings by setting the logging level to ERROR
logging.getLogger("pgmpy").setLevel(logging.ERROR)

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent
RELATIVE_VIS_DIR = Path("visualizations")
OBS_SAMPLES = 20000

def run_simulation(model, n_trials, nI_values, aI1_values, aI2_values, strategy="entropy"):
    true_graph = nx.DiGraph(model)
    true_edges = set(true_graph.edges())
    simulation_results = []

    print(f"Strategy: {strategy}")

    total_combinations = len(nI_values) * len(aI1_values) * len(aI2_values)
    #with tqdm(total=total_combinations, desc="Parameter combinations") as pbar_outer:
    for nI in nI_values:
        for aI1 in aI1_values:
            for aI2 in aI2_values:
                for _ in tqdm(range(n_trials), desc=f"nI={nI}, aI1={aI1}, aI2={aI2}"):
                    corr_right, corr_wrong = 0, 0
                    incorr_right, incorr_wrong = 0, 0
                    
                    obs_data = silent_simulate(model, OBS_SAMPLES, show_progress=False)

                    essential_graph = to_undirected_with_v_structures(true_graph)

                    is_correct = check_if_estimated_correctly(essential_graph, true_graph)

                    start_orient = time.time()
                    _, oriented, num_exp, fallback_perc, marg_perc, cond_perc = orient_with_logic_and_experiments(
                        essential_graph, obs_data, model, nI, aI1, aI2, strategy
                    )

                    #print(f"\nTrue: {sorted(true_edges)}")
                    #print(f"Oriented: {sorted(oriented)}")

                    end_orient = time.time()
                    time_orient = end_orient - start_orient

                    right = len(oriented & true_edges)
                    wrong = len(find_undirected_edges(essential_graph)) / 2 - right

                    if is_correct:
                        corr_right += right
                        corr_wrong += wrong
                    else:
                        incorr_right += right
                        incorr_wrong += wrong

                    rop = (corr_right + incorr_right) / (corr_right + corr_wrong + incorr_right + incorr_wrong) if (corr_right + corr_wrong + incorr_right + incorr_wrong) > 0 else 0

                    ropao = len(oriented & true_edges) / len(oriented) if len(oriented) > 0 else 0

                    simulation_results.append({
                        'nI': nI, 'aI1': aI1, 'aI2': aI2,
                        'rop': rop, #rightly oriented percentage
                        'm': is_correct,
                        "ropao": ropao, #rightly oriented percentage among oriented
                        'avg_exp': num_exp,
                        'time_orient': time_orient,
                        'fallback_perc': fallback_perc,
                        'marg_perc': marg_perc,
                        'cond_perc': cond_perc
                    })
                    #pbar_outer.update(1)

    return simulation_results

import numpy as np
from scipy.stats import wilcoxon

def run_conditional_test(data_a, data_b, data_name, alpha=0.05):
    """
    Performs a conditional statistical test (Wilcoxon Signed-Rank Test 
    if symmetry is assumed) and reports which data is statistically bigger.
    """
    diff_scores = [a - b for a, b in zip(data_a, data_b)]
    
    aver_diff = np.average(diff_scores) 
    
    print("\n--- Statistical Test Result ---")
    
    stat, p_val = wilcoxon(data_a, data_b, zero_method='wilcox', alternative='two-sided') 
    print(f"Test Used: Wilcoxon Signed-Rank Test")
    print(f"Result: W={stat:.4f}, p={p_val:.4e}")
    print(f"Average Differences (A - B): {aver_diff:.4f}")

    # --- Logic for determining statistical difference and direction ---
    if p_val < alpha:
        # Result is statistically significant
        if aver_diff > 0:
            print(f"✅ Conclusion: {data_name} Data A is statistically larger than Data B (p < {alpha}).")
        elif aver_diff < 0:
            print(f"✅ Conclusion: {data_name} Data A is statistically smaller than Data B (p < {alpha}).")
        else:
            print(f"⚠️ Conclusion: {data_name} There is a statistically significant difference (p < {alpha}), but the average difference is zero.")
    else:
        # Result is NOT statistically significant
        print(f"❌ Conclusion: {data_name} There is **no statistically significant difference** between Data A and Data B (p >= {alpha}).")

# --- 5. Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    import numpy as np
    from scipy.stats import ttest_rel, ranksums
    import json
    import os

    #'example', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'sprinkler', 'child', 'insurance', 'hailfinder', 'win95pts'
    #"minimax", "greedy", "entropy"
    model_name = 'child'
    strategy = "greedy"
    model = create_model(model_name)
    trials = 1

    results = run_simulation(model, trials, [5000], [0.01], [0.01], strategy)

    with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_{strategy}_{model_name}_{trials}.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n--- Simulation Results ---")
    print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'rop':<8}{'m':<8}{'ropao':<8}{'Avg Exp':<10}{'Avg Time Orient':<15}{'Fallback perc':<15}{'Marg perc':<15}{'Cond perc':<15}")
    for res in results:
        print(f"{res['nI']:<8}{res['aI1']:<8.2f}{res['aI2']:<8.2f}{res['rop']:<8.3f}{res['m']:<8}{res['ropao']:<8.3f}{res['avg_exp']:<10.2f}{res['time_orient']:<15.4f}{res['fallback_perc']:<15.4f}{res['marg_perc']:<15.4f}{res['cond_perc']:<15.4f}")
    
    total_rop = sum(res['rop'] for res in results) / len(results) if results else 0.0
    total_ropao = sum(res['ropao'] for res in results) / len(results) if results else 0.0
    total_avg_exp = sum(res['avg_exp'] for res in results) / len(results) if results else 0.0
    total_time_orient = sum(res['time_orient'] for res in results) / len(results) if results else 0.0
    total_fallback_perc = sum(res['fallback_perc'] for res in results) / len(results) if results else 0.0
    total_marg_perc = sum(res['marg_perc'] for res in results) / len(results) if results else 0.0
    total_cond_perc = sum(res['cond_perc'] for res in results) / len(results) if results else 0.0

    print(f"{'Total':<8}{'-':<8}{'-':<8}{total_rop:<8.2f}{'-':<8}{total_ropao:<8.2f}{total_avg_exp:<10.2f}{total_time_orient:<15.4f}{total_fallback_perc:<15.4f}{total_marg_perc:<15.4f}{total_cond_perc:<15.4f}")

    """if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_entropy_{model_name}.json'):
        results_entropy = run_simulation(model, 10, [5000], [0.01], [0.01], "entropy")
        with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_entropy_{model_name}.json', 'w') as f:
            json.dump(results_entropy, f, indent=4)
    if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_greedy_{model_name}.json'):
        results_greedy = run_simulation(model, 10, [5000], [0.01], [0.01], "greedy")
        with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_greedy_{model_name}.json', 'w') as f:
            json.dump(results_greedy, f, indent=4)
    if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_minimax_{model_name}.json'):
        results_minimax = run_simulation(model, 10, [5000], [0.01], [0.01], "minimax")
        with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_minimax_{model_name}.json', 'w') as f:
            json.dump(results_minimax, f, indent=4)

    with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_entropy_{model_name}.json', 'r') as f:
        loaded_entropy = json.load(f)
    with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_greedy_{model_name}.json', 'r') as f:
        loaded_greedy = json.load(f)
    with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_minimax_{model_name}.json', 'r') as f:
        loaded_minimax = json.load(f)

    greedy_exp = [res['avg_exp'] for res in loaded_greedy]
    entropy_exp = [res['avg_exp'] for res in loaded_entropy]
    minimax_exp = [res['avg_exp'] for res in loaded_minimax]
    greedy_lambda = [res['λ'] for res in loaded_greedy]
    entropy_lambda = [res['λ'] for res in loaded_entropy]
    minimax_lambda = [res['λ'] for res in loaded_minimax]

    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    print("="*60)
    print("ANALYSIS OF EFFICIENCY (avg_exp): Entropy vs. Greedy (Symmetry Check)")
    print("="*60)
    run_conditional_test(entropy_exp, greedy_exp, "Efficiency (avg_exp)")

    print("\n\n" + "="*60)
    print("ANALYSIS OF ACCURACY (λ): Entropy vs. Greedy (Symmetry Check)")
    print("="*60)
    run_conditional_test(entropy_lambda, greedy_lambda, "Accuracy (λ)")

    print("="*60)
    print("ANALYSIS OF EFFICIENCY (avg_exp): Minimax vs. Greedy (Symmetry Check)")
    print("="*60)
    run_conditional_test(minimax_exp, greedy_exp, "Efficiency (avg_exp)")

    print("\n\n" + "="*60)
    print("ANALYSIS OF ACCURACY (λ): Minimax vs. Greedy (Symmetry Check)")
    print("="*60)
    run_conditional_test(minimax_lambda, greedy_lambda, "Accuracy (λ)")

    print("="*60)
    print("ANALYSIS OF EFFICIENCY (avg_exp): Entropy vs. Minimax (Symmetry Check)")
    print("="*60)
    run_conditional_test(entropy_exp, minimax_exp, "Efficiency (avg_exp)")

    print("\n\n" + "="*60)
    print("ANALYSIS OF ACCURACY (λ): Entropy vs. Minimax (Symmetry Check)")
    print("="*60)
    run_conditional_test(entropy_lambda, minimax_lambda, "Accuracy (λ)")"""