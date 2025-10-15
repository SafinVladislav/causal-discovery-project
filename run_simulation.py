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
from core.algorithm import orient_with_logic_and_experiments
from models.graphs import create_model
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

from pathlib import Path

import logging
# Suppress pgmpy warnings by setting the logging level to ERROR
logging.getLogger("pgmpy").setLevel(logging.ERROR)

current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent
RELATIVE_VIS_DIR = Path("visualizations")
OBS_SAMPLES = 20000

def to_undirected_with_v_structures(directed_graph):
    """
    Convert a directed graph to an undirected graph, preserving v-structures as directed edges.
    
    Args:
        directed_graph (nx.DiGraph): The input directed graph (e.g., from a pgmpy model).
    
    Returns:
        nx.DiGraph: A graph where v-structure edges are directed, and other edges are undirected
                   (represented by bidirectional edges).
    """
    # Initialize a new directed graph
    result_graph = nx.DiGraph()
    result_graph.add_nodes_from(directed_graph.nodes())
    
    # Find all v-structures (X -> Y <- Z)
    v_structures = []
    for y in directed_graph.nodes():
        # Get parents (nodes with edges pointing to y, excluding successors)
        parents = list(set(directed_graph.predecessors(y)) - set(directed_graph.successors(y)))
        if len(parents) >= 2:
            import itertools
            for x, z in itertools.combinations(parents, 2):
                if not directed_graph.has_edge(x, z) and not directed_graph.has_edge(z, x):
                    v_structures.append((x, y, z))
    
    # Create undirected edges for all edges in the input graph
    undirected_edges = set()
    for u, v in directed_graph.edges():
        # Skip edges that are part of v-structures (to be added as directed later)
        is_v_structure_edge = False
        for x, y, z in v_structures:
            if (u == x and v == y) or (u == z and v == y) or (u == y and v == x) or (u == y and v == z):
                is_v_structure_edge = True
                break
        if not is_v_structure_edge:
            undirected_edges.add((u, v))
            undirected_edges.add((v, u))  # Add both directions for undirected
    
    # Add undirected edges (bidirectional) to the result graph
    result_graph.add_edges_from(undirected_edges)
    
    # Add directed edges for v-structures
    for x, y, z in v_structures:
        result_graph.add_edge(x, y)
        result_graph.add_edge(z, y)
    
    return result_graph


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
                    _, oriented, num_exp = orient_with_logic_and_experiments(
                        essential_graph, obs_data, model, nI, aI1, aI2, strategy
                    )
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

                    #time_orient
                    #oriented
                    rightly_oriented = corr_right + incorr_right
                    perc = rightly_oriented / len(oriented) if len(oriented) > 0 else 0

                    correct = corr_right + corr_wrong
                    conditional_perc = corr_right / correct if correct > 0 else 0

                    simulation_results.append({
                        'nI': nI, 'aI1': aI1, 'aI2': aI2,
                        'λ': perc,
                        'm': is_correct,
                        "λ'": conditional_perc,
                        'avg_exp': num_exp,
                        'time_orient': time_orient
                    })
                    #pbar_outer.update(1)

    return simulation_results


def visualize_graphs(true_model, essential_graph, oriented_graph):
    """
    Visualize the essential graph and the final oriented graph.
    
    Args:
        true_model: The true causal model (for title reference).
        essential_graph: The CPDAG from PC.
        oriented_graph: The oriented graph after algorithm.
        output_path (str): Path to save the figure.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    from networkx.drawing.nx_pydot import graphviz_layout
    pos = graphviz_layout(essential_graph, prog='dot')
    #pos = nx.spring_layout(essential_graph)
    nx.draw(
        essential_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightgreen',
        font_size=12,
        font_weight='bold',
        ax=ax[0]
    )
    ax[0].set_title('Essential Graph (CPDAG)')

    nx.draw(
        oriented_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightcoral',
        font_size=12,
        font_weight='bold',
        ax=ax[1]
    )
    ax[1].set_title('Oriented Graph (After Algorithm)')

    plt.tight_layout()
    output_path = PROJECT_ROOT / RELATIVE_VIS_DIR / 'output_graph.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    import numpy as np
    from scipy.stats import ttest_rel, ranksums
    import json
    import os
    model_name = 'win95pts'
    model = create_model(model_name)
    
    if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_entropy_{model_name}.json'):
        results_entropy = run_simulation(model, 50, [5000], [0.01], [0.01], "entropy")
        with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_entropy_{model_name}.json', 'w') as f:
            json.dump(results_entropy, f, indent=4)
    if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_greedy_{model_name}.json'):
        results_greedy = run_simulation(model, 50, [5000], [0.01], [0.01], "greedy")
        with open(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_greedy_{model_name}.json', 'w') as f:
            json.dump(results_greedy, f, indent=4)
    if not os.path.exists(f'/content/drive/MyDrive/causal-discovery-project/statistics/results_minimax_{model_name}.json'):
        results_minimax = run_simulation(model, 50, [5000], [0.01], [0.01], "minimax")
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

def check_symmetry(data, name, alpha=0.05):
    from scipy.stats import skewtest
    skew_stat, skew_p = skewtest(data)
    is_symmetric = skew_p > alpha
    
    print(f"\n--- Symmetry Check for {name} Differences (D'Agostino's Skewtest) ---")
    print(f"Skewtest Statistic: {skew_stat:.3f}")
    print(f"P-value: {skew_p:.3f}")
    
    if is_symmetric:
        print(f"Condition Met: P > {alpha}. We ASSUME symmetry (skewness is not significantly non-zero).")
    else:
        print(f"Condition Failed: P <= {alpha}. We REJECT symmetry (data is significantly skewed).")
        
    return is_symmetric, skew_p

import numpy as np
from scipy.stats import wilcoxon

# Assume check_symmetry(diff_scores, data_name) is defined elsewhere 
# and returns (is_symmetric: bool, skew_p: float)

def run_conditional_test(data_a, data_b, data_name, alpha=0.05):
    """
    Performs a conditional statistical test (Wilcoxon Signed-Rank Test 
    if symmetry is assumed) and reports which data is statistically bigger.
    """
    diff_scores = [a - b for a, b in zip(data_a, data_b)]
    
    # Assuming this function is correctly defined and available
    is_symmetric, skew_p = check_symmetry(diff_scores, data_name) 
    
    # Calculate the median of the differences for directionality
    median_diff = np.average(diff_scores) 
    
    print("\n--- Statistical Test Result ---")
    
    if is_symmetric:
        # Assumption met (Symmetry): Use the Wilcoxon Signed-Rank Test.
        stat, p_val = wilcoxon(data_a, data_b, zero_method='wilcox', alternative='two-sided') 
        print(f"Test Used: Wilcoxon Signed-Rank Test")
        print(f"Result: W={stat:.4f}, p={p_val:.4e}")
        print(f"Median of Differences (A - B): {median_diff:.4f}")

        # --- Logic for determining statistical difference and direction ---
        if p_val < alpha:
            # Result is statistically significant
            if median_diff > 0:
                print(f"✅ Conclusion: {data_name} **Data A** is statistically **significantly larger** than Data B (p < {alpha}).")
            elif median_diff < 0:
                print(f"✅ Conclusion: {data_name} **Data B** is statistically **significantly larger** than Data A (p < {alpha}).")
            else:
                # Should be rare, but handles the case where p < alpha but median_diff is 0
                # A more robust check might look at the sign of the test statistic or mean difference
                print(f"⚠️ Conclusion: {data_name} There is a statistically significant difference (p < {alpha}), but the median difference is zero.")
        else:
            # Result is NOT statistically significant
            print(f"❌ Conclusion: {data_name} There is **no statistically significant difference** between Data A and Data B (p >= {alpha}).")
            
    else:
        # If symmetry assumption is NOT met, you might want to use a different test 
        # (like the Sign Test) or flag a warning. For now, we only report the assumption failure.
        print("⚠️ Test not run: The symmetry assumption for Wilcoxon was not met.")

    # Plot for visual confirmation regardless of the result
    # plot_distributions(diff_scores, data_name)

# --- 5. Execution ---
if __name__ == "__main__":
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
    run_conditional_test(entropy_lambda, minimax_lambda, "Accuracy (λ)")
    
    """results_no_propagating = run_simulation(model, 50, [5000], [0.01], [0.01], "greedy")

    greedy_lambda = [res['λ'] for res in results_no_propagating]

    print(greedy_lambda)"""