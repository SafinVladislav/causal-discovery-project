# Standard library imports
import contextlib
import io
import logging
import warnings

# Third-party library imports
import networkx as nx
import pandas as pd
from tqdm import tqdm
from IPython.utils import io
from pgmpy.estimators import PC

# Local project imports
from core.algorithm import orient_with_logic_and_experiments
from models.graphs import create_true_model
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

import logging

# Suppress pgmpy warnings by setting the logging level to ERROR
logging.getLogger("pgmpy").setLevel(logging.ERROR)

def compute_shd(estimated_cpdag, true_cpdag):
    """
    Compute Structural Hamming Distance (SHD) between estimated and true CPDAGs.
    
    Args:
        estimated_cpdag: Estimated CPDAG (nx.DiGraph with possibly undirected edges).
        true_cpdag: True CPDAG (nx.DiGraph with possibly undirected edges).
    
    Returns:
        int: SHD (number of edge additions, deletions, or reversals needed).
    """
    # Convert to undirected graphs for skeleton comparison
    est_skeleton = estimated_cpdag.to_undirected()
    true_skeleton = true_cpdag.to_undirected()
    
    # Compute skeleton differences (additions and deletions)
    est_edges = set(est_skeleton.edges())
    true_edges = set(true_skeleton.edges())
    skeleton_additions = len(true_edges - est_edges)  # Edges in true but not estimated
    skeleton_deletions = len(est_edges - true_edges)  # Edges in estimated but not true
    
    # Compute orientation differences
    orientation_errors = 0
    for edge in est_edges & true_edges:  # Common edges in skeleton
        # Check if orientation differs
        est_has_undirected = (estimated_cpdag.has_edge(*edge) and 
                           estimated_cpdag.has_edge(edge[1], edge[0]))
        true_has_undirected = (true_cpdag.has_edge(*edge) and 
                            true_cpdag.has_edge(edge[1], edge[0]))
        
        if not est_has_undirected and not true_has_undirected:
            # Both directed, check if directions match
            est_dir = (estimated_cpdag.has_edge(*edge), 
                      estimated_cpdag.has_edge(edge[1], edge[0]))
            true_dir = (true_cpdag.has_edge(*edge), 
                       true_cpdag.has_edge(edge[1], edge[0]))
            if est_dir != true_dir:
                orientation_errors += 1
    
    return skeleton_additions + skeleton_deletions + orientation_errors

def tune_pc_parameters(model, n_trials_tune=10,
                      n_candidates=[500, 1000, 5000, 10000, 20000], 
                      sl_candidates=[0.001, 0.01, 0.05, 0.1]):
    """
    Tune the parameters for the PC algorithm by evaluating Structural Hamming Distance (SHD)
    on simulated data. The metric is the average SHD across trials, measuring the difference
    between the estimated CPDAG and the true CPDAG.
    
    Args:
        model: The true causal model.
        n_trials_tune (int): Number of trials per parameter combination.
        n_candidates (list): Candidate sample sizes for observational data.
        sl_candidates (list): Candidate significance levels for CI tests.
    
    Returns:
        tuple: Best (PC_N, PC_SL) based on lowest average SHD (ties broken by smaller N, then smaller SL).
    """
    true_graph = nx.DiGraph(model)
    results = []
    
    total_combinations = len(n_candidates) * len(sl_candidates)
    with tqdm(total=total_combinations, desc="Tuning PC parameters") as pbar:
        for n in n_candidates:
            for sl in sl_candidates:
                shd_sum = 0
                for _ in range(n_trials_tune):
                    obs_data = silent_simulate(model, n, show_progress=False)
                    pc_estimator = PC(data=obs_data)
                    with io.capture_output():
                        essential_graph = nx.DiGraph(pc_estimator.estimate(
                            variant='stable', ci_test='chi_square',
                            significance_level=sl, return_type="cpdag"
                        ))
                    shd = compute_shd(essential_graph, true_graph)
                    shd_sum += shd
                avg_shd = shd_sum / n_trials_tune if n_trials_tune > 0 else float('inf')
                results.append({'n': n, 'sl': sl, 'avg_shd': avg_shd})
                pbar.update(1)
    
    # Sort results by avg_shd (ascending), then n (ascending), then sl (ascending)
    sorted_results = sorted(results, key=lambda x: (x['avg_shd'], x['n'], x['sl']))
    
    # Print top k parameter sets
    top_k = 3
    print(f"\nTop {min(top_k, len(sorted_results))} PC parameter sets:")
    print(f"{'Rank':<6}{'N':<8}{'SL':<8}{'Avg SHD':<10}")
    print("-" * 32)
    for i, res in enumerate(sorted_results[:top_k], 1):
        print(f"{i:<6}{res['n']:<8}{res['sl']:<8.4f}{res['avg_shd']:<10.3f}")
    
    # Select best for return (same as before)
    best = sorted_results[0]  # First entry is the best (lowest SHD, then smallest n, then smallest sl)
    print(f"\nReturning best PC parameters: N={best['n']}, SL={best['sl']} with average SHD {best['avg_shd']:.3f}")
    return best['n'], best['sl']

def run_simulation(model, n_trials, nI_values, aI1_values, aI2_values, strategy="entropy", 
                   pc_n=20000, pc_sl=0.05):
    """
    Run simulations to evaluate the orientation algorithm after estimating the CPDAG with PC.
    
    Args:
        model: The true causal model.
        n_trials (int): Number of simulation trials.
        nI_values (list): Values for nI parameter in orientation.
        aI1_values (list): Values for aI1 parameter in orientation.
        aI2_values (list): Values for aI2 parameter in orientation.
        strategy (str): Strategy for orientation ('entropy' or 'greedy').
        pc_n (int): Sample size for PC observational data.
        pc_sl (float): Significance level for PC CI tests.
    
    Returns:
        list: List of result dictionaries for each parameter combination.
    """
    true_graph = nx.DiGraph(model)
    true_edges = set(true_graph.edges())
    simulation_results = []

    print(f"Strategy: {strategy}")
    print(f"PC parameters: N={pc_n}, SL={pc_sl}")
    
    total_combinations = len(nI_values) * len(aI1_values) * len(aI2_values)
    with tqdm(total=total_combinations, desc="Parameter combinations") as pbar_outer:
        for nI in nI_values:
            for aI1 in aI1_values:
                for aI2 in aI2_values:
                    #print(f"\nRunning: nI={nI}, aI1={aI1}, aI2={aI2}")
                    corr_right, corr_wrong = 0, 0
                    incorr_right, incorr_wrong = 0, 0
                    num_correct_essential_graphs = 0
                    total_experiments = 0

                    for _ in range(n_trials):
                        obs_data = silent_simulate(model, pc_n, show_progress=False)
                        pc_estimator = PC(data=obs_data)
                        with io.capture_output():
                            essential_graph = nx.DiGraph(pc_estimator.estimate(
                                variant='stable', ci_test='chi_square',
                                significance_level=pc_sl, return_type="cpdag"
                            ))

                        is_correct = check_if_estimated_correctly(essential_graph, true_graph)
                        if is_correct:
                            num_correct_essential_graphs += 1

                        _, oriented, num_exp = orient_with_logic_and_experiments(
                            essential_graph, obs_data, model, nI, aI1, aI2, strategy
                        )
                        total_experiments += num_exp

                        right = len(oriented & true_edges)
                        wrong = len(find_undirected_edges(essential_graph)) / 2 - right

                        if is_correct:
                            corr_right += right
                            corr_wrong += wrong
                        else:
                            incorr_right += right
                            incorr_wrong += wrong

                    avg_experiments = total_experiments / n_trials if n_trials > 0 else 0
                    total_oriented = corr_right + corr_wrong + incorr_right + incorr_wrong
                    total_rightly_oriented = corr_right + incorr_right
                    overall_perc = total_rightly_oriented / total_oriented if total_oriented > 0 else 0

                    total_correct = corr_right + corr_wrong
                    conditional_perc = corr_right / total_correct if total_correct > 0 else 0

                    simulation_results.append({
                        'nI': nI, 'aI1': aI1, 'aI2': aI2,
                        'λ': overall_perc,
                        'm': num_correct_essential_graphs,
                        "λ'": conditional_perc,
                        'avg_exp': avg_experiments
                    })
                    pbar_outer.update(1)

    return simulation_results

def visualize_graphs(true_model, essential_graph, oriented_graph, output_path):
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

    pos_essential = nx.spring_layout(essential_graph)
    nx.draw(
        essential_graph,
        pos=pos_essential,
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
        pos=pos_essential,
        with_labels=True,
        node_size=2000,
        node_color='lightcoral',
        font_size=12,
        font_weight='bold',
        ax=ax[1]
    )
    ax[1].set_title('Oriented Graph (After Algorithm)')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    # Suppress warnings if desired
    warnings.filterwarnings("ignore")
    
    N_TRIALS = 10  # Example: set to a positive value for actual runs
    NI_VALUES = [200, 500, 1000, 10000]
    AI1_VALUES = [0.01, 0.05, 0.1]
    AI2_VALUES = [0.01, 0.05, 0.1]
    STRATEGY = "minimax"
    VISUALIZATION_PATH = '/content/drive/MyDrive/causal-discovery-project/output_graph.png'

    true_model = create_true_model()
    print("True model edges:", true_model.edges())

    # Step 1: Tune PC parameters
    PC_N, PC_SL = 500, 0.01#tune_pc_parameters(true_model)
    # Step 2: Run the simulation with tuned parameters and activate orientation algorithm
    """results = run_simulation(
        model=true_model,
        n_trials=N_TRIALS,
        nI_values=NI_VALUES,
        aI1_values=AI1_VALUES,
        aI2_values=AI2_VALUES,
        strategy=STRATEGY,
        pc_n=PC_N,
        pc_sl=PC_SL
    )

    # Print results
    print("\n--- Simulation Results ---")
    print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'λ':<8}{'m':<8}{'λ\'':<8}{'Avg Exp':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['nI']:<8}{res['aI1']:<8.2f}{res['aI2']:<8.2f}{res['λ']:<8.3f}{res['m']:<8}{res['λ\'']:<8.3f}{res['avg_exp']:<10.2f}")
    """
    # Additional visualization with example parameters
    print("\nGenerating visualization...")
    obs_data = silent_simulate(true_model, PC_N, show_progress=False)
    print("Observational data sample:")
    print(obs_data.head())

    pc_estimator = PC(data=obs_data)
    with io.capture_output():
        essential_graph = nx.DiGraph(pc_estimator.estimate(
            variant='stable', ci_test='chi_square',
            significance_level=PC_SL, return_type="cpdag"
        ))

    oriented_graph, _, _ = orient_with_logic_and_experiments(
        essential_graph, obs_data, true_model, nI=5000, aI1=0.01, aI2=0.01, strategy="greedy"
    )

    visualize_graphs(true_model, essential_graph, oriented_graph, VISUALIZATION_PATH)