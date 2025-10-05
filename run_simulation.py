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
from models.graphs import create_true_model
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
                corr_right, corr_wrong = 0, 0
                incorr_right, incorr_wrong = 0, 0
                num_correct_essential_graphs = 0
                total_experiments = 0.0
                max_experiments = 0

                total_time_orient = 0.0

                for _ in tqdm(range(n_trials), desc=f"nI={nI}, aI1={aI1}, aI2={aI2}"):
                    obs_data = silent_simulate(model, OBS_SAMPLES, show_progress=False)

                    essential_graph = to_undirected_with_v_structures(true_graph)

                    is_correct = check_if_estimated_correctly(essential_graph, true_graph)
                    if is_correct:
                        num_correct_essential_graphs += 1

                    start_orient = time.time()
                    _, oriented, num_exp = orient_with_logic_and_experiments(
                        essential_graph, obs_data, model, nI, aI1, aI2, strategy
                    )
                    end_orient = time.time()
                    time_orient = end_orient - start_orient
                    total_time_orient += time_orient

                    total_experiments += num_exp
                    max_experiments = max(max_experiments, num_exp)

                    right = len(oriented & true_edges)
                    wrong = len(find_undirected_edges(essential_graph)) / 2 - right

                    if is_correct:
                        corr_right += right
                        corr_wrong += wrong
                    else:
                        incorr_right += right
                        incorr_wrong += wrong

                avg_experiments = total_experiments / n_trials if n_trials > 0 else 0
                avg_time_orient = total_time_orient / n_trials if n_trials > 0 else 0
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
                    'avg_exp': avg_experiments,
                    'max_exp': max_experiments,
                    'avg_time_orient': avg_time_orient
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
    # Suppress warnings if desired
    warnings.filterwarnings("ignore")
    
    N_TRIALS = 1
    NI_VALUES = [5000]#[1000, 5000, 10000]
    AI1_VALUES = [0.01]#[0.01, 0.05, 0.1]
    AI2_VALUES = [0.01]#[0.01, 0.05, 0.1]
    #STRATEGY = "greedy"
  
    true_model = create_true_model()
    print(f"Nodes in true model: {len(true_model.nodes())}")
    print(f"Edges in true model: {len(true_model.edges())}")
    from core.graph_utils import get_chain_components
    comps = get_chain_components(to_undirected_with_v_structures(nx.DiGraph(true_model)))
    print(f"There are: {len(comps)} components.")
    for i, comp in enumerate(comps):
        print(f"Size of component {i} is {len(comp)}.")
    
    for strategy in ["greedy", "entropy", "minimax"]:
        # Run the simulation and activate orientation algorithm
        results = run_simulation(
            model=true_model,
            n_trials=N_TRIALS,
            nI_values=NI_VALUES,
            aI1_values=AI1_VALUES,
            aI2_values=AI2_VALUES,
            strategy=strategy#STRATEGY,
        )

        # Calculate total averages
        total_avg_exp = sum(res['avg_exp'] for res in results) / len(results) if results else 0.0
        total_avg_time_orient = sum(res['avg_time_orient'] for res in results) / len(results) if results else 0.0

        # Print results
        print("\n--- Simulation Results ---")
        print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'λ':<8}{'m':<8}{'λ\'':<8}{'Avg Exp':<10}{'Max Exp':<10}{'Avg Time Orient':<15}")
        print("-" * 75)
        for res in results:
            print(f"{res['nI']:<8}{res['aI1']:<8.2f}{res['aI2']:<8.2f}{res['λ']:<8.3f}{res['m']:<8}{res['λ\'']:<8.3f}{res['avg_exp']:<10.2f}{res['max_exp']:<10.2f}{res['avg_time_orient']:<15.4f}")
        print("-" * 75)
        print(f"{'Total':<8}{'-':<8}{'-':<8}{'-':<8}{'-':<8}{'-':<8}{total_avg_exp:<10.2f}{'-':<10}{total_avg_time_orient:<15.4f}")   

    # Additional visualization with example parameters
    """print("\nGenerating visualization...")
    obs_data = silent_simulate(true_model, OBS_SAMPLES, show_progress=False)
    print("Observational data sample:")
    print(obs_data.head())

    essential_graph = to_undirected_with_v_structures(true_model)

    oriented_graph, _, _ = orient_with_logic_and_experiments(
        essential_graph, obs_data, true_model, nI=5000, aI1=0.05, aI2=0.05, strategy="greedy"
    )

    visualize_graphs(true_model, essential_graph, oriented_graph)"""