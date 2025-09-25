# Standard library imports
import contextlib
import io
import logging
import warnings

# Third-party library imports
import networkx as nx
import pandas as pd
import tqdm
from IPython.utils import io
from pgmpy.estimators import PC

# Local project imports
from core.algorithm import orient_with_logic_and_experiments
from models.graphs import create_true_model
from core.intervention import silent_simulate
from core.graph_utils import check_if_estimated_correctly, find_undirected_edges

PC_N = 10000
PC_SL = 0.01

def run_simulation(model, n_trials, nI_values, aI1_values, aI2_values, strategy="entropy"):
    true_graph = nx.DiGraph(model)
    true_edges = set(true_graph.edges())
    simulation_results = []

    print(f"Strategy is {strategy}.")
    for nI in nI_values:
        for aI1 in aI1_values:
          for aI2 in aI2_values:
              print(f"Running: nI={nI}, aI1={aI1}, aI2={aI2}")

              corr_right, corr_wrong = 0, 0
              incorr_right, incorr_wrong = 0, 0
              num_correct_essential_graphs = 0
              total_experiments = 0

              for _ in tqdm.tqdm(range(n_trials)):
                  obs_data = silent_simulate(model, PC_N, show_progress=False)

                  pc_estimator = PC(data=obs_data)
                  from IPython.utils import io
                  with io.capture_output():
                      essential_graph = nx.DiGraph(pc_estimator.estimate(
                          variant='stable', ci_test='chi_square',
                          significance_level=PC_SL, return_type="cpdag"
                      ))

                  is_correct = check_if_estimated_correctly(essential_graph, true_graph)
                  if is_correct:
                      num_correct_essential_graphs += 1

                  _, oriented, num_exp = orient_with_logic_and_experiments(essential_graph, obs_data, model, nI, aI1, aI2, strategy)
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

    return simulation_results

if __name__ == '__main__':
    N_TRIALS = 5
    NI_VALUES = [1000]#[200, 500, 1000]
    AI1_VALUES = [0.01]#[0.01, 0.05, 0.1]
    AI2_VALUES = [0.01]#[0.01, 0.05, 0.1]
    STRATEGY = "greedy"

    true_model = create_true_model()

    results = run_simulation(
        model=true_model,
        n_trials=N_TRIALS,
        nI_values=NI_VALUES,
        aI1_values=AI1_VALUES,
        aI2_values=AI2_VALUES,
        strategy=STRATEGY
    )

    print("\n--- Simulation Results ---")
    print(f"{'nI':<8}{'aI1':<8}{'aI2':<8}{'λ':<8}{'m':<8}{'λ\'':<8}{'Avg Exp':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['nI']:<8}{res['aI1']:<8.2f}{res['aI2']:<8.2f}{res['λ']:<8.3f}{res['m']:<8}{res['λ\'']:<8.3f}{res['avg_exp']:<10.2f}")

    #Additional part for visualization
    obs_data = silent_simulate(true_model, PC_N, show_progress=False)

    pc_estimator = PC(data=obs_data)
    from IPython.utils import io
    with io.capture_output():
        essential_graph = nx.DiGraph(pc_estimator.estimate(
            variant='stable', ci_test='chi_square',
            significance_level=PC_SL, return_type="cpdag"
        ))

    oriented_graph, _, _ = orient_with_logic_and_experiments(essential_graph, obs_data, true_model)

    #Displaying
    import matplotlib.pyplot as plt
    import networkx as nx

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
    ax[0].set_title('True Model (Essential Graph)')

    pos_final = nx.spring_layout(oriented_graph)
    nx.draw(
        oriented_graph,
        pos=pos_final,
        with_labels=True,
        node_size=2000,
        node_color='lightcoral', # Use a different color to distinguish
        font_size=12,
        font_weight='bold',
        ax=ax[1]
    )
    ax[1].set_title('Estimated Model (Final Graph)')

    # Adjust the layout to prevent titles from overlapping
    plt.tight_layout()

    plt.savefig('/content/drive/MyDrive/causal-discovery-project/output_graph.png')
