import cProfile
from core.graph_utils import find_undirected_edges, propagate_orientations, get_chain_components
from core.intervention import choose_intervention_variable, quasi_experiment, silent_simulate
from core.statistical_tests import robust_orientation_test
from models.graphs import create_model
from run_simulation import to_undirected_with_v_structures
from core.algorithm import orient_with_logic_and_experiments
import networkx as nx

OBS_SAMPLES = 10000

# Example setup (adapt to your context)
def setup_for_profiling():
    model_name = 'win95pts'
    model = create_model(model_name)
    obs_data = silent_simulate(model, OBS_SAMPLES)  # From intervention.py 
    essential_graph = to_undirected_with_v_structures(nx.DiGraph(model))  # From run_simulation.py
    return essential_graph, obs_data, model

def profile_orient():
    graph, observational_data, model = setup_for_profiling()
    # Directly call the function within the profile context
    cProfile.runctx('orient_with_logic_and_experiments(graph, observational_data, model, nI=500, aI1=0.05, aI2=0.05, strategy="greedy")',
                    globals(), locals())

if __name__ == '__main__':
    profile_orient()