from core.graph_utils import find_undirected_edges, propagate_orientations, get_chain_components
from core.intervention import choose_intervention_variable, quasi_experiment
from core.statistical_tests import robust_orientation_test

def orient_with_logic_and_experiments(graph, observational_data, model, nI=5000, aI1=0.01, aI2=0.01, strategy="greedy"):
    # Create a working copy of the input graph
    temp_graph = graph.copy()
    all_oriented = set()
    num_experiments = 0

    # Process each chain component
    for comp in get_chain_components(temp_graph):
        intervened_in_comp = set()

        while True:
            # Get undirected edges in the component
            comp_undirected = find_undirected_edges(comp)
            if not comp_undirected:
                break

            # Select variable for intervention
            variable_to_intervene = choose_intervention_variable(comp, intervened_in_comp, strategy=strategy)
            if variable_to_intervene is None:
                break

            intervened_in_comp.add(variable_to_intervene)
            num_experiments += 1
            exp_data = quasi_experiment(model, variable_to_intervene, samples=nI)

            # Filter edges involving the intervened variable
            edges_to_check = [(u, v) for u, v in comp_undirected if variable_to_intervene in (u, v) and u < v]
            for u, v in edges_to_check:
                if (not comp.has_edge(u, v)) or (not comp.has_edge(v, u)):
                    continue

                vk = v if u == variable_to_intervene else u
                predecessors = set(comp.predecessors(vk))
                orientation = robust_orientation_test(variable_to_intervene, vk, list(predecessors), observational_data, exp_data, alpha1=aI1, alpha2=aI2)
                if orientation:
                    comp.remove_edge(orientation[1], orientation[0])
                    all_oriented.add(orientation)
                    all_oriented.update(propagate_orientations(comp))

    for orientation in all_oriented:
        temp_graph.remove_edge(orientation[1], orientation[0])

    return temp_graph, all_oriented, num_experiments