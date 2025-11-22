from core.graph_utils import find_undirected_edges, propagate_orientations, get_chain_components
from core.intervention import choose_intervention_variable
from core.statistical_tests import robust_orientation_test

"""
Algorithm for orienting an essential graph.
It does so in each chain component separately, getting experimental
data and running statistical tests.
"""
def orient_with_logic_and_experiments(graph, observational_data, data_generator, nI, aI1, aI2, strategy):
    temp_graph = graph.copy()
    all_oriented = set()

    total_interventions = 0
    fallback_interventions = 0 

    chain_components = get_chain_components(graph)
    for num, comp in enumerate(chain_components):
        intervened_in_comp = set()

        while True:
            comp_undirected = find_undirected_edges(comp)
            if not comp_undirected:
                break

            variable_to_intervene, fallback = choose_intervention_variable(comp, intervened_in_comp, strategy=strategy)
            if variable_to_intervene is None:
                break
            total_interventions += 1
            fallback_interventions += fallback

            intervened_in_comp.add(variable_to_intervene)

            edges_to_check = [(u, v) for u, v in comp_undirected if variable_to_intervene in (u, v) and u < v]

            exp_data = data_generator.quasi_experiment(variable_to_intervene, nI)

            for u, v in edges_to_check:
                if (not comp.has_edge(u, v)) or (not comp.has_edge(v, u)):
                    continue

                Vk = v if u == variable_to_intervene else u

                B = list(temp_graph.predecessors(Vk))

                orientation = robust_orientation_test(variable_to_intervene, Vk, B, observational_data, exp_data, aI1, aI2)
                if orientation:
                    comp.remove_edge(orientation[1], orientation[0])
                    propagated = propagate_orientations(comp)
                    
                    temp_graph.remove_edge(orientation[1], orientation[0])
                    for prop in propagated:
                        temp_graph.remove_edge(prop[1], prop[0])

                    all_oriented.add(orientation)
                    all_oriented.update(propagated)

    fallback_perc = fallback_interventions / total_interventions if (total_interventions > 0) else 0
    return temp_graph, all_oriented, total_interventions, fallback_perc