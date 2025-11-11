from core.graph_utils import find_undirected_edges, propagate_orientations, get_chain_components
from core.intervention import choose_intervention_variable
from core.statistical_tests import robust_orientation_test

def orient_with_logic_and_experiments(graph, observational_data, data_generator, nI, aI1, aI2, strategy):
    print(f"All edges: {graph.edges()}")
    all_oriented = set()

    total_interventions = 0
    fallback_interventions = 0 

    chain_components = get_chain_components(graph)

    #print(f"Components: {len(chain_components)}")
    #for comp in chain_components:
    #    print(len(comp.nodes()))

    for num, comp in enumerate(chain_components):
        intervened_in_comp = set()

        while True:
            comp_undirected = find_undirected_edges(comp)

            if not comp_undirected:
                break

            #print("\nStart")
            variable_to_intervene, fallback = choose_intervention_variable(comp, intervened_in_comp, strategy=strategy)
            #print(f"Variable to intervene: {variable_to_intervene}")
            if variable_to_intervene is None:
                break
            total_interventions += 1
            fallback_interventions += fallback

            intervened_in_comp.add(variable_to_intervene)

            edges_to_check = [(u, v) for u, v in comp_undirected if variable_to_intervene in (u, v) and u < v]

            exp_data = data_generator.quasi_experiment(variable_to_intervene, nI)

            oriented_orig = 0
            oriented_prop = 0
            for u, v in edges_to_check:
                if (not comp.has_edge(u, v)) or (not comp.has_edge(v, u)):
                    continue

                vk = v if u == variable_to_intervene else u

                neighbors = set(graph.neighbors(vk))
                children_outside = {w for w in graph.successors(vk) if ((w not in comp.nodes()) and (w not in graph.predecessors(vk)))}
                B = list(neighbors - children_outside)

                #Rude violation
                import networkx as nx
                true_graph = nx.DiGraph(data_generator.model)
                true_edges = set(true_graph.edges())
                if (vk, variable_to_intervene) in true_edges:
                    print("\nMust be.")
                else:
                    print("\nMust not be.")

                orientation = robust_orientation_test(variable_to_intervene, vk, B, observational_data, exp_data, aI1, aI2)
                if orientation:
                    print(f"Current oriented: {orientation}")
                    comp.remove_edge(orientation[1], orientation[0])
                    all_oriented.add(orientation)
                    #oriented_orig += 1
                    propagated = propagate_orientations(comp)
                    print(f"Propagated: {propagated}")
                    for rib in propagated:
                        if rib not in true_edges:
                            print("Holy fuck!")
                            print(f"Troublemaker: {rib}")
                            while(True):
                                pass
                    all_oriented.update(propagated)
                    #oriented_prop += len(propagated)
            #print(f"Oriented orig: {oriented_orig}")
            #print(f"Oriented prop: {oriented_prop}")
            #print("End")

    temp_graph = graph.copy()
    for orientation in all_oriented:
        temp_graph.remove_edge(orientation[1], orientation[0])

    fallback_perc = fallback_interventions / total_interventions if (total_interventions > 0) else 0
    return temp_graph, all_oriented, total_interventions, fallback_perc