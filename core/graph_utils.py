# Standard library imports
import random
import copy

# Third-party library imports
import networkx as nx
import pandas as pd
from collections import defaultdict

def find_undirected_edges(graph):
    undirected = []
    for u, v in graph.edges():
        if graph.has_edge(v, u):
            undirected.append((u, v))
    return undirected

def has_directed_cycle(graph):
    undirected_edges = find_undirected_edges(graph)
    graph.remove_edges_from(undirected_edges)
    result = not nx.is_directed_acyclic_graph(graph)
    graph.add_edges_from(undirected_edges)
    return result

def has_v_structure(graph):
    for u, v in graph.edges():
        if not graph.has_edge(v, u):
            z_candidates = [
                z for z in graph.predecessors(v)
                if z != u and z not in graph.successors(v) and not graph.has_edge(z, u) and not graph.has_edge(u, z)
            ]
            if len(z_candidates) > 0:
                return True
    return False

def is_bad_graph(graph):
    print(f"Edges: {graph.edges()}")
    return has_directed_cycle(graph) or has_v_structure(graph)

def propagate_orientations(graph):
    all_oriented = set()
    while True:
        oriented_in_pass = False
        undirected_edges = [e for e in find_undirected_edges(graph)]

        for u, v in {tuple(sorted(e)) for e in undirected_edges}:
            graph.remove_edge(v, u)
            if is_bad_graph(graph):
                all_oriented.add((v, u))
                graph.remove_edge(u, v)
                oriented_in_pass = True
                graph.add_edge(v, u)
                continue
            graph.add_edge(v, u)

            graph.remove_edge(u, v)
            if is_bad_graph(graph):
                all_oriented.add((u, v))
                graph.remove_edge(v, u)
                oriented_in_pass = True
            graph.add_edge(u, v)

        if not oriented_in_pass:
            break
    return all_oriented

def get_chain_components(graph):
    undirected_graph = nx.Graph()
    undirected_graph.add_edges_from(find_undirected_edges(graph))
    node_components = list(nx.connected_components(undirected_graph))
    return [nx.DiGraph(undirected_graph.subgraph(nodes).copy()) for nodes in node_components]

def generate_dag_from_cpdag(graph, max_attempts=3):
    for _ in range(max_attempts):
        temp_graph = graph.copy()
        undirected_edges = find_undirected_edges(temp_graph)
        random.shuffle(undirected_edges)
        for u, v in {tuple(sorted(e)) for e in undirected_edges}:
            if not temp_graph.has_edge(u, v) or not temp_graph.has_edge(v, u):
                continue

            if random.choice([True, False]):
                temp_graph.remove_edge(v, u)
            else:
                temp_graph.remove_edge(u, v)
            
            propagate_orientations(temp_graph)
            if is_bad_graph(temp_graph):
                break
        
        if not is_bad_graph(temp_graph):
            return temp_graph
    return None

def sample_dags(graph, n_samples):
    dags = []
    for _ in range(n_samples):
        dag = generate_dag_from_cpdag(graph)
        if dag:
            dags.append(dag)
    #print(len(dags))
    return dags

def check_if_estimated_correctly(estimated, true_graph):
    if set(map(tuple, map(sorted, estimated.edges()))) != set(map(tuple, map(sorted, true_graph.edges()))):
        return False

    estimated_directed = {(u, v) for u, v in estimated.edges() if not estimated.has_edge(v, u)}
    return estimated_directed.issubset(set(true_graph.edges()))