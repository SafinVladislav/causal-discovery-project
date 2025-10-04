# Standard library imports
import random
import copy
import itertools

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

def has_directed_cycle(graph, new_oriented_edge):
    u, v = new_oriented_edge
    # Check if there is a path from v back to u in the graph (excluding the new edge)
    visited = set()
    stack = [v]
    while stack:
        node = stack.pop()
        if node == u:
            return True  # Cycle found
        if node not in visited:
            visited.add(node)
            for neighbor in (set(graph.successors(node)) - set(graph.predecessors(node))):
                stack.append(neighbor)
    return False

def has_v_structure(graph, new_oriented_edge):
    u, v = new_oriented_edge
    parents = list(set(graph.predecessors(v)) - set(graph.successors(v)) - {u})
    if len(parents) >= 1:
        for z in parents:
            if not graph.has_edge(u, z) and not graph.has_edge(z, u):
                return True
    return False

def is_bad_graph(graph, new_oriented_edge):
    return has_directed_cycle(graph, new_oriented_edge) or has_v_structure(graph, new_oriented_edge)

def propagate_orientations(graph):
    temp_graph = graph.copy()
    all_oriented = set()
    while True:
        oriented_in_pass = False
        undirected_edges = [e for e in find_undirected_edges(temp_graph)]

        for u, v in {tuple(sorted(e)) for e in undirected_edges}:
            bad_u_v, bad_v_u = None, None

            temp_graph.remove_edge(v, u)
            if is_bad_graph(temp_graph, (u, v)):
                bad_u_v = True
            temp_graph.add_edge(v, u)

            temp_graph.remove_edge(u, v)
            if is_bad_graph(temp_graph, (v, u)):
                bad_v_u = True
            temp_graph.add_edge(u, v)

            orientation = None
            if bad_u_v and not bad_v_u:
                orientation = (v, u)
            if not bad_u_v and bad_v_u:
                orientation = (u, v)

            if orientation is not None:
                all_oriented.add(orientation)
                temp_graph.remove_edge(orientation[1], orientation[0])
                oriented_in_pass = True

        if not oriented_in_pass:
            break
    return all_oriented

def get_chain_components(graph):
    undirected_graph = nx.Graph()
    undirected_graph.add_edges_from(find_undirected_edges(graph))
    node_components = list(nx.connected_components(undirected_graph))
    return [nx.DiGraph(undirected_graph.subgraph(nodes).copy()) for nodes in node_components]

def generate_dag_from_cpdag(graph):
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
        
    return temp_graph

def sample_dags(graph, n_samples):
    #print("Start")
    #print(is_bad_graph(graph, (None, None)))
    dags = []
    for _ in range(n_samples):
        dag = generate_dag_from_cpdag(graph)
        if dag:
            dags.append(dag)
    #print(f"Dags: {len(dags)}")
    #print("End")
    return dags

def check_if_estimated_correctly(estimated, true_graph):
    if set(map(tuple, map(sorted, estimated.edges()))) != set(map(tuple, map(sorted, true_graph.edges()))):
        return False

    estimated_directed = {(u, v) for u, v in estimated.edges() if not estimated.has_edge(v, u)}
    true_directed = {(u, v) for u, v in true_graph.edges() if not true_graph.has_edge(v, u)}
    return estimated_directed.issubset(set(true_directed))