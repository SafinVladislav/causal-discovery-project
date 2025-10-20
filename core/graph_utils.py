import random
import copy
import itertools
import networkx as nx
import pandas as pd
from collections import defaultdict

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

def find_undirected_edges(graph):
    undirected = []
    for u, v in graph.edges():
        if graph.has_edge(v, u):
            undirected.append((u, v))
    return undirected

def has_directed_cycle(graph, new_oriented_edge):
    u, v = new_oriented_edge

    visited = set()
    stack = [v]
    while stack:
        node = stack.pop()
        if node == u:
            return True
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

def recursive(graph):
    undirected_edges = find_undirected_edges(graph)
    if len(undirected_edges) == 0:
        return graph

    import random
    u, v = random.choice(undirected_edges)
    if random.random() < 0.5:
        u, v = v, u

    # First
    temp_graph = graph.copy()
    temp_graph.remove_edge(v, u)
    if not is_bad_graph(temp_graph, (u, v)):
        propagate_orientations(temp_graph)
        dag = recursive(temp_graph)
        if dag is not None:
            return dag
    # Second
    temp_graph = graph.copy()
    temp_graph.remove_edge(u, v)
    if not is_bad_graph(temp_graph, (v, u)):
        propagate_orientations(temp_graph)
        dag = recursive(temp_graph)
        if dag is not None:
            return dag
    # Default
    return None

def generate_dag_from_cpdag(graph):    
    return recursive(graph)

def sample_dags(graph, n_samples):
    #bad = 0
    dags = []
    for _ in range(n_samples):
        dag = generate_dag_from_cpdag(graph)
        if dag:
            dags.append(dag)
            """print(len(dag.edges()))
            print(len(find_undirected_edges(graph)))
            print(len(graph.edges()))
            print(len(find_undirected_edges(dag)))
            for u, v in dag.edges():
                if is_bad_graph(dag, (u, v)):
                  bad += 1
                  break"""
    """print(f"Undirected: {len(find_undirected_edges(graph))}")
    print(f"Good: {len(dags) - bad}; Bad: {bad}")
    print(f"Dags: {len(dags)}")"""
    return dags

from joblib import Parallel, delayed
def sample_dags(graph, n_samples):
    def generate_one():
        return generate_dag_from_cpdag(graph)
    dags = Parallel(n_jobs=-1)(delayed(generate_one)() for _ in range(n_samples))
    return [dag for dag in dags if dag]

def check_if_estimated_correctly(estimated, true_graph):
    if set(map(tuple, map(sorted, estimated.edges()))) != set(map(tuple, map(sorted, true_graph.edges()))):
        return False

    estimated_directed = {(u, v) for u, v in estimated.edges() if not estimated.has_edge(v, u)}
    true_directed = {(u, v) for u, v in true_graph.edges() if not true_graph.has_edge(v, u)}
    return estimated_directed.issubset(set(true_directed))