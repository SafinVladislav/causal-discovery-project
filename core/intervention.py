import contextlib
import io
from math import log
from collections import Counter
import numpy as np
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

from core.graph_utils import find_undirected_edges, sample_dags

N_SAMPLES = 1000
EST_ITERS = 1
THRESHOLD = 0.7

def choose_intervention_variable(graph, intervened, strategy):
    if strategy not in ["greedy", "minimax", "entropy"]:
        raise ValueError("Invalid strategy.")

    undirected_edges = find_undirected_edges(graph)
    if not undirected_edges:
        return None, None

    nodes_to_consider = sorted(set().union(*undirected_edges) - intervened)
    if not nodes_to_consider:
        return None, None

    if strategy == "greedy":
        node_counts = Counter(u for u, _ in undirected_edges if u in nodes_to_consider)
        return max(node_counts, key=node_counts.get, default=None), 0

    adj_undirected = {node: {tuple(sorted(e)) for e in undirected_edges if node in e} 
                      for node in nodes_to_consider}
    
    sampling_success = False
    orientation_totals = {node: {} for node in nodes_to_consider}
    for _ in range(EST_ITERS):
        dag_sample = sample_dags(graph, n_samples=N_SAMPLES)

        if len(dag_sample) / N_SAMPLES < THRESHOLD:
            continue
        else:
            sampling_success = True

        for node in nodes_to_consider:
            edges = adj_undirected[node]
            if not edges:
                continue

            orientation_classes = Counter(
                tuple("out" if dag.has_edge(node, v if node == u else u) else "in"
                      for u, v in edges)
                for dag in dag_sample
            )

            for orientation_tuple, count in orientation_classes.items():
                orientation_totals[node][orientation_tuple] = orientation_totals[node].get(orientation_tuple, 0) + count / len(dag_sample)

    if not sampling_success:
        return choose_intervention_variable(graph, intervened, strategy="greedy")[0], 1

    for node in nodes_to_consider:
        for orientation_tuple in orientation_totals[node].keys():
            orientation_totals[node][orientation_tuple] /= EST_ITERS

    node_metrics = {}
    for node in nodes_to_consider:
        if strategy == "entropy":
            entropy = -sum(p * log(p) 
                          for p in orientation_totals[node].values() if p > 0)
            node_metrics[node] = entropy
        elif strategy == "minimax":
            node_metrics[node] = max(orientation_totals[node].values())

    if not node_metrics:
        return None, None
    
    node = (max if strategy == "entropy" else min)(node_metrics, key=node_metrics.get)
    return node, 0