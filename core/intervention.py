# Standard library imports
import contextlib
import io
import random
from math import log
from collections import defaultdict
from core.graph_utils import find_undirected_edges, sample_dags

# Third-party library imports
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def silent_simulate(model, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return model.simulate(*args, **kwargs)

def quasi_experiment(model, var, samples):
    intervened_model = model.copy()
    old_cpd = intervened_model.get_cpds(var)
    old_values = old_cpd.values.copy().reshape(old_cpd.cardinality[0], -1)

    new_values = np.zeros_like(old_values)
    for i in range(old_values.shape[1]):
        shifted_probs = old_values[:, i].copy()
        shifted_probs[0] = round(shifted_probs[0])
        shifted_probs[1] = 1 - shifted_probs[0]
        new_values[:, i] = shifted_probs

    new_cpd = TabularCPD(
        variable=var,
        variable_card=old_cpd.cardinality[0],
        values=new_values,
        evidence=old_cpd.variables[1:],
        evidence_card=old_cpd.cardinality[1:]
    )
    try:
        intervened_model.remove_cpds(var)
    except Exception:
        pass
    intervened_model.add_cpds(new_cpd)
    return silent_simulate(intervened_model, samples, show_progress=False)

def choose_intervention_variable(graph: any, intervened: set, n_samples: int = 1000, strategy: str = "entropy") -> str | None:
    if strategy not in ["greedy", "minimax", "entropy"]:
      raise ValueError("No such strategy.")

    undirected_edges = find_undirected_edges(graph)
    if not undirected_edges:
        return None

    nodes_to_consider = sorted(list({n for edge in undirected_edges for n in edge} - intervened))
    if not nodes_to_consider:
        return None

    if strategy == "greedy":
        # The greedy strategy is a simple heuristic based on degree.
        # It's also used as a fallback if sampling yields too few DAGs.
        node_counts = defaultdict(int)
        for u, _ in undirected_edges:
            if u not in intervened:
                node_counts[u] += 1
        return max(node_counts, key=node_counts.get) if node_counts else None

    # Sample DAGs, this is common to both 'entropy' and 'minimax' strategies.
    dag_sample = sample_dags(graph, n_samples=n_samples)
    if not dag_sample:
        # Fallback to greedy if no samples could be generated.
        #print("fallback")
        return choose_intervention_variable(graph, intervened, strategy="greedy")

    node_metrics = {}
    for node in nodes_to_consider:
        adjacent_undirected = {tuple(sorted(e)) for e in undirected_edges if node in e}
        if not adjacent_undirected:
            continue

        # Map each DAG sample to an "orientation class" for the current node.
        orientation_classes = defaultdict(int)
        for dag in dag_sample:
            orientation_key = tuple(
                "out" if dag.has_edge(node, neighbor if node == u else u) else "in"
                for u, neighbor in adjacent_undirected
            )
            orientation_classes[orientation_key] += 1

        if strategy == "entropy":
            # Calculate the entropy of the orientation classes.
            entropy = -sum(
                (p / n_samples) * log(p / n_samples)
                for p in orientation_classes.values() if p > 0
            )
            node_metrics[node] = entropy
            #print(f"{node}{orientation_classes.values()}")
        elif strategy == "minimax":
            # Find the size of the largest orientation class.
            max_class_size = max(orientation_classes.values())
            node_metrics[node] = max_class_size

    if not node_metrics:
        return None

    # Return the best variable based on the chosen strategy.
    if strategy == "entropy":
        # Maximize entropy to gain the most information.
        return max(node_metrics, key=node_metrics.get)
    elif strategy == "minimax":
        # Minimize the size of the largest class to reduce worst-case uncertainty.
        return min(node_metrics, key=node_metrics.get)