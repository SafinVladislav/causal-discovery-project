import contextlib
import io
from math import log
from collections import Counter
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from core.graph_utils import find_undirected_edges, sample_dags
from pgmpy.sampling import BayesianModelSampling

def silent_simulate(model, samples, show_progress=False):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sampler = BayesianModelSampling(model)
        return sampler.forward_sample(size=samples)

def quasi_experiment(model, var, samples):
    intervened_model = model.copy()
    old_cpd = intervened_model.get_cpds(var)
    values = old_cpd.values.reshape(old_cpd.cardinality[0], -1)
    
    new_values = np.zeros_like(values)
    max_indices = np.argmin(values, axis=0)
    new_values[max_indices, np.arange(values.shape[1])] = 1.0
    #print(new_values.shape)
    #print(values[0])
    #print(values[1])

    state_names = {var: old_cpd.state_names[var]}
    evidence = old_cpd.variables[1:] if len(old_cpd.variables) > 1 else None
    if evidence:
        for ev_var in evidence:
            ev_cpd = model.get_cpds(ev_var)
            if ev_cpd:
                state_names[ev_var] = ev_cpd.state_names[ev_var]
            else:
                for node in model.nodes():
                    node_cpd = model.get_cpds(node)
                    if node_cpd and ev_var in node_cpd.state_names:
                        state_names[ev_var] = node_cpd.state_names[ev_var]
                        break

    new_cpd = TabularCPD(
        variable=var,
        variable_card=old_cpd.cardinality[0],
        values=new_values,
        evidence=old_cpd.variables[1:],
        evidence_card=old_cpd.cardinality[1:],
        state_names=state_names
    )
    intervened_model.remove_cpds(old_cpd)
    intervened_model.add_cpds(new_cpd)
    return silent_simulate(intervened_model, samples, show_progress=False)

N_SAMPLES = 1000
EST_ITERS = 3
THRESHOLD = 0.7

def choose_intervention_variable(graph, intervened, strategy="entropy"):
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
        #print(node_counts)
        return max(node_counts, key=node_counts.get, default=None), 1

    #dag_sample = sample_dags(graph, n_samples=N_SAMPLES)
    #if len(dag_sample) / N_SAMPLES < THRESHOLD:
    #    return choose_intervention_variable(graph, intervened, strategy="greedy")

    adj_undirected = {node: {tuple(sorted(e)) for e in undirected_edges if node in e} 
                      for node in nodes_to_consider}
    
    sampling_success = False
    orientation_totals = {node: {} for node in nodes_to_consider}
    for _ in range(EST_ITERS):
        dag_sample = sample_dags(graph, n_samples=N_SAMPLES)
        #for dag in dag_sample:
        #    print(dag.edges())

        if len(dag_sample) / N_SAMPLES < THRESHOLD:
                continue
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
        return choose_intervention_variable(graph, intervened, strategy="greedy")

    for node in nodes_to_consider:
        #print(node)
        for orientation_tuple in orientation_totals[node].keys():
            orientation_totals[node][orientation_tuple] /= EST_ITERS
            #print(orientation_totals[node][orientation_tuple])
    

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
    #print(f"\nNode to intervene: {node}. It's metric: {node_metrics[node]}.")
    #print(f"All metrics: {node_metrics}.")
    return node, strategy == "greedy"