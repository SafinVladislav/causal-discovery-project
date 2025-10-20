import networkx as nx
import math
import random
from collections import Counter
from pathlib import Path
import sys
from core.graph_utils import to_undirected_with_v_structures
from core.intervention import choose_intervention_variable

# Add the project root to Python path to enable imports
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent.parent  # Adjust based on your structure
sys.path.insert(0, str(PROJECT_ROOT))

# Import from existing modules
from core.graph_utils import (
    find_undirected_edges, 
    has_directed_cycle, 
    has_v_structure, 
    is_bad_graph,
    propagate_orientations,
    generate_dag_from_cpdag,
    sample_dags
)
from models.graphs import create_model
from core.intervention import N_SAMPLES

def all_dags(graph):
    """
    Exact enumeration of all DAGs in the Markov equivalence class.
    """
    def recursive_all(graph):
        undirected = find_undirected_edges(graph)
        if not undirected:
            yield graph.copy()
            return
        
        undirected_set = set(tuple(sorted(e)) for e in undirected)
        any_edge = sorted(undirected_set)[0]
        u, v = any_edge
        
        # Try u -> v
        temp = graph.copy()
        if temp.has_edge(v, u):
            temp.remove_edge(v, u)
            if not is_bad_graph(temp, (u, v)):
                temp2 = temp.copy()
                propagate_orientations(temp2)
                for dag in recursive_all(temp2):
                    yield dag
        
        # Try v -> u
        temp = graph.copy()
        if temp.has_edge(u, v):
            temp.remove_edge(u, v)
            if not is_bad_graph(temp, (v, u)):
                temp2 = temp.copy()
                propagate_orientations(temp2)
                for dag in recursive_all(temp2):
                    yield dag

    return list(recursive_all(graph))

def compute_exact_metrics(graph):
    """
    Compute exact theoretical metrics for intervention strategies.
    """
    all_dag_list = all_dags(graph)
    total = len(all_dag_list)
    
    if total == 0:
        return {}, {}, {}, 0
    
    undirected_edges = find_undirected_edges(graph)
    
    # Build adjacency for undirected edges per node
    adj_undirected = {}
    for node in sorted(graph.nodes()):
        node_edges = set()
        for u, v in set(tuple(sorted(e)) for e in undirected_edges):
            if (node == u) or (node == v):
                node_edges.add(tuple(sorted((u, v))))
        adj_undirected[node] = node_edges
    
    # Greedy metrics (simple edge counts)
    greedy_counts = Counter()
    for e in set(tuple(sorted(e)) for e in undirected_edges):
        u, v = e
        greedy_counts[u] += 1
        greedy_counts[v] += 1
    
    # Entropy and Minimax metrics
    entropy_metrics = {}
    minimax_metrics = {}
    
    for node in sorted(graph.nodes()):
        edges = list(adj_undirected[node])
        if not edges:
            continue
            
        # Count orientation patterns across all DAGs
        orientation_classes = Counter()
        for dag in all_dag_list:
            pattern = []
            for edge in edges:
                u, v = edge
                if node == u:  # edge is (node, v)
                    orientation = 'out' if dag.has_edge(node, v) and not dag.has_edge(v, node) else 'in'
                else:  # edge is (u, node)
                    orientation = 'out' if dag.has_edge(node, u) and not dag.has_edge(u, node) else 'in'
                pattern.append(orientation)
            orientation_classes[tuple(pattern)] += 1
        
        # Compute entropy
        probs = [count / total for count in orientation_classes.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        entropy_metrics[node] = entropy
        
        # Compute minimax (maximum class size)
        max_size = max(orientation_classes.values())
        minimax_metrics[node] = max_size
    
    return greedy_counts, entropy_metrics, minimax_metrics, total

def analyze_strategies_for_model(true_graph):
    """
    Analyze intervention strategies for a given model.
    """
    print(f"\n{'='*60}")
    essential_graph = to_undirected_with_v_structures(true_graph)
    
    print(f"Essential graph edges: {len(essential_graph.edges())}")
    print(f"Undirected edges: {len(find_undirected_edges(essential_graph))}")
    
    # Compute theoretical metrics
    print("\n--- THEORETICAL METRICS (Exact Enumeration) ---")
    greedy_counts, entropy_metrics, minimax_metrics, total_dags = compute_exact_metrics(essential_graph)
    
    print(f"Total DAGs in MEC: {total_dags}")
    if greedy_counts:
        print("Greedy counts (edge degree):", dict(greedy_counts))
        max_greedy = max(greedy_counts.values())
        best_greedy_nodes = [node for node, count in greedy_counts.items() if count == max_greedy]
        print(f"Best for greedy: {best_greedy_nodes} (edge count: {max_greedy})")

    if entropy_metrics:
        print("Entropy metrics:", {k: round(v, 4) for k, v in entropy_metrics.items()})
        max_entropy = max(entropy_metrics.values())
        best_entropy_nodes = [node for node, entropy in entropy_metrics.items() if entropy == max_entropy]
        print(f"Best for entropy: {best_entropy_nodes} (entropy: {max_entropy:.4f})")
    
    if minimax_metrics:
        print("Minimax metrics:", minimax_metrics)
        min_minimax = min(minimax_metrics.values())
        best_minimax_nodes = [node for node, minimax in minimax_metrics.items() if minimax == min_minimax]
        print(f"Best for minimax: {best_minimax_nodes} (max class: {min_minimax})")
    
    # Compute sampled metrics
    print("\n--- SAMPLED METRICS ---")
    
    greedy_choice, _ = choose_intervention_variable(essential_graph, set(), "greedy")
    print(f"Sampled greedy choice: {greedy_choice}")
    
    entropy_choice, _ = choose_intervention_variable(essential_graph, set(), "entropy")
    print(f"Sampled entropy choice: {entropy_choice}")
    
    minimax_choice, _ = choose_intervention_variable(essential_graph, set(), "minimax")
    print(f"Sampled minimax choice: {minimax_choice}")

    print("\n--- COMPARISON ---")
    strategies_match = {
        'greedy': greedy_choice in best_greedy_nodes if greedy_counts else False,
        'entropy': entropy_choice in best_entropy_nodes if entropy_metrics else False,
        'minimax': minimax_choice in best_minimax_nodes if minimax_metrics else False
    }
    
    for strategy, matches in strategies_match.items():
        if not matches:
            return False
        #print(f"{strategy.upper():<10} {status} Theoretical and sampled choices {'match' if matches else 'differ'}")
    return True