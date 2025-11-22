import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import os
import math

"""
Visualizing true (from our model), essential (after PC), oriented (after our algorithm).
"""
def visualize_graphs(true_graph, pc_essential_graph, oriented_graph, pic_path):
    n_true = true_graph.number_of_nodes()
    n_essential = pc_essential_graph.number_of_nodes()
    n_oriented = oriented_graph.number_of_nodes()
    n = max(n_true, n_essential, n_oriented)
    
    # Dynamically scale figure size (wider for 3 subplots)
    base_scale = math.sqrt(n) if n > 0 else 1
    fig_width = max(24, 3 * base_scale)
    fig_height = max(8, base_scale * 2)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height))
    
    # Dynamically scale visual elements
    node_size = max(50, 5000 / base_scale)
    font_size = max(6, min(12, 120 / base_scale))
    arrowsize = max(5, font_size * 1.5)
    
    # Use 'sfdp' for better scaling on large graphs; fallback to 'dot' if issues
    try:
        pos = graphviz_layout(true_graph, prog='sfdp')
    except:
        pos = graphviz_layout(true_graph, prog='dot')
    
    # Draw True Graph
    nx.draw(
        true_graph,
        pos=pos,
        with_labels=True,
        node_size=node_size,
        node_color='gold',
        font_size=font_size,
        font_weight='bold',
        arrowsize=arrowsize,
        ax=ax[0]
    )
    ax[0].set_title('True Graph')
    
    # Draw Essential Graph (reuse pos)
    nx.draw(
        pc_essential_graph,
        pos=pos,
        with_labels=True,
        node_size=node_size,
        node_color='lime',
        font_size=font_size,
        font_weight='bold',
        arrowsize=arrowsize,
        ax=ax[1]
    )
    ax[1].set_title('Essential Graph')
    
    # Draw Oriented Graph (reuse pos)
    nx.draw(
        oriented_graph,
        pos=pos,
        with_labels=True,
        node_size=node_size,
        node_color='olive',
        font_size=font_size,
        font_weight='bold',
        arrowsize=arrowsize,
        ax=ax[2]
    )
    ax[2].set_title('Oriented Graph')
    
    try:
        directory = os.path.dirname(pic_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save as PDF for vector/zoomable output; high DPI for clarity
        plt.savefig(str(pic_path) + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
        print(f"Graph visualization successfully saved to: {pic_path}.pdf")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    plt.close(fig)