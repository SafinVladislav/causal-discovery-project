import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import os

def visualize_graphs(true_graph, pc_essential_graph, oriented_graph, pic_path):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    
    pos = graphviz_layout(true_graph, prog='dot')

    nx.draw(
        true_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightgreen',
        font_size=12,
        font_weight='bold',
        ax=ax[0]
    )
    ax[0].set_title('True Graph')

    nx.draw(
        pc_essential_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightgreen',
        font_size=12,
        font_weight='bold',
        ax=ax[1]
    )
    ax[1].set_title('Essential Graph')

    nx.draw(
        oriented_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightcoral',
        font_size=12,
        font_weight='bold',
        ax=ax[2]
    )
    ax[2].set_title('Oriented Graph')

    try:
        directory = os.path.dirname(pic_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(pic_path, bbox_inches='tight')
        print(f"Graph visualization successfully saved to: {pic_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close(fig)