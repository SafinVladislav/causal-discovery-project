import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx

def visualize_graphs(essential_graph, oriented_graph, pic_path):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    
    pos = graphviz_layout(essential_graph, prog='dot')

    nx.draw(
        essential_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightgreen',
        font_size=12,
        font_weight='bold',
        ax=ax[0]
    )
    ax[0].set_title('Essential Graph')

    nx.draw(
        oriented_graph,
        pos=pos,
        with_labels=True,
        node_size=2000,
        node_color='lightcoral',
        font_size=12,
        font_weight='bold',
        ax=ax[1]
    )
    ax[1].set_title('Oriented Graph')

    try:
        plt.savefig(pic_path, bbox_inches='tight')
        print(f"Graph visualization successfully saved to: {pic_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close(fig)