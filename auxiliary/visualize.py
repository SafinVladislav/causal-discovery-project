import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import os
import math
from moviepy.editor import ImageSequenceClip
from pathlib import Path
from matplotlib.patches import FancyArrowPatch

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy")

def visualize_graphs(true_graph, pc_essential_graph, oriented_graph, pic_path):
    try:
        n_true = true_graph.number_of_nodes()
        n_essential = pc_essential_graph.number_of_nodes()
        n_oriented = oriented_graph.number_of_nodes()
        n = max(n_true, n_essential, n_oriented)
        
        base_scale = math.sqrt(n) if n > 0 else 1
        fig_width = max(24, 3 * base_scale)
        fig_height = max(8, base_scale * 2)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height))
        
        node_size = max(50, 5000 / base_scale)
        font_size = max(6, min(12, 120 / base_scale))
        arrowsize = max(5, font_size * 1.5)
        
        try:
            pos = graphviz_layout(true_graph, prog='sfdp')
        except Exception:
            try:
                pos = graphviz_layout(true_graph, prog='dot')
            except Exception:
                print("Graphviz has issues; using built-in layout for visualization.")
                pos = nx.kamada_kawai_layout(true_graph)
                
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
            pic_path.mkdir(parents=True, exist_ok=True)
            save_path = pic_path / f"pic.pdf"
            plt.savefig(str(save_path), format='pdf', dpi=300, bbox_inches='tight')
            print(f"Graph visualization successfully saved to: {save_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
        
        plt.close(fig)
    except Exception as e:
        print(f"Visualization error.")

def visualize_step(current_graph, intervened_node, all_oriented, step, pic_dir: Path):
    try:
        n = current_graph.number_of_nodes()
        
        base_scale = math.sqrt(n) if n > 0 else 1
        fig_width = max(8, base_scale * 2)
        fig_height = max(8, base_scale * 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        node_size = max(50, 5000 / base_scale)
        font_size = max(6, min(12, 120 / base_scale))
        arrowsize = max(5, font_size * 1.5)
        
        try:
            pos = graphviz_layout(current_graph, prog='sfdp')
        except Exception:
            try:
                pos = graphviz_layout(current_graph, prog='dot')
            except Exception:
                print("Graphviz has issues; using built-in layout for visualization.")
                pos = nx.kamada_kawai_layout(current_graph)
                
        # Node colors: yellow for intervened, olive for others
        node_colors = ['yellow' if n == intervened_node else 'olive' for n in current_graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            current_graph,
            pos,
            node_size=node_size,
            node_color=node_colors,
            ax=ax
        )
        
        for u, v in current_graph.edges():
            arrow = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle='->',
                mutation_scale=arrowsize,
                linewidth=4,
                color='black',
                shrinkA=(node_size**0.5)/2,
                shrinkB=(node_size**0.5)/2,
                connectionstyle="arc3,rad=0",
                zorder=2
            )
            ax.add_patch(arrow)

        for u, v in all_oriented:
            arrow = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle='->', 
                mutation_scale=arrowsize,
                linewidth=4,
                color='red',
                shrinkA=(node_size**0.5)/2,
                shrinkB=(node_size**0.5)/2,
                connectionstyle="arc3,rad=0",
                zorder=2
            )
            ax.add_patch(arrow)
        
        nx.draw_networkx_labels(
            current_graph,
            pos,
            font_size=font_size,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(f"Step {step}: Intervened on {intervened_node}")
        
        try:
            pic_dir.mkdir(parents=True, exist_ok=True)
            save_path = pic_dir / f"comp_step{step}.png"
            plt.savefig(str(save_path), format='png', dpi=300, bbox_inches='tight')
            print(f"Step visualization saved to: {save_path}")
            
        except Exception as e:
            print(f"Error saving file: {e}")
        
        plt.close(fig)
    except Exception as e:
        print(f"Visualization error: {e}")

def make_video_from_pngs(vis_dir: Path, fps: int = 0.3):
    """Compile all PNGs into an MP4 video."""
    try:
        png_files = sorted([str(p) for p in vis_dir.glob('*.png') if p.is_file()])
        if not png_files:
            print("No PNG files found for video compilation.")
            return
        
        clip = ImageSequenceClip(png_files, fps=fps)
        video_path = vis_dir / 'orientation_video.mp4'

        from moviepy.video.fx.all import resize
        clip = clip.resize(lambda t: (clip.w + 1 if clip.w % 2 else clip.w,
                             clip.h + 1 if clip.h % 2 else clip.h))

        clip.write_videofile(
            str(video_path),
            fps=fps,
            codec='libx264',
            audio_codec=None,
            preset='medium',
            bitrate="5000k",
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-movflags', '+faststart'
            ],
            threads=4,
            logger=None
        )
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error creating video: {e}")