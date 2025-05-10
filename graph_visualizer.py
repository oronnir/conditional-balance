import os
import glob
import torch
import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np
import argparse

import matplotlib.pyplot as plt

def min_max_scale(tensor):
    """Scale a tensor to the range [0, 1]."""
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def load_pt_files(directory):
    """Load all .pt files in the specified directory as PyTorch tensors."""
    tensors = {}
    pts = [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith('.pt')]
    for file_path in pts:
        file_name = os.path.basename(file_path)
        tensor = torch.load(file_path, weights_only=False, map_location='cpu').to(torch.float32)
        print(f"Loaded tensor {file_name} with shape {tensor.shape}")

        # # run a 2D convolution on the tensor
        # tensor = torch.nn.functional.conv2d(tensor, torch.ones(1, 1, 3, 3), padding=1)

        # scale tensor to 0-1 range
        tensor = (min_max_scale(tensor[0,:,:])*min_max_scale(tensor[2,:,:])-min_max_scale(tensor[1,:,:])*min_max_scale(tensor[3,:,:])).squeeze()
        tensors[file_name] = tensor

    style_tensors = {}
    for fn, tensor in tensors.items():
        artist = os.path.basename(file_path).split("_")[1]
        if artist == "" or fn.startswith("ref_") or "_\"\"_gen_s0_" in fn:
            print(f"Skipping tensor {fn} as it does not match the expected naming convention.")
            continue

        name_parts = os.path.basename(file_path).split("_")
        style_file_name = f"{name_parts[0]}_\"\"_gen_s0_{name_parts[-1]}"
        style_pt = tensors[style_file_name]
        style_tensors[fn] = tensor #- style_pt
    return style_tensors

def calculate_cosine_similarities(tensors):
    """Calculate cosine similarity between all pairs of tensors."""
    tensor_names = list(tensors.keys())
    n = len(tensor_names)
    similarities = {}
    
    for i in range(n):
        name_i = tensor_names[i]
        
        for j in range(i+1, n):
            name_j = tensor_names[j]
            
            try:
                # Convert to float32 or float64 to avoid precision issues
                tensor_i = np.asarray(tensors[name_i].flatten().cpu().detach().numpy(), dtype=np.float32)
                tensor_j = np.asarray(tensors[name_j].flatten().cpu().detach().numpy(), dtype=np.float32)
                
                # Check for numerical stability issues
                norm_i = np.linalg.norm(tensor_i)
                norm_j = np.linalg.norm(tensor_j)

                # Try normalizing vectors before computing similarity
                tensor_i_norm = tensor_i / (np.max(np.abs(tensor_i)) + 1e-8)
                tensor_j_norm = tensor_j / (np.max(np.abs(tensor_j)) + 1e-8)

                if np.isnan(tensor_i_norm).any() or np.isnan(tensor_j_norm).any():
                    print(f"Warning: NaN values found in normalized tensors {name_i} or {name_j}")
                    tensor_i_norm = np.nan_to_num(tensor_i_norm)
                    tensor_j_norm = np.nan_to_num(tensor_j_norm)
                if np.isinf(tensor_i_norm).any() or np.isinf(tensor_j_norm).any():
                    print(f"Warning: Inf values found in normalized tensors {name_i} or {name_j}")
                    tensor_i_norm = np.clip(tensor_i_norm, -1e8, 1e8)
                    tensor_j_norm = np.clip(tensor_j_norm, -1e8, 1e8)

                # Standard calculation
                similarity = 1 - cosine(tensor_i_norm, tensor_j_norm)
                similarities[(name_i, name_j)] = similarity
                # if np.isinf(norm_i) or np.isinf(norm_j) or norm_i == 0 or norm_j == 0:
                #     print(f"Warning: Numerical stability issues between {name_i} (norm: {norm_i}) and {name_j} (norm: {norm_j})")
                if np.isnan(similarity) or np.isinf(similarity):
                    print(f"Warning: Similarity calculation resulted in NaN or Inf between {name_i} and {name_j}")
                    similarities[(name_i, name_j)] = 0.0

            except Exception as e:
                print(f"Error calculating similarity between {name_i} and {name_j}: {e}")
                similarities[(name_i, name_j)] = 0.0
            
    return similarities

def build_graph(tensors, similarities, threshold=0.0):
    """Build a graph where nodes are tensors and edges are weighted by cosine similarity."""
    G = nx.Graph()
    
    # Add nodes
    for name in tensors.keys():
        tensor = tensors[name]
        G.add_node(name, shape=str(tensor.shape), dtype=str(tensor.dtype))
    
    # Add edges
    for (name_i, name_j), similarity in similarities.items():
        if similarity > threshold:
            G.add_edge(name_i, name_j, weight=similarity)
    
    return G

def visualize_graph(G, output_file=None):
    """Visualize the graph using NetworkX and Matplotlib."""
    plt.figure(figsize=(12, 10))
    
    # Node positions using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw edges with varying thickness based on weight
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edge labels (similarity values)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title('Tensor Similarity Graph')
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    
    plt.show()

def main(directory='.', threshold=0.75, output_file=None):
    """Main function to load tensors, build and visualize the graph."""
    print(f"Loading PyTorch tensors from {directory}...")
    tensors = load_pt_files(directory)
    print(f"Loaded {len(tensors)} tensors.")
    
    if not tensors:
        print("No tensors found. Make sure the directory contains .pt files.")
        return
    
    print("Calculating cosine similarities...")
    similarities = calculate_cosine_similarities(tensors)

    similarities = {k: np.exp(v) for k, v in similarities.items() if v > threshold}

    th = np.percentile(list(similarities.values()), threshold * 100)
    print(f"Max similarity: {max(similarities.values())}")
    
    print(f"Building graph with similarity threshold: {th}...")
    G = build_graph(tensors, similarities, th/100)
    
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    print("Visualizing graph...")
    visualize_graph(G, output_file)
    tsne_vis_name = os.path.join(directory, "tsne.png")
    mat, styles, contents = tensors_to_matrix(tensors)
    visualize_embeddings_tsne(mat, styles, contents, tsne_vis_name)

def tensors_to_matrix(tensors):
    """Convert tensors to a matrix and extract style/content labels."""
    tensor_list = []
    s_labels = []
    c_labels = []
    
    for name, tensor in tensors.items():
        tensor_list.append(tensor.flatten().cpu().detach().numpy())
        style = "_".join(name.split("_gen")[0].split("_")[1:])
        content = name.split("_")[0]
        s_labels.append(style)
        c_labels.append(content)
    
    return np.array(tensor_list), s_labels, c_labels

def visualize_embeddings_tsne(embeddings, style_labels, content_labels, output_file=None):
    """Visualize the embeddings using t-SNE. Color-code the points based on their labels."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.markers as mmarkers
    
    # Map content labels (integers) to different marker shapes
    content_labels = [int(cl) for cl in content_labels]
    unique_contents = sorted(list(set(content_labels)))
    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', '1', '2', '3', '4', 'P', '*', 'H', 'h', 'D', 'd', '|', '_']
    content_to_marker = {content: markers[i % len(markers)] for i, content in enumerate(unique_contents)}
    
    # Map style labels to numeric indices for colors
    unique_styles = list(set(style_labels))
    style_to_index = {label: i for i, label in enumerate(unique_styles)}
    style_indices = [style_to_index[label] for label in style_labels]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(14, 12))
    
    # Create separate scatter plots for each content category (to get different markers)
    for content in unique_contents:
        # Get indices for this content
        indices = [i for i, c in enumerate(content_labels) if c == content]
        
        # Plot points for this content with its marker
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1],
            c=[style_indices[i] for i in indices],
            cmap='tab20',
            marker=content_to_marker[content],
            s=100,
            alpha=0.7,
            label=f"Content {content}"
        )
    
    # Add text labels to each point
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(
            style_labels[i],
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha='center',
            fontsize=8,
            alpha=0.7
        )
    
    # Add legends
    # Style legend (colors)
    from matplotlib.lines import Line2D
    style_legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.tab20(style_to_index[style]/len(unique_styles)), 
                              markersize=10, label=style) 
                      for style in unique_styles]
    
    # Content legend (markers)
    content_legend_elements = [Line2D([0], [0], marker=content_to_marker[content], color='black',
                               markersize=8, label=f"Content {content}")
                      for content in unique_contents]
    
    # Create two legends
    plt.legend(handles=style_legend_elements, title="Artists", loc="upper right")
    plt.legend(handles=content_legend_elements, title="Content", loc="upper left", bbox_to_anchor=(1, 1), ncol=2)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    
    plt.title("t-SNE Visualization of Tensor Embeddings (Color: Artist, Shape: Content)")
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Visualize tensor similarities as a graph')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing .pt files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold for edges')
    parser.add_argument('--output', type=str, help='Output file for the graph visualization')
    
    args = parser.parse_args()
    main(args.dir, args.threshold, args.output)

    # example usage:
    # python graph_visualizer.py --dir /home/oron_nir/code/balance/conditional-balance/outputs/canny/42 --threshold 0.5 --output graph.png