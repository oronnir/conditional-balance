import torch
import torch.nn as nn
import torch.optim as optim
import os
from glob import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        self.device = device
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_and_process_tensors(directory):
    """Load tensors and calculate differences between styled and baseline versions"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_files = glob(os.path.join(directory, "*.pt"))
    
    # Group tensors by content ID
    content_groups = defaultdict(dict)
    style_names = set()
    content_ids = set()
    
    print("Processing files...")
    for file_path in tensor_files:
        filename = Path(file_path).stem
        if filename.startswith('ref'):
            continue
            
        # Parse filename
        content_id = filename.split('_')[0]
        if '""_gen' in filename:  # This is a baseline image
            style_name = "baseline"
        else:
            style_start = filename.find('_') + 1
            style_end = filename.find('_gen')
            if style_end == -1:
                continue
            style_name = filename[style_start:style_end - 1]
            
        print(f"File: {filename}, Content: {content_id}, Style: {style_name}")
        
        tensor = torch.load(file_path)
        if tensor.shape == (4, 128, 128):
            content_groups[content_id][style_name] = tensor.float()
            if style_name != "baseline":
                style_names.add(style_name)
            content_ids.add(content_id)
    
    # Calculate differences
    diff_tensors = []
    style_labels = []
    content_labels = []
    
    style_names = sorted(list(style_names))
    content_ids = sorted(list(content_ids))
    style_to_idx = {style: idx for idx, style in enumerate(style_names)}
    content_to_idx = {content: idx for idx, content in enumerate(content_ids)}
    
    print("\nCalculating differences...")
    print(f"Found {len(content_groups)} content groups")
    for content_id, style_dict in content_groups.items():
        print(f"Content {content_id}: {len(style_dict)} styles")
        if "baseline" in style_dict:
            baseline = style_dict["baseline"]
            for style_name, tensor in style_dict.items():
                if style_name != "baseline":
                    diff = tensor - baseline
                    diff_tensors.append(diff)
                    style_labels.append(style_to_idx[style_name])
                    content_labels.append(content_to_idx[content_id])
    
    if not diff_tensors:
        raise ValueError("No differences were calculated. Check file parsing logic.")
    
    diff_tensors = torch.stack(diff_tensors).to(device)
    style_labels = torch.tensor(style_labels, device=device)
    content_labels = torch.tensor(content_labels, device=device)
    
    return diff_tensors, style_labels, content_labels, style_names, content_ids

def train_model(model, tensors, labels, num_epochs=5000, name="", lambda_l1=0.5):
    ce_criterion = nn.CrossEntropyLoss().to(model.device)
    l1_criterion = nn.L1Loss().to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print(f"\nTraining {name} classifier...")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(tensors)
        
        # Calculate both losses
        ce_loss = ce_criterion(outputs, labels)
        l1_loss = l1_criterion(model.classifier.weight, torch.zeros_like(model.classifier.weight))
        
        # Combine losses
        total_loss = ce_loss + lambda_l1 * l1_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(tensors)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).sum().item() / len(labels)
                print(f'Epoch [{epoch+1}/{num_epochs}], CE Loss: {ce_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, Accuracy: {accuracy:.4f}')
            model.train()
    
    return model

def analyze_importance_ratio(style_model, content_model, shape=(4, 128, 128)):
    """Analyze ratio of style importance to content importance"""
    style_weights = style_model.classifier.weight.data.cpu()
    content_weights = content_model.classifier.weight.data.cpu()
    
    # Reshape weights to original tensor dimensions
    style_weights = style_weights.reshape(-1, *shape)
    content_weights = content_weights.reshape(-1, *shape)
    
    # Compute importance maps using L2 norm
    style_importance = torch.norm(style_weights, dim=0)
    content_importance = torch.norm(content_weights, dim=0)
    
    # Compute ratio (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    ratio_map = style_importance / (content_importance + epsilon)
    
    # Compute channel-wise statistics
    channel_ratios = []
    for c in range(4):
        channel_ratio = ratio_map[c].mean().item()
        channel_ratios.append(channel_ratio)
        print(f"Channel {c} average style/content ratio: {channel_ratio:.4f}")
    
    return ratio_map, style_importance, content_importance, channel_ratios

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    directory = "/home/oron_nir/code/balance/conditional-balance/outputs/canny/42/"
    
    print("Loading tensors and computing differences...")
    tensors, style_labels, content_labels, styles, contents = load_and_process_tensors(directory)
    print(f"Processed {len(tensors)} difference tensors")
    print(f"Number of styles: {len(styles)}")
    print(f"Number of content classes: {len(contents)}")
    
    input_dim = 4 * 128 * 128
    
    # Train style classifier
    style_model = Classifier(input_dim, len(styles), device)
    style_model = train_model(style_model, tensors, style_labels, name="style", lambda_l1=0.01)
    
    # Train content classifier
    content_model = Classifier(input_dim, len(contents), device)
    content_model = train_model(content_model, tensors, content_labels, name="content", lambda_l1=0.01)
    
    # Analyze importance ratios
    ratio_map, style_imp, content_imp, channel_ratios = analyze_importance_ratio(style_model, content_model)
    
    # Find top locations with highest style/content ratio
    flat_ratio = ratio_map.flatten()
    top_k = 20
    values, indices = torch.topk(flat_ratio, top_k)
    
    print("\nTop locations with highest style/content importance ratio:")
    for val, idx in zip(values, indices):
        c = idx.item() // (128 * 128)
        h = (idx.item() % (128 * 128)) // 128
        w = idx.item() % 128
        print(f"Channel {c}, Position ({h}, {w}): {val:.2f} ratio")
    
    # Visualize the results
    plt.figure(figsize=(15, 12))
    
    # Plot style importance
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.imshow(style_imp[i].numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Style Importance Ch.{i}\nAvg Ratio: {channel_ratios[i]:.3f}')
    
    # Create and save a visual representation of the aggregated importance map over all channels in greyscale from the top 256 locations
    # top_k = 256
    # values, indices = torch.topk(flat_ratio, top_k)
    # print(f"Top {top_k} locations with highest style/content importance ratio:")
    # for val, idx in zip(values, indices):
    #     c = idx.item() // (128 * 128)
    #     h = (idx.item() % (128 * 128)) // 128
    #     w = idx.item() % 128
    #     print(f"Channel {c}, Position ({h}, {w}): {val:.2f} ratio")
    
    # # Create a mask for the top locations
    # mask = torch.zeros_like(ratio_map)
    # for val, idx in zip(values, indices):
    #     c = idx.item() // (128 * 128)
    #     h = (idx.item() % (128 * 128)) // 128
    #     w = idx.item() % 128
    #     mask[c, h, w] = 1
    
    # # Create a visual representation of the aggregated importance map over all channels
    # summed_importance = torch.sum(mask, dim=0).numpy()
    # plt.figure(figsize=(15, 12))
    # plt.imshow(summed_importance, cmap='gray')
    # plt.colorbar()
    # plt.title('Sum of All Channels')
    
    # # Save the importance visualization
    # plt.tight_layout()
    # plt.savefig('style_content_ratio.png')
    # print("\nSaved visualization to 'style_content_ratio.png'")
    
    # # use the importance map to create a visual representation of the sum of all channels of size 256 and plot a tSNE visualization showing all images
    # # in a 2D scatterplot with content as shape and style as color with visualize_embeddings_tsne()
    # embedding_top_k = tensors[indices].cpu().numpy().reshape(-1, 256)
    # visualize_embeddings_tsne(embedding_top_k, style_labels.cpu().numpy(), content_labels.cpu().numpy(), output_file='tsne_visualization.png')
    # print("Saved t-SNE visualization to 'tsne_visualization.png'")

    # Modify this part in your main() function
    # First, get the mask indices for the top 256 locations
    flat_ratio = ratio_map.flatten()
    top_k = 256
    values, indices = torch.topk(flat_ratio, top_k)

    # Get the actual coordinates for these locations
    mask_coordinates = []
    for idx in indices:
        c = idx.item() // (128 * 128)
        h = (idx.item() % (128 * 128)) // 128
        w = idx.item() % 128
        mask_coordinates.append((c, h, w))

    # Extract just these locations from all tensors
    # Each tensor will be reduced to a vector of length 256
    reduced_tensors = torch.zeros((tensors.shape[0], top_k), device=tensors.device)

    for i, tensor in enumerate(tensors):
        for j, (c, h, w) in enumerate(mask_coordinates):
            reduced_tensors[i, j] = tensor[c, h, w]

    # Now use the reduced tensors for the t-SNE visualization
    visualize_embeddings_tsne(
        reduced_tensors.cpu().numpy(), 
        style_labels.cpu().numpy(), 
        content_labels.cpu().numpy(), 
        output_file='tsne_reduced_visualization.png'
    )
    print("Saved reduced dimension t-SNE visualization to 'tsne_reduced_visualization.png'")

    print("Done.")

if __name__ == "__main__":
    main()
