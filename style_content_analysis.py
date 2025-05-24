import torch
import torch.nn as nn
import torch.optim as optim
import os
from glob import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        self.device = device
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
        # tensor = (min_max_scale(tensor[0,:,:])*min_max_scale(tensor[2,:,:])-min_max_scale(tensor[1,:,:])*min_max_scale(tensor[3,:,:])).squeeze()
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
        content_num = name_parts[0]
        style_tensors[(artist, content_num)] = tensor - style_pt
    return style_tensors

def load_tensors(directory):
    """Load tensors and prepare both style and content labels"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_files = glob(os.path.join(directory, "*.pt"))
    tensors = []
    style_names = []
    content_ids = []
    
    for file_path in tensor_files:
        tensor = torch.load(file_path)
        if tensor.shape != (4, 128, 128):
            print(f"Skipping file {file_path} with shape {tensor.shape}")
            continue

        tensors.append(tensor.float())
        filename = Path(file_path).stem
        
        # Extract content ID (first part before underscore)
        content_id = filename.split("_")[0]
        if content_id == "ref":  # Handle reference images
            content_id = "ref"
        content_ids.append(content_id)
        
        # Extract style name
        parts = filename.split('_')
        if len(parts) > 1:
            style_name = '_'.join(parts[1:-1])
            style_name = style_name.replace('"', '').replace('.png', '')
        else:
            style_name = 'unknown'
        style_names.append(style_name)
    
    tensors = torch.stack(tensors).to(device)
    
    # Create style labels
    unique_styles = sorted(list(set(style_names)))
    style_to_idx = {style: idx for idx, style in enumerate(unique_styles)}
    style_labels = torch.tensor([style_to_idx[name] for name in style_names], device=device)
    
    # Create content labels
    unique_contents = sorted(list(set(content_ids)))
    content_to_idx = {content: idx for idx, content in enumerate(unique_contents)}
    content_labels = torch.tensor([content_to_idx[cid] for cid in content_ids], device=device)
    
    return tensors, style_labels, content_labels, unique_styles, unique_contents

def train_model(model, tensors, labels, num_epochs=1000):
    criterion = nn.CrossEntropyLoss().to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(tensors)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).sum().item() / len(labels)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
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
    
    return ratio_map, style_importance, content_importance

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    directory = "/home/oron_nir/code/balance/conditional-balance/outputs/canny/42/"
    
    print("Loading tensors...")
    # tensors, style_labels, content_labels, styles, contents = load_tensors(directory)

    tensors, style_labels, content_labels, styles, contents = load_pt_files(directory)

    print(f"Loaded {len(tensors)} tensors")
    print(f"Number of styles: {len(styles)}")
    print(f"Number of content classes: {len(contents)}")
    
    input_dim = 4 * 128 * 128
    
    # Train style classifier
    print("\nTraining style classifier...")
    style_model = Classifier(input_dim, len(styles), device)
    style_model = train_model(style_model, tensors, style_labels)
    
    # Train content classifier
    print("\nTraining content classifier...")
    content_model = Classifier(input_dim, len(contents), device)
    content_model = train_model(content_model, tensors, content_labels)
    
    # Analyze importance ratios
    ratio_map, style_imp, content_imp = analyze_importance_ratio(style_model, content_model)
    
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
        plt.title(f'Style Importance Channel {i}')
    
    # Plot content importance
    for i in range(4):
        plt.subplot(3, 4, i+5)
        plt.imshow(content_imp[i].numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Content Importance Channel {i}')
    
    # Plot ratio
    for i in range(4):
        plt.subplot(3, 4, i+9)
        plt.imshow(ratio_map[i].numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Style/Content Ratio Channel {i}')
    
    plt.tight_layout()
    plt.savefig('style_content_ratio.png')
    print("\nSaved visualization to 'style_content_ratio.png'")
    
    # Save the importance maps
    torch.save({
        'ratio_map': ratio_map,
        'style_importance': style_imp,
        'content_importance': content_imp
    }, 'importance_maps.pt')
    print("Saved importance maps to 'importance_maps.pt'")

if __name__ == "__main__":
    main()
