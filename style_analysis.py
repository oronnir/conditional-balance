import torch
import torch.nn as nn
import torch.optim as optim
import os
from glob import glob
import numpy as np
from pathlib import Path

class StyleAnalyzer(nn.Module):
    def __init__(self, input_dim, num_styles, device):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_styles).to(device)
        self.device = device
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_tensors(directory):
    """Load all .pt files and prepare labels based on artist names in filenames"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_files = glob(os.path.join(directory, "*.pt"))
    tensors = []
    filenames = []
    
    for file_path in tensor_files:
        tensor = torch.load(file_path)
        if tensor.shape == (4, 128, 128):  # Verify expected shape
            # Convert to float32 for compatibility
            tensors.append(tensor.float())
            filenames.append(Path(file_path).stem)
    
    # Convert to tensor
    tensors = torch.stack(tensors).to(device)
    
    # Extract artist names (style information) from filenames
    # Format: number_artistname or ref_artistname
    artist_names = []
    for name in filenames:
        parts = name.split('\"_gen_')[0].split("_42")[0].split('_')
        if len(parts) > 1:  # Skip if filename doesn't contain underscore
            # Extract artist name and clean it
            artist_name = '_'.join(parts[1:])  # Join middle parts (exclude number and extension)
            artist_name = artist_name.replace('"', '').replace('.png', '')  # Remove quotes and extension
            artist_names.append(artist_name)
        else:
            artist_names.append('unknown')
    
    # Create labels based on unique artist names
    style_names = sorted(list(set(artist_names)))
    style_to_idx = {style: idx for idx, style in enumerate(style_names)}
    labels = torch.tensor([style_to_idx[name] for name in artist_names], device=device)
    
    return tensors, labels, style_names

def analyze_weights(model, shape=(4, 128, 128)):
    """Analyze which locations in the tensor are most important for style classification"""
    # Get the weights from the linear layer and move to CPU for analysis
    weights = model.classifier.weight.data.cpu()  # Shape: [num_styles, 4*128*128]
    
    # Reshape weights to original tensor dimensions plus style dimension
    weights = weights.reshape(-1, *shape)  # Shape: [num_styles, 4, 128, 128]
    
    # Compute the overall importance of each location across all styles
    # Using L2 norm across style dimension
    importance_map = torch.norm(weights, dim=0)  # Shape: [4, 128, 128]
    
    # Analyze channel-wise contributions
    channel_importance = torch.sum(importance_map, dim=(1, 2))
    total_importance = torch.sum(channel_importance)
    channel_percentages = channel_importance / total_importance * 100
    
    print("\nChannel-wise contribution percentages:")
    for i, percentage in enumerate(channel_percentages):
        print(f"Channel {i}: {percentage:.2f}%")
    
    return importance_map

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory containing the tensor files
    directory = "/home/oron_nir/code/balance/conditional-balance/outputs/canny/42/"
    
    # Load data
    print("Loading tensors...")
    tensors, labels, style_names = load_tensors(directory)
    
    # Check if the loaded tensors match the expected shape
    print(f"Loaded {len(tensors)} tensors with {len(style_names)} unique styles")
    print(f"Styles found: {style_names}")
    
    # Create and train model
    input_dim = 4 * 128 * 128
    num_styles = len(style_names)
    model = StyleAnalyzer(input_dim, num_styles, device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining model...")
    num_epochs = 1000
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
    
    print("\nAnalyzing weights...")
    importance_map = analyze_weights(model)
    
    # Find top contributing locations per channel
    print("\nTop contributing locations per channel:")
    for channel in range(4):
        channel_importance = importance_map[channel]
        flat_channel = channel_importance.flatten()
        total_channel_contribution = flat_channel.sum()
        
        # Get top 5 locations for this channel
        values, indices = torch.topk(flat_channel, 5)
        print(f"\nChannel {channel} top locations:")
        for val, idx in zip(values, indices):
            h = idx.item() // 128
            w = idx.item() % 128
            contribution = (val / total_channel_contribution).item() * 100
            print(f"Position ({h}, {w}): {contribution:.2f}%")
    
    # Save the importance map for visualization
    torch.save(importance_map, 'style_importance_map.pt')
    print("\nSaved importance map to 'style_importance_map.pt'")
    
    # Create and save a visual representation of the importance map
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(importance_map[i].numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Channel {i}')
    
    plt.tight_layout()
    plt.savefig('importance_visualization.png')
    print("Saved importance visualization to 'importance_visualization.png'")

    # Create and save a visualization of the sum of all channels
    plt.figure(figsize=(10, 8))
    summed_importance = torch.sum(importance_map, dim=0).numpy()
    plt.imshow(summed_importance, cmap='gray')
    plt.colorbar()
    plt.title('Sum of All Channels')
    plt.tight_layout()
    plt.savefig('summed_importance_visualization.png')
    print("Saved summed importance visualization to 'summed_importance_visualization.png'")

if __name__ == "__main__":
    main()
