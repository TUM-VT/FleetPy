from sklearn.model_selection import train_test_split
import torch


def get_cons_tensor(size, fill_ones):
    if fill_ones:
        return torch.ones(size, dtype=torch.bool)
    else:
        return torch.zeros(size, dtype=torch.bool)

def get_masks(data):
    # Create masks based on time order
    num_graphs = len(data)
    
    # Use first 70% for training
    train_size = int(0.7 * num_graphs)
    # Use next 15% for validation
    val_size = int(0.15 * num_graphs)
    
    # Create masks
    train_mask = torch.zeros(num_graphs, dtype=torch.bool)
    val_mask = torch.zeros(num_graphs, dtype=torch.bool)
    test_mask = torch.zeros(num_graphs, dtype=torch.bool)
    
    # Assign masks in time order
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True
    
    print(f"Data split: Train={train_size}, Val={val_size}, Test={num_graphs - train_size - val_size}")
    return train_mask, val_mask, test_mask
