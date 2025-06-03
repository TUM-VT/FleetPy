import torch
from HeteroGAT import HeteroGAT

def test_hetero_gat():
    # Test parameters
    hidden_channels = 64
    out_channels = 2
    num_layers = 2
    
    # Create model
    model = HeteroGAT(hidden_channels, out_channels, num_layers)
    
    # Test case 1: Graph with edges
    x_dict = {
        'request': torch.randn(5, 10, dtype=torch.float),  # 5 request nodes with 10 features each
        'vehicle': torch.randn(3, 10, dtype=torch.float)   # 3 vehicle nodes with 10 features each
    }
    
    edge_index_dict = {
        ('request', 'connects', 'request'): torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),  # 3 edges between requests
        ('vehicle', 'connects', 'request'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),        # 2 edges from vehicles to requests
        ('request', 'rev_connects', 'vehicle'): torch.tensor([[], []], dtype=torch.long)             # No edges for this type
    }
    
    edge_attr_dict = {
        ('request', 'connects', 'request'): torch.randn(3, 5, dtype=torch.float),  # 3 edges with 5 features each
        ('vehicle', 'connects', 'request'): torch.randn(2, 5, dtype=torch.float),  # 2 edges with 5 features each
        ('request', 'rev_connects', 'vehicle'): torch.tensor([], dtype=torch.float) # No edge attributes for this type
    }
    
    # Test forward pass with edges
    out = model(x_dict, edge_index_dict, edge_attr_dict)
    print("Test case 1 (with edges):")
    print(f"Output shape: {out.shape}")
    print(f"Output values: {out}")
    
    # Test case 2: Graph without edges
    empty_edge_index_dict = {
        ('request', 'connects', 'request'): torch.tensor([[], []], dtype=torch.long),
        ('vehicle', 'connects', 'request'): torch.tensor([[], []], dtype=torch.long),
        ('request', 'rev_connects', 'vehicle'): torch.tensor([[], []], dtype=torch.long)
    }
    
    empty_edge_attr_dict = {
        ('request', 'connects', 'request'): torch.tensor([], dtype=torch.float),
        ('vehicle', 'connects', 'request'): torch.tensor([], dtype=torch.float),
        ('request', 'rev_connects', 'vehicle'): torch.tensor([], dtype=torch.float)
    }
    
    # Test forward pass without edges
    out_empty = model(x_dict, empty_edge_index_dict, empty_edge_attr_dict)
    print("\nTest case 2 (without edges):")
    print(f"Output shape: {out_empty.shape}")
    print(f"Output values: {out_empty}")

if __name__ == "__main__":
    test_hetero_gat() 