from collections import defaultdict

import torch
from torch.nn import Linear, BatchNorm1d, InstanceNorm1d, Dropout
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.instance_norms = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)
        
        for _ in range(num_layers):
            conv = HeteroConv({
                ('request', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
                ('vehicle', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
                ('request', 'rev_connects', 'vehicle'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
            
            # Add batch normalization for each node type
            batch_norm = torch.nn.ModuleDict({
                'request': BatchNorm1d(hidden_channels * 2),
                'vehicle': BatchNorm1d(hidden_channels * 2)
            })
            self.batch_norms.append(batch_norm)
            
            # Add instance normalization for each node type (used when batch size is 1)
            instance_norm = torch.nn.ModuleDict({
                'request': InstanceNorm1d(hidden_channels * 2),
                'vehicle': InstanceNorm1d(hidden_channels * 2)
            })
            self.instance_norms.append(instance_norm)
            
        # Linear layer will be initialized in forward pass with correct input size
        self.lin = None
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Convert all inputs to float32 and check for NaN values
        x_dict = {key: torch.nan_to_num(x.float(), nan=0.0) for key, x in x_dict.items()}
        edge_attr_dict = {key: torch.nan_to_num(attr.float(), nan=0.0) for key, attr in edge_attr_dict.items()}
        
        # Check if there are any edges in the graph
        has_edges = any(len(edges[0]) > 0 for edges in edge_index_dict.values())
        
        if not has_edges:
            # If no edges, return a tensor with requires_grad=True
            empty_tensor = torch.zeros((0, self.out_channels), device=x_dict['request'].device, dtype=torch.float32)
            empty_tensor.requires_grad_(True)
            return empty_tensor
            
        for conv, batch_norm, instance_norm in zip(self.convs, self.batch_norms, self.instance_norms):
            # Apply GAT convolution
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            # Apply normalization and activation for each node type
            x_dict = {
                key: torch.nn.functional.leaky_relu(
                    # Use instance norm if batch size is 1, otherwise use batch norm
                    instance_norm[key](x) if x.size(0) == 1 else batch_norm[key](x),
                    negative_slope=0.2
                ) for key, x in x_dict.items()
            }
            
            # Apply dropout
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Extract embeddings for edge classification, maintaining edge type structure
        edge_features_dict = {}
        for edge_type, edges in edge_index_dict.items():
            if len(edges[0]) == 0:  # Skip empty edge types
                empty_tensor = torch.zeros((0, self.out_channels), 
                                         device=x_dict['request'].device, 
                                         dtype=torch.float32)
                empty_tensor.requires_grad_(True)
                edge_features_dict[edge_type] = empty_tensor
                continue
                
            edge_features = []
            for edge_ind, edge in enumerate(zip(edges[0], edges[1])):
                # Get source and target node features
                src_features = x_dict[edge_type[0]][edge[0]]
                tgt_features = x_dict[edge_type[2]][edge[1]]
                # Get edge attributes
                edge_attrs = edge_attr_dict[edge_type][edge_ind]
                
                # Project edge attributes to match node feature dimensions if needed
                if edge_attrs.size(0) != self.hidden_channels:
                    if not hasattr(self, f'edge_proj_{edge_type}'):
                        setattr(self, f'edge_proj_{edge_type}', 
                               Linear(edge_attrs.size(0), self.hidden_channels).to(edge_attrs.device))
                    edge_attrs = getattr(self, f'edge_proj_{edge_type}')(edge_attrs)
                
                # Concatenate all features
                new_edge_features = torch.cat([src_features, edge_attrs, tgt_features])
                edge_features.append(new_edge_features)
            
            # Stack features for this edge type
            edge_features = torch.stack(edge_features)
            
            # Initialize linear layer if not already done
            if self.lin is None:
                input_size = edge_features.size(1)
                self.lin = Linear(input_size, self.out_channels).to(edge_features.device)
            
            # Get predictions for this edge type using softmax
            edge_features_dict[edge_type] = torch.nn.functional.softmax(self.lin(edge_features), dim=1)
        
        # Concatenate predictions in the same order as the target tensor
        final_output = torch.cat([edge_features_dict[edge_type] for edge_type in edge_index_dict.keys()])
        return final_output
