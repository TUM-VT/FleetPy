from collections import defaultdict

import torch
from torch.nn import Linear, BatchNorm1d, InstanceNorm1d, Dropout
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=1, num_layers=2, dropout=0.2):  # Changed out_channels to 1 for binary classification
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.instance_norms = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)
        
        # Initialize projection layers for each edge type
        self.edge_projs = torch.nn.ModuleDict()
        
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
            
        # Output layers with intermediate layer
        self.lin1 = None  # Will be initialized in forward pass
        self.lin2 = None  # Final layer
        self.final_norm = BatchNorm1d(hidden_channels)  # Intermediate batch norm
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        for conv in self.convs:
            for conv_layer in conv.convs.values():
                if hasattr(conv_layer, 'reset_parameters'):
                    conv_layer.reset_parameters()
        
        if self.lin1 is not None:
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.zeros_(self.lin1.bias)
        if self.lin2 is not None:
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Convert all inputs to float32 and handle NaN values
        x_dict = {key: torch.nan_to_num(x.float(), nan=0.0) for key, x in x_dict.items()}
        edge_attr_dict = {key: torch.nan_to_num(attr.float(), nan=0.0) for key, attr in edge_attr_dict.items()}
        
        # Check if there are any edges in the graph
        has_edges = any(len(edges[0]) > 0 for edges in edge_index_dict.values())
        
        if not has_edges:
            # Return empty tensor with requires_grad=True
            empty_tensor = torch.zeros((0, self.out_channels), device=x_dict['request'].device, dtype=torch.float32)
            empty_tensor.requires_grad_(True)
            return empty_tensor
            
        for conv, batch_norm, instance_norm in zip(self.convs, self.batch_norms, self.instance_norms):
            # Apply GAT convolution
            x_dict_out = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            # Apply normalization, activation, and residual connection for each node type
            for key in x_dict_out:
                if key in x_dict:  # Apply residual connection if shapes match
                    if x_dict[key].shape == x_dict_out[key].shape:
                        x_dict_out[key] = x_dict_out[key] + x_dict[key]
                
                # Normalize and activate
                x_dict_out[key] = torch.nn.functional.leaky_relu(
                    instance_norm[key](x_dict_out[key]) if x_dict_out[key].size(0) == 1 
                    else batch_norm[key](x_dict_out[key])
                )
                
                # Apply dropout
                x_dict_out[key] = self.dropout(x_dict_out[key])
            
            x_dict = x_dict_out

        # Extract embeddings for edge classification
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
                src_features = x_dict[edge_type[0]][edge[0]]
                tgt_features = x_dict[edge_type[2]][edge[1]]
                edge_attrs = edge_attr_dict[edge_type][edge_ind]
                
                # Get or create edge projection layer
                edge_proj_key = f'edge_proj_{edge_type}'
                if edge_attrs.size(0) != self.hidden_channels:
                    if edge_proj_key not in self.edge_projs:
                        self.edge_projs[edge_proj_key] = Linear(edge_attrs.size(0), self.hidden_channels).to(edge_attrs.device)
                    edge_attrs = self.edge_projs[edge_proj_key](edge_attrs)
                
                # Concatenate all features
                edge_features.append(torch.cat([src_features, edge_attrs, tgt_features]))
            
            # Stack features for this edge type
            edge_features = torch.stack(edge_features)
            
            # Initialize linear layers if not already done
            if self.lin1 is None:
                input_size = edge_features.size(1)
                self.lin1 = Linear(input_size, self.hidden_channels).to(edge_features.device)
                self.lin2 = Linear(self.hidden_channels, self.out_channels).to(edge_features.device)
                self._reset_parameters()
            
            # Two-layer MLP with batch norm and dropout
            edge_features = self.dropout(torch.nn.functional.leaky_relu(self.final_norm(self.lin1(edge_features))))
            edge_features_dict[edge_type] = self.lin2(edge_features)  # No activation here - using BCEWithLogitsLoss
        
        # Concatenate logits
        final_output = torch.cat([edge_features_dict[edge_type] for edge_type in edge_index_dict.keys()])
        return final_output
