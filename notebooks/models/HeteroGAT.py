import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=1, num_layers=2, dropout=0.6, heads=8):  # Changed out_channels to 1 for binary classification
        super().__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)
        
        # Initialize projection layers for each edge type
        self.edge_projs = torch.nn.ModuleDict()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                ('request', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
                ('vehicle', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
                ('request', 'rev_connects', 'vehicle'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
            
        # Output layers with intermediate layer
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lin1 = Linear(2176, hidden_channels)  # src, edge, tgt features concatenated
        self.lin2 = Linear(hidden_channels, out_channels)

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
        print(f"[DEBUG] Model mode: {'train' if self.training else 'eval'}, Dropout p: {self.dropout.p}")
        # Debug: Print input node and edge feature stats
        # for key, x in x_dict.items():
        #     print(f"[MODEL DEBUG] Input node '{key}' features: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}")
        # for key, attr in edge_attr_dict.items():
        #     print(f"[MODEL DEBUG] Input edge '{key}' attr: mean={attr.mean().item():.4f}, std={attr.std().item():.4f}, min={attr.min().item():.4f}, max={attr.max().item():.4f}")
        
        # Check if there are any edges in the graph
        has_edges = any(len(edges[0]) > 0 for edges in edge_index_dict.values())
        
        if not has_edges:
            # Return empty tensor with requires_grad=True
            empty_tensor = torch.zeros((0, self.out_channels), device=x_dict['request'].device, dtype=torch.float32)
            empty_tensor.requires_grad_(True)
            return empty_tensor
            
        for conv in self.convs:
            # Apply GAT convolution
            x_dict_out = conv(x_dict, edge_index_dict, edge_attr_dict)

            # Debug: Print intermediate node embedding stats
            # for key, x in x_dict_out.items():
            #     print(f"[MODEL DEBUG] After GAT '{key}' embedding: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            
            # Apply activation and dropout for each node type
            for key in x_dict_out:
                if key in x_dict:  # Apply residual connection if shapes match
                    if x_dict[key].shape == x_dict_out[key].shape:
                        # Use .clone() to avoid in-place modification for autograd safety
                        x_dict_out[key] = x_dict_out[key] + x_dict[key].clone()
                # Only activation and dropout
                x_dict_out[key] = torch.nn.functional.leaky_relu(x_dict_out[key])
                # Apply dropout (out-of-place)
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
            # Debug: Print edge features stats before MLP
            print(f"[DEBUG] {edge_type} - edge_features shape: {edge_features.shape}")
            print(f"[MODEL DEBUG] Edge type {edge_type} features before MLP: mean={edge_features.mean().item():.4f}, std={edge_features.std().item():.4f}, min={edge_features.min().item():.4f}, max={edge_features.max().item():.4f}")
            edge_features = self.dropout(torch.nn.functional.leaky_relu(self.lin1(edge_features)))
            edge_features_dict[edge_type] = self.lin2(edge_features)  # No activation here - using BCEWithLogitsLoss
            # Debug: Print logits for this edge type
            print(f"[MODEL DEBUG] Edge type {edge_type} logits: mean={edge_features_dict[edge_type].mean().item():.4f}, std={edge_features_dict[edge_type].std().item():.4f}, min={edge_features_dict[edge_type].min().item():.4f}, max={edge_features_dict[edge_type].max().item():.4f}")
        
        # Concatenate logits
        final_output = torch.cat([edge_features_dict[edge_type] for edge_type in edge_index_dict.keys()])
        # Debug: Print final output logits stats
        # print(f"[MODEL DEBUG] Final output logits: mean={final_output.mean().item():.4f}, std={final_output.std().item():.4f}, min={final_output.min().item():.4f}, max={final_output.max().item():.4f}")
        return final_output
