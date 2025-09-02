import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import HeteroConv, GATv2Conv, GATConv
import logging

logger = logging.getLogger(__name__)

class HeteroGAT(torch.nn.Module):
    EDGE_FEAT_DIM = 96

    def __init__(self, hidden_channels, out_channels=1, num_layers=2, dropout=0.3, heads=4):
        super().__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)
        # Add LayerNorms for each layer
        self.layernorms = torch.nn.ModuleList()
        # Initialize projection layers for each edge type
        self.edge_projs = torch.nn.ModuleDict()
        for _ in range(num_layers):
            self.layernorms.append(torch.nn.ModuleDict({
                ntype: torch.nn.LayerNorm(hidden_channels) for ntype in ['request', 'vehicle']
            }))
            # Create GAT convolutions for original and reversed edge types
            conv_dict = {
                ('request', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, concat=False, dropout=dropout, residual=True),
                ('vehicle', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, concat=False, dropout=dropout, residual=True),
                ('request', 'rev_connects', 'vehicle'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, concat=False, dropout=dropout, residual=True),
            }
            # The undirected transform will add these automatically with identical parameters
            # but we don't need to define them here
            
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)
            
        # Output layers with intermediate layer
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lin1 = Linear(HeteroGAT.EDGE_FEAT_DIM, hidden_channels)  # src, edge, tgt features concatenated. TODO parameterize
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
        # Check if there are any edges in the graph
        has_edges = any(len(edges[0]) > 0 for edges in edge_index_dict.values())
        
        if not has_edges:
            # Return empty tensor with requires_grad=True
            empty_tensor = torch.zeros((0, self.out_channels), device=x_dict['request'].device, dtype=torch.float32)
            empty_tensor.requires_grad_(True)
            return empty_tensor
        
        # DEBUG: check if there are NaNs in the input
        for key, x in x_dict.items():
            if torch.isnan(x).any():
                logger.warning(f"NaNs detected in input '{key}'")
                x_dict[key] = torch.nan_to_num(x)

        for i, conv in enumerate(self.convs):
            # Apply GAT convolution
            x_dict_out = conv(x_dict, edge_index_dict, edge_attr_dict)
            # for key, x in x_dict_out.items():
            #     logger.debug(f"After GAT '{key}' embedding: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            # Apply LayerNorm, activation, and dropout for each node type
            for key in x_dict_out:
                if key in x_dict:  # Apply residual connection if shapes match
                    if x_dict[key].shape == x_dict_out[key].shape:
                        x_dict_out[key] = x_dict_out[key] + x_dict[key].clone()
                # LayerNorm (if available for this node type)
                if key in self.layernorms[i]:
                    x_dict_out[key] = self.layernorms[i][key](x_dict_out[key])
                # Activation
                x_dict_out[key] = torch.nn.functional.leaky_relu(x_dict_out[key])
                # Dropout
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
            # logger.debug(f"{edge_type} - edge_features shape: {edge_features.shape}")
            # logger.debug(f"Edge type {edge_type} features before MLP: mean={edge_features.mean().item():.4f}, std={edge_features.std().item():.4f}, min={edge_features.min().item():.4f}, max={edge_features.max().item():.4f}")
            edge_features = self.dropout(torch.nn.functional.leaky_relu(self.lin1(edge_features)))
            edge_features_dict[edge_type] = self.lin2(edge_features)  # No activation here - using BCEWithLogitsLoss
            # logger.debug(f"Edge type {edge_type} logits: mean={edge_features_dict[edge_type].mean().item():.4f}, std={edge_features_dict[edge_type].std().item():.4f}, min={edge_features_dict[edge_type].min().item():.4f}, max={edge_features_dict[edge_type].max().item():.4f}")
        
        # Concatenate logits
        final_output = torch.cat([edge_features_dict[edge_type] for edge_type in edge_index_dict.keys()])
        # logger.debug(f"Final output logits: mean={final_output.mean().item():.4f}, std={final_output.std().item():.4f}, min={final_output.min().item():.4f}, max={final_output.max().item():.4f}")
        return final_output
