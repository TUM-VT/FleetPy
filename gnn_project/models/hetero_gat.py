import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import HeteroConv, GATv2Conv, GATConv
import logging

from gnn_project.defaults import RR_EDGE_NAME, VR_EDGE_NAME, RV_EDGE_NAME, REQUEST, VEHICLE

logger = logging.getLogger(__name__)

class HeteroGAT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rr_edge_dim is None or config.vr_edge_dim is None:
            raise ValueError(
                "config.rr_edge_dim/vr_edge_dim must be set before building HeteroGAT "
                "(train_utils infers them from loaded graphs or a saved checkpoint)"
            )
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(config.dropout)
        self.layernorms = torch.nn.ModuleList()
        # RV shares VR's edge_attr/dim - created from VR edges via T.ToUndirected(merge=True)
        self.edge_projs = torch.nn.ModuleDict()
        self.edge_stack_dim = config.hidden_channels * 3  # src + edge_attr + tgt
        edge_dim = {
            RR_EDGE_NAME: config.rr_edge_dim,
            VR_EDGE_NAME: config.vr_edge_dim,
            RV_EDGE_NAME: config.vr_edge_dim,
        }
        for et, dim in edge_dim.items():
            key = f"edge_proj_{et}"
            self.edge_projs[key] = Linear(dim, config.hidden_channels)

        for _ in range(config.num_layers):
            self.layernorms.append(torch.nn.ModuleDict({
                ntype: torch.nn.LayerNorm(config.hidden_channels) for ntype in [REQUEST, VEHICLE]
            }))
            # Create GAT convolutions for original and reversed edge types
            conv_dict = {
                RR_EDGE_NAME: GATConv((-1, -1), config.hidden_channels, heads=config.heads, add_self_loops=False, concat=False, dropout=config.dropout, residual=True),
                VR_EDGE_NAME: GATConv((-1, -1), config.hidden_channels, heads=config.heads, add_self_loops=False, concat=False, dropout=config.dropout, residual=True),
                RV_EDGE_NAME: GATConv((-1, -1), config.hidden_channels, heads=config.heads, add_self_loops=False, concat=False, dropout=config.dropout, residual=True),
            }
            
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)
            
        # Output layers with intermediate layer
        self.out_channels = config.num_classes
        self.hidden_channels = config.hidden_channels
        self.lin1 = Linear(self.edge_stack_dim, config.hidden_channels)
        self.lin2 = Linear(config.hidden_channels, config.num_classes)

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
            empty_tensor = torch.zeros((0, self.out_channels), device=x_dict[REQUEST].device, dtype=torch.float32)
            empty_tensor.requires_grad_(True)
            return empty_tensor
        
        for key, x in x_dict.items():
            if torch.isnan(x).any():
                logger.warning(f"NaNs detected in input {key}: {x}")

        for i, conv in enumerate(self.convs):
            # Apply GAT convolution
            x_dict_out = conv(x_dict, edge_index_dict, edge_attr_dict)
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
                                         device=x_dict[REQUEST].device, 
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
            edge_features = self.dropout(torch.nn.functional.leaky_relu(self.lin1(edge_features)))
            edge_features_dict[edge_type] = self.lin2(edge_features)  # No activation here - using BCEWithLogitsLoss

        # Concatenate logits
        final_output = torch.cat([edge_features_dict[edge_type] for edge_type in edge_index_dict.keys()])
        return final_output
