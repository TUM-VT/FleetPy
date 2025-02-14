from collections import defaultdict

import torch
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=2, num_layers=2):
        super().__init__()
        # TODO add convs for nodes
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('request', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
                ('vehicle', 'connects', 'request'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
                ('request', 'rev_connects', 'vehicle'): GATConv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)
        self.double()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Extract embeddings for edge classification
        edge_features = {}
        for edge_type, edges in edge_index_dict.items():
            edge_features[edge_type] = []
            for edge_ind, edge in enumerate(zip(edges[0], edges[1])):
                new_edge_features = torch.cat([x_dict[edge_type[0]][edge[0]], edge_attr_dict[edge_type][edge_ind], x_dict[edge_type[2]][edge[1]]])
                edge_features[edge_type].append(new_edge_features)
        return torch.sigmoid(self.lin(edge_features))
