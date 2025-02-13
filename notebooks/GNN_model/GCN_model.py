import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Compute edge embeddings by concatenating node embeddings
        edge_emb = x[edge_index[0]] * x[edge_index[1]]  # Element-wise product
        return torch.sigmoid(edge_emb.sum(dim=1))  # Binary classification output
