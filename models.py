from torch.functional import F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, MLP
import torch

class GIN(torch.nn.Module):
    def __init__(self, in_channels, mlp_hidden_channels, mlp_output_channels, mlp_num_layers, gin_hidden_channels, gin_output_channels, gin_num_layers, out_channels, dropout_rate):
        super(GIN, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.mlp_linear = MLP(in_channels=int(in_channels/2), hidden_channels=mlp_hidden_channels, out_channels=mlp_output_channels, num_layers=mlp_num_layers)
        
        self.gin_layers = torch.nn.ModuleList([
            GINConv(
                Sequential(
                    Linear(mlp_output_channels, gin_hidden_channels),
                    BatchNorm1d(gin_hidden_channels),
                    ReLU(),
                    Linear(gin_hidden_channels, gin_hidden_channels),
                    ReLU()
                )
            ) if i == 0 else
            GINConv(
                Sequential(
                    Linear(gin_hidden_channels, gin_hidden_channels),
                    BatchNorm1d(gin_hidden_channels),
                    ReLU(),
                    Linear(gin_hidden_channels, gin_output_channels),
                    ReLU()
                )
            ) if i == gin_num_layers-1 else
            GINConv(
                Sequential(
                    Linear(gin_hidden_channels, gin_hidden_channels),
                    BatchNorm1d(gin_hidden_channels),
                    ReLU(),
                    Linear(gin_hidden_channels, gin_hidden_channels),
                    ReLU()
                )
            )
            for i in range(gin_num_layers)
        ])
        
        self.out = MLP(in_channels=gin_output_channels, hidden_channels=int(gin_output_channels/3), out_channels=out_channels, num_layers=3)

    def forward(self, x, edge_index):
        x1 = self.mlp_linear(x[:,:int(len(x[0])/2)])
        x2 = self.mlp_linear(x[:,int(len(x[0])/2):])
        x = x1+x2
    
        # gin
        for conv in self.gin_layers:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Classifier
        x = self.out(x)
        
        return x