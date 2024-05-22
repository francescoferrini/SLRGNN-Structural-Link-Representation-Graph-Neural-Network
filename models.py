from torch.functional import F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, MLP
import torch

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_mlp_layers, num_gin_layers, dropout_rate, num_initial_gin_layers):
        super(GIN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_gin_layers = num_gin_layers
        
        self.initial_gin_layers = torch.nn.ModuleList([
                GINConv(
                        Sequential(Linear(in_channels, hidden_channels),
                        BatchNorm1d(hidden_channels), ReLU(),
                        Linear(hidden_channels, hidden_channels), ReLU()))
                if i ==0 else 
                GINConv(
                        Sequential(Linear(hidden_channels, hidden_channels),
                        BatchNorm1d(hidden_channels), ReLU(),
                        Linear(hidden_channels, hidden_channels), ReLU()))
                for i in range(num_initial_gin_layers)
            ])
        
        self.mlp_linear = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=num_mlp_layers)
        self.gin_layers = torch.nn.ModuleList([
                GINConv(
                        Sequential(Linear(hidden_channels, hidden_channels),
                        BatchNorm1d(hidden_channels), ReLU(),
                        Linear(hidden_channels, hidden_channels), ReLU()))
                for i in range(num_gin_layers)
            ])
        
        self.out = Linear(hidden_channels, out_channels)

    def forward(self, edge_index, x_original, edge_index_original, edge_dict_original):
        for conv in self.initial_gin_layers:
            x_original = conv(x_original, edge_index_original)
            x_original = F.dropout(x_original, p=self.dropout_rate, training=self.training)
            
        x = torch.stack([torch.cat((x_original[value[0]], x_original[value[1]]), dim=0) for value in edge_dict_original.values()])
        
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