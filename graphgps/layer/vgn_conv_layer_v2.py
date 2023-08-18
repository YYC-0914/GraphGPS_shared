import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg


class VGNConvLayer_v2(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers, scaling, dropout=0., train_eps=True, batch_norm=True, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.hidden_dim = dim_hidden
        self.train_eps = train_eps
        self.model = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.scaling = scaling

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = nn.Sequential(pyg_nn.Linear(dim_in, self.hidden_dim), nn.ReLU(),
                                    pyg_nn.Linear(self.hidden_dim, self.hidden_dim))
                if self.train_eps:
                    self.model.append(pyg_nn.GINEConv(mlp, True))
                else:
                    self.model.append(pyg_nn.GINEConv(mlp))
            else:
                mlp = nn.Sequential(pyg_nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                    pyg_nn.Linear(self.hidden_dim, self.hidden_dim))
                if self.train_eps:
                    self.model.append(pyg_nn.GINEConv(mlp, True))
                else:
                    self.model.append(pyg_nn.GINEConv(mlp))
            self.bn.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, batch):
        masks = batch.masks
        # device = batch.x.device
        masks = self.scaling * masks
        for layer in range(self.num_layers):
            x_in = batch.x.clone()
            batch.x = self.model[layer](batch.x, batch.edge_index, batch.edge_attr)
            if self.residual:
                batch.x = torch.einsum('i,ij->ij', masks[layer], batch.x) + x_in
            else:
                batch.x = torch.einsum('i,ij->ij', masks[layer], batch.x)
 
            if self.batch_norm:
                batch.x = self.bn[layer](batch.x)
            batch.x = F.relu(batch.x)
            batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        return batch
        
            
        

        
            








