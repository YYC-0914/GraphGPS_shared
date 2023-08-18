import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg


class VGNConvELapSE(pyg_nn.conv.MessagePassing):
    def __init__(self, nn, eps=0., train_eps=False, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

        if hasattr(self.nn[0], 'in_features'):
            out_dim = self.nn[0].out_features
        else:
            out_dim = self.nn[0].out_channels
        
        self.mlp_r_ij = torch.nn.Sequential(
            torch.nn.Linear(1, out_dim), torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid())
        
    
    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        pyg_nn.inits.reset(self.mlp_r_ij)


    def forward(self, x, edge_index, edge_attr=None, pe_LapPE=None, size=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             PE=pe_LapPE, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)
    


    def message(self, x_j, edge_attr, PE_i, PE_j):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim

        return ((x_j + edge_attr).relu()) * r_ij

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'
        



class VGNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_clusters, dropout, train_eps=True, 
                 init_eps = 0., batch_norm=True, sublayer_residual=True, residual=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.sublayer_residual = sublayer_residual
        self.init_eps = init_eps
        self.num_clusters = num_clusters
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([init_eps]))

        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))
        
        self.model = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()

        for cluster in range(num_clusters):
            if cluster == 0:
                mlp = nn.Sequential(pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
                                    pyg_nn.Linear(dim_out, dim_out))
                if train_eps:
                    self.model.append(pyg_nn.GINEConv(mlp, True))
                else:
                    self.model.append(pyg_nn.GINEConv(mlp))
            else:
                mlp = nn.Sequential(pyg_nn.Linear(dim_out, dim_out), nn.ReLU(),
                                    pyg_nn.Linear(dim_out, dim_out))
                if train_eps:
                    self.model.append(pyg_nn.GINEConv(mlp, True))
                else:
                    self.model.append(pyg_nn.GINEConv(mlp))
            self.bn.append(nn.BatchNorm1d(dim_out))
        
    def forward(self, batch):
        x_in = batch.x
        masks = batch.masks
        complement_masks = batch.complement_masks
        for cluster in  range(self.num_clusters):
            x_tmp = batch.x.clone()
            batch.x = self.model[cluster](batch.x, batch.edge_index, batch.edge_attr)
            if self.sublayer_residual:
                batch.x = torch.einsum('i,ij->ij', masks[cluster], batch.x) + x_tmp
            else:
                batch.x = torch.einsum('i,ij->ij', masks[cluster], batch.x) + torch.einsum('i,ij->ij', complement_masks[cluster], x_tmp)

            if self.batch_norm:
                batch.x = self.bn[cluster](batch.x)
        
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        if self.residual:
            batch.x = x_in + batch.x # residual connection
        
        return batch
    

class VGNConvELapSELayer(nn.Module):
    pass
    # def __init__(self, dim_in, dim_out, num_clusters, dropout, train_eps=True, batch_norm=True,
    #              sublayer_residual=True, residual=True):
    #     super().__init__()
    #     self.dim_in = dim_in
    #     self.dim_out = dim_out
    #     self.dropout = dropout
    #     self.residual = residual
    #     self.batch_norm = batch_norm
    #     self.sublayer_residual = sublayer_residual
    #     self.num_clusters = num_clusters

    #     self.model = torch.nn.ModuleList()
    #     self.bn = torch.nn.ModuleList()

        