import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.register import register_network
from graphgps.layer.vgn_conv_layer import VGNConvLayer


class MaskEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(MaskEncoder, self).__init__()
        self.dim_in = dim_in
        print("Dim_in: ", dim_in)
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('VGNModel')
class VGNModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mask_encoder = MaskEncoder(dim_in)
        dim_in = self.mask_encoder.dim_in

        layers = []
        hidden_dim = cfg.gnn.dim_inner
        num_clusters = cfg.gnn.num_clusters
        for layer in range(cfg.gnn.layers_mp):
            if layer == 0:
                layers.append(VGNConvLayer(
                    dim_in=dim_in,
                    dim_out=hidden_dim,
                    num_clusters=num_clusters,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual
                    ))
            else:
                layers.append(VGNConvLayer(
                    dim_in=hidden_dim,
                    dim_out=hidden_dim,
                    num_clusters=num_clusters,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual
                    ))
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
    
    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

# @register_network('VGNModel_v2')
# class VGNModel_v2(torch.nn.module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.mask_encoder = MaskEncoder(dim_in)
#         self.dim_in = self.mask_encoder.dim_in
#         layers = []
#         self.hidden_dim = cfg.gnn.dim_inner
#         self.train_eps = cfg.gnn.train_eps
#         self.residual = cfg.gnn.residual
#         self.dropout = cfg.gnn.dropout
#         self.batch_norm = cfg.gnn.batchnorm
#         self.bn = torch.nn.ModuleList()
#         self.num_layers = cfg.gnn.layers_mp

#         for layer in range(self.num_layers):
#             if layer == 0:
#                 mlp = nn.Sequential(pyg_nn.Linear(dim_in, self.hidden_dim), nn.ReLU(),
#                                     pyg_nn.Linear(self.hidden_dim, self.hidden_dim))
#                 if self.train_eps:
#                     layers.append(pyg_nn.GINEConv(mlp, True))
#                 else:
#                     layers.append(pyg_nn.GINEConv(mlp))
#             else:
#                 mlp = nn.Sequential(pyg_nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
#                                     pyg_nn.Linear(self.hidden_dim, self.hidden_dim))
#                 if self.train_eps:
#                     layers.append(pyg_nn.GINEConv(mlp, True))
#                 else:
#                     layers.append(pyg_nn.GINEConv(mlp))
#             self.bn.append(nn.BatchNorm1d(self.hidden_dim))
#         self.layers = torch.nn.Sequential(*layers)
#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

#     def forward(self, batch):
#         masks = batch.masks
#         for layer in range(self.num_layers):
#             x_in = batch.x.clone()
#             batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr)
#             if self.residual:
#                 batch.x = torch.einsum('i,ij->ij', masks[layer], batch.x) + x_in
#             else:
#                 batch.x = torch.einsum('i,ij->ij', masks[layer], batch.x)
 
#             if self.batch_norm:
#                 batch.x = self.bn[layer](batch.x)
#             batch.x = F.relu(batch.x)
#             batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        