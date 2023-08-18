import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
import networkx as nx
import numpy as np
import torch.nn.functional as F


@register_node_encoder('GIMaskEncoder')
class GraphInvariantMaskEncoder(torch.nn.Module):
    def __init__(self, dim_emb=None):
        super().__init__()
        pecfg = cfg.posenc_GIMaskEncoder
        model_type = pecfg.model

        if model_type not in ['mlp', 'Linear']:
            raise ValueError(f"Unexpected PE model {model_type}")
        
        self.model_type = model_type

        n_layers = pecfg.layers  # Num. layers in PE encoder model
        num_clusters = cfg.gnn.num_clusters
        graph_invariant = pecfg.graph_invariant
        dim_in = len(graph_invariant)
        hidden_dim = pecfg.hidden_dim
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.batch_norm = pecfg.batch_norm
        self.n_layers = n_layers

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        self.bn = nn.ModuleList()
        self.model = nn.ModuleList()
        if n_layers == 1:
            self.model.append(nn.Linear(dim_in, num_clusters))
            self.bn.append(nn.BatchNorm1d(num_clusters))
        else:
            self.model.append(nn.Linear(dim_in, hidden_dim))
            self.bn.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(n_layers - 2):
                self.model.append(nn.Linear(dim_in, hidden_dim))
                self.bn.append(nn.BatchNorm1d(hidden_dim))
            self.model.append(nn.Linear(hidden_dim, num_clusters))
            self.bn.append(nn.BatchNorm1d(num_clusters))



        # if model_type == 'mlp':
        #     layers = []
        #     if n_layers == 1:
        #         layers.append(nn.Linear(dim_in, num_clusters))
        #         layers.append(activation())
        #         self.bn.append(nn.BatchNorm1d(num_clusters))
        #     else:
        #         layers.append(nn.Linear(dim_in, hidden_dim))
        #         self.bn.append(nn.BatchNorm1d(hidden_dim))
        #         layers.append(activation()) 
        #         for _ in range(n_layers - 2):
        #             layers.append(nn.Linear(hidden_dim, hidden_dim))
        #             layers.append(activation())
        #             self.bn.append(nn.BatchNorm1d(hidden_dim))
        #         layers.append(nn.Linear(hidden_dim, num_clusters))
        #         layers.append(activation())
        #         self.bn.append(nn.BatchNorm1d(hidden_dim))
        #     self.encoder = nn.Sequential(*layers)
        # elif model_type == 'linear':
        #     self.encoder = nn.Linear(dim_in, num_clusters)
        # else:
        #     raise ValueError(f"{self.__class__.__name__}: Does not support "
        #             f"'{model_type}' encoder model.")
    
    def forward(self, batch):
        if not (hasattr(batch, 'encoding')):
            raise ValueError("Precomputed graph-invariant encoding is "
                f"required for {self.__class__.__name__}; "
                "set config 'posenc_GIMaskEncoder.enable' to True")
        encoding = batch.encoding
        if self.raw_norm:
            encoding = self.raw_norm(encoding)
        for i in range(self.n_layers):
            encoding = self.model[i](encoding)
            if self.batch_norm:
                encoding = self.bn[i](encoding)
        # encoding = self.encoder(encoding)
            encoding = F.relu(encoding)
        raw_masks = F.softmax(encoding, dim=-1)
        masks = torch.transpose(raw_masks, 0, 1)
        batch.masks = masks
        device =batch.x.device
        batch.complement_masks = torch.ones(masks.shape).to(device) - masks
        return batch
