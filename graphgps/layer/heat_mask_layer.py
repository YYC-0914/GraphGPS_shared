import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg



class HeatConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers):
        super().__init__()
        self.mask_encoder = MaskEncoderLayer()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        for layer in range(n_layers):
            if layer == 0:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters
                ))
            else:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_out,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters
                ))
            self.bns.append(nn.BatchNorm1d(dim_out))

    def forward(self, batch, cur_layer):
        self.mask_encoder(batch, cur_layer)
        for i in range(self.n_layers):
            self.layers[i](batch, cur_layer)
        return batch
                

class MaskEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        n_layers = cfg.gnn.GIMaskEncoder_layers
        num_clusters = cfg.gnn.num_clusters
        graph_invariant = cfg.gnn.GIMaskEncoder_graph_invariant
        dim_in = len(graph_invariant)
        hidden_dim = cfg.gnn.GIMaskEncoder_hidden_dim
        norm_type = cfg.gnn.GIMaskEncoder_raw_norm_type
        self.batch_norm = cfg.gnn.GIMaskEncoder_batch_norm
        self.n_layers = n_layers


        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.raw_norm = None

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

    def forward(self, batch, cur_layer):
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
        cur_mask = "masks_" + str(cur_layer)
        # if not (hasattr(batch, cur_mask)):
        setattr(batch, cur_mask, masks)
        device =batch.x.device
        batch.complement_masks = torch.ones(masks.shape).to(device) - masks
        return batch



class HeatConvLayer(nn.Module):
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
        
    def forward(self, batch, cur_layer):
        x_in = batch.x
        cur_mask = "masks_" + str(cur_layer)
        masks = getattr(batch, cur_mask)
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
        
