import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
import networkx as nx
import numpy as np
import torch.nn.functional as F

# 使用laplacian eigenvectos作为vgn masks的输入
@register_node_encoder('VGNMaskEncoder')
class VGNMaskEncoder(torch.nn.Module):
    def __init__(self, dim_emb=None):
        super().__init__()
        pecfg = cfg.posenc_VGNMaskEncoder
        dim_pe = pecfg.dim_pe # Size of Laplace PE embedding
        model_type = pecfg.model # Encoder NN model type for PEs
        
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type

        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder
        num_clusters = cfg.gnn.num_clusters
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable
        

        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = nn.ReLU

        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder =  nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            layers = [] 
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

        
        self.post_mlp = None
        layers = []
        if post_n_layers <= 1:
            layers.append(nn.Linear(dim_pe, num_clusters))
            layers.append(activation())
        else:
            layers.append(nn.Linear(dim_pe, 2 * dim_pe))
            layers.append(activation())
            for _ in range(post_n_layers - 2):
                layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                layers.append(activation())
            layers.append(nn.Linear(2 * dim_pe, num_clusters))
            layers.append(activation())
        self.post_mlp = nn.Sequential(*layers)



        
    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                        f"required for {self.__class__.__name__}; "
                        "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals_mask
        EigVecs = batch.EigVecs_mask
        
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)


        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x num_clusters
        # raw_masks = torch.normalize(pos_enc, p=1, dim=1)
        # raw_masks = F.normalize(torch.abs(pos_enc), p=1, dim=1) # 稍微没那么极端的normalization
        raw_masks = F.softmax(pos_enc, dim=-1) # (Num nodes) x num_clusters as probability 
        masks = torch.transpose(raw_masks, 0, 1)
        batch.masks = masks
        device = batch.x.device
        batch.complement_masks = torch.ones(masks.shape).to(device) - masks
        print("Learned masks: ", masks)
        return batch
        

# adding more graph_invariant options for the vgn_mask_encoder
