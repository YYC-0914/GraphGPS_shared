import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder

class KernelMaskEncoder(torch.nn.Module):
    
    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb=None):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}MaskEncoder")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable
        num_clusters = cfg.gnn.num_clusters


        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]

        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, num_clusters))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation()) 
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, num_clusters))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, num_clusters)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                    f"'{model_type}' encoder model.")
            


    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}_mask"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")
        # pos_enc = batch.pestat_var
        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        # print("pos_enc: ", pos_enc)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        raw_masks = F.softmax(pos_enc, dim=-1)
        masks = torch.transpose(raw_masks, 0, 1)
        batch.masks = masks
        device = batch.x.device
        batch.complement_masks = torch.ones(masks.shape).to(device) - masks
        # print("Learned mask for RWSEMaskEncoder: ", masks)
        return batch


@register_node_encoder('RWSEMaskEncoder')
class RWSEMaskEncoder(KernelMaskEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'


@register_node_encoder('HKdiagSEMaskEncoder')
class HKdiagSEMaskEncoder(KernelMaskEncoder):
    """Heat kernel (diagonal) Structural Encoding node encoder.
    """
    kernel_type = 'HKdiagSE'


@register_node_encoder('ElstaticSEMaskEncoder')
class ElstaticSEMaskEncoder(KernelMaskEncoder):
    """Electrostatic interactions Structural Encoding node encoder.
    """
    kernel_type = 'ElstaticSE'
