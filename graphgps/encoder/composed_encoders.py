import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from graphgps.encoder.ast_encoder import ASTNodeEncoder
from graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder, \
    HKdiagSENodeEncoder, ElstaticSENodeEncoder
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.ppa_encoder import PPANodeEncoder
from graphgps.encoder.signnet_pos_encoder import SignNetNodeEncoder
from graphgps.encoder.voc_superpixels_encoder import VOCNodeEncoder
from graphgps.encoder.type_dict_encoder import TypeDictNodeEncoder
from graphgps.encoder.linear_node_encoder import LinearNodeEncoder
from graphgps.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from graphgps.encoder.graphormer_encoder import GraphormerEncoder
from graphgps.encoder.vgn_mask_encoder import VGNMaskEncoder
from graphgps.encoder.kernel_mask_encoder import RWSEMaskEncoder
from graphgps.encoder.graph_invariant_mask_encoder import GraphInvariantMaskEncoder


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # print("Dim embedding: ", dim_emb)
            
            if cfg.posenc_EquivStableLapPE.enable or cfg.posenc_VGNMaskEncoder.enable: # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
                self.encoder1 = self.enc1_cls(dim_emb)
                self.encoder2 = self.enc2_cls(dim_emb)
            else:
                # PE dims can only be gathered once the cfg is loaded.
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe

                self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
                self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            # print("Dimension after first encoder: ", batch.x.shape)
            batch = self.encoder2(batch)
            # print("Dimension after second encoder: ", batch.x.shape)
            return batch
    
    class Concat3NodeEncoder_with_mask(torch.nn.Module):
        """Used specifically for VGN model to learn a maskencoding
        """
        enc1_cls = None
        mask_enc_cls = None
        mask_enc_name = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None
        def __init__(self, dim_emb):
            super().__init__()
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe) # dataset-specific node feature encoder
            self.mask_encoder = self.mask_enc_cls() # encoder for masks specifically for VGN
            self.encoder2 = self.enc2_cls(dim_emb - enc2_dim_pe, expand_x=False) 
            self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)
        
        def forward(self, batch):
            batch = self.encoder1(batch)
            # print("Encoding dimension after encoder1: ", batch.x.shape)
            batch = self.mask_encoder(batch)
            # print("Encoding dimension after mask_encoder: ", batch.x.shape)
            batch = self.encoder2(batch)
            # print("Encoding dimension after encoder2: ", batch.x.shape)
            batch = self.encoder3(batch)
            # print("Encoding dimension after encoder3: ", batch.x.shape)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # batch.pe_LapPE
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            # print("Encoding dimension of encoder1: ", batch.x.shape)
            batch = self.encoder2(batch)
            # print("Encoding dimension of encoder2: ", batch.x.shape)
            batch = self.encoder3(batch)
            # print("Encoding dimension of encoder3: ", batch.x.shape)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    elif len(encoder_classes) == 4:
        Concat3NodeEncoder_with_mask.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder_with_mask.mask_enc_cls = encoder_classes[1]
        Concat3NodeEncoder_with_mask.enc2_cls = encoder_classes[2]
        Concat3NodeEncoder_with_mask.enc3_cls = encoder_classes[3]
        Concat3NodeEncoder_with_mask.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder_with_mask.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder_with_mask
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")



# Dataset-specific node encoders.
ds_encs = {'Atom': AtomEncoder,
           'ASTNode': ASTNodeEncoder,
           'PPANode': PPANodeEncoder,
           'TypeDictNode': TypeDictNodeEncoder,
           'VOCNode': VOCNodeEncoder,
           'LinearNode': LinearNodeEncoder}

# Positional Encoding node encoders.
pe_encs = {'LapPE': LapPENodeEncoder,
           'RWSE': RWSENodeEncoder,
           'HKdiagSE': HKdiagSENodeEncoder,
           'ElstaticSE': ElstaticSENodeEncoder,
           'SignNet': SignNetNodeEncoder,
           'EquivStableLapPE': EquivStableLapPENodeEncoder,
           'GraphormerBias': GraphormerEncoder,
           'VGNMaskEncoder': VGNMaskEncoder}


# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

# Combine both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )

# Combine both SignNet and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SignNet+RWSE",
        concat_node_encoders([ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder],
                             ['SignNet', 'RWSE'])
    )

# Combine GraphormerBias with LapPE or RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+LapPE",
        concat_node_encoders([ds_enc_cls, GraphormerEncoder, LapPENodeEncoder],
                             ['GraphormerBias', 'LapPE'])
    )
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+RWSE",
        concat_node_encoders([ds_enc_cls, GraphormerEncoder, RWSENodeEncoder],
                             ['GraphormerBias', 'RWSE'])
    )


# combine VGNMaskEncoder, LapPE and RWSE together
for ds_enc_name, ds_enc_cls in ds_encs.items():
    # print("registering masked node encoder")
    register_node_encoder(
        f"{ds_enc_name}+VGNMaskEncoder+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, VGNMaskEncoder, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )
    # print(f"{ds_enc_name}+VGNMaskEncoder+LapPE+RWSE")


# combine RWSEMaskEncoder, LapPE and RWSE together
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+RWSEMaskEncoder+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, RWSEMaskEncoder, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )

# combine GIMaskEncoder, LapPE and RWSE together

for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+GIMaskEncoder+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, GraphInvariantMaskEncoder, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )
