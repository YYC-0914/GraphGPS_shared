from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.sublayer_residual = False
    cfg.gnn.num_clusters = 3
    cfg.gnn.train_eps = False
    cfg.gnn.scaling = 3