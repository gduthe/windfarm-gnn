import torch.nn as nn
from .basics import MLP
import torch

class Encoder(nn.Module):
    ''' Encoder class, uses MLPs for each graph attribute.
        In this framework the inflow conditions are provided as global attributes and the relative geometry of the farm
        are given as edge attributes. There are no node attributes initially, so the globals are used to create initial 
        encoded node features. 
        
        $\mathbf{h}_{i}^{(0)} = \mathrm{MLP}_{ENC, node} (\textit{U})$
        $\mathbf{a}_{i, j} = \mathrm{MLP}_{ENC, edge} (\mathbf{e}_{i, j})$

        args:
        node_enc_mlp_layers: list of ints, the dimensions of the MLP layers for the node encoder
        node_latent_dim: int, the dimension of the node latent space
        edge_feature_dim: int, the dimension of the edge features
        edge_enc_mlp_layers: list of ints, the dimensions of the MLP layers for the edge encoder
        edge_latent_dim: int, the dimension of the edge latent space
        dropout: float, the dropout probability, used in all MLPs
        
    '''

    def __init__(self, **kwargs):
        super().__init__()
        # make sure the required arguments are provided
        assert {'node_feature_dim', 'node_enc_mlp_layers', 'node_latent_dim', 'edge_feature_dim', 'edge_enc_mlp_layers',
                'edge_latent_dim', 'encode_globals', 'glob_feature_dim', 'glob_enc_mlp_layers', 'glob_latent_dim', 'dropout'}.issubset(kwargs)
        
        # Flag to encode global attributes
        self.encode_globals = kwargs['encode_globals']
        
        # initialize the node, edge and global params encoder MLPs
        # In this case, we encode global attributes and yaw angles to both node and global features
        self.node_encoder           = MLP(kwargs['node_feature_dim'] + kwargs['glob_feature_dim'],
                                          kwargs['node_enc_mlp_layers'] + [kwargs['node_latent_dim']],
                                          activation_type='LeakyReLU', norm_type='LayerNorm', dropout=kwargs['dropout'])
        self.edge_encoder           = MLP(kwargs['edge_feature_dim'],
                                          kwargs['edge_enc_mlp_layers'] + [kwargs['edge_latent_dim']],
                                          activation_type='ReLU', norm_type='LayerNorm', dropout=kwargs['dropout'])
        if kwargs['encode_globals']:
            self.global_encoder     = MLP(kwargs['glob_feature_dim'],
                                          kwargs['glob_enc_mlp_layers'] + [kwargs['glob_latent_dim']],
                                          activation_type='ReLU', norm_type='LayerNorm', dropout=kwargs['dropout'])

    def forward(self, node_attr, edge_attr, global_attr, batch):
        if self.encode_globals:
            global_attr_enc = self.global_encoder(global_attr)
        else:
            global_attr_enc = global_attr

        if node_attr is not None:
            node_attr_enc = self.node_encoder(torch.cat([node_attr, global_attr[batch]], dim=1))
        else:
            node_attr_enc = self.node_encoder(global_attr)[batch]

        edge_attr_enc = self.edge_encoder(edge_attr)

        return node_attr_enc, edge_attr_enc, global_attr_enc