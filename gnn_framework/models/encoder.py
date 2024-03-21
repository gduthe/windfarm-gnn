import torch.nn as nn
from .basics import MLP

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

        assert {'node_enc_mlp_layers', 'node_latent_dim',
                'edge_feature_dim', 'edge_enc_mlp_layers', 'edge_latent_dim',
                'encode_globals', 'glob_enc_mlp_layers', 'glob_latent_dim', 'dropout'}.issubset(kwargs)
        
        # initialize the node, edge and global params encoder MLPs
        self.node_encoder = MLP(kwargs['glob_feature_dim'],
                                kwargs['node_enc_mlp_layers'] + [kwargs['node_latent_dim']],
                                activation_type='ReLU', norm_type='LayerNorm', dropout=kwargs['dropout'])
        self.edge_encoder = MLP(kwargs['edge_feature_dim'],
                                kwargs['edge_enc_mlp_layers'] + [kwargs['edge_latent_dim']],
                                activation_type='ReLU', norm_type='LayerNorm', dropout=kwargs['dropout'])

    def forward(self, edge_attr, global_attr, batch):
        x_enc = self.node_encoder(global_attr)[batch] # the batch index is needed for multi-graph training
        edge_attr_enc = self.edge_encoder(edge_attr)

        return x_enc, edge_attr_enc