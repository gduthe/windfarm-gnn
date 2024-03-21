import torch.nn as nn
from .basics import MLP

class Decoder(nn.Module):
    ''' Decoder class, uses MLPs for each graph attribute
        In this framework, we are only interested in retrieving the node attributes, so we only use a 
        node decoder.
        
        $\mathbf{x}_{i} = \mathrm{MLP}_{DEC, node} (\mathbf{h}_i^{(L)})$
        
        args:
        node_latent_dim: int, the dimension of the node latent space
        node_dec_mlp_layers: list of ints, the dimensions of the MLP layers for the node decoder
        node_out_dim: int, the dimension of the node output space
        dropout: float, the dropout probability, used in all MLPs    
    '''

    def __init__(self, **kwargs):
        super().__init__()

        assert {'node_latent_dim', 'node_dec_mlp_layers', 'node_out_dim', 'dropout'}.issubset(kwargs)

        # initialize the node and global params decoder MLPs, without any normalization
        self.node_decoder = MLP(kwargs['node_latent_dim'],
                                kwargs['node_dec_mlp_layers'] + [kwargs['node_out_dim']],
                                activation_type='ReLU', norm_type=None, dropout=kwargs['dropout'])

    def forward(self, x):
        x = self.node_decoder(x)
        return x