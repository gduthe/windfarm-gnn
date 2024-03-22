import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GENConv, GINEConv, MLP, Linear
from torch_geometric.utils import dropout_edge, contains_isolated_nodes

class Processor(nn.Module):
    """ Processor class, uses a stack of message-passing blocks to update the node latent features.
        Three message-passing blocks are implemented:
            - GAT: Graph Attention Network (Veličković et al., 2018)
            - GEN: GENeralized Graph Convolution (Li et al., 2020)
            - GINE: modified GIN (Xu et al., 2019) that uses edge features in the message-passing (Hu et al., 2020)
            
        args:
        mp_type: str, the type of message-passing block to use, one of ['GAT', 'GEN', 'GINE']
        mp_aggr: str, the type of aggregation to use in the message-passing block, one of ['softmax', 'powermean', 'add', 'mean', 'max']
                 softmax and pwermean are only available for GEN
        num_mp_steps: int, the number of message-passing steps to use (depth of the processor)
        node_latent_dim: int, the dimension of the node latent space
        edge_latent_dim: int, the dimension of the edge latent space
        dropout: float, the edge dropout probability, used in all message-passing blocks 
    """
    
    def __init__(self, **kwargs):
        super().__init__()

        assert {'mp_type', 'mp_aggr', 'num_mp_steps', 'node_latent_dim', 'edge_latent_dim', 'dropout'}.issubset(kwargs)

        # initialize the message-passing blocks
        self.mp_type = kwargs['mp_type']
        self.mp_layers = nn.ModuleList()
        for i in range(kwargs['num_mp_steps']):
            mp_layer = ProcessorBlock(kwargs['node_latent_dim'], kwargs['edge_latent_dim'], kwargs['mp_type'], kwargs['mp_aggr'])
            self.mp_layers.append(mp_layer)

        # if using GIN, we need to add a linear layer to combine the history of node embeddings
        if self.mp_type == 'GINE':
            self.lin_out = Linear(kwargs['node_latent_dim']*kwargs['num_mp_steps'], kwargs['node_latent_dim'])

        self.dropout = kwargs['dropout']


    def forward(self, x, edge_index, edge_attr):
        # if using GIN, we need to store history of node embeddings:
        if self.mp_type == 'GINE':
            h_list = []

        # apply edge dropout
        if self.dropout > 0:
            while True:
                edge_index_, edge_mask = dropout_edge(edge_index, p=self.dropout, training=self.training)
                if not contains_isolated_nodes(edge_index_, num_nodes=x.shape[0]):
                    break
            edge_index = edge_index_
            edge_attr = edge_attr[edge_mask]
        
        # update the node latent features through the message-passing
        for layer in self.mp_layers:
            x = layer(x, edge_index, edge_attr)
            if self.mp_type == 'GINE':
                h_list.append(x)

        if self.mp_type == 'GINE':
            x = self.lin_out(torch.cat(h_list, dim=1))

        return x

class ProcessorBlock(nn.Module):
    """ ProcessorBlock class, implements a single message-passing layer that updates node latent features.
    
        args:
        node_latent_dim: int, the dimension of the node latent space
        edge_latent_dim: int, the dimension of the edge latent space
        mp_type: str, the type of message-passing block to use, one of ['GAT', 'GEN', 'GINE']
        mp_aggr: str, the type of aggregation to use in the message-passing block, one of ['softmax', 'powermean', 'add', 'mean', 'max']
                 softmax and powermean are only available for GEN
    
    """
    
    def __init__(self, node_latent_dim, edge_latent_dim, mp_type='GAT', mp_aggr='softmax'):
        super().__init__()
        assert (mp_type in ['GAT', 'GEN', 'GINE'])
        if mp_type == 'GEN': # gen supports different kinds of aggregation schemes
            assert (mp_aggr in ['softmax', 'powermean', 'add', 'mean', 'max'])
        
        self.mp_type = mp_type
        
        # iniitalize the message-passing conv layer
        if mp_type == 'GAT':
            self.conv = GATConv(in_channels=node_latent_dim, out_channels=node_latent_dim, edge_dim=edge_latent_dim, heads=1, add_self_loops=True, concat=False)
        elif mp_type == 'GEN':
            self.conv = GENConv(in_channels=node_latent_dim, out_channels=node_latent_dim, norm='layer', aggr=mp_aggr, num_layers=2)
        elif mp_type == 'GINE':
            # for GINE we need to define the update function of the node features (a neural network), we use a 2-layer MLP
            gin_mlp =   MLP(in_channels = node_latent_dim, hidden_channels = node_latent_dim*2, out_channels = node_latent_dim,
                            num_layers = 2, act='relu', norm='LayerNorm')
            self.conv = GINEConv(gin_mlp, node_latent_dim, node_latent_dim, edge_dim=edge_latent_dim)
        else:
            raise NotImplementedError

        if mp_type == 'GAT':  # apply layer norm to the GAT output only (GINE and GEN already have)
            self.norm = nn.LayerNorm(node_latent_dim, elementwise_affine=True)
        else:
            self.norm = nn.Identity()

    def reset_parameters(self):
        # method to reset the parameters of the message-passing block
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.norm(self.conv(x, edge_index, edge_attr))