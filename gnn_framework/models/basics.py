import torch.nn as nn

class MLP(nn.Module):
    ''' A basic MLP class with a number of available normalizations and activations.'''

    def __init__(self, in_dim, layer_dim_list, norm_type='LayerNorm', activation_type='ReLU', dropout=0.0):
        super().__init__()
        # get the type of activation
        if activation_type is not None:
            assert (activation_type in ['ReLU', 'LeakyReLU', 'Tanh'])
            activation = getattr(nn, activation_type)
        else:
            activation = nn.Sequential

        # get the dims of the fully connected layer
        dim_list = [in_dim] + layer_dim_list
        fc_layers = []
        for i in range(len(dim_list) - 2):
            fc_layers += [nn.Linear(dim_list[i], dim_list[i + 1]), activation(), nn.Dropout(p=dropout)]


        # add the output layer without activation
        fc_layers += [nn.Linear(dim_list[-2], dim_list[-1])]

        # get the normalization type to add to the output of the MLP
        if norm_type is not None:
            assert (norm_type in ['LayerNorm', 'BatchNorm', 'GraphNorm', 'InstanceNorm', 'MessageNorm'])
            norm_layer = getattr(nn, norm_type)
            fc_layers.append(norm_layer(dim_list[-1]))

        # init the fully connected layers
        self.__fcs = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.__fcs(x)