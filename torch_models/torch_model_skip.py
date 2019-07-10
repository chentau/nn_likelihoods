import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualNet(nn.Module):
    def __init__(self, input_dim=5, num_layers=3, layer_sizes=[20,80,1],
            layer_activations=["relu", "relu", "linear"]):
        # number of hidden layers = num_layer - 1
        super(DDM_NN, self).__init__()
        self.num_layers = num_layers

        # Initialize layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_sizes[0])])
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        # Define nonlinearities
        self.layer_activations = []
        for activation in layer_activations:
            if activation == "relu":
                self.layer_activations.append(F.relu)
            elif activation == "sigmoid":
                self.layer_activations.append(F.sigmoid)
            elif activation == "linear":
                continue

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Last layer is linear and has no nonlinearity
            if i == self.num_layers - 1:
                break
            x = self.layer_activations[i](layer(x)) + x # skip connections
        x = self.layers[-1](x)
        return x




