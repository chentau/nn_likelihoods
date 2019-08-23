import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormNet(nn.Module):
    def __init__(self, input_dim=5, layer_sizes=[20,80,1],
            layer_activations=["relu", "relu", "linear"]):
        # number of hidden layers = num_layer - 1
        super(BatchNormNet, self).__init__()

        # Initialize layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_sizes[0])])
        self.layers.append(nn.BatchNorm1d(layer_sizes[0]))
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.layers.append(nn.BatchNorm1d(layer_sizes[i]))
        self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.num_layers = len(self.layers)

        # Define nonlinearities
        self.layer_activations = []
        for activation in layer_activations:
            if activation == "relu":
                self.layer_activations.append(F.relu)
            elif activation == "sigmoid":
                self.layer_activations.append(F.sigmoid)
            elif activation == "leakyrelu":
                self.layer_activations.append(F.leaky_relu)
            elif activation == "linear":
                continue

    def forward(self, x):
        for i in range(0, self.num_layers - 1, 2):
            x = self.layer_activations[i // 2](self.layers[i](x))
            x = self.layers[i + 1](x)
        x = self.layers[-1](x)
        return x




