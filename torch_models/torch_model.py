import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim=5, layer_sizes=[20,80,1],
            layer_activations=["relu", "relu", "linear"]):
        # number of hidden layers = num_layer - 1
        super(Net, self).__init__()
        self.num_layers = len(layer_sizes)

        # Initialize layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_sizes[0])])
        for i in range(1, self.num_layers - 1):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])

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
            x = self.layer_activations[i](layer(x))
        x = self.output(x)
        return x

