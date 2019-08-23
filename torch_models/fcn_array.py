import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullyConvolutionalArray(nn.Module):
    def __init__(self, input_shape, filters=[64, 64, 128, 128, 128],
            strides=[1, 1, 1, 1, 1], hidden_activations=["relu", "relu", 
            "relu", "relu", "relu", "linear"]):

        super(FullyConvolutionalArray, self).__init__()
        self.n_conv_layers = len(filters)

        self.layers = nn.ModuleList([nn.Conv1d(input_shape, filters[0],
            kernel_size=1, stride=strides[0])])
        for i, layer in enumerate(zip(filters[1:], strides[1:]), 1):
            self.layers.append(nn.Conv1d(filters[i-1], layer[0],
                kernel_size=1, stride=layer[1]))
        self.output_conv = nn.Conv1d(filters[-1], 128, kernel_size=1, stride=1)

        self.activations = []
        for activation in hidden_activations:
            if activation == "relu":
                self.activations.append(F.relu)
            elif activation == "sigmoid":
                self.activations.append(torch.sigmoid)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        x = self.output_conv(x)
        return x.mean(1)
