import torch
import torch.nn as nn
import torch.nn.Functional as F
import numpy as np

class FullyConvolutional(nn.Module):
    def __init__(self, n_features, input_shape, filters=[64, 64, 128, 128, 128],
            strides=[1, 2, 2, 2, 2]):
        super(FullyConvolutional, self).__init__()
        self.n_conv_layers = len(filters)
        self.layers = nn.ModuleList([nn.Conv1d(input_shape, filters[0],
            stride=stride[0])])
        for i, filter, stride in enumerate(zip(filters[1:], strides[1:])):
            self.layers.append(nn.Conv1d(filters[i-1], filter, stride=stride))
        self.layers.append(nn.AvgPool1d(filters[-1]))
        self.layers.append(nn.Linear(filters[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
