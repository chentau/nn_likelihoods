import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullyConvolutional(nn.Module):
    def __init__(self, input_shape, filters=[64, 64, 128, 128, 128],
            strides=[1, 1, 1, 1, 1]):
        super(FullyConvolutional, self).__init__()
        self.n_conv_layers = len(filters)
        self.layers = nn.ModuleList([nn.Conv1d(input_shape, filters[0],
            kernel_size=1, stride=strides[0])])
        for i, layer in enumerate(zip(filters[1:], strides[1:]), 1):
            self.layers.append(nn.Conv1d(filters[i-1], layer[0],
                kernel_size=1, stride=layer[1]))
        self.fc = nn.Linear(filters[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.sum(2)
        return self.fc(x)
