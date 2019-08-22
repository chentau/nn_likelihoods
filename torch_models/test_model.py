import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet(nn.Module):
    def __init__(self, input_dim):
        # number of hidden layers = num_layer - 1
        super(TestNet, self).__init__()

        # Initialize layers
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 60)
        self.layer3 = nn.Linear(60, 50)
        self.layer4 = nn.Linear(50, 50)
        self.layer5 = nn.Linear(50, 60)
        self.layer6 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        return x
