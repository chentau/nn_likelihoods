import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
from cddm_data_simulation import ddm_simulate
from cdweiner import fptd
import os

from torch_models import Net

def generate_input_grid():
    grid_v = np.random.uniform(-1, 1, 10)
    grid_a = np.random.uniform(0, 1.5, 10)
    grid_z = np.random.uniform(0, 1, 10)
    grid_rt = np.random.uniform(0, 5, 10)
    grid_choice = [-1, 1]
    return torch.tensor(list(product(grid_v, grid_a, grid_z, grid_rt, grid_choice)))
    
def generate_labels(input_grid):
    out = [fptd(t=feature[4] * feature[3], v=feature[0], a=feature[1], w=feature[2] * feature[1], 
        eps = 1e-8) for feature in input_grid]
    return torch.tensor(out)

def train_test(X, y, X_val, y_val, learning_rate=.002, n_batches=5, n_epochs=20, save=True):
    PATH = "/home/tony/repos/temp_models/torch_dnnregressor_ddm"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    loss_fn = nn.MSELoss()
    network = Net()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    batch_size = X.shape[0] // n_batches

    for epoch in range(n_epochs):
        for i in range(n_batches):
            X_batch = X[i * batch_size:(i+1) * batch_size]
            y_batch = y[i * batch_size:(i+1) * batch_size]

            optimizer.zero_grad()
            out = network(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()

            out = network(X_val)
            loss = loss_fn(out, y_val)
            print("epoch {}, batch {}, loss: {}".format(epoch, i, loss))

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": network.state_dict,
                "optim_state_dict": optimizer.state_dict,
                "loss": loss
                }, PATH + "/checkpoint{}.pt".format(epoch // 10))
            torch.save(network, PATH + "/model.pt")

    torch.save({
        "epoch": epoch,
        "model_state_dict": network.state_dict,
        "optim_state_dict": optimizer.state_dict,
        "loss": loss
        }, PATH + "/checkpointfinal.pt")
    torch.save(network, PATH + "/model.pt")


    return network

X = generate_input_grid()
y = generate_labels(X)
X_val = X[18000:]
y_val = y[18000:]
X = X[0:18000]
y = y[0:18000]
