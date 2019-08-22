import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import time
import pickle
from datetime import datetime
import yaml
from torch_models import Net, FullyConvolutional
from torch_load_data import KdeDataset, KdeFCNDataset
from torch.utils.data import DataLoader

device = torch.device("cuda")
method = "ddm_fcn"
method_params = pickle.load(open("kde_stats.pickle", "rb"))[method]

params = yaml.load(open("hyperparameters.yaml"))

dataset = KdeFCNDataset(method_params["data_folder"])
# dataset = KdeDataset(method_params["data_folder"])
dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

input_shape = dataset.X.shape[1]
timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

# Data params
model_path  = "/users/afengler/data/tony/kde/ddm/torch_models"
model_path += "/" + params["model_type"] + params["data_type_signature"] + timestamp

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Log hyperparameters
os.system("cp {} {}".format("/users/afengler/git_repos/tony/nn_likelihoods/hyperparameters.yaml", model_path + "/"))

loss_fn = nn.MSELoss()
loss_fn = nn.SmoothL1Loss()
val_loss = nn.MSELoss()
# network = Net(input_dim=input_shape, layer_sizes= params["hidden_layers"],
#     layer_activations=params["hidden_activations"])
network = FullyConvolutional(input_shape=input_shape, filters=params["filters"],
    strides=params["stride"], hidden_activations=params["hidden_activations"])

network = network.to(device)
print("is model on CUDA? {}".format(next(network.parameters()).is_cuda))
optimizer = optim.Adam(network.parameters(), lr=params["learning_rate"])

print(network)

for epoch in range(params["n_epochs"]):
    print("epoch {}".format(epoch))
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        out = network(batch[0].to(device))
        loss = loss_fn(out, batch[1].to(device))
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("loss: {}".format(loss.item()))

    with torch.no_grad():
        out_val = network(dataset.X_val.to(device))
        loss_val = val_loss(out_val, dataset.y_val.to(device))
        print("-" * 50)
        print("validation loss: {}".format(loss_val.item()))
        print("-" * 50)

    torch.save({
    "epoch": epoch,
    "model_state_dict": network.state_dict(),
    "optim_state_dict": optimizer.state_dict(),
    "loss": loss
    }, model_path + "/checkpoint{}.pt".format(epoch // 10))
    torch.save(network, model_path + "/model.pt")

torch.save({
"epoch": epoch,
"model_state_dict": network.state_dict(),
"optim_state_dict": optimizer.state_dict(),
"loss": loss
}, model_path + "/checkpointfinal.pt")
torch.save(network, model_path + "/model.pt")

import cddm_data_simulation as cd
import boundary_functions as bf
from scipy.optimize import differential_evolution

def target(param):
    grid = np.concatenate([np.tile(param, (data.shape[0], 1)), data], axis=1)
    out = cpu_network(torch.tensor(grid.T, dtype=torch.float)[None, :, :])
    return -out.item()

cpu_network = network.cpu()

v, a, w = np.random.uniform(np.array([-2, .6, .3]), np.array([2, 1.5, .7]))
rts, choices, _ = cd.ddm_flexbound(v, a, w, n_samples=1000, delta_t=.01, boundary_fun=bf.constant)
data = np.concatenate([rts, choices], axis=1)

result = differential_evolution(target, bounds=[[-2, 2], [.6, 1.5], [.3, .7]])

