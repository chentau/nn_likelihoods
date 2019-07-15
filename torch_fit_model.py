import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cddm_data_simulation import ddm_simulate
from cdwiener import array_fptd
import os
import pandas as pd
import scipy as scp
import scipy.stats as scps
import time
from datetime import datetime
import yaml

from torch_models import Net
from kde_training_utilities import kde_load_data
from kde_training_utilities import kde_make_train_test_split

data_folder = "/users/afengler/data/kde/ddm/train_test_data/"

device = torch.device("cuda")

# kde_make_train_test_split(folder = data_folder,
#                           p_train = 0.8)

# Load train test split
X, _, X_val, __ = kde_load_data(folder = data_folder, log=True, prelog_cutoff='none')

# Use the analytical likelihoods for training instead
X = np.array(X)
y = np.log(array_fptd(X[:, 3] * X[:, 4] * (-1), X[:, 0], X[:, 1], X[:, 2]))
X_val = np.array(X_val)
y_val = np.log(array_fptd(X_val[:, 3] * X_val[:, 4] * (-1), X_val[:, 0], X_val[:, 1], X_val[:, 2]))

# Convert to pytorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32)

X = X.to(device)
y = y.to(device)
# X_val = X_val.to(device)
# y_val = y_val.to(device)

with open("/users/afengler/git_repos/tony/nn_likelihoods/hyperparameters.yaml") as f:
    params = yaml.load(f)

input_shape = X.shape[1]
timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

# Data params

model_path  = "/users/afengler/data/tony/kde/ddm/torch_models"
model_path += "/" + params["model_type"] + params["data_type_signature"] + timestamp

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Log hyperparameters
os.system("cp {} {}".format("/users/afengler/git_repos/tony/nn_likelihoods/hyperparameters.yaml", model_path + "/"))

loss_fn = nn.MSELoss()
network = Net(input_dim=input_shape, num_layers=len(params["hidden_layers"]), 
	layer_sizes= params["hidden_layers"], layer_activations=params["hidden_activations"])
network = network.cuda()
print("is model on CUDA? {}".format(next(network.parameters()).is_cuda))
optimizer = optim.Adam(network.parameters(), lr=params["learning_rate"])

print(network)

n_batches = X.shape[0] // params["batch_size"]

# Instead of moving the entire dataset onto the GPU, move each batch to the GPU
# at each epoch

for epoch in range(params["n_epochs"]):
    for i in range(n_batches):
        X_batch = X[i * params["batch_size"]:(i+1) * params["batch_size"]]
        y_batch = y[i * params["batch_size"]:(i+1) * params["batch_size"]]

        # X_batch = X_batch.to(device)
        # y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = network.forward(X_batch)
        loss = loss_fn(out, y_batch)
        print("epoch {}, batch {}, loss: {}".format(epoch, i, loss))
        loss.backward()
        optimizer.step()

        # out_val = network(X_val)
        # loss_val = loss_fn(out, y_val)
        # print("epoch {}, batch {}, loss: {}".format(epoch, i, loss_val))

        if epoch % 10 == 0:
            torch.save({
            "epoch": epoch,
            "model_state_dict": network.state_dict,
            "optim_state_dict": optimizer.state_dict,
            "loss": loss
            }, model_path + "/checkpoint{}.pt".format(epoch // 10))
            torch.save(network, model_path + "/model.pt")

torch.save({
"epoch": epoch,
"model_state_dict": network.state_dict,
"optim_state_dict": optimizer.state_dict,
"loss": loss
}, model_path + "/checkpointfinal.pt")
torch.save(network, model_path + "/model.pt")

print("Finished!")

