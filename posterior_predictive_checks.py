import os
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt

import cddm_data_simulation as cd
import boundary_functions as bf

method = "ddm"

if method == "ddm":
    dgp = cd.ddm_flexbound_simulate
    boundary = bf.constant
    params=["v", "a", "w"]
    data_folder = "/users/afengler/data/tony/kde/ddm/method_comparison/"
elif method == "ornstein":
    dgp = cd.ornstein_uhlenbeck
    boundary = bf.constant
    params = ["v", "a", "w", "g"]
    data_folder = "/users/afengler/data/tony/kde/ornstein_uhlenbeck/method_comparison/"

files = os.listdir(data_folder)
files = [f for f in files if re.match("kde_sim_random.*", f)]

true_params, samples = pickle.load(open(data_folder + files[0], "rb"))
for f in files[1:]:
    param_tmp, samples_tmp = pickle.load(open(data_folder + f, "rb"))
    true_params = np.concatenate([true_params, param_tmp])
    samples = np.concatenate([samples, samples_tmp])

eap = np.zeros((true_params.shape[0], len(params)))
for i in range(samples.shape[0]):
    eap[i] = samples[i][500:].mean(axis=0)

ix = np.random.choice(true_params.shape[0], size=9)
ppc_params = true_params[ix]
ppc_samples = eap[ix]

fig, ax = plt.subplots(3, 3)

for i in range(3):
    for j in range(3):
        sim_data = dgp(*ppc_params[i*3 + j], n_samples=2000, boundary_fun=boundary)
        sim_data = sim_data[0][sim_data[1] == 1]
        ax[i][j].hist(sim_data, color="blue", bins=30)
        ppc_data = dgp(*ppc_samples[i*3 + j], n_samples=2000, boundary_fun=boundary)
        ppc_data = ppc_data[0][ppc_data[1] == 1]
        ax[i][j].hist(ppc_data, color="red", bins=30)

fig.savefig(data_folder + "ppc.png")
