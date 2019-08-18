# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import pandas as pd
from itertools import product
from samplers import SliceSampler
import pickle

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from np_network import np_predict

method = "linear_collapse"
n_data_samples = 2000
n_slice_samples = 2000

part = 0
print(part)

if method == "ddm":
    dgp = cd.ddm_flexbound_simulate
    boundary = bf.constant
    network_path = "/users/afengler/data/tony/kde/ddm/keras_models/\
dnnregressor_ddm_08_08_19_23_37_48/"
#     network_path = "/users/afengler/data/tony/kde/ddm/keras_models/\
# dnnregressor_ddm_08_06_19_13_22_02/"
#     custom_objects = {"huber_loss": tf.losses.huber_loss}
#     fcn_path = "/users/afengler/data/tony/kde/ddm/keras_models/\
# deep_inference08_12_19_11_15_06/model.h5"
#    fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}
    output_folder = "/users/afengler/data/tony/kde/ddm/method_comparison/"
    param_names = ["v", "a", "w"]
    param_bounds = np.array([[-2, .6, .3], [2, 1.5, .7]])
elif method == "linear_collapse":
    dgp = cd.ddm_flexbound_simulate
    boundary = bf.linear_collapse
    network_path = "/users/afengler/data/tony/kde/linear_collapse/keras_models/\
dnnregressor_ddm_08_08_19_19_05_39/"
    output_folder = "/users/afengler/data/tony/kde/linear_collapse/method_comparison/"
    param_names = ["v", "a", "w", "node", "theta"]
    param_bounds = np.array([[-2, .6, .3, 0, 0], [2, 1.5, .7, 2, 1.37]])

# model = keras.models.load_model(network_path, custom_objects=custom_objects)
# fcn = keras.models.load_model(fcn_path, custom_objects=fcn_custom_objects)

weights = pickle.load(open(network_path + "weights.pickle", "rb"))
biases = pickle.load(open(network_path + "biases.pickle", "rb"))
activations = pickle.load(open(network_path + "activations.pickle", "rb"))

def extract_info(model):
    biases = []
    activations = []
    weights = []
    for layer in model.layers:
        if layer.name == "input_1":
            continue
        weights.append(layer.get_weights()[0])
        biases.append(layer.get_weights()[1])
        activations.append(layer.get_config()["activation"])
    return weights, biases, activations

# weights, biases, activations = extract_info(model)
def target(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    return out.sum()

# n_sims_per_param = 10
# n_sims = n_sims_per_param ** len(param_names)
n_sims = 250

# data_grid = np.zeros((n_sims, n_data_samples, 2))
# data_grid = data_grid[(data_grid.shape[0] // 2):]
# data_grid = data_grid[(part * data_grid.shape[0] // 4):((part + 1) * data_grid.shape[0] // 4)]
data_grid = np.zeros((n_sims, n_data_samples, 2))

# param_grid = np.linspace(param_bounds[0], param_bounds[1], num=n_sims_per_param)
# param_grid = np.array(list(product(*param_grid.T)))
# param_grid = param_grid[(part * param_grid.shape[0] // 4):((part + 1) * param_grid.shape[0] // 4)]
param_grid = np.random.uniform(param_bounds[0], param_bounds[1], size=(n_sims, len(param_names)))

for i, param in enumerate(param_grid):
    param_dict_temp = dict(zip(param_names, param)) 
    rts, choices, _ = dgp(**param_dict_temp, boundary_fun=boundary, 
        n_samples=n_data_samples, delta_t=.01)
    data = np.concatenate([rts, choices], axis=1)
    data_grid[i] = data

# test kde method
def test_kde(data):
    model = SliceSampler(bounds=param_bounds.T,
                         target=target, w = .4 / 1024, p = 8)
    model.sample(data, num_samples=n_slice_samples)
    return model.samples

def nf_target(params, data):
    return np.log(batch_fptd(data[:, 0] * data[:, 1] * (-1), params[0],
         params[1] * 2, params[2])).sum()

#test navarro-fuss
def test_nf(data):
    model = SliceSampler(bounds=param_bounds.T,
                         target=nf_target, w = .4 / 1024, p = 8)
    model.sample(data, num_samples=n_slice_samples)
    return model.samples

print("testing kde:")

p = mp.Pool(mp.cpu_count())

kde_results = np.array(p.map(test_kde, data_grid))

print("kde finished!")

# if method == "ddm":
#     nf_results = p.map(test_nf, data_grid)

# print("nf finished!")

# fcn_results = fcn.predict(data_grid)

# print("fcn finished!")


pickle.dump((param_grid, kde_results), open(output_folder + "kde_sim_random{}.pickle".format(part), "wb"))
# pickle.dump((param_grid, fcn_results), open(output_folder + "fcn_sim_random{}.pickle".format(part), "wb"))

# if method == "ddm":
#     pickle.dump((param_grid, nf_results), open(output_folder + "nf_sim{}.pickle".format(part), "wb"))

