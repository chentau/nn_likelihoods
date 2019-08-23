# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
from samplers import SliceSampler
import pickle
import uuid

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
from np_network import np_predict
from kde_info import KDEStats

method = "full"
n_data_samples = 2000
n_slice_samples = 2000
n_sims = 100

stats = pickle.load(open("kde_stats.pickle", "rb"))
method_params = stats[method]

network_path = yaml.load(open("model_paths.yaml"))[method]

# model = keras.models.load_model(network_path, custom_objects=custom_objects)
# fcn = keras.models.load_model(fcn_path, custom_objects=fcn_custom_objects)

weights = pickle.load(open(network_path + "weights.pickle", "rb"))
biases = pickle.load(open(network_path + "biases.pickle", "rb"))
activations = pickle.load(open(network_path + "activations.pickle", "rb"))

def target(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    return out.sum()

# n_sims_per_param = 10
# n_sims = n_sims_per_param ** len(param_names)

# data_grid = np.zeros((n_sims, n_data_samples, 2))
# data_grid = data_grid[(data_grid.shape[0] // 2):]
# data_grid = data_grid[(part * data_grid.shape[0] // 4):((part + 1) * data_grid.shape[0] // 4)]

# param_grid = np.linspace(param_bounds[0], param_bounds[1], num=n_sims_per_param)
# param_grid = np.array(list(product(*param_grid.T)))
# param_grid = param_grid[(part * param_grid.shape[0] // 4):((part + 1) * param_grid.shape[0] // 4)]
param_grid = np.random.uniform(method_params["param_bounds"][0], 
    method_params["param_bounds"][1], size=(n_sims, len(param_names)))

if len(boundary_param_bounds) > 0:
    boundary_param_grid = np.random.uniform(method_params["boundary_param_bounds"][0],
 method_params["boundary_param_bounds"][1], size=(n_sims, len(method_params["boundary_param_bounds"])))
else:
    boundary_param_grid = []

def generate_data_grid(param_grid, boundary_param_grid):
    data_grid = np.zeros((n_sims, n_data_samples, 2))
    for i in range(n_sims):
        param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
        if boundary_param_grid:
            boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_grid))
        else:
            boundary_dict_tmp = {}
        rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
            boundary_fun=method_params["boundary"], n_samples=n_data_samples,
             delta_t=.01, boundary_params=boundary_dict_tmp)
        data_grid[i] = np.concatenate([rts, choices], axis=1)
    return data_grid

data_grid = generate_data_grid(param_grid, boundary_param_grid)

# test kde method
def test_kde(data):
    model = SliceSampler(bounds=method_params["param_bounds"].T,
                         target=target, w = .4 / 1024, p = 8)
    model.sample(data, num_samples=n_slice_samples)
    return model.samples

def nf_target(params, data):
    return np.log(batch_fptd(data[:, 0] * data[:, 1] * (-1), params[0],
         params[1] * 2, params[2])).sum()

#test navarro-fuss
def test_nf(data):
    model = SliceSampler(bounds=method_params["param_bounds"].T,
                         target=nf_target, w = .4 / 1024, p = 8)
    model.sample(data, num_samples=n_slice_samples)
    return model.samples

p = mp.Pool(mp.cpu_count())

kde_results = np.array(p.map(test_kde, data_grid))

if method == "ddm":
    nf_results = p.map(test_nf, data_grid)

# print("nf finished!")

# fcn_results = fcn.predict(data_grid)

# print("fcn finished!")


pickle.dump((param_grid, data_grid, kde_results), open(output_folder + "kde_sim_random{}.pickle".format(uuid.uuid1()), "wb"))
# pickle.dump((param_grid, fcn_results), open(output_folder + "fcn_sim_random{}.pickle".format(part), "wb"))

if method == "ddm":
    pickle.dump((param_grid, data_grid, nf_results), open(output_folder + "nf_sim{}.pickle".format(uuid.uuid1()), "wb"))

