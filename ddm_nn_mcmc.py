import numpy as np
import pickle

from cddm_data_simulation import ddm_simulate
from samplers import *
from np_model import np_predict
# from tensorflow import keras

print("loading neural network model")

network_path = "/home/tony/repos/temp_models/keras_models/\
dnnregressoranalytical_ddm_07_26_19_15_43_44/"

# model = keras.models.load_model(network_path)

print("successfully loaded")

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
weights = pickle.load(open(network_path + "weights.pickle", "rb"))
biases = pickle.load(open(network_path + "biases.pickle", "rb"))
activations = pickle.load(open(network_path + "activations.pickle", "rb"))

def informative_wrapper(true_params):
    v_dist = ss.norm(true_params[0], .1)
    a_dist = ss.norm(true_params[1], .1)
    z_dist = ss.norm(true_params[2], .1)
    def out(fn):
        def informative(params, data):
            out = fn(params, data)
            out += v_dist.logpdf(params[0])
            out += a_dist.logpdf(params[1])
            out += z_dist.logpdf(params[2])
            return out
        return informative
    return out

def informative_target(params, data):
    out = target(params, data)
    out += v_dist.logpdf(params[0])
    out += a_dist.logpdf(params[1])
    out += z_dist.logpdf(params[2])
    return out

def target(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    return out.sum()

def target_array(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    return out

def generate_data(n = 2000):
    v = np.random.uniform(-1, 1)
    a = np.random.uniform(.5, 1.5)
    w = np.random.uniform(.3, .7)
    rts, choices, _ = ddm_simulate(v, a, w, n_samples=2000)
    data = np.concatenate([rts, choices], axis=1)
    return ((v,a,w), data)

def test_mh_sampling(num_iter=1000, proposal_var=.001):
    true_params, data = generate_data()
    model = MetropolisHastingsParallel(dims=3, num_chains=4,
            bounds=[np.array([-1, 0, .3]), np.array([1, 1.5, .7])],
            target=target, proposal_var=proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def test_mh_adaptive(num_iter=1000, proposal_var=.001):
    true_params, data = generate_data()
    model = MetropolisHastingsAdaptive(dims=3, num_chains=4,
            bounds=[np.array([-1, 0, .3]), np.array([1, 1.5, .7])],
            target=target, proposal_var=proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def test_mh_componentwise(num_iter=1000, proposal_var=.001):
    true_params, data = generate_data()
    model = MetropolisHastingsComponentwise(dims=3, num_chains=4,
            bounds=[np.array([-1, .5, .3]), np.array([1, 1.5, .7])],
            target=target, proposal_var=proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def test_de_sampling(NP=7, num_iter=200, gamma=.1, proposal_var=.001):
    true_params, data = generate_data()
    model = DifferentialEvolutionParallel(dims=3,
            bounds=[np.array([-1, 0, .3]), np.array([1, 1.5, .7])],
            NP=NP, target=target, gamma=gamma, proposal_var=proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def test_particle_filter(num_particles=5000, num_iter=10, proposal_var=.001):
    true_params, data = generate_data()
    model = ParticleFilterParallel(num_particles, [(-1, 1), (.5, 2), (.3, .7)], 
                target, proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def test_slice_sampling(num_samples = 2000, w = .4 / 256, p = 8):
    true_params, data = generate_data()
    model = SliceSampler(bounds=np.array([[-1, 1], [.5, 1.5], [.3, .7]]),
            target=target, w = w, p = p)
    model.sample(data, num_samples=num_samples)
    return model, true_params

def test_importance_sampling(num_particles=10000, max_iter=20):
    true_params, data = generate_data()
    model = ImportanceSampler(bounds=np.array([[-1, .5, .3], [1, 1.5, .7]]), target = target)
    model.sample(data, num_particles=num_particles)
    return model, true_params
