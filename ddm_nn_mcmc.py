import numpy as np
import pickle
from cddm_data_simulation import ddm_simulate
from mh_mcmc_parallel import MH_MCMC
from de_mcmc_parallel import DE_MCMC
from de_crossover_mcmc_parallel import DE_CROSS_MCMC
import scipy.stats as ss

print("loading neural network model")
with open("dnn_weights.pickle", 'rb') as f:
    weights = pickle.load(f)

with open("dnn_biases.pickle", 'rb') as f:
    biases = pickle.load(f)

with open("dnn_activations.pickle", 'rb') as f:
    activations = pickle.load(f)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))

def linear(x):
    return x

activation_fns = {
        "relu":relu,
        "sigmoid":sigmoid,
        "linear":linear
        }

def np_predict(x, weights, biases, activations):
    """
    Redefine the keras .predict function in numpy

    Params
    ------
    x: np.array((*batch_size, n_input))
        inputs to calculate the prediction for.
        batch size is optional
    weights: list of np.ndarray((n_input, n_hidden))
        list of the weight matrices for each layer
    biases: list of np.ndarray((n_hidden, ))
        list of the biases for each layer. Must be of
        same length as the weights list
    activations: list of strings
        list of the activations to apply for each layer
    """
    num_layers = len(weights)
    out = x
    for l in range(num_layers):
        out = np.dot(out, weights[l])
        out += biases[l]
        out = activation_fns[activations[l]](out)
    return out

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

def target(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    out[out <= 0] = 1e-14
    return np.log(out).sum()

def target_array(params, data):
    params_rep = np.tile(params, (data.shape[0], 1))
    input_batch = np.concatenate([params_rep, data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    out[out <= 0] = 1e-14
    return out

def generate_data(n = 2000):
    v = np.random.uniform(-1, 1)
    a = np.random.uniform(0, 1.5)
    w = np.random.uniform(.3, .7)
    rts, choices, _ = ddm_simulate(v, a, w, n_samples=2000)
    choices = choices * (-1)
    data = np.concatenate([rts, choices], axis=1)
    return ((v,a,w), data)

def test_mh_sampling(num_iter=200):
    v = np.random.uniform(-1, 1)
    a = np.random.uniform(0, 1.5)
    w = np.random.uniform(.3, .7)
    print("v: {}, a: {}, w: {}".format(v, a, w))
    rts, choices, _ = ddm_simulate(v, a, w, n_samples=2000)
    choices = choices * (-1)
    data = np.concatenate([rts, choices], axis=1)

    model = MH_MCMC(dims=3, num_chains=4, bounds = [np.array([-1, 0, .3]), np.array([1, 1.5, .7])], target=target, proposal_var=.001)
    model.sample(data, num_iter=num_iter)
    return model

def test_de_sampling(NP=7, num_iter=200, gamma=.1, proposal_var=.001):
    true_params, data = generate_data()
    model = DE_CROSS_MCMC(dims=3, bounds = [np.array([-1, 0, .3]), np.array([1, 1.5, .7])], NP=NP,target=target, gamma=gamma, crossover_prob=.2, proposal_var=proposal_var)
    model.sample(data, num_iter=num_iter)
    return model, true_params

def generate_likelihood(params, choice):
    dat = np.random.uniform(0, 5, (10000, 1))
    data = np.concatenate([dat, np.tile(choice, (10000, 1))], axis=1)
    input_batch = np.concatenate([np.tile(params, (10000, 1)), data], axis=1)
    out = np_predict(input_batch, weights, biases, activations)
    return (dat, out)

true_params, data = generate_data()

v_dist = ss.norm(true_params[0], .001)
a_dist = ss.norm(true_params[1], .001)
z_dist = ss.norm(true_params[2], .001)

def informative_target(params, data):
    out = target(params, data)
    out += v_dist.logpdf(params[0])
    out += a_dist.logpdf(params[1])
    out += z_dist.logpdf(params[2])
    return out

model = DE_MCMC(dims=3, bounds = [np.array([-1, 0, .3]), np.array([1, 1.5, .7])], NP=10,target=informative_target, gamma=.2, proposal_var=.001)
model.sample(data, num_iter=400)

