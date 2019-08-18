import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))

def linear(x):
    return x

activation_fns = {
        "relu": relu,
        "sigmoid": sigmoid,
        "linear": linear
        }

def np_predict(x, weights, biases, activations):
    num_layers = len(weights)
    out = x
    for l in range(num_layers):
        out = np.dot(out, weights[l])
        out += biases[l]
        out = activation_fns[activations[l]](out)
    return out

