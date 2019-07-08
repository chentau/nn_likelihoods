import numpy as np
from sklearn.neighbors.kde import KernelDensity
from cddm_data_simulation import ddm_simulate
from de_mcmc_parallel import DE_MCMC

# np.seterr("raise")

def target(params, data):
    """
    params is a numpy vector for v, a, and w, respectively
    data is a numpy array, where the first element is rt, second is choice
    """
    sim_rts, sim_choices, _ = ddm_simulate(v=params[0], a=params[1], w=params[2], n_samples=5000)
    correct_choices = (sim_choices == -1) # Correct choices    
    p_choice0 = correct_choices.mean()
    p_choice1 = 1 - p_choice0
    
    kde0 = KernelDensity(kernel='gaussian', bandwidth=.05)
    kde0.fit(sim_rts[correct_choices].reshape((-1, 1)))
    kde1 = KernelDensity(kernel='gaussian', bandwidth=.05)
    kde1.fit(sim_rts[~correct_choices].reshape((-1, 1)))
    
    correct_filter = (data[:, 1] == -1)
    lp = correct_filter.sum() * np.log(p_choice0) + kde0.score(data[:, 0][correct_filter].reshape((-1, 1)))
    lp += (data.shape[0] - correct_filter.sum()) * np.log(p_choice1) + kde1.score(data[:, 0][~correct_filter].reshape((-1, 1)))
    
    # if not np.isfinite(lp):
    #     print(params)
    #     print(p_choice0)
    return lp

def test_sampling():
    v = np.random.uniform(-1, 1)
    a = np.random.uniform(0, 1.5)
    w = np.random.uniform()
    print("v: {}, a: {}, w: {}".format(v, a, w))
    rts, choices, _ = ddm_simulate(v, a, w, n_samples=1000)
    data = np.concatenate([rts, choices], axis=1)
    
    model = DE_MCMC(dims=3, bounds = [np.array([-1, 0, 0]), np.array([1, 1.5, 1])], 
            NP=15, target=target, gamma = .4)
    model.sample(data, num_iter=200)
    return model

# model = test_sampling()    
    
