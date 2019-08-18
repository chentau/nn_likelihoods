import numpy as np
from sklearn.neighbors.kde import KernelDensity
import uuid
from time import time
import pickle

from cddm_data_simulation import ddm_flexbound_simulate
from boundary_functions import constant
from samplers import SliceSampler
import kde_class as kde

# np.seterr("raise")

# def target(params, data):
#     """
#     params is a numpy vector for v, a, and w, respectively
#     data is a numpy array, where the first element is rt, second is choice
#     """
#     sim_rts, sim_choices, sim_info = ddm_simulate(v=params[0], a=params[1], w=params[2], n_samples=5000)
#     correct_choices = (sim_choices == -1) # Correct choices    
#     p_choice0 = correct_choices.mean()
#     p_choice1 = 1 - p_choice0
#     
#     kde0 = KernelDensity(kernel='gaussian', bandwidth=.05)
#     kde0.fit(sim_rts[correct_choices].reshape((-1, 1)))
#     kde1 = KernelDensity(kernel='gaussian', bandwidth=.05)
#     kde1.fit(sim_rts[~correct_choices].reshape((-1, 1)))
#     
#     correct_filter = (data[:, 1] == -1)
#     lp = correct_filter.sum() * np.log(p_choice0) + kde0.score(data[:, 0][correct_filter].reshape((-1, 1)))
#     lp += (data.shape[0] - correct_filter.sum()) * np.log(p_choice1) + kde1.score(data[:, 0][~correct_filter].reshape((-1, 1)))
#     
#     # if not np.isfinite(lp):
#     #     print(params)
#     #     print(p_choice0)
#     return lp

def target(params, data):
    sim_rts, sim_choices, sim_info = ddm_flexbound_simulate(v=params[0], a=params[1], w=params[2], n_samples=2000, boundary_fun=constant)
    tmp_kde = kde.logkde((sim_rts, sim_choices, sim_info))
    return tmp_kde.kde_eval((sim_rts, sim_choices)).sum()

def test_sampling(n_samples=2000):
    v = np.random.uniform(-1, 1)
    a = np.random.uniform(0, 1.5)
    w = np.random.uniform()
    true_params = np.array([v, a, w])
    rts, choices, _ = ddm_flexbound_simulate(v, a, w, n_samples=2000, boundary_fun=constant)
    data = np.concatenate([rts, choices], axis=1)
    
    # model = DifferentialEvolutionParallel(dims=3, bounds=np.array([[-1, .5, 0],
    #     [1, 1.5, .7]]), NP=15, target=target, gamma=.4)
    model = SliceSampler(bounds = np.array([[-2, .55, .3],
        [2, 1.5, .7]]).T, target=target)
    model.sample(data, num_samples=n_samples)
    return model, true_params

start = time()
model, true_params = test_sampling()
end = time()

target_folder = "/users/afengler/data/tony/kde/ddm/kde_simulations/"

pickle.dump((true_params, model.samples, end - start), open(target_folder + "sim{}.pickle".format(uuid.uuid1()), "wb")) 
