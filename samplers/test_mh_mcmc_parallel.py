from mh_mcmc_parallel import *
import numpy as np
import scipy.stats as ss

# Try to find the mean of a normal distribution
def target_normal(params, data):
    return (- (data - params[0])**2 / (2 * params[1]**2) - np.log(params[1])).sum()

def test_naive():
    data = np.random.normal(loc=10, scale=1, size=1000)

    test = MH_MCMC(dims=2, num_chains=4, bounds=(np.array([-np.inf, 1e-14]), np.array([np.inf, np.inf])), target=target_normal)
    test.sample(data)
    return (test, test.samples)

model, samples = test_naive()

