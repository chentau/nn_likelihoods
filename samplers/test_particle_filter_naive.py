import numpy as np

from particle_filter_naive import ParticleFilterSequential
from particle_filter_parallel import ParticleFilterParallel

def target(params, data):
    return - ((data - params[0])**2 / 2).sum()

def test_sampling():
    a = np.random.uniform()
    print(a)
    data = np.random.normal(loc=a, scale=1, size=5000)

    test = ParticleFilterSequential(20000, [(0, 1)], target)
    test.sample(data, num_iter=30)
    return test

def test_parallel():
    a = np.random.uniform()
    print(a)
    data = np.random.normal(loc=a, scale=1, size=5000)

    test = ParticleFilterParallel(50000, [(0, 1)], target)
    test.sample(data, num_iter=30)
    return test

test = test_sampling()
# test = test_parallel()
