import numpy as np
from slice_sampler import SliceSampler

def target_normal(params, data):
    return (- (data - params[0])**2 / (2 * params[1]**2) - np.log(params[1])).sum()

def test_slice_rejection():
    a = np.random.uniform(-2, 2)
    b = np.random.uniform(.5, 2)
    data = np.random.normal(loc=a, scale=b, size=2000)

    test = SliceRejection(np.array([[-2, 2], [.5, 2]]), target_normal)
    test.sample(data, num_samples = 5000)
    return test, (a, b)

model, true_params = test_slice_rejection()
