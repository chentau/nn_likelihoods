import numpy as np
from importance import ImportanceSampler

def target_normal(params, data):
    return (- (data - params[0])**2 / (2 * params[1]**2) - np.log(params[1])).sum()

def test_importance():
    a = np.random.uniform(-2, 2)
    b = np.random.uniform(.5, 2)
    data = np.random.normal(loc=a, scale=b, size=2000)

    test = ImportanceSampler(np.array([[-2, .5], [2, 2]]), target_normal)
    test.sample(data)
    return test, (a, b)

model, true_params = test_importance()
