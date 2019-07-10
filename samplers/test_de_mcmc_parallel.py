from de_mcmc_parallel import *
import scipy.stats as ss

def target_normal(params, data):
    return (- (data - params[0])**2 / (2 * params[1]**2) - np.log(params[1])).sum()

def test_parallel():
    data = np.random.normal(loc=10, scale=1, size=1000)

    test = DE_MCMC(dims=2, bounds=(np.array([-np.inf, 1e-14]), np.array([np.inf, np.inf])),
            NP=50, target=target_normal, gamma = .4)
    samples = test.sample(data)
    return (test, samples)

def target_hierarchical(params, data):
    # params = [tau_1, tau_2, phi, mu, sigma]
    lp = ss.norm().logpdf(params[3])
    lp += ss.gamma(a=2).logpdf(params[4])
    lp += ss.norm(params[3], params[4]).logpdf([params[0], params[1]]).sum()
    lp += ss.gamma(a = 2).logpdf(params[2])
    lp += ss.norm(params[0], params[2]).logpdf(data[:500]).sum()
    lp += ss.norm(params[1], params[2]).logpdf(data[500:]).sum()
    return lp
    
def test_hierarchical_normal():
    mu = np.random.normal()
    print("mu = {}".format(mu))
    sigma = np.random.gamma(shape=2)
    print("sigma = {}".format(sigma))
    tau_1, tau_2 = np.random.normal(loc=mu, scale=sigma, size=2)
    print("tau_1 = {}".format(tau_1))
    print("tau_2 = {}".format(tau_2))
    phi = np.random.gamma(shape=2)
    print("phi = {}".format(phi))
    data_1 = np.random.normal(loc=tau_1, scale=phi, size=500)
    data_2 = np.random.normal(loc=tau_2, scale=phi, size=500)
    data = np.concatenate([data_1, data_2])
    
    test = DE_MCMC(dims=5, bounds=(np.array([-np.inf, -np.inf, 1e-14, -np.inf, 1e-14]),
        np.array([np.inf, np.inf, np.inf, np.inf, np.inf])), NP=50,
        target=target_hierarchical, gamma = .4)
    samples = test.sample(data)
    return test, samples

sampler, samples = test_hierarchical_normal()
