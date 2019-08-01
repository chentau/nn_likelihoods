import numpy as np
import multiprocessing as mp
import scipy.stats as ss
from scipy.special import logsumexp

class ImportanceSampler:
    def __init__(self, bounds, target):
        self.dims = bounds.shape[0]
        self.bounds = bounds
        self.target = target
        self.proposal = ss.multivariate_normal
        self.eff = 0

    def calculate_weights(self):
        out = self.p.starmap(self.target, zip(self.particles, self.data_tile))
        return np.fromiter(out, np.float)

    def sample(self, data, num_particles=10000, max_iter=20):
        self.particles = np.zeros((num_particles, self.dims))
        # Start off with uniform proposal
        self.particles = np.random.uniform(self.bounds[0], self.bounds[1],
                size=(num_particles, self.bounds[0].shape[0]))

        self.data = data
        self.data_tile = np.tile(self.data, (self.particles.shape[0], 1, 1))
        self.p = mp.Pool(4)

        # Proposal 0
        self.weights = self.calculate_weights()
        self.weights = np.exp(self.weights - logsumexp(self.weights))
        self.eff = self.weights.var() / np.power(self.weights.mean(), 2) + 1
        ratio = self.eff

        iteration = 1
        while ratio > 1e-4 and iteration < max_iter:
            print("iteration {}".format(iteration))
            mask = np.isclose(self.weights, 0) 
            samples = self.particles[~mask, :]
            print(samples.shape[0])
            sample_mu = samples.mean(axis=0)
            sample_cov = np.cov(samples.T)

            self.particles = self.proposal(mean=sample_mu, cov=sample_cov).rvs(
                    num_particles)
            self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
            self.weights = self.calculate_weights()
            self.weights = np.exp(self.weights - logsumexp(self.weights))

            eff = self.weights.var() / np.power(self.weights.mean(), 2) + 1
            ratio = eff / self.eff
            self.eff = eff
            iteration += 1
