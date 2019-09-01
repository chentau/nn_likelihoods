import numpy as np
from scipy.special import logsumexp
from scipy.optimize import differential_evolution
import scipy.stats as ss
import multiprocessing as mp

class PopulationMC:
    def __init__(self, bounds, target, k=5, init_mean=None, init_var=np.array([.5, .1, .05])):
        self.dims = bounds[0].shape[0]
        self.bounds = bounds
        self.target = target

        self.k = k
        self.mixture_p = np.ones(k) / k
        self.ess = []

        self.mixture_means = np.zeros((self.k, self.dims))
        if init_mean is None:
            optim = differential_evolution(target, bounds=bounds)
            self.mixture_means[:] = optim["x"]
        else:
            self.mixture_means[:] = init_mean

        self.mixture_cov = np.array([np.diag(init_var) for _ in range(k)], 
                dtype=np.float64)
        self.mixture_dists = [ss.multivariate_normal(self.mixture_means[i],
            self.mixture_cov[i]) for i in range(self.k)]

        self.p = mp.Pool(3)

    def update_importance_dist(self):
        self.rho_components = np.array([self.mixture_p[i] 
            * self.mixture_dists[i].pdf(self.particles) for i in range(self.k)])
        self.rho_components = self.rho_components \
                / self.rho_components.sum(axis=0)

        for i in range(self.k):
            sum_tmp = self.weights * self.rho_components[i]

            self.mixture_p[i] = sum_tmp.sum()
            self.mixture_means[i] = (sum_tmp[:, None] 
                    * self.particles).sum(axis=0) / self.mixture_p[i]

            centered_particles = self.particles - self.mixture_means[i]
            cov_tmp = (sum_tmp[:, None, None] * (centered_particles[:, :, None]
                * centered_particles[:, None, :])) / self.mixture_p[i]
            self.mixture_cov[i] = cov_tmp.sum(axis=0)

        self.mixture_dists = [ss.multivariate_normal(self.mixture_means[j],
            self.mixture_cov[j]) for j in range(self.k)]

    def draw_importance_samples(self):
        self.assignments = np.random.choice(self.k, p=self.mixture_p, size=self.n_particles)
        for i in range(self.k):
            mask = self.assignments == i
            self.particles[mask] = self.mixture_dists[i].rvs(mask.sum())
        self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])

    def calculate_weights(self):
        lp = self.target(self.particles, self.data)
        # lp = np.fromiter(self.p.starmap(self.target,
        #     zip(self.particles, self.data_tile)))
        importance_lp = np.log(self.mixture_p[0]) + self.mixture_dists[0].logpdf(self.particles)
        for i in range(1, self.k):
            importance_lp += np.log(self.mixture_p[i]) + self.mixture_dists[i].logpdf(self.particles)
        self.weights =  lp - importance_lp
        self.weights = np.exp(self.weights - logsumexp(self.weights))

    def sample(self, n_particles, data, num_iter):
        self.n_particles = n_particles
        self.data = data
        self.weights = np.zeros(self.n_particles)
        self.particles = np.zeros((self.n_particles, self.dims))

        for i in range(num_iter):
            self.draw_importance_samples()
            self.calculate_weights()
            self.update_importance_dist()
            self.ess.append(1 / np.power(self.weights, 2).sum())

