import numpy as np
from scipy.special import logsumexp

class ParticleFilterSequential():

    def __init__(self, num_particles, bounds, target, proposal_var=.001):
        # Importance sampling with uniform importance density
        self.n = num_particles
        self.weights = np.ones(num_particles)
        self.particles = np.zeros((num_particles, len(bounds)))
        self.proposal_var = proposal_var
        self.target = target
        self.indices = [0 for _ in range(num_particles)] # indices for resampling

        for i, param_bound in enumerate(bounds):
            self.particles[:, i] = np.random.uniform(param_bound[0], param_bound[1],
                    num_particles)

    def _importance_sample(self):
        self.particles += np.random.uniform((-1) * self.proposal_var,
                self.proposal_var, size = self.particles.size).reshape((
                        self.particles.shape[0], self.particles.shape[1]))

    def _update_weights(self):
        log_prob = self._lp_array()
        self.weights = log_prob - logsumexp(log_prob)

    def _lp_array(self):
        out = [self.target(particle, self.data) for particle in self.particles]
        return np.array(out)

    def sample(self, data, num_iter=100):
        self.data = data

        for i in range(num_iter):
            print("Iteration: {}".format(i))
            self._importance_sample()
            self._update_weights()
            ess = 1. / (self.weights ** 2).sum()
            if ess < self.n / 2:
                u_1 = np.random.uniform(0, 1. / self.n)
                u = np.arange(0, self.n) / self.n
                u = u_1 + u

                cum_weights = np.cumsum(np.exp(self.weights))
                i = 0
                j = 0

                while (i < self.n):
                    if u[i] < cum_weights[j]:
                        self.indices[i] = j
                        i += 1
                    else:
                        j += 1

                self.particles = self.particles[self.indices]
                # self.particles = self.particles[np.random.choice(np.arange(self.n),
                #     size=self.n, p = np.exp(self.weights))]
                self.weights = np.repeat(1. / self.n, self.n)


