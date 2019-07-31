import numpy as np
import multiprocessing as mp

class ImportanceSampling:
    def __init__(self, bounds, target):
        self.dims = bounds.shape[0]
        self.bounds = bounds
        self.target = target

    def calculate_weights(self):
        out = self.p.starmap(self.target, zip(self.particles, self.data_tile))
        return np.fromiter(out, np.float)

    def sample(self, data, num_particles=10000):
        self.particles = np.zeros((num_particles, self.dims))
        for i in range(self.dims):
            self.particles[:, i] = np.random.uniform(self.bounds[i][0],
                    self.bounds[i][1], size=num_particles)

        self.data = data
        self.data_tile = np.tile(self.data, (self.particles.shape[0], 1, 1))
        self.p = mp.Pool(4)

        self.weights = np.exp(self.calculate_weights())
        self.weights = self.weights / self.weights.sum()
