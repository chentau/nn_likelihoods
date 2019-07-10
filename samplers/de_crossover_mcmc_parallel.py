import numpy as np
import multiprocessing as mp
import ctypes

class DifferentialEvolutionCrossover:
    def __init__(self, dims, bounds, NP, target, gamma, crossover_prob, proposal_var=.01):
        """
        Params
        -----
        dims: int
            dimension of the parameter space
        bounds: list of np.ndarrays
            The first element is a numpy array of lower bounds, the
            second element is the numpy array of upper bounds for the params
        NP: int
            number of particles to use
        target: function(ndarray params, ndarray data) -> float
            function that takes in two arguments: params, and data, and returns
            the log likelihood of the data conditioned on the parameters.
        gamma: float
            gamma parameter to mediate the magnitude of the update
        """
        self.dims = dims
        self.NP = NP
        self.target = target
        self.gamma = gamma
        self.bound_min = bounds[0]
        self.bound_max = bounds[1]
        self.proposal_var = proposal_var
        self.crossover_prob = crossover_prob

        self.cross_length_dist = crossover_prob ** np.arange(0, self.dims)
        self.cross_length_dist = self.cross_length_dist / self.cross_length_dist.sum()

    def propose(self, ix):
        i, j = np.random.choice(self.NP, size=2, replace=False)
        n = np.random.choice(np.arange(0, self.dims))
        l = np.random.choice(np.arange(0, self.dims), p=self.cross_length_dist)
        index = np.sort(np.arange(n, n + l + 1) % self.dims)
        proposal = self.chains[ix].copy()
        proposal[index] = self.chains[ix, index] + self.gamma * (self.chains[i, index] - \
                self.chains[j, index]) + np.random.multivariate_normal(mean = 
                        np.zeros(index.shape[0]), cov=self.proposal_var * np.eye(index.shape[0]))
        proposal = np.clip(proposal, self.bound_min, self.bound_max)
        proposal_lp = self.target(proposal, data=self.data)
        acceptance_prob = proposal_lp - self.chains_lp[ix]
        if np.log(np.random.uniform()) < acceptance_prob:
            return (proposal, proposal_lp)
        else:
            return (self.chains[ix], self.chains_lp[ix])

    def sample(self, data, burn=0, num_iter = 800, n_cores = 4):
        self.data = data

        # Define array of chain value to be shared in memory
        starting_vals_base = mp.Array(ctypes.c_double, self.NP * self.dims)
        # Uniform initialization
        starting_vals_base[:] = list(np.random.uniform(size = self.NP * self.dims))
        self.chains = np.ctypeslib.as_array(starting_vals_base.get_obj())
        self.chains = self.chains.reshape((self.NP, self.dims))

        # Define array of chain log_likelihoods to be shared in memory
        starting_lp_base = mp.Array(ctypes.c_double, self.NP)
        starting_lp_base[:] = [self.target(self.chains[i, :], self.data) for i in range(self.NP)]
        self.chains_lp = np.ctypeslib.as_array(starting_lp_base.get_obj())

        self.samples = np.zeros((num_iter, self.NP, self.dims))

        print("Beginning sampling")
        p = mp.Pool(n_cores)
        for t in range(num_iter):
            if t % 200 == 0:
                print("Iteration: {}".format(t))

            out = p.map(self.propose, range(self.NP))
            # unzip the output
            out_unzip = [list(t) for t in zip(*out)]
            self.chains = np.array(out_unzip[0])
            self.chains_lp = np.array(out_unzip[1])
            self.samples[t, :, :] = self.chains

            # unzip the output - there has to be a more efficient way to do this
            # self.chains = np.array([temp[0] for temp in out])
            # self.chains_lp = np.array([temp[1] for temp in out])
            # self.samples[t, :, :] = self.chains

        p.close()
        p.terminate()
        self.samples = self.samples.reshape((self.NP * num_iter, self.dims))

    # def sample(self, data, burn=0, num_iter = 800):
    #     self.data = data
    #     # List of lists. Each lists is of the form (params, lp) where params=parameters
    #     # and lp=likelihood of data conditioned on the parameters
    #     init = []

    #     starting_vals = np.random.uniform(size=(self.NP, self.dims))
    #     for i in range(self.NP):
    #         temp_lp = self.target(starting_vals[i, :], data = self.data)
    #         init.append([starting_vals[i, :], temp_lp])

    #     manager = mp.Manager()
    #     self.chains = manager.list(init)
    #     self.samples = np.zeros((num_iter, self.NP, self.dims))
    #     p = mp.Pool(4)

    #     print("Beginning sampling:")
    #     for t in range(num_iter):
    #         if (t % 200 == 0):
    #             print("Iteration {}".format(t))

    #         self.samples[t, :] = np.array(p.map(self.propose, range(self.NP)))

    #     aggregated_samples = self.samples.reshape((self.NP * num_iter, self.dims))
    #     return aggregated_samples
