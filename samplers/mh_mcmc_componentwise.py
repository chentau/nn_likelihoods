import numpy as np
import multiprocessing as mp
import ctypes

class MetropolisHastingsComponentwise:

    def __init__(self, dims, num_chains, bounds, target, proposal_var=.01):
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
        self.target = target
        self.num_chains = num_chains
        self.bound_min = bounds[0]
        self.bound_max = bounds[1]
        self.proposal_var = proposal_var

    def propose(self, chain):
        proposal = chain[0]
        proposal[np.random.choice(3)] += np.random.normal(
                loc=0, scale=self.proposal_var)
        proposal = np.clip(proposal, self.bound_min, self.bound_max)
        proposal_lp = self.target(proposal, data=self.data)
        acceptance_prob = proposal_lp - chain[1]
        if np.log(np.random.uniform()) < acceptance_prob:
            return (proposal, proposal_lp)
        else:
            return (chain[0], chain[1])

    def sample(self, data, burn=0, num_iter = 800, n_cores = 4):
        self.data = data
        self.chains = []

        # Define array of chain value to be shared in memory
        for i in range(self.num_chains):
            temp = np.random.uniform(0, 1, self.dims)
            self.chains.append((temp, self.target(temp, self.data)))

        self.samples = np.zeros((num_iter, self.num_chains, self.dims))

        print("Beginning sampling")
        p = mp.Pool(n_cores)
        for t in range(num_iter):
            if t % 100 == 0:
                self.proposal_var = self.proposal_var / 5
                print("Iteration: {}".format(t))

            out = p.map(self.propose, self.chains)
            # unzip the output
            out_unzip = [list(t) for t in zip(*out)]
            self.samples[t, :, :] = np.array(out_unzip[0])
            self.chains = out

            # unzip the output - there has to be a more efficient way to do this
            # self.chains = np.array([temp[0] for temp in out])
            # self.chains_lp = np.array([temp[1] for temp in out])
            # self.samples[t, :, :] = self.chains

        p.close()
        p.terminate()
        # self.samples = self.samples.reshape((self.num_chains * num_iter, self.dims))

