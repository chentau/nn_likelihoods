import numpy as np
import multiprocessing as mp
import ctypes

class MetropolisHastingsAdaptive:

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
        proposal = chain[0] + np.random.multivariate_normal(mean=np.zeros(self.dims), 
                                                                   cov=self.proposal_var * \
                                                                   np.eye(self.dims))
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
        rejection_rate = np.zeros(4)

        # Define array of chain value to be shared in memory
        for i in range(self.num_chains):
            temp = np.random.uniform(0, 1, self.dims)
            self.chains.append((temp, self.target(temp, self.data)))

        self.samples = np.zeros((num_iter, self.num_chains, self.dims))

        print("Beginning sampling")
        p = mp.Pool(n_cores)
        for t in range(num_iter):
            if t % 100 == 0:
                print("Iteration: {}".format(t))
            out = p.map(self.propose, self.chains)
            # unzip the output
            out_unzip = [list(t) for t in zip(*out)]
            self.samples[t, :, :] = np.array(out_unzip[0])
            self.chains = out

            if t % 50 == 0 and t != 0:
                for i in range(4):
                    rejection_rate[i] = np.unique(self.samples[(t - 50):t, i, :], 
                            axis=0).shape[0] / 50
                print(rejection_rate)
                if np.mean(rejection_rate < .20) > .5:
                    # self.proposal_var /= min(2, 1 / np.sqrt(t))
                    self.proposal_var /= 2.
                    print(self.proposal_var)
                elif np.mean(rejection_rate >=.5) >= .5:
                    self.proposal_var *= 1.5
                    print(self.proposal_var)
            # unzip the output - there has to be a more efficient way to do this
            # self.chains = np.array([temp[0] for temp in out])
            # self.chains_lp = np.array([temp[1] for temp in out])
            # self.samples[t, :, :] = self.chains

        p.close()
        p.terminate()
        self.samples = self.samples.reshape((self.num_chains * num_iter, self.dims))

