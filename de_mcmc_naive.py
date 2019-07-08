import numpy as np
import scipy

class DE_MCMC():
    
    def __init__(self, dims, bounds, NP, target, gamma):
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
        
    def propose(self, chain):
        """
        Takes in a chain, and updates the chain parameters and log-likelihood
        """
        i, j = np.random.choice(self.NP, size=2, replace=False)
        proposal = chain[0] + self.gamma * (self.chains[i][0] - self.chains[j][0]) + \
                np.random.multivariate_normal(mean = np.zeros(self.dims), cov=.05 * np.eye(self.dims))
        proposal = np.clip(proposal, self.bound_min, self.bound_max)
        proposal_lp = self.target(proposal, data=self.data)
        acceptance_prob = proposal_lp - chain[1]
        if np.log(np.random.uniform()) < acceptance_prob:
            chain[0] = proposal
            chain[1] = proposal_lp
            
    def sample(self, data, burn=0, num_iter = 800):
        self.data = data
        # List of lists. Each inner list is of the form (params, lp) where params=parameters
        # and lp=likelihood of data conditioned on the parameters
        self.chains = []
        self.lps = np.zeros(num_iter)
        
        temp = np.random.uniform(size=(self.NP, self.dims))
        for i in range(self.NP):
            temp_lp = self.target(temp[i, :], data = self.data)
            self.chains.append([temp[i, :], temp_lp])
            
        self.samples = np.zeros((num_iter, self.NP, self.dims))
        print("Beginning sampling:")
        for t in range(num_iter):
            if (t % 200 == 0):
                print("Iteration {}".format(t))
                
            for i in range(self.NP):
                self.propose(self.chains[i])
                self.samples[t, i, :] = self.chains[i][0]
                self.lps[i] = self.chains[i][1]
            
        aggregated_samples = self.samples.reshape((self.NP * num_iter, self.dims))
        return aggregated_samples
