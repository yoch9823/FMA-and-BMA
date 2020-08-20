# Generic Bayesian Model Averaging - core.py 
## compute the posterior distribution Pr[M|D]=(p[D|M]*P[M])/P(D) 
## where M is a mber of some model space and D is observed data. 
## Pr(D|M) (the marginal likelihood) is computed by using MCMC simulation for better scaling. 

import numpy as np 
import mcmc_sim as mcmc ## I need to code mcmc from other mcmc.py 
from collections import Counter # need Counter to be installed
from itertools import product # need product to be installed
from scipy.special import binom # to import binomial distribution for prior model probability P(M)
#import float_range


def log_sum_exp(sequence):
    """ compute the logarithm of a sum of exponential in a stable way.
     if the difference between the smallest and largest exponential is larger than the float-64 range,
     the routine will drop low-order terms. 


    Parameters
    ----------
    sequence: array_like 
    sequence of numbers to be exponentiated and added 

    Returns
    -------
    float 
        logarithm of the sum of exponentials of the sequence's elements 
    """

    float_range = (-745,705) 
    lower = np.min(sequence)
    upper = np.max(sequence)

    if upper - lower < float_range[1] - float_range[0]:
        offset = (lower + upper)/2
    else: 
        print("Warning: Some very small terms have been dropped from a sum of exponentials to prevent overflow.")
        offset = upper - float_range[1]
        sequence = sequence[sequence - offset > float_range[0]]
        
    return offset + np.log(np.sum(np.e ** (sequence - offset))) # final version of the function log_sum_exp()

class InputError(Exception):

    def __init__(self,expr,msg):  # initialize the class, without this cannot operate
        self.expr = expr
        self.msg = msg
    
    def __str__(self): # override to print a readable string presentation of your object
        return str(self.expr) + ": " + str(self.msg)



class Enumerator(object):
    """ Generic model averaging routine. 

    Computes the posterior distribution over the model space. 
    Full enumration requires the computation of 2^ndim (2 square of the number of regressors) probabilities. 
    The method does not scale well beyond 15 dimensions due to too large computation. 

    Parameters 
    ----------
    X : np.ndarray in R^(n x k)
    y : np.ndarray in R^n
    likelihood_func : Pr(D|M)
    prior_func : Pr(M)

    Attributes 
    ----------
    nobs : int, number of observation 
    ndim : int, number of regressors (predictors) 
    X : prediction matrix in R^(nxk)
    y : response vector in R^n 
    posterior : Counter (container that keeps track of how many times equivalent values are added)
        posterior distribution Pr(M|D) over the model space where 
        tuple(model) is the key and the posterior probability is weight in our final goal. 
    """

    def __init__(self, X, y, likelihood_func, prior_func):

        self.X = X
        self.y = y 
        self.nobs, self.ndim = X.shape 

        self._get_likelihood = likelihood_func
        self._get_prior= prior_func

    def select(self):
        """ Compute the posterior probability distribution Pr(M|D) by enumerating all 2^k models. 
        """

        models = product((0,1), repeat=self.ndim)
        # models = np.array(list(product((0, 1), repeat = self.ndim))) 
        # make array with 0 and 1 combination of 2^k 

        priors = np.array([
            np.log(self._get_prior_prob(np.sum(np.array(list(model))),self.ndim))
            for model in models
            ])

        # print('P(M) is computed.')

        # ad-hoc: re-construct models
        del models
        models = product((0,1), repeat=self.ndim)

        likelihoods = np.array([
            self._get_likelihood(np.array(list(model)))
            for model in models
            ])

        # print('P(D|M) is computed.')

        posteriors = np.e ** (priors +likelihoods - log_sum_exp(priors+ likelihoods))

        # print('P(M|D) is computed.')

        # summarize to count that maps model and posterior with iteration of the length of the list model
        # print(len(priors))

        self.posterior = Counter({
            tuple(models[i]):posteriors[i]
            for i in range(len(np.array(list(models))))
        })


    def test_single_coefficients(self):
        """ Evaluate the inclusion probability of single coefficients 

        Returns 
        -------
        np.ndarray 
            vector of individual inclusion probabilities 
        """

        weighted_models = np.array([
            weight * np.array(model)
            for model, weight in self.posterior.items()
        ])

        return np.sum(weighted_models, 0) 
        print(np.sum(weighted_models,0))

    def test_joint_coefficients(self, indices):
        """ Evaluate the joint inclusion probability of multiple coefficients. 

        Parameters
        ----------
        indices : array_like in {0,..., ndim-1}
            indicies of variables in X to be included in the test 

        Returns 
        -------
        float 
            joint inclusion probability 
        """ 

        return sum(
            weight
            for model, weight in self.posterior.items()
            if np.all(np.array(model)[indices])
        )
    

    def get_model_size_dist(self):
        """ Evaluate the posterior model size distribution.
        Returns
        -------
        np.ndarray
            (ndim+1X1) posterior model size probabilities
        """

        dist = np.zeros(self.ndim +1)

        for model, weight in self.posterior.items():
            dist[sum(model)] += weight 

        return dist 


class MC3(Enumerator, mcmc.MetropolisSampler):
    """ Generic model averaging routine based on the Metropolis-Hastings algorithm 

    Approximates the posterior distribution over the model space. 
    Scales well to high dimensions. 

    Parameters 
    ----------
    X : np.ndarray in R^(nobs x ndim) 
        predictor matrix 
    y : np.ndarray in R^nobs 
        response vector
    likelihood_func : func 
        function that returns the log marginal likelihood of a give model 
    prior_func : fuc 
        function that returns the prior probability of a given model 

    Attributes
    ----------
    nobs : int 
        number of observation
    ndim : int 
        number of regressors
    X : np.ndarray 
    y : np.ndarray 
    posterior : Counter
        posterior distribution over the model space wehre tuple(model) is the key and the 
        posterior probability is the value 
    """

    def select(self, niter = 10000, method="random"):
        """ Estimate the posterior proability distribution through MCMC simulation. 

        Parameters
        ----------
        niter : int{1,..., inf}
            number of draws from the distribution 
        proposal : str{"random", "prior"}
            strategy that determines MCMC proposal probabilities 
        """

        # execute MCMC search 
        self.method = method
        self._run(np.zeros(self.ndim), niter)

        # summarize 
        counts = Counter()
        for i in range(self.draws.shape[0]):
            counts[tuple(self.draws[i,:])] += 1

        self.posterior = Counter({
            key:(counts[key] / sum(counts.values()))
            for key in counts
        })


    def _get_rv_prob(self, model):
        """ Compute the posterior proability of a given model up to the normalizing constant

        Parameter
        --------
        model : np.ndarray in {0,1}^ndim 
            vector of variable inclusion indicators

        Returns
        -------
        float
            log posterior probability
        """

        prior = np.log(self._get_prior_prob(np.sum(model), self.ndim))
        likelihood = self._get_likelihood(model)

        return likelihood + prior


    def _propose(self,state):
        """ Draw a candidate from the proposal distribution. 

        Parameters
        ----------
        state : np.ndarray in {0,1}^ndim 
            current MC state
        
        Returns
        -------
        np.ndarray 
            candidate vector of variable inclusion indicators
        """ 

        if self.method == "prior":
            prob_dplus = binom(len(state),np.sum(state) + 1) * self._get_prior_prob(
                np.sum(state) + 1,len(state))
            prob_dminus = binom(len(state),np.sum(state) - 1) * self._get_prior_prob(
                np.sum(state) - 1,len(state))
            growth_prob = prob_dplus / (prob_dplus+prob_dminus)
        else: 
            growth_prob = 1 - np.sum(state)/len(state)

        # decide on an action
        add = np.random.binomial(1,growth_prob)

        # pick entering/leaving variable 
        pick = np.random.choice(np.arange(len(state))[state!=add])
        candidate = np.copy(state)
        candidate[pick] = not candidate[pick]

        return candidate
    
    def _get_proposal_prob(self,proposal,state):
        """ Compute the propobability of proposing "proposal" given "state"

        Parameters
        ----------
        proposal : np.ndarray in {0,1}^ndim 
            candidate MC state 
        state: np.ndarray in {0,1}^ndim 
            current MC state 

        Returns
        -------
        float
            probability of proposal
        """ 

        if self.method == "prior":
            prob_dplus = binom(len(state),np.sum(state) + 1) * self._get_prior_prob(
                np.sum(state) + 1,len(state))
            prob_dminus = binom(len(state),np.sum(state) - 1) * self._get_prior_prob(
                np.sum(state) - 1,len(state))
            growth_prob = prob_dplus / (prob_dplus+prob_dminus)
        else: 
            growth_prob = 1 - np.sum(state)/len(state)

        if np.sum(state) < np.sum(proposal):
            forward_prob = growth_prob / (len(state)-np.sum(state))
        else:
            forward_prob = (1-growth_prob)/ np.sum(state)
            
        return np.log(forward_prob)    
