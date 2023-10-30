import numpy as np
from tqdm import tqdm

class MCMC:
    def __init__(self,prob_func,params):
        '''
        A class for performing MCMC sampling of a given probability function based on SampleExperiment class object
        Parameters
        ----------
        prob_func : function
            A function that takes in a dictionary of parameters and returns a tuple of (probability, value)
        params : ParamDict
            A dictionary of parameters to be varied
        '''
        self._prob_func = prob_func
        self.params = params
        self._keys = self.params.keys()
        bounds = self.params._bounds_vec()
        self._bounds = [np.array(bounds[0]), np.array(bounds[1])]
        self.std = (self._bounds[1]-self._bounds[0])/20.
        #Test that the prob_func gives a float
        prob_result, value = self._prob_func(self.params._val())
        if not isinstance(prob_result,float):
            raise ValueError("prob_func should output a float. Instead it gave", prob_result)
        print("MCMC instance created with the following settings")
        print(self.params)
        print(self.params._n(),"parameters will be varied.")

        self._samples = None
        self._probs = None
        self._values = None

    def _proposal(self):
        dx = np.random.normal(size=self.params._n())*self.std #could multiply by a vector of standard deviations for each parameter
        return self.c0 + dx

    def run(self,n):
        '''
        Run the MCMC sampling
        Parameters
        ----------
        n : int
            Number of MCMC iterations to perform (number of samples will be less than n due to rejection sampling)
        '''
        self.c0 = self.params._vec()
        #bounds_vec = self.params._bounds_vec()
        #self.std = (bounds_vec[1]-bounds_vec[0])/20.
        bounds = self.params._bounds_vec()
        self._bounds = [np.array(bounds[0]), np.array(bounds[1])]
        old_prob, old_value = self._prob_func(self.params._val())
        if self._samples is None:
            self._samples = [self.c0]
            self._probs = [old_prob]
            self._values = [old_value]
        for i in tqdm(range(n)):
            new = self._proposal()
            #print(new)
            if np.any(new < self._bounds[0]) or np.any(new > self._bounds[1]):
                continue
            self.params._set(new) 
            new_prob, new_value = self._prob_func(self.params._val())
            if new_prob>=old_prob:
                alpha = 1.
            else:
                alpha = min(new_prob/old_prob,1)
            if np.random.uniform() < alpha:
                self.c0 = new
                old_prob = new_prob
                self._samples.append(self.c0)
                self._probs.append(old_prob)
                self._values.append(new_value)

        print("MCMC sampling completed. Acceptance rate:",len(self._samples)/n)
        print("Number of samples:",len(self._samples))
        print("To access the samples, use get_samples()")

    def get_samples(self):
        '''
        Return the samples
        '''
        if self._samples is None:
            raise ValueError("MCMC has not been run yet")
        return self._samples.copy()
    
    def get_probs(self):
        '''
        Return the probabilities
        '''
        if self._probs is None:
            raise ValueError("MCMC has not been run yet")
        return self._probs.copy()
    
    def get_values(self):
        '''
        Return the values
        '''
        if self._values is None:
            raise ValueError("MCMC has not been run yet")
        return self._values.copy()