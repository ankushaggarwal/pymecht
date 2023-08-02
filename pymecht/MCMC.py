import numpy as np
from tqdm import tqdm

class MCMC:
    def __init__(self,prob_func,params,params_lb=None,params_ub=None,params_fix=None):
        '''
        A class for performing MCMC sampling of a given probability function based on SampleExperiment class object
        Parameters
        ----------
        prob_func : function
            A function that takes in a dictionary of parameters and returns a tuple of (probability, value)
        params : dict
            A dictionary of parameters to be varied
        params_lb : dict
            A dictionary of lower bounds of the parameters to be varied, default is -inf
        params_ub : dict
            A dictionary of upper bounds of the parameters to be varied, default is inf
        params_fix : dict
            A dictionary of boolean values indicating whether the parameters are fixed or not, default is False
        '''
        self._prob_func = prob_func
        self.params,self._lb,self._ub,self._params_fix = params,params_lb,params_ub,params_fix
        self._keys = self.params.keys()
        self.update_ranges()
        #Test that the prob_func gives a float
        prob_result, value = self._prob_func(self.params)
        if not isinstance(prob_result,float):
            raise ValueError("prob_func should output a float. Instead it gave", prob_result)
        print("MCMC instance created with the following settings")
        self._param_print_transpose()
        print(self._n_params,"parameters will be varied. For changing any of the bounds/fixed, use update_ranges function")

        self._samples = None
        self._probs = None
        self._values = None

    def _format_float(self,x):
        if (x>0.01 and x<100) or x==0:
            return "{:<12}".format("{:.2f}".format(x))
        else:
            return "{:<12}".format("{:.2e}".format(x))

    def _param_print_transpose(self):
        header = "{:<18}".format("Keys")
        header += "{:<12}".format("Initial value")
        header += "{:<12}".format("Fixed?")
        header += "{:<12}".format("Lower bound")
        header += "{:<12}".format("Upper bound")
        print("-"*len(header))
        print(header)
        print("-"*len(header))
        for k in self._keys:
            line = "{:<18}".format(k)
            line += self._format_float(self.params[k])
            line += "{:<12}".format(str(self._params_fix[k]))
            if self._lb is None:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("-inf")
            else:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self._lb[k])
            if self._ub is None:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("inf")
            else:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self._ub[k])
            print(line)
        print("-"*len(header))

    def _param_print(self):
        header = "{:<18}".format("Keys")
        for k in self._keys:
            header += "{:<12}".format(k)
        print("-"*len(header))
        print(header)
        print("-"*len(header))
        line = "{:<18}".format("Initial value")
        for k in self._keys:
            line += self._format_float(self.params[k])
        print(line)
        line = "{:<18}".format("Fixed?")
        for k in self._keys:
            line += "{:<12}".format(str(self._params_fix[k]))
        print(line)
        line = "{:<18}".format("Lower bound")
        if self._lb is None:
            for k in self._keys:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("-inf")
        else:
            for k in self._keys:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self._lb[k])
        print(line)
        line = "{:<18}".format("Upper bound")
        if self._ub is None:
            for k in self._keys:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("inf")
        else:
            for k in self._keys:
                if self._params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self._ub[k])
        print(line)
        print("-"*len(header))

    def update_ranges(self,params_lb=None,params_ub=None,params_fix=None):
        '''
        Update the bounds and fixed parameters
        Parameters
        ----------
        params_lb : dict
            A dictionary of lower bounds of the parameters to be varied, default is -inf
        params_ub : dict
            A dictionary of upper bounds of the parameters to be varied, default is inf
        params_fix : dict
            A dictionary of boolean values indicating whether the parameters are fixed or not, default is False
            '''
        if params_lb is not None:
            self._lb = params_lb
            if self._lb.keys() != self._keys:
                raise ValueError("Keys of the input lower bound of parameters do not match the model parameter keys")

        if params_ub is not None:
            self._ub = params_ub
            if self._ub.keys() != self._keys:
                raise ValueError("Keys of the input upper bound of parameters do not match the model parameter keys")

        if params_fix is not None:
            self._params_fix = params_fix
            if self._params_fix.keys() != self._keys:
                raise ValueError("Keys of the input params_fix do not match the model parameter keys")

        #set the values and bounds
        if self._params_fix is None:
            self._params_fix = self.params.copy()
            for k in self._keys:
                self._params_fix[k] = False #by default fit all parameters
        self._n_params = np.sum(~self._dict2list(self._params_fix))
        if self._lb is not None:
            lb  = self._vec(self._lb)
        else:
            lb  = np.array([-np.inf]*self._n_params)
        if self._ub is not None:
            ub      = self._vec(self._ub)
        else:
            ub  = np.array([np.inf]*self._n_params)
        self._bounds = (lb,ub)
        self.std = (ub-lb)/20.

    def _vec(self,x):
        return np.array([value for key, value in x.items() if not self._params_fix[key]])

    def _dict2list(self,xdict):
        x = []
        for k in xdict.keys():
            x.append(xdict[k])
        return np.array(x)

    def _proposal(self):
        dx = np.random.normal(size=self._n_params)*self.std #could multiply by a vector of standard deviations for each parameter
        return self.c0 + dx

    def run(self,n):
        '''
        Run the MCMC sampling
        Parameters
        ----------
        n : int
            Number of MCMC iterations to perform (number of samples will be less than n due to rejection sampling)
        '''
        self.c0 = self._vec(self.params)
        old_prob, old_value = self._prob_func(self.params)
        if self._samples is None:
            self._samples = [self.c0]
            self._probs = [old_prob]
            self._values = [old_value]
        for i in tqdm(range(n)):
            new = self._proposal()
            #print(new)
            if np.any(new < self._bounds[0]) or np.any(new > self._bounds[1]):
                continue
            self._complete_params(new) 
            new_prob, new_value = self._prob_func(self.params)
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


    def _complete_params(self,cval):
        i=0
        for key,value in self.params.items():
            if not self._params_fix[key]:
                try:
                    self.params[key] = cval[i]
                    i += 1
                except IndexError as err:
                    print("Non-fixed parameters and cval are of different length",err)
        return
