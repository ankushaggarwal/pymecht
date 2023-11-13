import numpy as np
from scipy.optimize import least_squares

class ParamFitter:
    '''
    A class to fit parameters of a simulation function to a given output

    Parameters
    ----------
    sim_func : function
        A function that takes in a dictionary of parameters and returns a numpy array of the same shape as the output

    output : numpy array
        A numpy array of the same shape as the output of the sim_func to be fitted

    params : ParamDict
        A dictionary of parameters to be varied

    '''
    def __init__(self,sim_func,output,params=None):
        self._sim_func = sim_func
        self._output = output
        self.params = params
        self._keys = self.params.keys()
        #Test that the sim_func gives the same sized result as the "output"
        sim_result = self._sim_func(self.params._val())
        if sim_result.shape != self._output.shape:
            raise ValueError("Output and simulation results are of different shape", sim_result.shape, self._output.shape)
        print("Parameter fitting instance created with the following settings")
        print(self.params)
        print(self.params._n(),"parameters will be fitted.")

    def _residual(self,cval):
        assert(len(cval)==self.params._n())
        p = self.params._dict(cval)
        return self._sim_func(p) - self._output

    def fit(self):
        '''
        Perform the fitting

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result of the fitting
        '''
        self.c0 = self.params._vec()
        result = least_squares(self._residual,x0=self.c0,bounds=self.params._bounds_vec(),verbose=2)
        p = self.params._dict(result.x)
        for k in p:
            self.params[k].set(p[k])
        print("Fitting completed, with the following results")
        print(self.params)
        return result
