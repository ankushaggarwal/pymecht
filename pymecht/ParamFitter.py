import numpy as np
from scipy.optimize import least_squares

class ParamFitter:
    def __init__(self,sim_func,output,params,params_lb=None,params_ub=None,params_fix=None):
        '''
        A class to fit parameters of a simulation function to a given output
        Parameters
        ----------
        sim_func : function
            A function that takes in a dictionary of parameters and returns a numpy array of the same shape as the output
        output : numpy array
            A numpy array of the same shape as the output of the sim_func to be fitted
        params : dict
            A dictionary of parameters to be varied
        params_lb : dict
            A dictionary of lower bounds of the parameters to be varied, default is -inf
        params_ub : dict
            A dictionary of upper bounds of the parameters to be varied, default is inf
        params_fix : dict
            A dictionary of boolean values indicating whether the parameters are fixed or not, default is False
        '''
        self._sim_func = sim_func
        self._output = output
        self.params,self._lb,self._ub,self._params_fix = params,params_lb,params_ub,params_fix
        self._keys = self.params.keys()
        self.update_ranges()
        #Test that the sim_func gives the same sized result as the "output"
        sim_result = self._sim_func(self.params)
        if sim_result.shape != self._output.shape:
            raise ValueError("Output and simulation results are of different shape", sim_result.shape, self._output.shape)
        print("Parameter fitting instance created with the following settings")
        self._param_print_transpose()
        print(self._n_params,"parameters will be fitted. For changing any of the bounds/fixed, use update_ranges function")

    def _format_float(self,x):
        if (x>0.01 and x<100) or x==0:
            return "{:<12}".format("{:.2f}".format(x))
        else:
            return "{:<12}".format("{:.2e}".format(x))

    def _param_print_transpose(self):
        header = "{:<18}".format("Keys")
        header += "{:<12}".format("Value")
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
        #update the bounds and fix if provided
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

    def _vec(self,x):
        return np.array([value for key, value in x.items() if not self._params_fix[key]])

    def _dict2list(self,xdict):
        x = []
        for k in xdict.keys():
            x.append(xdict[k])
        return np.array(x)

    def _residual(self,cval):
        assert(len(cval)==self._n_params)
        self._complete_params(cval)
        return self._sim_func(self.params) - self._output

    def fit(self):
        '''
        Perform the fitting
        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the fitting
        '''
        self.c0 = self._vec(self.params)
        result = least_squares(self._residual,x0=self.c0,bounds=self._bounds,verbose=2)
        print("Fitting completed, with the following results")
        self._param_print_transpose()
        return result

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
