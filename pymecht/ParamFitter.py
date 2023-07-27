import numpy as np
from scipy.optimize import least_squares

class ParamFitter:
    def __init__(self,sim_func,output,params,params_lb=None,params_ub=None,params_fix=None):
        self.sim_func = sim_func
        self.output = output
        self.params,self.lb,self.ub,self.params_fix = params,params_lb,params_ub,params_fix
        self.keys = self.params.keys()
        self.update_ranges()
        #Test that the sim_func gives the same sized result as the "output"
        sim_result = self.sim_func(self.params)
        if sim_result.shape != self.output.shape:
            raise ValueError("Output and simulation results are of different shape", sim_result.shape, self.output.shape)
        print("Parameter fitting instance created with the following settings")
        self._param_print_transpose()
        print(self.n_params,"parameters will be fitted. For changing any of the bounds/fixed, use update_ranges function")

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
        for k in self.keys:
            line = "{:<18}".format(k)
            line += self._format_float(self.params[k])
            line += "{:<12}".format(str(self.params_fix[k]))
            if self.lb is None:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("-inf")
            else:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self.lb[k])
            if self.ub is None:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("inf")
            else:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self.ub[k])
            print(line)
        print("-"*len(header))

    def _param_print(self):
        header = "{:<18}".format("Keys")
        for k in self.keys:
            header += "{:<12}".format(k)
        print("-"*len(header))
        print(header)
        print("-"*len(header))
        line = "{:<18}".format("Initial value")
        for k in self.keys:
            line += self._format_float(self.params[k])
        print(line)
        line = "{:<18}".format("Fixed?")
        for k in self.keys:
            line += "{:<12}".format(str(self.params_fix[k]))
        print(line)
        line = "{:<18}".format("Lower bound")
        if self.lb is None:
            for k in self.keys:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("-inf")
        else:
            for k in self.keys:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self.lb[k])
        print(line)
        line = "{:<18}".format("Upper bound")
        if self.ub is None:
            for k in self.keys:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += "{:<12}".format("inf")
        else:
            for k in self.keys:
                if self.params_fix[k]:
                    line += "{:<12}".format("-")
                else:
                    line += self._format_float(self.ub[k])
        print(line)
        print("-"*len(header))

    def update_ranges(self,params_lb=None,params_ub=None,params_fix=None):
        #update the bounds and fix if provided
        if params_lb is not None:
            self.lb = params_lb
            if self.lb.keys() != self.keys:
                raise ValueError("Keys of the input lower bound of parameters do not match the model parameter keys")

        if params_ub is not None:
            self.ub = params_ub
            if self.ub.keys() != self.keys:
                raise ValueError("Keys of the input upper bound of parameters do not match the model parameter keys")

        if params_fix is not None:
            self.params_fix = params_fix
            if self.params_fix.keys() != self.keys:
                raise ValueError("Keys of the input params_fix do not match the model parameter keys")

        #set the values and bounds
        if self.params_fix is None:
            self.params_fix = self.params.copy()
            for k in self.keys:
                self.params_fix[k] = False #by default fit all parameters
        self.n_params = np.sum(~self._dict2list(self.params_fix))
        if self.lb is not None:
            lb  = self._vec(self.lb)
        else:
            lb  = np.array([-np.inf]*self.n_params)
        if self.ub is not None:
            ub      = self._vec(self.ub)
        else:
            ub  = np.array([np.inf]*self.n_params)
        self.bounds = (lb,ub)

    def _vec(self,x):
        return np.array([value for key, value in x.items() if not self.params_fix[key]])

    def _dict2list(self,xdict):
        x = []
        for k in xdict.keys():
            x.append(xdict[k])
        return np.array(x)

    def _residual(self,cval):
        assert(len(cval)==self.n_params)
        self._complete_params(cval)
        return self.sim_func(self.params) - self.output

    def fit(self):
        self.c0 = self._vec(self.params)
        result = least_squares(self._residual,x0=self.c0,bounds=self.bounds,verbose=2)
        return result

    def _complete_params(self,cval):
        i=0
        for key,value in self.params.items():
            if not self.params_fix[key]:
                try:
                    self.params[key] = cval[i]
                    i += 1
                except IndexError as err:
                    print("Non-fixed parameters and cval are of different length",err)
        return
