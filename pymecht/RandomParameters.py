import numpy as np
import warnings
from collections import OrderedDict
from pyDOE import lhs
from torch.quasirandom import SobolEngine

class Parameter:
    def __init__(self,low,high,value,rtype):
        self.value = value
        if rtype.lower() == 'fixed':
            self.rtype = 'fixed'
            return
        if low>high:
            print(low,high)
            raise ValueError("The upper bound of parameter cannot be lower than the lower bound")
        if (high-low) < 0.01*low+1e-10:
            warnings.warn("High and low values are within 1%, while the parameter is not considered fixed. It might be better to do that instead.")

        if rtype.lower() == 'uniform':
            self.low = low
            self.high = high
            self.rtype = 'uniform'
            return

        elif rtype.lower() == 'normal':
            self.mean = (high+low)/2.
            self.std = (high-low)/4. #4*std considered to cover the provided range
            self.rtype = 'normal'
            return

        elif rtype.lower() == 'lognormal':
            self.mean = (np.log(high)+np.log(low))/2.
            self.std = (np.log(high)-np.log(low))/4. #4*std considered to cover the provided range
            self.rtype = 'lognormal'
        else:
            raise ValueError(rtype,"not implemented")

    def __repr__(self):
        if self.rtype=='fixed':
            return 'Fixed:{0}'.format(self.value)
        if self.rtype=='uniform':
            return 'Uniform between {0} and {1}'.format(self.low,self.high)
        if self.rtype=='normal':
            return 'Normal with mean {0} and standard deviation {1}'.format(self.mean,self.std)
        if self.rtype=='lognormal':
            return 'LogNormal with log mean {0} and log standard deviation {1}'.format(self.mean,self.std)
        else:
            raise ValueError('Unknown type')

    def __str__(self):
        return self.__repr__

    def sample(self,x):
        if self.rtype == 'fixed':
            return self.value
        if self.rtype == 'uniform':
            return self.low+x*(self.high-self.low)
        if self.rtype == 'normal':
            return self.mean + x*self.std
        if self.rtype == 'lognormal':
            return np.exp(self.mean + x*self.std)

    def fix(self,x=None):
        self.rtype = 'fixed'
        if x is not None:
            self.value = x

    def make_normal(self,x=None):
        self.rtype = 'normal'
        self.mean = (self.high+self.low)/2.
        self.std = (self.high-self.low)/4.
        if x is not None:
            self.mean = x

    def make_lognormal(self,x=None):
        self.rtype = 'lognormal'
        self.mean = (np.log(self.high)+np.log(self.low))/2.
        self.std = (np.log(self.high)-np.log(self.low))/4.
        if x is not None:
            self.mean = np.log(x)

    def prob(self,x):
        if self.rtype=='fixed':
            return 1.
        if self.rtype=='uniform':
            if x>self.low and x<self.high:
                return 1./(self.high-self.low)
            return 0.
        if self.rtype=='normal':
            return 1./(self.std*np.sqrt(2*np.pi))*np.exp(-(x-self.mean)**2/(2*self.std**2))
        if self.rtype=='lognormal':
            return 1./(self.std*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-self.mean)**2/(2*self.std**2))

class RandomParameters:
    def __init__(self,params):
        '''
        A class to generate random parameters from sample parameters
        Parameters
        ----------
        params : ParamDict
            A dictionary of parameters to be varied
        '''
        param = params.val()
        param_low = params.lb()
        param_up = params.ub()
        param_type = params.fixed()

        self._param_sample = param.copy()
        for p in param_type.keys():
            if param_type[p]:
                param_type[p]='fixed'
                param_low[p]=param[p]
                param_up[p]=param[p]
            else:
                param_type[p]='uniform' #by default all parameters are created uniform
        set_keys = set(param.keys())
        if set(param_low.keys()) != set_keys or set(param_up.keys()) != set_keys or set(param_type.keys()) != set_keys:
            raise ValueError("The dictionaries of parameter default, upper, and lower values have different set of keys",param_low,param_up,param)
        self._parameters=OrderedDict([])
        for k in param.keys():
            self._parameters[k]=Parameter(param_low[k],param_up[k],param[k],param_type[k])
        #print(self._parameters)
    def _format_float(self,x):
        if (x>0.01 and x<100) or x==0:
            return "{:<18}".format("{:.2f}".format(x))
        else:
            return "{:<18}".format("{:.2e}".format(x))
        
    def __str__(self):
        #return str(self._parameters)
        lines = ''
        header = "{:<18}".format("Keys")
        header += "{:<18}".format("Type")
        header += "{:<18}".format("Lower/mean")
        header += "{:<18}".format("Upper/std")
        lines = header + '\n' + '-'*len(header) + '\n'
        #print("-"*len(header))
        #print(header)
        #print("-"*len(header))
        for k in self._parameters.keys():
            line = "{:<18}".format(k)
            line += "{:<18}".format(self._parameters[k].rtype)
            if self._parameters[k].rtype == 'fixed':
                line += self._format_float(self._parameters[k].value)
                line += self._format_float(self._parameters[k].value)
            elif self._parameters[k].rtype == 'uniform':
                line += self._format_float(self._parameters[k].low)
                line += self._format_float(self._parameters[k].high)
            elif self._parameters[k].rtype == 'normal':
                line += self._format_float(self._parameters[k].mean)
                line += self._format_float(self._parameters[k].std)
            elif self._parameters[k].rtype == 'lognormal':
                line += self._format_float(np.exp(self._parameters[k].mean))
                line += self._format_float(np.exp(self._parameters[k].std))
            #print(line)
            lines += line + '\n'
        lines += '-'*len(header) + '\n'
        #print("-"*len(header))
        return lines
    
    def __repr__(self):
        return self.__str__()

    def sample(self,N=1,sample_type=None):
        '''
        Returns a list of N samples of the parameters
        Parameters
        ----------
        N : int
            Number of samples
        sample_type : str
            Type of sampling to be performed. 
            Options are None (default), 'lhcube', and 'sobol'
            If None, random sampling is performed.
        Returns
        -------
        all_samples : A list of N dictionaries with random values of the parameters
        '''
        var_type = np.array([param.rtype for param in self._parameters.values()])
        ndim = len(var_type)
        nNorm = sum((var_type=='normal')+(var_type=='lognormal'))
        nUni  = sum(var_type=='uniform')
        nFix  = sum(var_type=='fixed')
        all_presamples = np.zeros([N,ndim])

        if sample_type is None:
            #perform random sampling
            normal_samples = np.random.normal(0,1,size=(N,nNorm))
            uniform_samples = np.random.random(size=(N,nUni))

        elif sample_type.lower() == 'lhcube':
            #Perform Latin-hypercube sampling
            if nUni != ndim-nFix:
                raise ValueError("For Latin Hypercube sampling, only uniform distributions are currently implemented")
            normal_samples = []
            uniform_samples = lhs(nUni,N)

        elif sample_type.lower() == 'sobol':
            #Perform Latin-hypercube sampling
            if nUni != ndim-nFix:
                raise ValueError("For Sobol sequence sampling, only uniform distributions are currently implemented")
            normal_samples = []
            uniform_samples = SobolEngine(nUni,True).draw(N).detach().numpy().astype('float64')

        all_presamples[:,var_type=='fixed'] = 0.
        all_presamples[:,(var_type=='normal')+(var_type=='lognormal')] = normal_samples
        all_presamples[:,var_type=='uniform'] = uniform_samples
        all_samples = []
        for i in range(N):
            sample = self._param_sample.copy()
            for j,k in enumerate(self._parameters.keys()):
                sample[k] = self._parameters[k].sample(all_presamples[i,j]) 
            all_samples.append(sample)
        return all_samples

    def fix(self,keys,x=None):
        '''
        Fix the parameters to the value x
        Parameters
        ----------
        keys : str or list of str
            The keys of the parameters to be fixed
        x : float or list of float
            The value to which the parameters are fixed
        '''
        if type(keys) is list:
            if x is None:
                for key in keys:
                    self._parameters[key].fix(x)
            else:
                assert(len(keys)==len(x))
                for i,key in enumerate(keys):
                    self._parameters[key].fix(x[i])
        else:
            self._parameters[keys].fix(x)

    def make_normal(self,key,x=None):
        '''
        Make the parameter normal with mean x
        Parameters
        ----------
        key : str
            The key of the parameter to be made normal
        x : float
            The mean of the normal distribution
        '''
        self._parameters[key].make_normal(x)

    def make_lognormal(self,key,x=None):
        '''
        Make the parameter lognormal with mean x
        Parameters
        ----------
        key : str
            The key of the parameter to be made lognormal
        x : float
            The mean of the lognormal distribution
        '''
        self._parameters[key].make_lognormal(x)

    def add(self,key,value,low,high,rtype):
        '''
        Add a new parameter
        Parameters
        ----------
        key : str
            The key of the parameter
        value : float
            The default value of the parameter
        low : float
            The lower bound of the parameter
        high : float
            The upper bound of the parameter
        rtype : str
            The type of the parameter. Options are 'fixed', 'uniform', 'normal', and 'lognormal'
        '''
        if key in self._parameters.keys():
            raise ValueError("The key already exists")
        self._parameters[key]=Parameter(low,high,value,rtype)

    def prob(self,param_sample):
        '''
        Returns the probability of the parameter sample
        Parameters
        ----------
        param_sample : dict
            A dictionary of the parameter sample
        Returns
        -------
        p : float
            The probability of the parameter sample
        '''
        if set(param_sample.keys()) != set(self._parameters.keys()):
            raise ValueError("The parameter sample has different variables")

        p=1.
        for k in self._parameters.keys():
            p *= self._parameters[k].prob(param_sample[k])

        return p
