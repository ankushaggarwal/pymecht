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

    def make_normal(self,key,x=None):
        self.rtype = 'normal'
        self.mean = (self.high+self.low)/2.
        self.std = (self.high-self.low)/4.
        if x is not None:
            self.mean = x

    def make_lognormal(self,key,x=None):
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
    def __init__(self,param,param_low,param_up,param_type=None):
        self.param_sample = param.copy()
        if param_type is None:
            param_type = param.copy()
            for p in param_type.keys():
                param_type[p]='uniform' #by default all parameters are created uniform
        set_keys = set(param.keys())
        if set(param_low.keys()) != set_keys or set(param_up.keys()) != set_keys or set(param_type.keys()) != set_keys:
            raise ValueError("The dictionaries of parameter default, upper, and lower values have different set of keys",param_low,param_up,param)
        self.parameters=OrderedDict([])
        for k in param.keys():
            self.parameters[k]=Parameter(param_low[k],param_up[k],param[k],param_type[k])
        #print(self.parameters)

    def __str__(self):
        return str(self.parameters)

    def sample(self,N=1,sample_type=None):
        var_type = np.array([param.rtype for param in self.parameters.values()])
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
            sample = self.param_sample.copy()
            for j,k in enumerate(self.parameters.keys()):
                sample[k] = self.parameters[k].sample(all_presamples[i,j]) 
            all_samples.append(sample)
        return all_samples

    def fix(self,keys,x=None):
        if type(keys) is list:
            if x is None:
                for key in keys:
                    self.parameters[key].fix(x)
            else:
                assert(len(keys)==len(x))
                for i,key in enumerate(keys):
                    self.parameters[key].fix(x[i])
        else:
            self.parameters[keys].fix(x)

    def make_normal(self,key,x=None):
        self.parameters[key].make_normal(x)

    def add(self,key,value,low,high,rtype):
            self.parameters[key]=Parameter(low,high,value,rtype)

    def prob(self,param_sample):
        if set(param_sample.keys()) != set(self.parameters.keys()):
            raise ValueError("The parameter sample has different variables")

        p=1.
        for k in self.parameters.keys():
            p *= self.parameters[k].prob(param_sample[k])

        return p
