import numpy as np
from collections import UserDict
import pandas as pd

class Param:
    '''
    A data structure for a single parameter. 
    ParamDict is a dictionary of Param objects.

    This class is used internally by ParamDict and is not intended to be used directly by the user.
    '''
    def __init__(self,value: float = 0., low: float = -np.inf, high: float = np.inf, fixed: bool = False):
        self.value = value
        self.low = low
        self.high = high
        self.fixed = fixed
    
    def _format_float(self,x):
        if (x>0.01 and x<100) or x==0:
            return "{:<12}".format("{:.2f}".format(x))
        else:
            return "{:<12}".format("{:.2e}".format(x))

    def __str__(self):
        #if self.fixed:
        #    return str(self.value) + "\t" + str(self.low) + "\t" + str(self.high) + "\tFixed"
        #else:
        #    return str(self.value) + "\t" + str(self.low) + "\t" + str(self.high) + "\tNot fixed"
        line = self._format_float(self.value)
        if self.fixed:
            line += "{:<12}".format("Yes")
        else:
            line += "{:<12}".format("No")
        if self.fixed:
            line += "{:<12}".format("-")
            line += "{:<12}".format("-")
        else:
            line += self._format_float(self.low)
            line += self._format_float(self.high)
        return line

    def __repr__(self):
        return self.__str__()
    
    def set(self,value: float):
        '''
        Set the value of the parameter

        Parameters
        ----------
        value : float
            Value of the parameter

        Returns
        -------
        None
        '''
        self.value = value
        if self.fixed:
            self.low = self.high = self.value

    def fix(self, x=None):
        '''
        Fix the parameter to the current value

        Parameters
        ----------
        x : float, optional
            Value of the parameter to be fixed to. If not provided, the current value is used.

        Returns
        -------
        None
        '''
        if x is not None:
            self.value = x  
        self.fixed = True
        self.low, self.high = self.value, self.value

    def __mul__(self,x):
        self.value *= x
        return self

    def __rmul__(self,x):
        self.value *= x
        return self

class ParamDict(dict):
    '''
    A custom dictionary for storing parameters. The keys are strings and the values are Param objects.
    The mat_model.parameters and sample_experiment.parameters are instances of ParamDict.
    '''
    def __init__(self, mapping=None, /, **kwargs):
        if mapping is not None:
            mapping = {
                str(key): (value if type(value) is Param else Param(value)) for key, value in mapping.items()
            }
        else:
            mapping = {}
        if kwargs:
            mapping.update(
                {str(key): (value if type(value) is Param else Param(value)) for key, value in kwargs.items()}
            )
        super().__init__(mapping)

    def __setitem__(self, key, value: Param):
        assert(type(value) is Param)
        super().__setitem__(key, value)

    def update(self, mapping=None, **kwargs):
        if mapping is not None:
            #mapping = {
            #    str(key): (value if type(value) is Param else Param(value)) for key, value in mapping.items()
            #}
            for value in mapping.values():
                assert(type(value) is Param)
        else:
            mapping = {}
        if kwargs:
            mapping.update(
                {str(key): (value if type(value) is Param else Param(value)) for key, value in kwargs.items()}
            )
        super().__init__(mapping)

    def _val(self):
        return {k: p.value for k,p in self.items()}

    def _fixed(self):
        return {k: p.fixed for k,p in self.items()}

    def _lb(self):
        return {k: p.low for k,p in self.items()}

    def _ub(self):
        return {k: p.high for k,p in self.items()}

    def _bounds(self):
        return {k: (p.low,p.high) for k,p in self.items()}

    def fix(self,k: str, x=None):
        '''
        Fix a parameter

        Parameters
        ----------
        k : str
            Key of the parameter to be fixed

        x : float, optional
            Value of the parameter to be fixed to. If not provided, the current value is used.

        Returns
        -------
        None
        '''
        self[k].fix(x)

    def _vec(self):
        return [p.value for k,p in self.items() if not p.fixed]

    def _bounds_vec(self):
        lb = [p.low for k,p in self.items() if not p.fixed]
        ub = [p.high for k,p in self.items() if not p.fixed]
        return (lb,ub)

    def set(self,k: str,x: float):
        '''
        Set the value of a parameter

        Parameters
        ----------
        k : str
            Key of the parameter to be set

        x : float
            Value of the parameter
        
        Returns
        -------
        None
        '''
        if k not in self.keys():
            raise ValueError(k,"not a key in the ParamDict object")
        self[k].set(x)

    def set_lb(self,k: str,x: float):
        '''
        Set the lower bound of a parameter

        Parameters
        ----------
        k : str
            Key of the parameter to be set

        x : float
            Lower bound of the parameter
        
        Returns
        -------
        None
        '''
        if k not in self.keys():
            raise ValueError(k,"not a key in the ParamDict object")
        self[k].low = x

    def set_ub(self,k: str,x: float):
        '''
        Set the upper bound of a parameter

        Parameters
        ----------
        k : str
            Key of the parameter to be set

        x : float
            Upper bound of the parameter

        Returns
        -------
        None
        '''
        if k not in self.keys():
            raise ValueError(k,"not a key in the ParamDict object")
        self[k].high = x
        
    def _set(self, x: np.array):
        if len(x) != self._n():
            raise ValueError("ParamDict._set: length of the input array is not consistent with the number of non-fixed parameters")
        i = 0
        for k in self.keys():
            if not self[k].fixed:
                self[k].set(x[i])
                i += 1

    def _dict(self,vec: np.ndarray) -> dict:
        x = dict()
        j = 0
        for k in self.keys():
            if self[k].fixed:
                x[k] = self[k].value
            else:
                x[k] = vec[j]
                j += 1
        return x

    def to_pandas(self):
        '''
        Convert the parameters to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parameters
        '''
        p = dict()
        for k,v in self.items():
            p[k] = [v.value, v.low, v.high, v.fixed]
        return pd.DataFrame.from_dict(p,orient="index", columns=['Initial value', 'Lower bound', 'Upper bound', 'Fixed'])

    def save(self,fname):
        '''
        Save the parameters to a csv file

        Parameters
        ----------
        fname : str
            Name of the file to be saved
        
        Returns
        -------
        None
        '''
        df = self.to_pandas()
        df.to_csv(fname)

    def read(self,fname):
        '''
        Read the parameters from a csv file

        Parameters
        ----------
        fname : str
            Name of the file to be read

        Returns
        -------
        None
        '''
        df = pd.read_csv(fname,index_col=0)
        for k in df.index:
            self[k]=Param(df.loc[k,'Initial value'],df.loc[k,'Lower bound'],df.loc[k,'Upper bound'],df.loc[k,'Fixed'])

        for k,p in self.items():
            if p.fixed:
                self[k].low = self[k].high = self[k].value

    def _n(self):
        i = 0
        for k in self.keys():
            if not self[k].fixed:
                i += 1
        return i

    def __str__(self):
        header = "{:<18}".format("Keys")
        header += "{:<12}".format("Value")
        header += "{:<12}".format("Fixed?")
        header += "{:<12}".format("Lower bound")
        header += "{:<12}".format("Upper bound")
        line = "-"*len(header) + "\n" + header + "\n" + "-"*len(header) + "\n"
        for k in self.keys():
            line += "{:<18}".format(k)+ self[k].__str__() + "\n"
        line += "-"*len(header) + "\n" 
        return line

    def __repr__(self):
        return self.__str__()

