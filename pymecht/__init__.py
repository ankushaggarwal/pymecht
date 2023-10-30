'''
pyMechT is a Python package for mechanics of tissue.
==================================
It is a collection of modules for:
    - defining material models
    - simulating experiments, such as uniaxial extension, biaxial extension, and inflation-extension
    - fitting parameters to experimental data
    - running Monte Carlo simulations

The package is developed by the Computational Biomechanics Research Group at the University of Glasgow.
The focus is on simplified models,rather than finite element models.
Thus, varying parameters and running simulations is extremely fast.
'''

from .ParamDict import *
from .MatModel import *
from .RandomParameters import *
from .SampleExperiment import *
from .ParamFitter import *
from .MCMC import *
