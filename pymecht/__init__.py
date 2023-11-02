'''
pyMechT 
=======
pyMechT is a Python package for simulating the mechanical response of soft biological tissues.

It is a collection of modules for:

    * defining material models
    * simulating experiments, such as uniaxial extension, biaxial extension, and inflation-extension
    * fitting parameters to experimental data
    * Bayesian inference by running Monte Carlo (MC) and Markov chain Monte Carlo (MCMC) simulations

This package is developed and maintained by the `Computational Biomechanics Research Group <https://userweb.eng.gla.ac.uk/ankush.aggarwal/>`_ at the University of Glasgow.

The focus is on simplified models,rather than finite element analysis. 
Thus, varying parameters and running simulations is much faster, allowing Bayesian inference etc.

'''

from .ParamDict import *
from .MatModel import *
from .RandomParameters import *
from .SampleExperiment import *
from .ParamFitter import *
from .MCMC import *
