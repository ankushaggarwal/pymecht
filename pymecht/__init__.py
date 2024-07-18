'''
pyMechT is a Python package for simulating the mechanical response of soft biological tissues. To start using, install it via `pip` as follows

.. code-black:: sh
   $ pip install pymecht

and being with the Tutorials in this documentation. 

pyMechT is a collection of modules for:

    * :class:`MatModels': defining material models
    * :class:`SampleExperiment': simulating experiments, such as uniaxial extension, biaxial extension, and inflation-extension. Simulations can be either :method:`disp_controlled' or :method:`force_controlled'
    * :class:`ParamFitter': fitting parameters to experimental data
    * :class:`RandomParameters' and :class:`MCMC': Bayesian inference by running Monte Carlo (MC) and Markov chain Monte Carlo (MCMC) simulations

.. image:: _static/my_figure.png
  :width: 300
  :alt: Structure of pyMechT
  :align: center

This package is developed and maintained by the 

.. raw::html
  <a href="https://userweb.eng.gla.ac.uk/ankush.aggarwal" target="_blank">Computational Biomechanics Research Grou </a>

at the University of Glasgow, and is hosted at 

.. raw::html
  <a href="https://github.com/ankushaggarwal/pymecht" target="_blank">github.</a>

The ethos of pyMechT is to create simplified virtual experimental setups, rather than finite element analysis. Thus, varying parameters and running simulations is much faster, allowing Bayesian inference and Markov Chain Monte Carlo.

'''

from .ParamDict import *
from .MatModel import *
from .RandomParameters import *
from .SampleExperiment import *
from .ParamFitter import *
from .MCMC import *

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pymecht")
except PackageNotFoundError:
    # package is not installed
    pass

