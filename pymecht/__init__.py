'''
pyMechT is a Python package for simulating the mechanical response of soft biological tissues. To start using, install it via `pip` as follows

.. code-block:: sh

   $ pip install pymecht

Tutorial provided in this documentation is a good place to start, followed by several detailed examples. 

pyMechT is a collection of modules for:

    * :class:`MatModels`: defining material models
    * :class:`SampleExperiment`: simulating experiments, such as uniaxial extension, biaxial extension, and inflation-extension. Simulations can be either :meth:`disp_controlled` or :meth:`force_controlled`
    * :class:`ParamFitter`: fitting parameters to experimental data
    * :class:`RandomParameters` and :class:`MCMC`: Bayesian inference by running Monte Carlo (MC) and Markov chain Monte Carlo (MCMC) simulations

.. image:: drawing-1.svg 
  :width: 300
  :alt: Structure of pyMechT
  :align: center

This package is developed and maintained by the `Computational Biomechanics Research Group <https://userweb.eng.gla.ac.uk/ankush.aggarwal/>`_ at the University of Glasgow and is hosted at `github <https://github.com/ankushaggarwal/pymecht/>`_.

The ethos of pyMechT is to create simplified virtual experimental setups, rather than finite element analysis. Thus, varying parameters and running simulations is much faster, making it feasible to perform Bayesian inference and Markov Chain Monte Carlo analysis.

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

