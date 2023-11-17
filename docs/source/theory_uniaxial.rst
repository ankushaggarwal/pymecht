==============================================
UniaxialExtension and LayeredUniaxialExtension
==============================================

.. figure:: uniax.svg
   :alt:
   :width: 50.0%
   :align: center

We consider a tissue sample with reference length :math:`L_0` and
cross-sectional area :math:`A_0`. For uniaxial extension, we stretch
along the first coordinate direction. Furthermore, we assume that the
material is incompressible and transversely isotropic (i.e., the
directions perpendicular to the stretch direction are
indistinguishable). Thus, the deformation gradient is:

.. math:: \mathbf{F} = \mathop{\mathrm{diag}}\left[\lambda,\frac{1}{\sqrt{\lambda}},\frac{1}{\sqrt{\lambda}}\right].

The resulting (normal) stress is non-zero along the first axis but zero
along the other two. Thus, we can calculate the Lagrange multiplier 
:math:`p` by equating
:math:`\sigma_{22}=\sigma_{33}=0`. From the calculated stress, the force
can be calculated as :math:`P_{11}A_0`. From stretch, other deformation
metrics can be calculated. Deformed length :math:`l=\lambda L_0`, change
in length :math:`\Delta l = (\lambda-1)L_0`, and strain
:math:`\epsilon = (\lambda-1)`. Given deformation (in terms of
stretch/change in length, strain, or deformed length), the stress or
force can be calculated using :py:meth:`SampleExperiment.disp_controlled`
function. Conversely, given stress or force, any of the deformation metric are solved
iteratively via :py:meth:`SampleExperiment.force_controlled` function.

:py:class:`UniaxialExtension` samples can be “layered” via :py:class:`LayeredUniaxialExtension`.
Such a setup can be used for representing, for example, tissues that
have multiple layers with different material models and possibly even
different reference lengths. The result would be that there is no zero
stress state for the layered sample. One has to be careful with the
inputs and outputs of the layered samples though. It is required to use
lengths (rather than stretches or strains) as the deformation metric.
Similarly, the force measure should not be stresses since the stresses
would not simply add up (instead they would be weighted by each layer’s
cross-sectional area).
