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
along the other two. Thus, we can :math:`p` by equating
:math:`\sigma_{22}=\sigma_{33}=0`. From the calculated stress, the force
can be calculated as :math:`P_{11}A_0`. From stretch, other deformation
metrics can be calculated. Deformed length :math:`l=\lambda L_0`, change
in length :math:`\Delta l = (\lambda-1)L_0`, and strain
:math:`\epsilon = (\lambda-1)`. Given deformation (in terms of
stretch/change in length, strain, or deformed length), the stress or
force can be calculated using disp_controlled function. Conversely,
given stress or force, any of the deformation metric are solved
iteratively.
