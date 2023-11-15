=========================================================
UniformAxisymmetricTubeInflationExtension and LayeredTube
=========================================================


.. figure:: tube.svg
   :alt: 
   :width: 50.0%
   :align: center

We consider a cylinder of reference inner radius :math:`R_i`, thickness
:math:`H`, opening angle :math:`\omega` (i.e., the stress-free state is
an “open” cylinder of an angle :math:`2\pi-\omega`), and length
:math:`L_0`. When deformed state, the cylinder is “closed” and stretched
along the length by :math:`\lambda_Z`. Cylindrical coordinate system is
used with the first coordinate being radial, second being the
circumferential, and third along the length. The material is assumed to
be incompressible. As a result, when the deformed inner radius is
:math:`r_i`, any point which was at radius :math:`R` in the reference
configuration moves to a deformed radius given by

.. math:: r(R) = \sqrt{ r_i^2 + \frac{R^2-R_i^2}{\kappa\lambda_Z} },

where :math:`\kappa=2\pi/(2\pi-\omega)`. Thus, in cylinderical
coordinates, the deformation gradient at any point is given by

.. math:: \mathbf{F} = \mathop{\mathrm{diag}}\left[\frac{R}{r \kappa \lambda_Z }, \frac{r\kappa}{R}, \lambda_Z\right].

As a result, the Cauchy stress tensor can be calculated at any point
(without the Lagrange multiplier :math:`p` term). The pressure
difference between inside and outside of the artery can then be written
as:

.. math::

   \Delta p = -\int\limits_{R_i}^{R_i+H}\frac{R}{\lambda_Z r^2} \left( \bar{\sigma}_{rr} - \bar{\sigma}_{\theta\theta} \right){\textrm{d}R}.
   \label{main-eq}

This integral is evaluated numerically. If the fiber directions are not
symmetric about the length of the cylinder, there could be shear stress
components. However, these are neglected. Lastly, from the pressure
difference :math:`\Delta p`, force acting on the cylinder can be
calculated as :math:`f = 2\pi r_i L_0 \lambda_Z`. The deformation can be
written in terms of either the inner radius :math:`r_i`, the radius
stretch :math:`r_i/R_i`, change in internal radius
:math:`\Delta r_i = r_i - R_i`, or the deformed (internal) luminal area
of the cylinder :math:`A = \pi r_i^2`.

Given deformation (in terms of inner radius/radius stretch/change in
radius, or deformed luminal area), the force measure can be calculated
using disp_controlled function. Conversely, given pressure difference or
force, any of the deformation metric are solved iteratively.

Lastly, the Cauchy stress tensor can be calculated by calculating the
Lagrange multiplier :math:`p` (which will vary across the thickness) by 
assuming the pressure on the external surface as zero, thus:

.. math:: {p}(R) = \bar{\sigma}_{rr}(R) + \Delta p +\int\limits_{R_i}^{R} \frac{RH}{r^2}\left[\sigma_{rr}-\sigma_{\theta\theta}\right] \textrm{d}\xi,\label{lagrange-multiplier2}

where :math:`\bar{\sigma}_{rr}(R)` is the Cauchy normal stress in the
(first) radial direction without the Lagrange multiplier term. Once
:math:`{p}(R)` is known, all the stresses can be calculated using the
usual definition of Cauchy stress.

UniformAxisymmetricTubeInflationExtension samples can be “layered” via
LayeredTube. Such a setup can be used for representing, for example,
tissues that have multiple layers with different material models and
possibly even incompatible reference radius. The result would be that
there is no zero stress state for the layered sample. If the reference
compatibility is desired (i.e., if the reference state of each layer is
desired to also be the equilibrium of the combined layered state), then
the radius and thickness of each layer should be chosen appropriately.
Specifically, the outer reference radius of the innermost layer should
be the inner radius of the second layer, and so on.
