============
LinearSpring
============

For samples with single input/output, a locally linear spring is provided. 
It does not have any material and its equations are simply governed as a 
linear approximation

.. math:: f(x) = k(x-x_0) + f_0,

where :math:`f` is the force and :math:`x` is the deformation measure, 
:math:`x_0` is the deformation measure at the point about which we have 
linearized, and :math:`f_0` is the force at that point. The force can be 
converted into stress/pressure by dividing by an effective area :math:`A_0`.

:py:class:`LinearSpring` can be linked to other samples by "layering" them. 
For example, using a :py:class:`LayeredTube` the springs acts as a Robin-type 
boundary condition on the next sample.

