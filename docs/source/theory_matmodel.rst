========
MatModel
========


MatModel class is designed to allow adding multiple material models
together. Each of the model defines a strain energy density function
:math:`\Psi_i`, so that the total strain energy density is

.. math:: \Psi(\mathbf{F},\boldsymbol{\theta}) =  \sum\limits_i \Psi_i(\mathbf{F},\boldsymbol{\theta}_i).

Here, :math:`\boldsymbol{\theta} = \cup_i \boldsymbol{\theta}_i` is the
union of all material parameters and :math:`\boldsymbol{\theta}_i` are
the parameters of :math:`i`-th material. A key point here is that each
material experiences the same deformation. Currently, only hyperelastic
material models (invariant-based and structural) are implemented. For a
compressible material, three types of stress tensors can be written as:

.. math::

   \begin{aligned}
   \mathbf{S} &= 2 \dfrac{\partial \Psi}{\partial \mathbf{C}} \\
   \mathbf{P} &= \mathbf{F}\cdot\mathbf{S} \\
   \boldsymbol{\sigma} &= \frac{1}{J} \mathbf{F} \cdot \mathbf{S} \cdot \mathbf{F}^{\top},
   \end{aligned}

where :math:`\mathbf{S}` is the second Piola-Kirchhoff stress tensor,
:math:`\mathbf{P}` is the first Piola-Kirchhoff stress tensor,
:math:`\boldsymbol{\sigma}` is the Cauchy stress tensor,
:math:`\mathbf{C}=\mathbf{F}^\top\mathbf{F}` is the right Cauchy-Green
deformation tensor, and
:math:`J=\det(\mathbf{F})=\sqrt{\det(\mathbf{C})}` is the Jacobian of
the deformation gradient.

For incompressible case, we need to add a Lagrange multiplier term that
enforces :math:`J=1`. Thus,

.. math::

   \begin{aligned}
   \mathbf{S} &= 2 \dfrac{\partial \Psi}{\partial \mathbf{C}} - p {\mathbf{C}}^{-1}, \\
   \mathbf{P} &= \mathbf{F}\cdot\mathbf{S} - p {\mathbf{F}}^{-\top}\\
   \boldsymbol{\sigma} &= \mathbf{F} \cdot \mathbf{S} \cdot \mathbf{F}^{\top} - p \mathbf{I}.
   \end{aligned}

When calling the stress or energy functions in the code, the deformation
gradient and model parameters are needed. In addition, whether the
deformation gradient can be assumed to be diagonal or not can be
specified (since a diagonal tensor is easier to invert). Lastly, whether
the material is compressible or not can be specified. For incompressible
material, the Lagrange multiplier :math:`p` is calculated by setting
:math:`\sigma_{33}=0`. Generally speaking, while one can use the
MatModel class directly, it is meant to be used by embedding it into a
SampleExperiment.

For invariant-based material models that depend on the first four strain
invariants

.. math::

   \begin{aligned}
   I_1 &= \mathop{\mathrm{tr}}(\mathbf{C}) \\
   I_2 &= \frac{1}{2} \left[ \mathop{\mathrm{tr}}(\mathbf{C}^2) - \left(\mathop{\mathrm{tr}}(\mathbf{C}) \right)^2 \right] \\
   J &= \sqrt{\det(\mathbf{C})} \\
   I_4 &= \boldsymbol{M}\cdot\mathbf{C}\boldsymbol{M},
   \end{aligned}

we can write

.. math:: \dfrac{\partial \Psi}{\partial \mathbf{C}} = \dfrac{\partial \Psi}{\partial I_1} \mathbf{I} + \dfrac{\partial \Psi}{\partial I_2} \left(I_1\mathbf{I} - \mathbf{C}\right) + \dfrac{\partial \Psi}{\partial J} \frac{J}{2} \mathbf{C}^{-1} + \dfrac{\partial \Psi}{\partial I_4}  \boldsymbol{M}\otimes\boldsymbol{M}.

Each material can have several fiber families, each with a direction
(:math:`\boldsymbol{M}`) specified, and the stress contribution from
each fiber direction is added.

In the future, we hope to extend to inelastic models.
