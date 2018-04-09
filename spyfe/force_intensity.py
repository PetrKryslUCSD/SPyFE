import numpy

class ForceIntensity:
    """
    Distributed force (force intensity) class.

    The force intensity class. The physical units are
    force per unit volume, where volume depends on to which manifold
    the force is applied:
    force/length^3 (when applied to a 3-D solid),
    force/length^2 (when applied to a surface),
    force/length^1 (when applied along a curve),
    or force/length^0 (when applied at a point).

    """

    def __init__(self, magn=lambda x, J: 1.0):
        self.magn = magn

    def get_magn(self, x, J):
        return self.magn(x, J)
