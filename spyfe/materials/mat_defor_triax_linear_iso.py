import numpy
from spyfe.materials.mat_base import ErrorUnknownOutput
from spyfe.materials.mat_defor import ID3, OUTPUT_CAUCHY
from spyfe.materials.mat_defor_triax import MatDeforTriax, M1, M1M1T, MI

defaultkind = 0
lambdakind = 1
lambda_shearkind = 2
bulkkind = 3
bulk_shearkind = 4
constrainedkind = 5
unconstrainedkind = 6


class MatDeforTriaxLinearIso(MatDeforTriax):
    """Triaxial deformation material class.


    """

    def __init__(self, e, nu, alpha=0.0, rho=0.0):
        super().__init__(rho=rho)
        self.e = e
        self.nu = nu
        self.alpha = alpha
        lamb = self.e * self.nu / (1 + self.nu) / (1 - 2 * (self.nu))
        mu = self.e / (2 * (1 + self.nu))
        self._d = lamb * M1M1T + 2 * mu * MI

    def moduli_are_constant(self):
        return True

    def tangent_moduli(self, d_out, xyz=None, kind=defaultkind):
        """Compute the tangent moduli.
        
        :param d_out: The moduli matrix, 6 x 6.
        :param xyz: Location at which the moduli are to be computed.
        :param kind: Kind of the moduli (see above).
        :return: 
        """
        d_out[:,:] = self._d[:,:]

    def thermal_strain(self, dtemp=0.0):
        """Compute the thermal strain.

        :param dtemp: temperature increment from reference temperature, either a vector
        or None
        :return: vector of thermal strains if temperature increments applied as not None
        """
        return (-dtemp * self.alpha) * M1

    def thermal_stress(self, dtemp, xyz=None):
        """Compute the thermal stress.

        :param dtemp: temperature increment from reference temperature
        :return: vector of thermal stresses 
        """
        d = numpy.zeros((6,6))
        self.tangent_moduli(d, xyz=xyz, kind=defaultkind)
        epsth = self.thermal_strain(dtemp=dtemp)
        return numpy.dot(d, epsth)

    def state(self, msinout,
              fn1=None, fn=None, dt=None, strain=None, dtemp=None,
              output=OUTPUT_CAUCHY, quantityout=None):
        """Retrieve material state.

        :param msinout: Material state object. Input/output.
        :param fn1: current deformation gradient (at time t_n+1).
        :param fn: previous converged deformation gradient (at time t_n)
        :param dt: Time step, dt=t_n+1 - t_n
        :param strain: For small-strain materials this may be supplied
        instead of the deformation gradients.
        :param dtemp: temperature increment from the reference temperature
        :param output: enumeration, which output is desired.
        :param quantityout: array buffer for the output.
        :return: quantityout: If quantityout comes in as None, we will allocate 
        it and return it.
        """
        if strain is not None:
            ev = numpy.reshape(strain, (6, 1))
        else:
            ev = numpy.zeros((6, 1))
            gradu = fn1 - ID3
            self.str((gradu + gradu.T) / 2, ev)

        d = numpy.zeros((6,6))
        self.tangent_moduli(d)
        t_ev = self.thermal_strain(dtemp=dtemp)  # stress in local coordinates
        if t_ev is not None:
            stress = numpy.dot(d, (ev + t_ev))
        else:
            stress = numpy.dot(d, ev)

        if output == OUTPUT_CAUCHY:
            if quantityout is None:
                quantityout = numpy.zeros((6, 1))
            quantityout[:] = stress[:]
        else:
            raise ErrorUnknownOutput
        return quantityout
