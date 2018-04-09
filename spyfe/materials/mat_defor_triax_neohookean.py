import numpy
from spyfe.materials.mat_base import ErrorUnknownOutput
from spyfe.materials.mat_defor import ID3, OUTPUT_CAUCHY, strain_3x3t_to_6v
from spyfe.materials.mat_defor_triax import MatDeforTriax, M1, M1M1, MI
import math


class MatDeforTriaxNeohookean(MatDeforTriax):
    """Triaxial deformation material class for the Neohookean model.


    """

    def __init__(self, e, nu, alpha=0.0, rho=0.0):
        super().__init__(rho=rho)
        self.e = e
        self.nu = nu
        self.alpha = alpha
        self._lamb = self.e * self.nu / (1 + self.nu) / (1 - 2 * (self.nu))
        self._mu = self.e / (2 * (1 + self.nu))

    def moduli_are_constant(self):
        return False

    def tangent_moduli(self, xyz=None, fn1=None, fn=None, dt=None):
        """
        
        :param xyz: 
        :param fn1: 
        :param fn: 
        :param dt: 
        :return: 
        """
        jac = numpy.linalg.det(fn1)
        d = (self._lamb / jac) * M1M1 \
            + 2. * (self._mu - self._lamb * math.log(jac)) / jac * MI
        return d

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
        d = self.tangent_moduli(xyz=xyz, kind=defaultkind)
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
        :return: quantityout: If out comes in as None, we will allocate it and return it.
        """
        if strain is not None:
            ev = numpy.reshape(strain, (6, 1))
        else:
            ev = numpy.zeros((6, 1))
            gradu = fn1 - ID3
            strain_3x3t_to_6v((gradu + gradu.T) / 2, ev)

        d = self.tangent_moduli()
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
