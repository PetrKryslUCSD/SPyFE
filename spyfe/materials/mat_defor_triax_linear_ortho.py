import numpy
from spyfe.materials.mat_base import ErrorUnknownOutput
from spyfe.materials.mat_defor import ID3, OUTPUT_CAUCHY
from spyfe.materials.mat_defor import strain_6v_from_3x3t
from spyfe.materials.mat_defor_triax import MatDeforTriax

defaultkind = 0

class MatDeforTriaxLinearOrtho(MatDeforTriax):
    """
    Triaxial deformation material class for linear elastic orthotropic models.

     E1, E2, E3=Young's modulus
     nu12, nu13, nu23=Poisson ratio
     G12,  G13, G23=Shear modulus
    """
    def __init__(self, e1=0.0, e2=0.0, e3=0.0,
                 g12=0.0, g13=0.0, g23=0.0,
                 nu12=0.0, nu13=0.0, nu23=0.0,
                 alpha1=0.0, alpha2=0.0, alpha3=0.0, rho=0.0):
        super().__init__(rho=rho)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.g12 = g12
        self.g13 = g13
        self.g23 = g23
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        compliance = numpy.array([[1 / e1, -nu12 / e1, -nu13 / e1, 0., 0., 0.],
                                  [-nu12 / e1, 1 / e2, -nu23 / e2, 0., 0., 0.],
                                  [-nu13 / e1, -nu23 / e2, 1 / e3, 0., 0., 0.],
                                  [0., 0., 0., 1 / g12, 0., 0.],
                                  [0., 0., 0., 0., 1 / g13, 0.],
                                  [0., 0., 0., 0., 0., 1 / g23]])
        assert numpy.linalg.matrix_rank(compliance) == 6
        assert numpy.all(numpy.linalg.eigvals(compliance) >0)
        self._d = numpy.linalg.inv(compliance)

    def moduli_are_constant(self):
        return True

    def tangent_moduli(self, d_out, xyz=None, kind=defaultkind):
        """Compute the tangent moduli.

                :param d_out: The moduli matrix, 6 x 6.
                :param xyz: Location at which the moduli are to be computed.
                :param kind: Kind of the moduli (see above).
                :return: 
                """
        d_out[:, :] = self._d[:, :]

    def thermal_strain(self, dtemp=0.0):
        """Compute the thermal strain.

        :param dtemp: temperature increment from reference temperature, either a vector
        or None
        :return: vector of thermal strains if temperature increments applied as not None
        """
        return (-dtemp) * numpy.array([self.alpha1, self.alpha2, self.alpha3, 0.0, 0.0, 0.0]).reshape(6,1)

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
        :return: quantityout: If quantityout comes in as None, we will allocate 
        it and return it.
        """
        if strain is not None:
            ev = strain.reshape(6,1)
        else:
            ev = numpy.zeros((6,))
            gradu = fn1 - ID3
            self.strain_6v_from_3x3t(ev, (gradu+gradu.T)/2)

        d = numpy.zeros((6, 6))
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