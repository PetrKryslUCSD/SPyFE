import numpy
from numpy import dot
from spyfe.csys import CSys
from spyfe.femms.femm_defor_linear_ms import FEMMDeforLinearMS
from spyfe.integ_rules import GaussRule

class FEMMDeforLinearH8MSGSO(FEMMDeforLinearMS):
    """
    Class for small-strain linear deformation based on the mean-strain
     hexahedra and stabilization by Gaussian quadrature.

     Reference:
     Krysl, P.,  Mean-strain eight-node hexahedron with stabilization by
     energy sampling, INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN
     ENGINEERING Article first published online : 24 JUN 2014, DOI:
     10.1002/nme.4721

     Krysl, P.,  Optimized Energy-Sampling Stabilization of the
     Mean-strain 8-node Hexahedron, submitted to IJNME 23 August 2014.

     Krysl, P.,  Mean-strain 8-node Hexahedron with Optimized
     Energy-Sampling Stabilization for Large-strain Deformation, submitted
     to IJNME 18 September 2014.
    """

    def __init__(self, material=None, fes=None, integration_rule=None,
                 material_csys=CSys(), assoc_geom=None):
        """Constructor.

        :param material: Material object.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        integration_rule = GaussRule(dim=3, order=2) if integration_rule is None else integration_rule
        super().__init__(fes=fes, material=material, integration_rule=integration_rule,
                         material_csys=material_csys, assoc_geom=assoc_geom)

    def associate_geometry(self, geom):
        """Associate geometry.

        :param geom: Geometry field.
        :return: Nothing.  The object is modified.
        """
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        jacmat = numpy.zeros((geom.dim, self.fes.dim))
        gradbfun = numpy.zeros((self.fes.nfens, geom.dim))
        self._phis = numpy.zeros((self.fes.conn.shape[0],))
        for i in range(self.fes.conn.shape[0]):
            x = geom.values[self.fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                self.fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                h2 = numpy.diag(numpy.dot(jacmat.T, jacmat))
                cap_phi = (2 * (1 + self.nu) * (min(h2) / max(h2)))  # Plane stress
                phi = cap_phi / (1 + cap_phi)
                self._phis[i] = max(self._phis[i], phi)
