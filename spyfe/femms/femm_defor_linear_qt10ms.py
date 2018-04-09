import numpy
from numpy import dot
from spyfe.femms.femm_defor_linear_ms import FEMMDeforLinearMS
from spyfe.csys import CSys
from spyfe.integ_rules import TetRule


class FEMMDeforLinearQT10MS(FEMMDeforLinearMS):
    """
    Class for small-strain linear deformation based on the mean-strain
     quadratic tetrahedron and stabilization by full four-point quadrature.

    """

    def __init__(self, material=None, fes=None, integration_rule=None,
                 material_csys=CSys(), assoc_geom=None):
        """Constructor.

        :param material: Material object.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        integration_rule = TetRule(npts=4) if integration_rule is None else integration_rule
        super().__init__(fes=fes, material=material, integration_rule=integration_rule,
                         material_csys=material_csys, assoc_geom=assoc_geom)
        self._gamma = 2.6 #one of the stabilization parameters
        self._C = 1.e4 #the other stabilization parameter


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
                condjacmat = numpy.linalg.cond(jacmat)
                cap_phi = self._C * (1. / condjacmat)** (self._gamma)
                phi = cap_phi / (1. + cap_phi)
                self._phis[i] = max(self._phis[i], phi)
