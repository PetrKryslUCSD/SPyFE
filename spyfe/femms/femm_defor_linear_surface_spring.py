import numpy
from numpy import dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm
from spyfe.meshing.transformation import skewmat
from spyfe.femms.femm_defor import FEMMDefor
from spyfe.csys import CSys


class FEMMDeforLinearSurfaceSpring(FEMMDefor):
    def __init__(self, material=None, fes=None, integration_rule=None,
                 material_csys=CSys(), assoc_geom=None,
                 surface_normal_spring_coefficient=1.0, surface_normal=None):
        """Constructor.

        :param material: Material object.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        super().__init__(fes=fes, integration_rule=integration_rule,
                         material_csys=material_csys, assoc_geom=assoc_geom)
        self.surface_normal_spring_coefficient = surface_normal_spring_coefficient
        self.surface_normal = surface_normal

    def evaluate_normal(self, location, jacmat):
        """Compute local normal. 
        
        :param location: spatial location
        :param jacmat: Jacobian matrix
        :return: 
        """
        if self.surface_normal is None:  # Produce a default normal
            if jacmat.shape[0] == 3 and jacmat.shape[1] == 2:  # surface in three dimensions
                n = numpy.dot(skewmat(jacmat[:, 0]), jacmat[:, 1])  # outer normal to the surface
            elif jacmat.shape[0] == 2 and jacmat.shape[1] == 1:  # curve in two dimensions
                n = numpy.array([jacmat[1, 0], -jacmat[0, 0]])  # outer normal to the surface
            else:
                raise Exception('No definition of normal vector available')
        else:
            n = self.surface_normal(location, jacmat)
        return n / numpy.linalg.norm(n)

    def stiffness_normal(self, geom, u):
        """Compute the stiffness matrix of surface normal spring.
        
        Rationale: consider continuously distributed springs between the surface of 
        the solid body and the 'ground', in the direction normal to the surface. 
        If the spring coefficient becomes large, we have an approximate
        method of enforcing the normal displacement to the surface.
        :param geom: 
        :param u: 
        :return: 
        """
        k = self.surface_normal_spring_coefficient
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        hedim = u.dim * fes.nfens
        jacmat = numpy.zeros((geom.dim, self.fes.dim))
        assm = SysmatAssemblerSparseFixedSymm(fes, u)
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                jac = fes.jac_surface(fes.conn[i, :], bfuns[j], jacmat, x)
                c = dot(bfuns[j].T, x)
                n = self.evaluate_normal(c, jacmat)
                n.shape = (1, -1)  # into a row vector
                nn = (bfuns[j] * n).reshape(hedim, 1)
                assm.elmtx[i, :, :] += nn * (nn.T * (k * jac * w[j]))
        return assm.make_matrix()
