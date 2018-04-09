import numpy
from math import pi
from numpy import linalg, dot
from spyfe.fesets.fe_set_base import FESet, ErrorWrongDimension, ErrorWrongNumberOfNodes
from spyfe.fesets.curvelike import FESetL2, FESetL3
#from spyfe.cyfuns import gradbfun2dcy, matrix_2_x_2_det


class FESet2Manifold(FESet):
    """
    Finite element 2-manifold set (surface).
    """

    # Manifold dimension
    dim = 2

    def __init__(self, nfens=0, conn=None, label=None, axisymm=False, other_dimension=lambda conn, N, x: 1.0):
        """
        Constructor of surface-like manifold element set.
        :param nfens: Number of finite nodes connected by the elements.
        :param conn: Connectivity.
        :param axisymm: Is this an axially symmetric model?
        :param other_dimension: This can be length. When we ask for
        (a) a volume Jacobian then the other dimension is 1.0 for axisymmetric, and
        length otherwise;
        (b) a surface Jacobian then the other dimension is 1.0.
        """
        super().__init__(nfens=nfens, conn=conn, label=label)
        self.axisymm = axisymm
        self.other_dimension = other_dimension

    def jac_surface(self, conn, bfunval, jacmat, x):
        """
        Evaluate the surface Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        sdim, ntan = jacmat.shape
        if sdim == ntan:
            return jacmat[0, 0] * jacmat[1, 1] - jacmat[0, 1] * jacmat[1, 0] # matrix_2_x_2_det(jacmat) # 
        else:
            return linalg.norm(numpy.cross(jacmat[:, 0], jacmat[:, 1]))

    def jac_volume(self, conn, bfunval, jacmat, x):
        """
        Evaluate the volume Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        Jac_surface = self.jac_surface(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return Jac_surface * 2 * pi * xyz[0] * other_dim
        else:
            return Jac_surface * other_dim

    def jac_mdim(self, conn, bfunval, jacmat, x, m):
        """
        Evaluate the m-manifold Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :param m: manifold dimension
        :return: Jacobian
        """
        if m == 3:
            return self.jac_volume(conn, bfunval, jacmat, x)
        elif m == 2:
            return self.jac_surface(conn, bfunval, jacmat, x)
        else:
            raise ErrorWrongDimension

    def jac(self, conn, bfunval, jacmat, x):
        """
        Evaluate the intrinsic Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        return self.jac_surface(conn, bfunval, jacmat, x)

    def gradbfun(self, gradbfunout, gradbfunpar, redjacmat):
        """Compute the gradient of the basis functions with the respect to
        the "reduced" spatial coordinates.

        :param gradbfunpar: matrix of gradients with respect to parametric coordinates, one per row
        :param redjacmat: reduced Jacobian matrix redjacmat=Rm'*J
        :param gradbfunout: the results will be placed in this matrix
        :return: Nothing
        """
        # gradbfun2dcy(gradbfunpar, redjacmat, gradbfunout)  # Cython implementation
        # This is the unrolled version that avoids allocation of a 2 x 2 matrix
        invdet = 1.0 / (redjacmat[0, 0] * redjacmat[1, 1] - redjacmat[0, 1] * redjacmat[1, 0])
        invredJ11 =  (redjacmat[1, 1]) * invdet
        invredJ12 = -(redjacmat[0, 1]) * invdet
        invredJ21 = -(redjacmat[1, 0]) * invdet
        invredJ22 =  (redjacmat[0, 0]) * invdet
        for r in range(gradbfunout.shape[0]):
            gradbfunout[r, 0] = gradbfunpar[r, 0] * invredJ11 + gradbfunpar[r, 1] * invredJ21
            gradbfunout[r, 1] = gradbfunpar[r, 0] * invredJ12 + gradbfunpar[r, 1] * invredJ22

#    def gradN(self, gradNparams, redJ, gradNout):
#        """Compute the gradient of the basis functions with the respect to
#        the "reduced" spatial coordinates.
#
#        :param gradNparams: matrix of gradients with respect to parametric coordinates, one per row
#        :param redJ: reduced Jacobian matrix redJ=Rm'*J
#        :param gradNout: the results will be placed in this matrix
#        :return: Nothing
#        """
#        def fun(gradNparams, nr, redJ, gradNout):
#            # This is the unrolled version that avoids allocation of a 2 x 2 matrix
#            invdet = 1.0 / (redJ[0, 0] * redJ[1, 1] - redJ[0, 1] * redJ[1, 0])
#            invredJ11 = (redJ[1, 1]) * invdet
#            invredJ12 = -(redJ[0, 1]) * invdet
#            invredJ21 = -(redJ[1, 0]) * invdet
#            invredJ22 = (redJ[0, 0]) * invdet
#            for r in range(nr):
#                gradNout[r, 0] = gradNparams[r, 0] * invredJ11 + gradNparams[r, 1] * invredJ21
#                gradNout[r, 1] = gradNparams[r, 0] * invredJ12 + gradNparams[r, 1] * invredJ22
#        #fun_jit = numba.jit("void(f8[:, :], i4, f8[:, :], f8[:, :])")(fun)
#        fun(gradNparams, gradNparams.shape[0], redJ, gradNout) 

class FESetT3(FESet2Manifold):
    """
    Finite element triangular surface piece with three nodes.
    """

    # Number of nodes
    nfens = 3

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetT3.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        return numpy.array([1. - param_coords[0] - param_coords[1],
                            param_coords[0],
                            param_coords[1]]).reshape(3, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        return numpy.array([[-1., -1.],
                            [1., 0.],
                            [0., 1.]
                            ]).reshape(3, 2)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 1)], c[:, (1, 2)], c[:, (2, 0)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetL2


class FESetQ4(FESet2Manifold):
    """
    Finite element quadrilateral surface piece with 4 nodes.
    """

    # Number of nodes
    nfens = 4

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetQ4.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        one_minus_xi = (1 - param_coords[0])
        one_plus_xi = (1 + param_coords[0])
        one_minus_eta = (1 - param_coords[1])
        one_plus_eta = (1 + param_coords[1])
        return numpy.array([0.25 * one_minus_xi * one_minus_eta,
                            0.25 * one_plus_xi * one_minus_eta,
                            0.25 * one_plus_xi * one_plus_eta,
                            0.25 * one_minus_xi * one_plus_eta]).reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        return numpy.array([[-(1. - eta) * 0.25, -(1. - xi) * 0.25],
                            [(1. - eta) * 0.25, -(1. + xi) * 0.25],
                            [(1. + eta) * 0.25, (1. + xi) * 0.25],
                            [-(1. + eta) * 0.25, (1. - xi) * 0.25]]).reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 1)], c[:, (1, 2)], c[:, (2, 3)], c[:, (3, 0)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetL2


class FESetQ8(FESet2Manifold):
    """
    Finite element quadrilateral surface piece with 4 nodes.
    """

    # Number of nodes
    nfens = 8

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetQ8.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        return numpy.array([[-(xi / 4 - 1 / 4) * (eta - 1) * (eta + xi + 1)],
                            [(xi / 4 + 1 / 4) * (eta - 1) * (eta - xi + 1)],
                            [(xi / 4 + 1 / 4) * (eta + 1) * (eta + xi - 1)],
                            [(xi / 4 - 1 / 4) * (eta + 1) * (xi - eta + 1)],
                            [(xi / 2 - 1 / 2) * (eta - 1) * (xi + 1)],
                            [-(eta / 2 - 1 / 2) * (eta + 1) * (xi + 1)],
                            [-(xi / 2 - 1 / 2) * (eta + 1) * (xi + 1)],
                            [(eta / 2 - 1 / 2) * (eta + 1) * (xi - 1)]
                            ]).reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        return numpy.array([[- (xi / 4 - 1 / 4) * (eta - 1) - (eta / 4 - 1 / 4) * (eta + xi + 1),
                             - (xi / 4 - 1 / 4) * (eta - 1) - (xi / 4 - 1 / 4) * (eta + xi + 1)],
                            [(eta / 4 - 1 / 4) * (eta - xi + 1) - (xi / 4 + 1 / 4) * (eta - 1),
                             (xi / 4 + 1 / 4) * (eta - xi + 1) + (xi / 4 + 1 / 4) * (eta - 1)],
                            [(xi / 4 + 1 / 4) * (eta + 1) + (eta / 4 + 1 / 4) * (eta + xi - 1),
                             (xi / 4 + 1 / 4) * (eta + 1) + (xi / 4 + 1 / 4) * (eta + xi - 1)],
                            [(eta / 4 + 1 / 4) * (xi - eta + 1) + (xi / 4 - 1 / 4) * (eta + 1),
                             (xi / 4 - 1 / 4) * (xi - eta + 1) - (xi / 4 - 1 / 4) * (eta + 1)],
                            [(xi / 2 - 1 / 2) * (eta - 1) + (xi / 2 + 1 / 2) * (eta - 1), (xi / 2 - 1 / 2) * (xi + 1)],
                            [-(eta / 2 - 1 / 2) * (eta + 1),
                             - (xi / 2 + 1 / 2) * (eta - 1) - (xi / 2 + 1 / 2) * (eta + 1)],
                            [- (xi / 2 - 1 / 2) * (eta + 1) - (xi / 2 + 1 / 2) * (eta + 1),
                             -(xi / 2 - 1 / 2) * (xi + 1)],
                            [(eta / 2 - 1 / 2) * (eta + 1), (xi / 2 - 1 / 2) * (eta - 1) + (xi / 2 - 1 / 2) * (eta + 1)]
                            ]).reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 1, 4)], c[:, (1, 2, 5)], c[:, (2, 3, 6)], c[:, (3, 0, 7)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetL3


class FESetT6(FESet2Manifold):
    """
    Finite element triangular surface piece with 6 nodes.
    """

    # Number of nodes
    nfens = 6

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetQ8.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        r = param_coords[0]
        s = param_coords[1]
        t = 1 - r - s
        return numpy.array([[t * (2 * t - 1)],
                            [r * (2 * r - 1)],
                            [s * (2 * s - 1)],
                            [4 * r * t],
                            [4 * r * s],
                            [4 * s * t]
                            ]).reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        r = param_coords[0]
        s = param_coords[1]
        return numpy.array([[4 * r + 4 * s - 3., 4 * r + 4 * s - 3.],
                            [4 * r - 1., 0.],
                            [0., 4 * s - 1],
                            [4 - 4 * s - 8 * r, -4 * r],
                            [4 * s, 4 * r],
                            [-4 * s, 4 - 8 * s - 4 * r]
                            ]).reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 1, 3)], c[:, (1, 2, 4)], c[:, (2, 0, 5)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetL3
