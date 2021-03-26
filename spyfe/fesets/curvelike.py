import numpy
from math import pi
from numpy import linalg, dot
from spyfe.fesets.fe_set_base import FESet, ErrorWrongDimension, ErrorWrongNumberOfNodes, ErrorWrongJacobianMatrix
from spyfe.fesets.pointlike import FESetP1


class FESet1Manifold(FESet):
    """
    Constructor of curve-like manifold element set.
    :param nfens: Number of finite nodes connected by the elements.
    :param conn: Connectivity.
    :param axisymm: Is this an axially symmetric model?
    :param other_dimension: This can be surface, length. When we ask for
    (a) a volume Jacobian then the other dimension is length for axisymmetric, and
    area otherwise
    (b) a surface Jacobian then the other dimension is 1.0 for axisymmetric, and
    length otherwise
    (c) a curve Jacobian then the other dimension is 1.0.
    """
    # Manifold dimension
    dim = 1

    def __init__(self, nfens=0, conn=None, label=None, axisymm=False, other_dimension=lambda conn, N, x: 1.0):
        super().__init__(nfens=nfens, conn=conn, label=label)
        self.axisymm = axisymm
        self.other_dimension = other_dimension

    def jac_curve(self, conn, bfunval, jacmat, x):
        """
        Evaluate the curve Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        sdim, ntan = jacmat.shape
        if ntan == 1:
            return numpy.linalg.norm(jacmat)
        else:
            raise ErrorWrongJacobianMatrix

    def jac_surface(self, conn, bfunval, jacmat, x):
        """
        Evaluate the surface Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        jac_curve = self.jac_curve(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return jac_curve * 2 * pi * xyz[0] * other_dim
        else:
            return jac_curve * other_dim

    def jac_volume(self, conn, bfunval, jacmat, x):
        """
        Evaluate the volume Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        jac_curve = self.jac_curve(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return jac_curve * 2 * pi * xyz[0] * other_dim
        else:
            return jac_curve * other_dim

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
        elif m == 1:
            return self.jac_curve(conn, bfunval, jacmat, x)
        else:
            raise ErrorWrongDimension

    def jac(self, conn, bfunval, J, x):
        """
        Evaluate the intrinsic Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param J: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        return self.jac_curve(conn, bfunval, J, x)

    def gradbfun(self, gradbfunout, gradbfunpar, redjacmat):
        """Compute the gradient of the basis functions with the respect to
        the "reduced" spatial coordinates.

        :param gradbfunpars: matrix of gradients with respect to parametric coordinates, one per row
        :param redjacmat: reduced Jacobian matrix redJ=Rm'*J
        :param gradbfunout: the results will be placed in this matrix
        :return: Nothing
        """
        for r in range(gradbfunout.shape[0]):
            gradbfunout[r, 0] = gradbfunpar[r, 0] / redjacmat[0, 0]


class FESetL2(FESet1Manifold):
    """
    Finite element curve with two nodes.
    """

    # Number of nodes
    nfens = 2

    def __init__(self, conn, label=None):
        super().__init__(conn=conn, nfens=FESetL2.nfens, label=label)

    def bfun(self, param_coords):
        """
        Evaluate the basis function matrix for an 2-node line element (bar).
        """
        return numpy.array([(1. - param_coords[0]) / 2., (1. + param_coords[0]) / 2.]).reshape(2, 1)

    def gradbfunpar(self, param_coords):
        """
        Evaluate the parametric derivatives of the
        basis function matrix for an 2-node line element (bar).
        """
        return numpy.array([-1.0 / 2., +1.0 / 2.]).reshape(2, 1)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0,)], c[:, (1,)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetP1


class FESetL3(FESet1Manifold):
    """
    Finite element curve with three nodes.
    """

    # Number of nodes
    nfens = 3

    def __init__(self, conn, label=None):
        super().__init__(conn=conn, nfens=FESetL3.nfens, label=label)

    def bfun(self, param_coords):
        """
        Evaluate the basis function matrix for an 3-node line element (bar).
        """
        xi = param_coords[0]
        return numpy.array([[(xi * (xi - 1)) / 2],
                            [(xi * (xi + 1)) / 2],
                            [-(xi - 1) * (xi + 1)]
                            ]).reshape(3, 1)

    def gradbfunpar(self, param_coords):
        """
        Evaluate the parametric derivatives of the
        basis function matrix for an 3-node line element (bar).
        """
        xi = param_coords[0]
        return numpy.array([[xi - 1 / 2],
                            [xi + 1 / 2],
                            [-2 * xi]
                            ]).reshape(3, 1)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, 0], c[:, 1]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetP1
