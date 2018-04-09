import numpy
from math import pi
from numpy import linalg, dot
from spyfe.fesets.fe_set_base import FESet, ErrorWrongDimension, ErrorWrongNumberOfNodes, ErrorWrongJacobianMatrix


class FESet0Manifold(FESet):
    """
    Finite element 0-manifold set (point).
    """
    
    # Manifold dimension
    dim = 0

    def __init__(self, nfens=0, conn=None, label=None, axisymm=False, other_dimension=lambda conn, N, x: 1.0):
        """
        Constructor.
        :param nfens: Number of finite nodes connected by the elements.
        :param conn: Connectivity.
        :param axisymm: Is this an axially symmetric model?
        :param other_dimension: This can be volume, surface, length. When we ask for
        (a) a volume Jacobian then the other dimension is area for axisymmetric, and
        volume otherwise;
        (b) a surface Jacobian then the other dimension is length for axisymmetric, and
        area otherwise;
        (c) a curve Jacobian then the other dimension is 1.0 for axisymmetric, and
        length otherwise;
        (d) a point Jacobian then the other dimension is 1.0.
        """
        super().__init__(nfens=nfens, conn=conn, label=label)
        self.axisymm = axisymm
        self.other_dimension = other_dimension

    def jac_point(self, conn, bfunval, jacmat, x):
        """
        Evaluate the point Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        return 1.0

    def jac_curve(self, conn, bfunval, jacmat, x):
        """
        Evaluate the curve Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        Jac_point = self.jac_point(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return Jac_point * 2 * pi * xyz[0] * other_dim
        else:
            return Jac_point * other_dim

    def jac_surface(self, conn, bfunval, jacmat, x):
        """
        Evaluate the surface Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        Jac_point = self.jac_point(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return Jac_point * 2 * pi * xyz[0] * other_dim
        else:
            return Jac_point * other_dim

    def jac_volume(self, conn, bfunval, jacmat, x):
        """
        Evaluate the volume Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        Jac_point = self.jac_point(conn, bfunval, jacmat, x)
        other_dim = self.other_dimension(conn, bfunval, x)
        if self.axisymm:
            xyz = dot(bfunval.T, x)
            return Jac_point * 2 * pi * xyz[0] * other_dim
        else:
            return Jac_point * other_dim

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
            return self.jac_point(conn, bfunval, jacmat, x)

    def jac(self, conn, bfunval, jacmat, x):
        """
        Evaluate the intrinsic Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        return self.jac_point(conn, bfunval, jacmat, x)

    def gradbfun(self, gradbfunpars, redjacmat, gradbfunout):
        """Compute the gradient of the basis functions with the respect to
        the "reduced" spatial coordinates.

        :param gradNparams: matrix of gradients with respect to parametric coordinates, one per row
        :param redJ: reduced Jacobian matrix redJ=Rm'*J
        :param gradNout: the results will be placed in this matrix
        :return: Nothing
        """
        for r in range(gradbfunout.shape[0]):
            gradbfunout[r, 0] = 0.0

class FESetP1(FESet0Manifold):
    """
    Finite element point element with one node.
    """

    # Number of nodes
    nfens = 1

    def __init__(self, conn, label=None):
        super().__init__(conn=conn, nfens=FESetP1.nfens, label=label)

    def bfun(self, param_coords):
        """
        Evaluate the basis function matrix for an 2-node line element (bar).
        """
        return numpy.array([1.0])

    def gradbfunpar(self, param_coords):
        """
        Evaluate the parametric derivatives of the
        basis function matrix for an 2-node line element (bar).
        """
        return numpy.array([0.0])



