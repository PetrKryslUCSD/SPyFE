import numpy
from numpy import linalg, dot
from spyfe.fesets.fe_set_base import FESet, ErrorWrongDimension, ErrorWrongJacobianMatrix
from spyfe.fesets.surfacelike import FESetQ4, FESetQ8, FESetT3, FESetT6
#from spyfe.cyfuns import gradbfun3dcy


class FESet3Manifold(FESet):
    """
    Finite element 3-manifold set (volume).
    """

    # Manifold dimension
    dim = 3

    def __init__(self, nfens=0, conn=None, label=None):
        """
        Constructor of volume-like manifold element set.
        :param nfens: Number of finite nodes connected by the elements.
        :param conn: Connectivity.
        """
        super().__init__(nfens=nfens, conn=conn, label=label)

    def jac_volume(self, conn, bfunval, jacmat, x):
        """
        Evaluate the volume Jacobian.
        :param conn: input connectivity (one-based)
        :param bfunval: matrix of basis function values at the integration point
        :param jacmat: Jacobian matrix at the integration point
        :param x: array of node locations, one per row
        :return: Jacobian
        """
        sdim, ntan = jacmat.shape
        if sdim == ntan:
            # return linalg.det(J)
            # The unrolled version
            return (+ jacmat[0, 0] * (jacmat[1, 1] * jacmat[2, 2] - jacmat[2, 1] * jacmat[1, 2])
                    - jacmat[0, 1] * (jacmat[1, 0] * jacmat[2, 2] - jacmat[1, 2] * jacmat[2, 0])
                    + jacmat[0, 2] * (jacmat[1, 0] * jacmat[2, 1] - jacmat[1, 1] * jacmat[2, 0]))
        else:
            raise ErrorWrongJacobianMatrix

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
        return self.jac_volume(conn, bfunval, jacmat, x)

    def gradbfun(self, gradbfunout, gradbfunpar, redjacmat):
        """Compute the gradient of the basis functions with the respect to
        the "reduced" spatial coordinates.

        :param gradbfunout: the results will be placed in this matrix
        :param gradbfunpar: matrix of gradients with respect to parametric coordinates, one per row
        :param redjacmat: reduced Jacobian matrix redJ=Rm'*J
        :return: Nothing
        """
        #        gradbfun3dcy(gradbfunpar, redjacmat, gradbfunout)
        # This is the unrolled version that avoids allocation of a 3 x 3 matrix
        invdet = 1.0 / (+redjacmat[0, 0] * (redjacmat[1, 1] * redjacmat[2, 2] - redjacmat[2, 1] * redjacmat[1, 2])
            - redjacmat[0, 1] * (redjacmat[1, 0] * redjacmat[2, 2] - redjacmat[1, 2] * redjacmat[2, 0])
            + redjacmat[0, 2] * (redjacmat[1, 0] * redjacmat[2, 1] - redjacmat[1, 1] * redjacmat[2, 0]))
        invredJ11 = (redjacmat[1, 1] * redjacmat[2, 2] - redjacmat[2, 1] * redjacmat[1, 2]) * invdet
        invredJ12 = -(redjacmat[0, 1] * redjacmat[2, 2] - redjacmat[0, 2] * redjacmat[2, 1]) * invdet
        invredJ13 = (redjacmat[0, 1] * redjacmat[1, 2] - redjacmat[0, 2] * redjacmat[1, 1]) * invdet
        invredJ21 = -(redjacmat[1, 0] * redjacmat[2, 2] - redjacmat[1, 2] * redjacmat[2, 0]) * invdet
        invredJ22 = (redjacmat[0, 0] * redjacmat[2, 2] - redjacmat[0, 2] * redjacmat[2, 0]) * invdet
        invredJ23 = -(redjacmat[0, 0] * redjacmat[1, 2] - redjacmat[1, 0] * redjacmat[0, 2]) * invdet
        invredJ31 = (redjacmat[1, 0] * redjacmat[2, 1] - redjacmat[2, 0] * redjacmat[1, 1]) * invdet
        invredJ32 = -(redjacmat[0, 0] * redjacmat[2, 1] - redjacmat[2, 0] * redjacmat[0, 1]) * invdet
        invredJ33 = (redjacmat[0, 0] * redjacmat[1, 1] - redjacmat[1, 0] * redjacmat[0, 1]) * invdet
        
        for r in range(gradbfunout.shape[0]):
            gradbfunout[r, 0] = gradbfunpar[r, 0] * invredJ11 + gradbfunpar[r, 1] * invredJ21 + gradbfunpar[r, 2] * invredJ31
            gradbfunout[r, 1] = gradbfunpar[r, 0] * invredJ12 + gradbfunpar[r, 1] * invredJ22 + gradbfunpar[r, 2] * invredJ32
            gradbfunout[r, 2] = gradbfunpar[r, 0] * invredJ13 + gradbfunpar[r, 1] * invredJ23 + gradbfunpar[r, 2] * invredJ33

    def bmatdata(self, nfn):
        """Return function to construct the strain-displacement matrix and 
        an array buffer for it.
        
        The function to compute the strain-displacement matrix needs to accept
        displacements and the global Cartesian coordinate systems and output
        strains in the same global Cartesian coordinate system.
        
        :param nfn: Number of basis functions per element, ==gradbfun.shape[0]
        :return: the B matrix function and an appropriately-sized buffer 
        (array) for it.
        """
        r0 = numpy.arange(0, 3 * nfn, 3)
        r1 = numpy.arange(1, 3 * nfn, 3)
        r2 = numpy.arange(2, 3 * nfn, 3)

        def blmat3(bout, bfunval, gradbfun, c):
            """Compute the strain-displacement matrix for a three-manifold element.

            :param bfunval: matrix of basis function values
            :param gradbfun: matrix of basis function gradients with respect to the
                   Cartesian coordinates in the directions of the material orientation
            :param c: array of spatial coordinates of the evaluation point
                 in the global Cartesian coordinates
            :return: nothing
            """
            bout.fill(0.0)
            bout[0, r0] = gradbfun[:, 0]
            bout[1, r1] = gradbfun[:, 1]
            bout[2, r2] = gradbfun[:, 2]
            bout[3, r0] = gradbfun[:, 1]
            bout[3, r1] = gradbfun[:, 0]
            bout[4, r0] = gradbfun[:, 2]
            bout[4, r2] = gradbfun[:, 0]
            bout[5, r1] = gradbfun[:, 2]
            bout[5, r2] = gradbfun[:, 1]
            # if csmtx is None:  # there is no global-to-local transformation
            #     bout[0, r0] = gradbfun[:, 0]
            #     bout[1, r1] = gradbfun[:, 1]
            #     bout[2, r2] = gradbfun[:, 2]
            #     bout[3, r0] = gradbfun[:, 1]
            #     bout[3, r1] = gradbfun[:, 0]
            #     bout[4, r0] = gradbfun[:, 2]
            #     bout[4, r2] = gradbfun[:, 0]
            #     bout[5, r1] = gradbfun[:, 2]
            #     bout[5, r2] = gradbfun[:, 1]
            #     # for i in numpy.arange(nfn):
            #     #     k = 3 * i
            #     #     bout[0, k + 0] = gradbfun[i, 0]
            #     #     bout[1, k + 1] = gradbfun[i, 1]
            #     #     bout[2, k + 2] = gradbfun[i, 2]
            #     #     bout[3, k + 0] = gradbfun[i, 1]
            #     #     bout[3, k + 1] = gradbfun[i, 0]
            #     #     bout[4, k + 0] = gradbfun[i, 2]
            #     #     bout[4, k + 2] = gradbfun[i, 0]
            #     #     bout[5, k + 1] = gradbfun[i, 2]
            #     #     bout[5, k + 2] = gradbfun[i, 1]
            # else:  # execute global-to-local transformation
            #     for i in numpy.arange(nfn):
            #         k = 3 * i
            #         bout[0, k + 0] = gradbfun[i, 0]
            #         bout[1, k + 1] = gradbfun[i, 1]
            #         bout[2, k + 2] = gradbfun[i, 2]
            #         bout[3, k + 0] = gradbfun[i, 1]
            #         bout[3, k + 1] = gradbfun[i, 0]
            #         bout[4, k + 0] = gradbfun[i, 2]
            #         bout[4, k + 2] = gradbfun[i, 0]
            #         bout[5, k + 1] = gradbfun[i, 2]
            #         bout[5, k + 2] = gradbfun[i, 1]
            #         bout[:, k:k + 3] = dot(bout[:, k:k + 3], csmtx.T)
            return

        b = numpy.zeros((6, self.nfens * 3))
        return blmat3, b


class FESetH8(FESet3Manifold):
    """
    Finite element hexahedral volume with 8 nodes.
    """

    # Number of nodes
    nfens = 8

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetH8.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        one_minus_xi = (1 - param_coords[0])
        one_minus_eta = (1 - param_coords[1])
        one_minus_theta = (1 - param_coords[2])
        one_plus_xi = (1 + param_coords[0])
        one_plus_eta = (1 + param_coords[1])
        one_plus_theta = (1 + param_coords[2])
        val = numpy.array([[one_minus_xi * one_minus_eta * one_minus_theta],
                           [one_plus_xi * one_minus_eta * one_minus_theta],
                           [one_plus_xi * one_plus_eta * one_minus_theta],
                           [one_minus_xi * one_plus_eta * one_minus_theta],
                           [one_minus_xi * one_minus_eta * one_plus_theta],
                           [one_plus_xi * one_minus_eta * one_plus_theta],
                           [one_plus_xi * one_plus_eta * one_plus_theta],
                           [one_minus_xi * one_plus_eta * one_plus_theta]]) / 8.
        return val.reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        one_minus_xi = (1 - param_coords[0])
        one_minus_eta = (1 - param_coords[1])
        one_minus_theta = (1 - param_coords[2])
        one_plus_xi = (1 + param_coords[0])
        one_plus_eta = (1 + param_coords[1])
        one_plus_theta = (1 + param_coords[2])
        val = numpy.array(
            [[-one_minus_eta * one_minus_theta, -one_minus_xi * one_minus_theta, -one_minus_xi * one_minus_eta],
             [one_minus_eta * one_minus_theta, -one_plus_xi * one_minus_theta, -one_plus_xi * one_minus_eta],
             [one_plus_eta * one_minus_theta, one_plus_xi * one_minus_theta, -one_plus_xi * one_plus_eta],
             [-one_plus_eta * one_minus_theta, one_minus_xi * one_minus_theta, -one_plus_eta * one_minus_xi],
             [-one_minus_eta * one_plus_theta, -one_minus_xi * one_plus_theta, one_minus_xi * one_minus_eta],
             [one_minus_eta * one_plus_theta, -one_plus_xi * one_plus_theta, one_plus_xi * one_minus_eta],
             [one_plus_eta * one_plus_theta, one_plus_xi * one_plus_theta, one_plus_xi * one_plus_eta],
             [-one_plus_eta * one_plus_theta, one_minus_xi * one_plus_theta, one_plus_eta * one_minus_xi]]) / 8.0
        return val.reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 3, 2, 1)],
                             c[:, (0, 1, 5, 4)],
                             c[:, (1, 2, 6, 5)],
                             c[:, (2, 3, 7, 6)],
                             c[:, (3, 0, 4, 7)],
                             c[:, (5, 6, 7, 4)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetQ4


class FESetH20(FESet3Manifold):
    """
    Finite element hexahedral volume with 20 nodes.
    """

    # Number of nodes
    nfens = 20

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetH20.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        zeta = param_coords[2]
        xim = (-1. + xi)
        etam = (-1. + eta)
        zetam = (-1. + zeta)
        xip = (1. + xi)
        etap = (1. + eta)
        zetap = (1. + zeta)
        val = numpy.array([[(etam * xim * zetam * (eta + xi + zeta + 2)) / 8],
                           [-(etam * xip * zetam * (eta - xi + zeta + 2)) / 8],
                           [-(etap * xip * zetam * (eta + xi - zeta - 2)) / 8],
                           [-(etap * xim * zetam * (xi - eta + zeta + 2)) / 8],
                           [-(etam * xim * zetap * (eta + xi - zeta + 2)) / 8],
                           [(etam * xip * zetap * (eta - xi - zeta + 2)) / 8],
                           [(etap * xip * zetap * (eta + xi + zeta - 2)) / 8],
                           [-(etap * xim * zetap * (eta - xi + zeta - 2)) / 8],
                           [-(etam * xim * xip * zetam) / 4],
                           [(etam * etap * xip * zetam) / 4],
                           [(etap * xim * xip * zetam) / 4],
                           [-(etam * etap * xim * zetam) / 4],
                           [(etam * xim * xip * zetap) / 4],
                           [-(etam * etap * xip * zetap) / 4],
                           [-(etap * xim * xip * zetap) / 4],
                           [(etam * etap * xim * zetap) / 4],
                           [-(etam * xim * zetam * zetap) / 4],
                           [(etam * xip * zetam * zetap) / 4],
                           [-(etap * xip * zetam * zetap) / 4],
                           [(etap * xim * zetam * zetap) / 4]
                           ])
        return val.reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        zeta = param_coords[2]
        xim = -(-1. + xi)
        etam = -(-1. + eta)
        zetam = -(-1. + zeta)
        xip = (1. + xi)
        etap = (1. + eta)
        zetap = (1. + zeta)
        twoppp = (2.0 + xi + eta + zeta)
        twompp = (2.0 - xi + eta + zeta)
        twopmp = (2.0 + xi - eta + zeta)
        twoppm = (2.0 + xi + eta - zeta)
        twommp = (2.0 - xi - eta + zeta)
        twopmm = (2.0 + xi - eta - zeta)
        twompm = (2.0 - xi + eta - zeta)
        twommm = (2.0 - xi - eta - zeta)
        val = numpy.array([[(etam * twoppp * zetam) / 8 - (etam * xim * zetam) / 8,
                            (twoppp * xim * zetam) / 8 - (etam * xim * zetam) / 8,
                            (etam * twoppp * xim) / 8 - (etam * xim * zetam) / 8],
                           [(etam * xip * zetam) / 8 - (etam * twompp * zetam) / 8,
                            (twompp * xip * zetam) / 8 - (etam * xip * zetam) / 8,
                            (etam * twompp * xip) / 8 - (etam * xip * zetam) / 8],
                           [(etap * xip * zetam) / 8 - (etap * twommp * zetam) / 8,
                            (etap * xip * zetam) / 8 - (twommp * xip * zetam) / 8,
                            (etap * twommp * xip) / 8 - (etap * xip * zetam) / 8],
                           [(etap * twopmp * zetam) / 8 - (etap * xim * zetam) / 8,
                            (etap * xim * zetam) / 8 - (twopmp * xim * zetam) / 8,
                            (etap * twopmp * xim) / 8 - (etap * xim * zetam) / 8],
                           [(etam * twoppm * zetap) / 8 - (etam * xim * zetap) / 8,
                            (twoppm * xim * zetap) / 8 - (etam * xim * zetap) / 8,
                            (etam * xim * zetap) / 8 - (etam * twoppm * xim) / 8],
                           [(etam * xip * zetap) / 8 - (etam * twompm * zetap) / 8,
                            (twompm * xip * zetap) / 8 - (etam * xip * zetap) / 8,
                            (etam * xip * zetap) / 8 - (etam * twompm * xip) / 8],
                           [(etap * xip * zetap) / 8 - (etap * twommm * zetap) / 8,
                            (etap * xip * zetap) / 8 - (twommm * xip * zetap) / 8,
                            (etap * xip * zetap) / 8 - (etap * twommm * xip) / 8],
                           [(etap * twopmm * zetap) / 8 - (etap * xim * zetap) / 8,
                            (etap * xim * zetap) / 8 - (twopmm * xim * zetap) / 8,
                            (etap * xim * zetap) / 8 - (etap * twopmm * xim) / 8],
                           [(etam * xim * zetam) / 4 - (etam * xip * zetam) / 4, -(xim * xip * zetam) / 4,
                            -(etam * xim * xip) / 4],
                           [(etam * etap * zetam) / 4, (etam * xip * zetam) / 4 - (etap * xip * zetam) / 4,
                            -(etam * etap * xip) / 4],
                           [(etap * xim * zetam) / 4 - (etap * xip * zetam) / 4, (xim * xip * zetam) / 4,
                            -(etap * xim * xip) / 4],
                           [-(etam * etap * zetam) / 4, (etam * xim * zetam) / 4 - (etap * xim * zetam) / 4,
                            -(etam * etap * xim) / 4],
                           [(etam * xim * zetap) / 4 - (etam * xip * zetap) / 4, -(xim * xip * zetap) / 4,
                            (etam * xim * xip) / 4],
                           [(etam * etap * zetap) / 4, (etam * xip * zetap) / 4 - (etap * xip * zetap) / 4,
                            (etam * etap * xip) / 4],
                           [(etap * xim * zetap) / 4 - (etap * xip * zetap) / 4, (xim * xip * zetap) / 4,
                            (etap * xim * xip) / 4],
                           [-(etam * etap * zetap) / 4, (etam * xim * zetap) / 4 - (etap * xim * zetap) / 4,
                            (etam * etap * xim) / 4],
                           [-(etam * zetam * zetap) / 4, -(xim * zetam * zetap) / 4,
                            (etam * xim * zetam) / 4 - (etam * xim * zetap) / 4],
                           [(etam * zetam * zetap) / 4, -(xip * zetam * zetap) / 4,
                            (etam * xip * zetam) / 4 - (etam * xip * zetap) / 4],
                           [(etap * zetam * zetap) / 4, (xip * zetam * zetap) / 4,
                            (etap * xip * zetam) / 4 - (etap * xip * zetap) / 4],
                           [-(etap * zetam * zetap) / 4, (xim * zetam * zetap) / 4,
                            (etap * xim * zetam) / 4 - (etap * xim * zetap) / 4]
                           ])
        return val.reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 3, 2, 1, 11, 10, 9, 8)],
                             c[:, (0, 1, 5, 4, 8, 17, 12, 16)],
                             c[:, (1, 2, 6, 5, 9, 18, 13, 17)],
                             c[:, (2, 3, 7, 6, 10, 19, 14, 18)],
                             c[:, (3, 0, 4, 7, 11, 16, 15, 19)],
                             c[:, (5, 6, 7, 4, 13, 14, 15, 12)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetQ8


class FESetT4(FESet3Manifold):
    """
    Finite element tetrahedral volume with 4 nodes.
    """

    # Number of nodes
    nfens = 4

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetT4.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        zeta = param_coords[2]
        val = numpy.array([1 - xi - eta - zeta, xi, eta, zeta])
        return val.reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        xi = param_coords[0]
        eta = param_coords[1]
        zeta = param_coords[2]
        val = numpy.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return val.reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 2, 1)],
                             c[:, (0, 1, 3)],
                             c[:, (1, 2, 3)],
                             c[:, (0, 3, 2)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetT3


class FESetT10(FESet3Manifold):
    """
    Finite element tetrahedral volume with 10 nodes.
    """

    # Number of nodes
    nfens = 10

    def __init__(self, conn, label=None):
        """Constructor.

        :param conn: Connectivity of the finite element set.
        """
        super().__init__(conn=conn, nfens=FESetT10.nfens, label=label)

    def bfun(self, param_coords):
        """Evaluate the basis function matrix.

        :param param_coords: Parametric coordinates within the element where the function is evaluated.
        :return: array of basis function values.
        """
        r = param_coords[0]
        s = param_coords[1]
        t = param_coords[2]
        val = numpy.array([[(r + s + t - 1) * (2 * r + 2 * s + 2 * t - 1)],
                           [r * (2 * r - 1)],
                           [s * (2 * s - 1)],
                           [t * (2 * t - 1)],
                           [-r * (4 * r + 4 * s + 4 * t - 4)],
                           [4 * r * s],
                           [-4 * s * (r + s + t - 1)],
                           [-t * (4 * r + 4 * s + 4 * t - 4)],
                           [4 * r * t],
                           [4 * s * t]
                           ])
        return val.reshape(self.nfens, 1)

    def gradbfunpar(self, param_coords):
        """Evaluate the parametric derivatives of the basis functions.

        :param param_coords: Parametric coordinates within the element where the gradient is evaluated.
        :return: array of basis function gradient values, one basis function gradients per row.
        """
        r = param_coords[0]
        s = param_coords[1]
        t = param_coords[2]
        val = numpy.array([[4 * r + 4 * s + 4 * t - 3, 4 * r + 4 * s + 4 * t - 3, 4 * r + 4 * s + 4 * t - 3],
                           [4 * r - 1, 0, 0],
                           [0, 4 * s - 1, 0],
                           [0, 0, 4 * t - 1],
                           [4 - 4 * s - 4 * t - 8 * r, -4 * r, -4 * r],
                           [4 * s, 4 * r, 0],
                           [-4 * s, 4 - 8 * s - 4 * t - 4 * r, -4 * s],
                           [-4 * t, -4 * t, 4 - 4 * s - 8 * t - 4 * r],
                           [4 * t, 0, 4 * r],
                           [0, 4 * t, 4 * s]
                           ])
        return val.reshape(self.nfens, self.dim)

    def boundary_conn(self):
        """Return the boundary connectivity.

        :return: Array of zero-based connectivities.
        """
        c = self._conn
        return numpy.vstack((c[:, (0, 2, 1, 6, 5, 4)],
                             c[:, (0, 1, 3, 4, 8, 7)],
                             c[:, (1, 2, 3, 5, 9, 8)],
                             c[:, (2, 0, 3, 6, 7, 9)]))

    def boundary_fe_type(self):
        """ Get the type of the  boundary finite element.
        :return: Type of the boundary finite element.
        """
        return FESetT6
