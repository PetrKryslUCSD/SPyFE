from spyfe.csys import CSys
import scipy.sparse
import numpy

class FEMMBase:
    """
    The class is for assembling of a dense global matrix from
    elementwise matrices.

    """

    def __init__(self, fes=None, integration_rule=None, material_csys=CSys(), assoc_geom=None):
        self.fes = None
        self.fes = fes
        self.integration_rule = integration_rule
        self.material_csys = material_csys
        self.assoc_geom = assoc_geom

    def integration_data(self):
        """Calculate data for numerical integrations.
        """
        pc = self.integration_rule.param_coords
        w = self.integration_rule.weights
        npts = self.integration_rule.npts
        Ns = list()
        gradNpars = list()
        for j in range(self.integration_rule.npts):
            Ns.append(self.fes.bfun(pc[j, :]))
            gradN = self.fes.gradbfunpar(pc[j, :])
            gradNpars.append(gradN)
        return Ns, gradNpars, npts, pc, w

    def connection_matrix(self, u):
        """Compute the connection matrix.

        Compute the connection matrix by computing and assembling the
        matrices of the individual FEs.

        Return a sparse matrix representing the connections between the finite 
        element nodes expressed by the  finite elements. The matrix holds 
        a one (1.0) for nodes that are connected by at least one finite
        element, and zero (0.0) otherwise.
        """

        # Much, much faster implementation
        c = self.fes.conn.copy()
        c.shape = (c.shape[0], c.shape[1], 1)
        B = numpy.tile(c, (1, 1, c.shape[1]))
        cb = B.transpose((0, 1, 2))
        rb = B.transpose((0, 2, 1))
        rb = rb.ravel()
        cb = cb.ravel()
        v = numpy.ones((c.shape[0]*c.shape[1]**2,))
        m = scipy.sparse.coo_matrix((v.ravel(), (rb.ravel(), cb.ravel())),
                                    shape=(u.nfens, u.nfens))
        return m.tocsr()
