import numpy
import numba

NoChange = Exception('Dimension and number of entities cannot be changed')


class GenField:
    """
    Class to represent general fields.

    Class that represents general fields: geometry, displacement, 
    incremental rotation,  temperature,... defined by values associated
    with nodes or elements.

    """

    def __init__(self, nents=0, dim=0, data=None):
        if nents != 0 and data is None:
            assert dim != 0
            self.values = numpy.zeros((nents, dim), dtype=numpy.float64)
            self.dofnums = numpy.zeros((nents, dim), dtype=int)
            self.fixed_values = numpy.zeros((nents, dim), dtype=numpy.float64)
            self.is_fixed = numpy.zeros((nents, dim), dtype=bool)
        else:
            assert data is not None
            nents, dim = data.shape
            self.values = data.copy()
            self.dofnums = numpy.zeros((nents, dim), dtype=int)
            self.fixed_values = numpy.zeros((nents, dim), dtype=numpy.float64)
            self.is_fixed = numpy.zeros((nents, dim), dtype=bool)
        self.nfreedofs = 0
        self._nents, self.dim = self.values.shape

    def get_nents(self):
        return self._nents

    def set_nents(self, value):
        raise NoChange

    nents = property(get_nents, set_nents)

    def numberdofs(self, node_perm=None):
        node_order = numpy.arange(self.nents)
        if node_perm is not None:
            assert len(node_perm) == self.nents
            znode_perm = node_perm  # adjust to zero-based indexing
            node_order = node_order[znode_perm]
        self.nfreedofs = 0
        for i in range(0, self.nents):
            k = node_order[i]
            for j in range(0, self.dim):
                if not self.is_fixed[k, j]:
                    self.dofnums[k, j] = self.nfreedofs
                    self.nfreedofs += 1
        N = self.nfreedofs
        for i in range(0, self.nents):
            for j in range(0, self.dim):
                if self.is_fixed[i, j]:
                    self.dofnums[i, j] = N
                    N += 1

    def set_ebc(self, entids=None, is_fixed=True, comp=0, val=0.0):
        self.nfreedofs = 0  # previous numbering is not valid anymore
        for index in entids:
            self.is_fixed[index, comp] = is_fixed
            if is_fixed:
                self.fixed_values[index, comp] = val

    def apply_ebc(self):
        self.values[self.is_fixed] = self.fixed_values[self.is_fixed]

    def scatter_sysvec(self, vec):
        for i in numpy.arange(0, self.nents):
            for j in numpy.arange(0, self.dim):
                dn = self.dofnums[i, j]
                if dn < self.nfreedofs:
                    self.values[i, j] = vec[dn]

    def gather_sysvec(self):
        vec = numpy.zeros((self.nfreedofs,), dtype=numpy.float64)
        for i in range(0, self.nents):
            for j in range(0, self.dim):
                dn = self.dofnums[i, j]
                if dn < self.nfreedofs:
                    vec[dn] = self.values[i, j]
        return vec

    def gather_all_dofnums(self, zconn, dofnumsout):
        sh = dofnumsout.shape
        dofnumsout = dofnumsout.ravel()

        def fun(dofnums, dofnumsr, dofnumsc, zconn, zconnr, zconnc, dofnumsout):
            n = 0
            for i in range(zconnr):
                for k in range(zconnc):
                    for j in range(dofnumsc):
                        dofnumsout[n] = dofnums[zconn[i, k], j]
                        n += 1

        fun_jit = numba.jit("void(i4[:, :], i8, i8, i4[:, :], i8, i8, i4[:])")(fun)
        fun_jit(self.dofnums, self.dofnums.shape[0], self.dofnums.shape[1],
                zconn, zconn.shape[0], zconn.shape[1], dofnumsout)
        dofnumsout.shape = sh
        return True

    #    #@numba.autojit
    #    def gather_all_dofnums(self, zconn, dofnumsout):
    #        sh = dofnumsout.shape
    #        dofnumsout = dofnumsout.ravel()
    #        n = 0
    #        for i in range(zconn.shape[0]):
    #            for k in range(zconn.shape[1]):
    #                for j in range(0, self.dim):
    #                    dofnumsout[n] = self.dofnums[zconn[i, k], j]
    #                    n += 1
    #        dofnumsout.shape = sh
    #        return True

    #    @numba.autojit
    #    def gather_all_dofnums(self, zconn, dofnumsout):
    #        for i in range(zconn.shape[0]):
    #            dofnumsout[i, :] = self.dofnums[zconn[i, :], :].ravel()
    #        return True

    def gather_dofnums_vec(self, zconn, vecout):
        n = 0
        for i in range(len(zconn)):
            for j in range(0, self.dim):
                vecout[n] = self.dofnums[zconn[i], j]
                n += 1
        return vecout

    def gather_values_vec(self, zconn, vecout):
        """Gather an elementwise vector of field values.

        :param zconn: Element connectivity.
        :param vecout: Output, vector of len(zconn)*self.dim entries.
        :return: Nothing.
        """
        n = 0
        for i in range(len(zconn)):
            for j in range(0, self.dim):
                vecout[n] = self.values[zconn[i], j]
                n += 1
        return True

    def gather_fixed_values_vec(self, zconn, vecout):
        """Gather an elementwise vector of field fixed values.

        :param zconn: Element connectivity.
        :param vecout: Output, vector of len(zconn)*self.dim entries.
        :return: Nothing.
        """
        n = 0
        for i in range(len(zconn)):
            for j in range(0, self.dim):
                vecout[n] = self.fixed_values[zconn[i], j]
                n += 1
            # This did not work: it was unbelievably slow.
            #        def fun(fixed_values, zconn, zconnlen, dim, vecout):
            #            n = 0
            #            for i in range(zconnlen):
            #                for j in range(dim):
            #                    vecout[n] = fixed_values[zconn[i], j]
            #                    n += 1
            #        fun_jit = numba.jit("void(f8[:, :], i4[:], i4, i4, f8[:])")(fun)
            #        fun_jit(self.fixed_values, zconn.ravel(), len(zconn), self.dofnums.shape[1], vec)
        return True

    def fun_set_values(self, xyz, fun):
        """Set the values in the field using a function of the coordinates of the entities.
        
        :param xyz: array of coordinates
        :param fun: function
        :return: 
        """
        for i  in range(self.values.shape[0]):
            self.values[i,:] = fun(xyz[i,:])