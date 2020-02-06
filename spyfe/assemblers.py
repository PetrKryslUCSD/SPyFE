import numpy
from numpy import ix_
import scipy.sparse
import numba


class SysmatAssemblerSparseFixedSymm:
    """
    Class for assembling of a sparse global matrix from
    elementwise matrices.
    
    The matrix is assembled from fixed-size symmetric elementwise matrices.
    The global matrix itself will therefore be symmetric.

    """

    def __init__(self, fes, fld):
        """Create and initialize the assembler.
        
        :param fes: finite element set
        :param fld: field for the unknowns
        :return: Nothing. The assembler is modified.
        """
        # These are the dimensions
        elem_mat_nrowcol = fld.dim * fes.nfens
        elem_mat_nmatrices = fes.conn.shape[0]
        nrowcol = fld.nfreedofs
        # Allocate the buffers
        # Buffer for the degrees of freedom
        self.dofnums = numpy.zeros((elem_mat_nmatrices, elem_mat_nrowcol), dtype=int)
        fld.gather_all_dofnums(fes.conn, self.dofnums)
        # Buffer for the element matrices
        self.elmtx = numpy.zeros((elem_mat_nmatrices, elem_mat_nrowcol, elem_mat_nrowcol))
        # Dimensions: dimensions of the elementwise matrices and the number of rows and columns of the global matrix
        self.elem_mat_nrowcol = elem_mat_nrowcol
        self.nrowcol = nrowcol

    def make_matrix(self):
        """Make a system matrix.

        :return: The matrix.
        """
        J = numpy.zeros((self.dofnums.shape[0], self.dofnums.shape[1] ** 2), dtype=int)

        #        col = 0
        #        for j in range(self.dofnums.shape[1]):
        #            for m in range(self.dofnums.shape[1]):
        #                J[:, col] = self.dofnums[:, m]
        #                col += 1
        def fun(nc, dofnums, J):
            col = 0
            for j in range(nc):
                for m in range(nc):
                    J[:, col] = dofnums[:, m]
                    col += 1

        fun(self.dofnums.shape[1], self.dofnums, J)
        I = self.dofnums.ravel()  # I is now an alias for self.dofnums
        I.shape = (len(I), 1)
        I = I * numpy.ones((1, self.elem_mat_nrowcol), dtype=int)
        I = I.ravel()
        J = J.ravel()
        V = self.elmtx.ravel()
        b = (I < self.nrowcol) & (J < self.nrowcol)
        A = scipy.sparse.coo_matrix((V[b], (I[b], J[b])), shape=(self.nrowcol, self.nrowcol))
        I = None
        J = None
        V = None
        self.elmtx = None
        return A.tocsc()


class SysvecAssembler:
    """
    Class for assembling a system vector from elementwise vectors.
    """

    def __init__(self, fes, fld):
        """Create and initialize the assembler.

        :param fes: finite element set
        :param fld: field for the unknowns
        :return: Nothing. The assembler is modified.
        """
        # These are the dimensions
        elem_mat_nrowcol = fld.dim * fes.nfens
        elem_mat_nmatrices = fes.conn.shape[0]
        nrowcol = fld.nfreedofs
        # Buffer for the degrees of freedom
        self.dofnums = numpy.zeros((elem_mat_nmatrices, elem_mat_nrowcol), dtype=int)
        fld.gather_all_dofnums(fes.conn, self.dofnums)
        # Buffer for the elementwise vectors
        self.elvec = numpy.zeros((elem_mat_nmatrices, elem_mat_nrowcol))
        self.elem_mat_nrowcol = elem_mat_nrowcol
        self.nrowcol = nrowcol

    def make_vector(self):
        """Make a system vector.

        :return: The vector.
        """
        I = self.dofnums.ravel()
        V = self.elvec.ravel()
        b = (I < self.nrowcol)
        I = I[b]
        V = V[b]
        vec = numpy.zeros((self.nrowcol,))
        for j in range(len(I)):
            vec[I[j]] += V[j]
        return vec

# class SysmatAssemblerDense:
#     """
#     Class for assembling of a dense global matrix from
#     elementwise matrices.
#
#     """
#
#     def __init__(self):
#         self.matrix = None
#         self.ndofs_row = 0
#         self.ndofs_col = 0
#
#     def start_assembly(self, elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col):
#         """Start the matrix assembly.
#
#         :param elem_mat_nrows: Number of rows in a typical element matrix.
#         :param elem_mat_ncols: Number of columns in a typical element matrix.
#         :param elem_mat_nmatrices: Number of element matrices.
#         :param ndofs_row: Number of free degrees of freedom in the row direction.
#         :param ndofs_col: Number of free degrees of freedom in the column direction.
#         :return: Nothing.
#         """
#         self.matrix = numpy.zeros((ndofs_row+1, ndofs_col+1))
#         self.ndofs_row = ndofs_row
#         self.ndofs_col = ndofs_col
#
#     def assemble(self, mat, dofnums_row, dofnums_col):
#         """ Assemble one elementwise matrix.
#
#         :param mat: Elementwise matrix.
#         :param dofnums_row: Degrees of freedom for the rows.
#         :param dofnums_col: Degrees of freedom for the columns.
#         :return: nothing.
#
#         :note: When the degrees of freedom are supplied greater than
#         the number of free degrees of freedom, the corresponding rows and columns
#         of the elementwise matrix are ignored.
#         """
#         for ro in range(len(dofnums_row)):
#             gro = numpy.min([dofnums_row[ro], self.ndofs_row])
#             for co in range(len(dofnums_col)):
#                 gco = numpy.min([dofnums_col[co], self.ndofs_col])
#                 self.matrix[gro, gco] += mat[ro, co]
#
#     def make_matrix(self):
#         """Make a system matrix.
#
#         :return: The matrix.
#         """
#         # We must trim off the fake row at the bottom and the fake column on the right
#         freerowdof = range(self.ndofs_row)
#         freecoldof = range(self.ndofs_col)
#         return self.matrix[ix_(freerowdof, freecoldof)]
#
# class SysvecAssembler:
#     """
#     Class for assembling a system vector from elementwise vectors.
#     """
#
#     def __init__(self):
#         self.vector = None
#         self.ndofs_row = 0
#
#     def start_assembly(self,  ndofs_row):
#         self.vector = numpy.zeros((ndofs_row, ))
#         self.ndofs_row = ndofs_row
#
#     def assemble(self, mat, dofnums_row):
#         b = (dofnums_row<self.ndofs_row)
#         self.vector[dofnums_row[b]] += mat[b]
#         # for ro in range(len(dofnums_row)):
#         #     gro = numpy.min([dofnums_row[ro], self.ndofs_row])
#         #     self.vector[gro] += mat[ro]
#
#     def make_vector(self):
#         """Make a system vector.
#
#         :return: The vector.
#         """
#         return self.vector
#
# class SysmatAssemblerSparse:
#     """
#     Class for assembling of a sparse global matrix from
#     elementwise matrices.
#
#     """
#
#     def __init__(self):
#         self.I = None
#         self.J = None
#         self.V = None
#         self.bp = 0
#         self.ndofs_row = 0
#         self.ndofs_col = 0
#
#     def start_assembly(self, elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col):
#         """Start the matrix assembly.
#
#         :param elem_mat_nrows: Number of rows in a typical element matrix.
#         :param elem_mat_ncols: Number of columns in a typical element matrix.
#         :param elem_mat_nmatrices: Number of element matrices.
#         :param ndofs_row: Number of free degrees of freedom in the row direction.
#         :param ndofs_col: Number of free degrees of freedom in the column direction.
#         :return: Nothing.
#         """
#         N = elem_mat_nrows * elem_mat_ncols * elem_mat_nmatrices
#         self.I = numpy.zeros(N, dtype=int)
#         self.J = numpy.zeros(N, dtype=int)
#         self.V = numpy.zeros(N)
#         self.bp = 0
#         self.ndofs_row = ndofs_row
#         self.ndofs_col = ndofs_col
#
#     def assemble(self, mat, dofnums_row, dofnums_col):
#         """ Assemble one elementwise matrix.
#
#         :param mat: Elementwise matrix.
#         :param dofnums_row: Degrees of freedom for the rows.
#         :param dofnums_col: Degrees of freedom for the columns.
#         :return: nothing.
#
#         :note: When the degrees of freedom are supplied greater than
#         the number of free degrees of freedom, the corresponding rows and columns
#         of the elementwise matrix are ignored.
#         """
#         nrows = len(dofnums_row)
#         ncols = len(dofnums_col)
#         ntotal = nrows * ncols
#         buffer_range = range(self.bp, self.bp + ntotal)
#         self.V[buffer_range] = mat.ravel()
#         exp_dofnums_row = numpy.ones((nrows, 1)) * dofnums_row.reshape((1, nrows))
#         self.I[buffer_range] = exp_dofnums_row.ravel()
#         exp_dofnums_col = dofnums_col.reshape((ncols, 1)) * numpy.ones((1, ncols))
#         self.J[buffer_range] = exp_dofnums_col.ravel()
#         self.bp += ntotal
#
#     def assemble_symmetric(self, mat, dofnums):
#         """ Assemble one symmetric elementwise matrix.
#
#         :param mat: Elementwise symmetric matrix.
#         :param dofnums: Degrees of freedom for the rows and the columns.
#         :return: nothing.
#
#         :note: When the degrees of freedom are supplied greater than
#         the number of free degrees of freedom, the corresponding rows and columns
#         of the elementwise matrix are ignored.
#         """
#         self.assemble(mat, dofnums, dofnums)
#
#     def make_matrix(self):
#         """Make a system matrix.
#
#         :return: The matrix.
#         """
#         # We must trim off the fake row at the bottom and the fake column on the right
#         self.I = self.I[0:self.bp]
#         self.J = self.J[0:self.bp]
#         self.V = self.V[0:self.bp]
#         b = (self.I<self.ndofs_row) & (self.J<self.ndofs_col)
#         A = scipy.sparse.coo_matrix((self.V[b], (self.I[b], self.J[b])), shape=(self.ndofs_row, self.ndofs_col))
#         return A.tocsr()
#
# class SysmatAssemblerSparseFixedSymmetric:
#     """
#     Class for assembling of a sparse global matrix from
#     elementwise matrices.
#
#     """
#
#     def __init__(self):
#         self.I = None
#         self.J = None
#         self.V = None
#         self.bp = 0
#         self.ndofs_row = 0
#         self.ndofs_col = 0
#         self.onesa = None
#         self.bpchunk = 0
#
#     def start_assembly(self, elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col):
#         """Start the matrix assembly.
#
#         :param elem_mat_nrows: Number of rows in a typical element matrix.
#         :param elem_mat_ncols: Number of columns in a typical element matrix.
#         :param elem_mat_nmatrices: Number of element matrices.
#         :param ndofs_row: Number of free degrees of freedom in the row direction.
#         :param ndofs_col: Number of free degrees of freedom in the column direction.
#         :return: Nothing.
#         """
#         if elem_mat_nrows != elem_mat_ncols:
#             raise Exception("Expected the elementwise matrices to be symmetric")
#         if ndofs_row != ndofs_col:
#             raise Exception("Expected the sparse matrix to be symmetric")
#         N = elem_mat_nrows * elem_mat_ncols * elem_mat_nmatrices
#         self.I = numpy.zeros(N, dtype=int)
#         self.J = numpy.zeros(N, dtype=int)
#         self.V = numpy.zeros(N)
#         self.bp = 0
#         self.ndofs_row = ndofs_row
#         self.ndofs_col = ndofs_col
#         self.onesa = numpy.ones((1, elem_mat_nrows))
#         self.bpchunk = elem_mat_nrows*elem_mat_ncols
#
#     def assemble(self, mat, dofnums_row, dofnums_col):
#         """ Assemble one elementwise matrix.
#
#         :param mat: Elementwise matrix.
#         :param dofnums_row: Degrees of freedom for the rows.
#         :param dofnums_col: Degrees of freedom for the columns.
#         :return: nothing.
#
#         :note: When the degrees of freedom are supplied greater than
#         the number of free degrees of freedom, the corresponding rows and columns
#         of the elementwise matrix are ignored.
#         """
#         raise Exception("This method  is not available since the matrix is assumed symmetric")
#
#     def assemble_symmetric(self, mat, dofnums):
#         """ Assemble one symmetric elementwise matrix.
#
#         :param mat: Elementwise symmetric matrix.
#         :param dofnums: Degrees of freedom for the rows and the columns.
#         :return: nothing.
#
#         :note: When the degrees of freedom are supplied greater than
#         the number of free degrees of freedom, the corresponding rows and columns
#         of the elementwise matrix are ignored.
#         """
#         buffer_range = range(self.bp, self.bp + self.bpchunk)
#         self.V[buffer_range] = mat.ravel()
#         exp_dofnums = dofnums * self.onesa
#         self.I[buffer_range] = exp_dofnums.T.ravel()
#         self.J[buffer_range] = exp_dofnums.ravel()
#         self.bp += self.bpchunk
#
#     def make_matrix(self):
#         """Make a system matrix.
#
#         :return: The matrix.
#         """
#         # We must trim off the fake row at the bottom and the fake column on the right
#         self.I = self.I[0:self.bp]
#         self.J = self.J[0:self.bp]
#         self.V = self.V[0:self.bp]
#         b = (self.I<self.ndofs_row) & (self.J<self.ndofs_col)
#         A = scipy.sparse.coo_matrix((self.V[b], (self.I[b], self.J[b])), shape=(self.ndofs_row, self.ndofs_col))
#         return A.tocsr()
