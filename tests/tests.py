import unittest

import numpy
from context import spyfe

class AllTests(unittest.TestCase):

    def test_feset(self):
        from spyfe.fesets.fe_set_base import FESet
        fes = FESet()
        fes.conn = numpy.array([[1, 2], [2, 3]])
        print(FESet.nfens, fes.conn)
        fes.conn = numpy.array([[1, 2], [2, 3], [8, 4], [5, 4]])
        print(fes.conn.shape)
        print(fes.label)
        fes.label=1
        print(fes.label)
        fes.label=numpy.array([1, 2, 2, 3])
        numpy.array([1, 2, 2])

    def test_fesetl2_conn(self):
        from spyfe.fesets.curvelike import FESetL2
        fes = FESetL2(numpy.array([[1, 2], [2, 3], [8, 4]]))
        print(fes.conn)
        print(fes.gradbfunpar(0.0))

    def test_gen_field(self):
        from spyfe.fields.gen_field import GenField
        f = GenField(nents=4, dim=3)
        zconn = numpy.array([1, 2], dtype=int)
        vec = numpy.zeros((6,), dtype=int)
        f.gather_dofnums_vec(zconn, vec)
        print(vec)

    def test_gen_field(self):
        from spyfe.fields.gen_field import GenField
        f = GenField(nents=4, dim=3)
        zconn=numpy.array([1, 2], dtype=int)
        dofnums = numpy.zeros((2*3, 1))
        f.gather_dofnums_vec(zconn, dofnums)
        print(dofnums)
        f.set_ebc(entids=[1,3], is_fixed=True, comp=1, val=3.0)
        f.apply_ebc()
        f.numberdofs()
        print(f.dofnums)
        print(f.nfreedofs)
        print(f.values)
        print(f.gather_sysvec())

    def test_Gauss(self):
        from numpy.testing import assert_array_almost_equal
        from spyfe.integ_rules import GaussRule
        ir = GaussRule(dim=3, order=2)
        assert_array_almost_equal(ir.param_coords,
                                  numpy.array([[-0.57735027, - 0.57735027, - 0.57735027],
                                               [-0.57735027, - 0.57735027, 0.57735027],
                                               [-0.57735027, 0.57735027, - 0.57735027],
                                               [-0.57735027, 0.57735027, 0.57735027],
                                               [0.57735027, - 0.57735027, - 0.57735027],
                                               [0.57735027, - 0.57735027, 0.57735027],
                                               [0.57735027, 0.57735027, - 0.57735027],
                                               [0.57735027, 0.57735027, 0.57735027]]))

    def test_csys(self):
        from spyfe.csys import CSys
        from numpy import linalg
        assert CSys().isconstant
        assert CSys().isidentity
        q, r = linalg.qr(numpy.random.rand(3, 3))
        assert CSys(matrix=q).isidentity == False

    def test_tri_rule(self):
        from numpy.testing import assert_array_almost_equal
        from spyfe.integ_rules import TriRule
        ir = TriRule(npts=3)
        assert_array_almost_equal(ir.param_coords,
                                  numpy.array([[0.666666666666667, 0.166666666666667],
                                               [0.166666666666667, 0.666666666666667],
                                               [0.166666666666667, 0.166666666666667]
                                               ]))


    def test_first_example(self):
        import math
        import numpy
        from numpy import array
        from spyfe.materials.mat_heatdiff import MatHeatDiff
        from spyfe.fesets.surfacelike import FESetT3
        from spyfe.femms.femm_heatdiff import FEMMHeatDiff
        from spyfe.fields.nodal_field import NodalField
        from spyfe.integ_rules import TriRule
        from spyfe.fenode_set import FENodeSet
        # These are the constants in the problem, k is kappa
        a = 2.5  # radius on the columnthe
        dy = a / 2 * math.sin(15. / 180 * math.pi)
        dx = a / 2 * math.cos(15. / 180 * math.pi)
        Q = 4.5  # internal heat generation rate
        k = 1.8  # thermal conductivity
        m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]))
        Dz = 1.0  # thickness of the slice
        xall = array([[0, 0], [dx, -dy], [dx, dy], [2*dx, -2*dy], [2*dx, 2*dy]])
        fes = FESetT3(array([[1, 2, 3], [2, 4, 5], [2, 5, 3]])-1)

        femm = FEMMHeatDiff(material=m, fes=fes, integration_rule=TriRule(npts=1))
        fens = FENodeSet(xyz=xall)
        geom = NodalField(fens=fens)
        temp = NodalField(nfens=xall.shape[0], dim=1)
        temp.set_ebc([3, 4])
        temp.apply_ebc()
        temp.numberdofs(node_perm=[1, 2, 0, 4, 3])
        print(temp.dofnums)
        K = femm.conductivity(geom, temp)
        print(K)

    def test_material_base(self):
        from spyfe.materials.mat_base import MatBase
        m = MatBase(rho=3)
        assert m.rho== 3


    def test_fens_1(self):
        import math
        import numpy
        from spyfe.fenode_set import FENodeSet
        a = 2.5  # radius on the columnthe
        dy = a / 2 * math.sin(15. / 180 * math.pi)
        dx = a / 2 * math.cos(15. / 180 * math.pi)
        fens = FENodeSet(xyz=numpy.array([[0, 0], [dx, -dy], [dx, dy], [2*dx, -2*dy], [2*dx, 2*dy]]))
        print(fens.xyz)

    def test_zero_one_based(self):
        from spyfe.onebased import OneBased2DArray, range_1based
        a = OneBased2DArray((5, 2))
        print(a)
        for index  in range_1based(1,5):
            a[index,0] = index
        print(a)

    def test_T3_ablock(self):
        from spyfe.meshing.generators.triangles import t3_ablock
        Length, Width, nL, nW = 2.0, 1.0, 4, 3
        fens, fes = t3_ablock(Length, Width, nL, nW)
        print(fes.conn)

    def test_one_based_initial(self):
        import numpy
        from spyfe.meshing.generators.triangles import t3_ablock
        Length, Width, nL, nW = 3.1, 2.2, 3, 2
        fens, fes = t3_ablock(Length, Width, nL, nW)
        print(fes.conn)
        assert fes.conn[0, 2]== 5
        assert fes.conn[2, 0] == 4
        #fes.conn[0, 0]
        print(fens.xyz[0,:])
    def test_boundary(self):
        import numpy
        from spyfe.meshing.generators.triangles import t3_ablock
        from spyfe.meshing.modification import mesh_boundary
        Length, Width, nL, nW = 3.1, 2.2, 3, 2
        fens, fes = t3_ablock(Length, Width, nL, nW)
        #   print(fes.conn)
        bfes = mesh_boundary(fes)

    def test_sparse(self):
        from scipy import sparse
        from numpy import array
        I = array([0, 3, 1, 0])
        J = array([0, 3, 1, 2])
        V = array([4, 5, 7, 9])
        A = sparse.coo_matrix((V, (I, J)), shape=(4, 4))
        #print(A)

#    def test_sparse_assemble(self):
#        from numpy import array
#        from spyfe.assemblers import SysmatAssemblerSparse
#        a = SysmatAssemblerSparse()
#        elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col = 3, 3, 4, 7, 7
#        a.start_assembly(elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col)
#        mat = array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
#        dofnums_row = array([3, 0, 5])
#        dofnums_col = array([3, 0, 5])
#        a.assemble(mat, dofnums_row, dofnums_col)
#        dofnums_row = array([3, 0, 5])+5
#        dofnums_col = array([3, 0, 5])+5
#        a.assemble(mat, dofnums_row, dofnums_col)
        
    def test_Poisson_t3(self):
        from context import spyfe
        from spyfe.meshing.generators.triangles import t3_ablock
        from spyfe.meshing.modification import mesh_boundary
        from spyfe.meshing.selection import connected_nodes
        from numpy import array
        from spyfe.materials.mat_heatdiff import MatHeatDiff
        from spyfe.femms.femm_heatdiff import FEMMHeatDiff
        from spyfe.fields.nodal_field import NodalField
        from spyfe.integ_rules import TriRule
        from spyfe.force_intensity import ForceIntensity
        from scipy.sparse.linalg import spsolve
        import time
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        
        start0 = time.time()
        
        # These are the constants in the problem, k is kappa
        boundaryf = lambda x, y: 1.0 + x ** 2 + 2 * y ** 2
        Q = -6  # internal heat generation rate
        k = 1.0  # thermal conductivity
        m = MatHeatDiff(thermal_conductivity=array([[k, 0.0], [0.0, k]]))
        Dz = 1.0  # thickness of the slice
        start = time.time()
        N =5
        Length, Width, nL, nW = 1.0, 1.0, N, N
        fens, fes = t3_ablock(Length, Width, nL, nW)
        print('Mesh generation',time.time() - start)
        bfes = mesh_boundary(fes)
        cn = connected_nodes(bfes)
        geom = NodalField(fens=fens)
        temp = NodalField(nfens=fens.count(), dim=1)
        for index  in cn:
            temp.set_ebc([index], val=boundaryf(fens.xyz[index, 0], fens.xyz[index, 1]))
        temp.apply_ebc()
        temp.numberdofs()
        femm = FEMMHeatDiff(material=m, fes=fes, integration_rule=TriRule(npts=1))
        start = time.time()
        fi = ForceIntensity(magn=lambda x, J: Q)
        F = femm.distrib_loads(geom, temp, fi, 3)
        print('Heat generation load',time.time() - start)
        start = time.time()
        F += femm.nz_ebc_loads_conductivity(geom, temp)
        print('NZ EBC load',time.time() - start)
        start = time.time()
        K = femm.conductivity(geom, temp)
        print('Matrix assembly',time.time() - start)
        start = time.time()
        temp.scatter_sysvec(spsolve(K, F))
        print('Solution',time.time() - start)
        print(temp.values)
        print('Done',time.time() - start0)
        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(temp.values[-4:-1], array([[ 3.16],
           [ 3.36],
           [ 3.64]]))

#    def test_sparse_assemble(self):
#        from numpy import array, ones
#        from spyfe.assemblers import SysmatAssemblerSparseFixedSymm
#        elem_mat_nrows, elem_mat_ncols, elem_mat_nmatrices, ndofs_row, ndofs_col = 3, 3, 4, 7, 7
#        a = SysmatAssemblerSparseFixedSymm(elem_mat_nrows, elem_mat_nmatrices, ndofs_row)
#        mat = array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
#        dofnums_row = array([3, 0, 5])
#        a.dofnums[0, :] = dofnums_row
#        #        a.elmatrices[0, :, :] = mat
#        dofnums_row = array([3, 0, 5]) + 5
#        a.dofnums[1, :] = dofnums_row
#        #        a.elmatrices[1, :, :] = mat
#        I = a.dofnums.ravel()
#        I.shape = (len(I), 1)
#        I = I * ones((1, elem_mat_ncols), dtype=int)
#        print(I.shape)

    def test_connection_matrix(self):
        from context import spyfe
        from spyfe.meshing.generators.triangles import t3_ablock
        from spyfe.femms.femm_base import FEMMBase
        from spyfe.fields.nodal_field import NodalField
        from spyfe.integ_rules import TriRule
        N = 5
        Length, Width, nL, nW = 1.0, 1.0, N, N
        fens, fes = t3_ablock(Length, Width, nL, nW)
        geom = NodalField(fens=fens)
        femm = FEMMBase(fes=fes, integration_rule=TriRule(npts=1))
        S = femm.connection_matrix(geom)

    def test_subset(self):
        from context import spyfe
        from spyfe.meshing.generators.triangles import t3_ablock
        from spyfe.femms.femm_base import FEMMBase
        from spyfe.fields.nodal_field import NodalField
        from spyfe.integ_rules import TriRule
        N = 2
        Length, Width, nL, nW = 1.0, 1.0, N, N
        fens, fes = t3_ablock(Length, Width, nL, nW)
        print(fes.conn)
        # print(fes.label)
        fes.subset([0, 3, 5])
        print(fes.conn)

    def test_fenode_map(self):
        from spyfe.fenode_to_fe_map import fenode_to_fe_map
        from spyfe.meshing.generators.triangles import t3_ablock
        N = 2
        Length, Width, nL, nW = 1.0, 1.0, N, N
        fens, fes = t3_ablock(Length, Width, nL, nW)
        print(fes.conn)
        femap = fenode_to_fe_map(fens.count(), fes.conn)
        print(femap)

    def test_Q8_meshing(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        from spyfe.meshing.generators.quadrilaterals import q4_block, q4_to_q8
        N = 2
        Length, Width, nL, nW = 10.0, 7.0, N, N
        fens, fes = q4_block(Length, Width, nL, nW)
        fens, fes = q4_to_q8(fens, fes)
        print(fes.conn)
        geom = NodalField(fens=fens)
        vtkexport("test_Q8_meshing_mesh", fes, geom)

    def test_fusing_nodes(self):
        import numpy
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        from spyfe.meshing.generators.quadrilaterals import q4_block, q4_to_q8
        from spyfe.meshing.modification import fuse_nodes, merge_meshes
        N = 10
        Length, Width, nL, nW = 2.0, 3.0, N, N
        fens, fes = q4_block(Length, Width, nL, nW)
        fens1, fes1 = q4_to_q8(fens, fes)
        fens1.xyz[:, 0] += Length
        fens, fes = q4_block(Length, Width, nL, nW)
        fens2, fes2 = q4_to_q8(fens, fes)
        tolerance = Length / 1000
        fens, new_indexes_of_fens1_nodes = fuse_nodes(fens1, fens2, tolerance)
        print(fens.xyz)
        print(new_indexes_of_fens1_nodes)
        fens, fes1, fes2 = merge_meshes(fens1, fes1, fens2, fes2, tolerance)
        fes = fes1.cat(fes2)
        print(fens.xyz)
        print(fes.conn)
        geom = NodalField(fens=fens)
        vtkexport("test_fusing_nodes_mesh", fes, geom)

    def test_nastran_importer(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.importers import nastran_importer
        from spyfe.fesets.volumelike import FESetT4, FESetT10
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        fens, fes = nastran_importer.import_mesh('Slot-coarser.nas')
        print(fes.count())
        geom = NodalField(fens=fens)
        vtkexport("test_nastran_importer", fes, geom)
        print( 'Done' )

    def test_nastran_importer(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.importers import nastran_importer
        from spyfe.fesets.volumelike import FESetT4, FESetT10
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        fens, feslist = nastran_importer.import_mesh('Slot-coarser.nas')
        print(feslist[0].count())
        geom = NodalField(fens=fens)
        vtkexport("test_nastran_importer", feslist[0], geom)
        print('Done')

    def test_Abaqus_importer(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.importers import abaqus_importer
        from spyfe.meshing.exporters.vtkexporter import vtkexport
        fens, feslist = abaqus_importer.import_mesh('LE11_H20.inp')
        for fes in feslist:
            print(fes.count())
        fes = feslist[0]
        geom = NodalField(fens=fens)
        vtkexport("test_Abaqus_importer", fes, geom)
        print('Done')


if __name__ == "__main__":
    unittest.main()