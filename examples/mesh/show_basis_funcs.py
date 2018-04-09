import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.triangles import t3_ablock
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
from numpy import array
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.fields.nodal_field import NodalField
from spyfe.fields.gen_field import GenField
from spyfe.integ_rules import TriRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import spsolve, minres
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

Length, Width, nL, nW = 10.0, 10.0, 7, 8
fens, fes = t3_ablock(Length, Width, nL, nW)
geom = NodalField(nfens=fens.count(), dim=3)
for index  in range(fens.count()):
    for j  in range(2):
        geom.values[index, j] = fens.xyz[index, j]
vtkexport("show_basis_funcs-geom", fes, geom)

bf1 = GenField(data = geom.values)
bf1.values[0, 2] = 1.0
vtkexport("show_basis_funcs-bf1", fes, bf1)

bf13 = GenField(data = geom.values)
bf13.values[12, 2] = 1.0
vtkexport("show_basis_funcs-bf13", fes, bf13)

bf16 = GenField(data = geom.values)
bf16.values[15, 2] = 1.0
vtkexport("show_basis_funcs-bf16", fes, bf16)

bf30 = GenField(data = geom.values)
bf30.values[29, 2] = 1.0
vtkexport("show_basis_funcs-bf30", fes, bf30)

bf31 = GenField(data = geom.values)
bf31.values[30, 2] = 1.0
vtkexport("show_basis_funcs-bf31", fes, bf31)

bf32 = GenField(data = geom.values)
bf32.values[31, 2] = 1.0
vtkexport("show_basis_funcs-bf32", fes, bf32)

bf39 = GenField(data = geom.values)
bf39.values[38, 2] = 1.0
vtkexport("show_basis_funcs-bf39", fes, bf39)

bf40 = GenField(data = geom.values)
bf40.values[39, 2] = 1.0
vtkexport("show_basis_funcs-bf40", fes, bf40)

bf31p32p30 = GenField(data = geom.values)
bf31p32p30.values[29, 2] = 1.0
bf31p32p30.values[30, 2] = 1.0
bf31p32p30.values[31, 2] = 1.0
vtkexport("show_basis_funcs-bf31p32p30", fes, bf31p32p30)
