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
from spyfe.integ_rules import TriRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import spsolve, minres
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
N = 500
Length, Width, nL, nW = 1.0, 1.0, N, N
fens, fes = t3_ablock(Length, Width, nL, nW)
print('Mesh generation', time.time() - start)
bfes = mesh_boundary(fes)
cn = connected_nodes(bfes)
geom = NodalField(fens=fens)
temp = NodalField(nfens=fens.count(), dim=1)
for index in cn:
    temp.set_ebc([index], val=boundaryf(fens.xyz[index, 0], fens.xyz[index, 1]))
temp.apply_ebc()
temp.numberdofs()
femm = FEMMHeatDiff(material=m, fes=fes, integration_rule=TriRule(npts=1))
start = time.time()
fi = ForceIntensity(magn=lambda x, J: Q)
F = femm.distrib_loads(geom, temp, fi, 3)
print('Heat generation load', time.time() - start)
start = time.time()
F += femm.nz_ebc_loads_conductivity(geom, temp)
print('NZ EBC load', time.time() - start)
start = time.time()
K = femm.conductivity(geom, temp)
print('Matrix assembly', time.time() - start)
start = time.time()
temp.scatter_sysvec(spsolve(K, F))
# T, info = minres(K, F)
# print(info)
# temp.scatter_sysvec(T)
print('Solution', time.time() - start)

print('Done', time.time() - start0)

vtkexport("Poisson_fe_T3_results", fes, geom, {'temperature': temp})
