import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.hexahedra import h8_blockx
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
import numpy
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import splu, spsolve, minres
from scipy.sparse.csgraph import reverse_cuthill_mckee
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

start0 = time.time()
# These are the constants in the problem, k is kappa
boundaryf = lambda x, y, z: 1.0 + x ** 2 + 2 * y ** 2
Q = -6  # internal heat generation rate
k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=k * numpy.identity(3))
start = time.time()
N =40
xs = numpy.linspace(0.0, 1.0, N+1)
ys = numpy.linspace(0.0, 1.0, N+1)
zs = numpy.linspace(0.0, 1.0, N+1)
fens, fes = h8_blockx(xs, ys, zs)
print('Mesh generation',time.time() - start)
bfes = mesh_boundary(fes)
cn = connected_nodes(bfes)
geom = NodalField(fens=fens)
temp = NodalField(nfens=fens.count(), dim=1)
for j  in cn:
    temp.set_ebc([j], val=boundaryf(fens.xyz[j, 0], fens.xyz[j, 1], fens.xyz[j, 2]))
temp.apply_ebc()
femm = FEMMHeatDiff(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
S = femm.connection_matrix(geom)
perm = reverse_cuthill_mckee(S,symmetric_mode=True)
temp.numberdofs(node_perm=perm)
#temp.numberdofs()
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
#lu = splu(K)
#T = lu.solve(F)
#T = spsolve(K, F, use_umfpack=True)
T, info = minres(K, F)
print(info)
temp.scatter_sysvec(T)
print('Solution',time.time() - start)

print('Done',time.time() - start0)

vtkexport("Poisson_fe_H8_results", fes, geom, temp, fldname="temp")
