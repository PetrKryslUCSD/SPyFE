import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import context
import math
import scipy.io
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.hexahedra import h8_block, h8_to_h20
import numpy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.femms.femm_defor_linear_h8msgso import FEMMDeforLinearH8MSGSO
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from spyfe.bipwr import gepbinvpwr2
from scipy.sparse.linalg import eigsh, eigs
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

start0 = time.time()
E = 1.0
nu = 0.499
rho = 1.0
a = 1.0
b = a
h = a

htol = h / 1000
na, nb, nh = 5,5,5
m = MatDeforTriaxLinearIso(rho=rho, e=E, nu=nu)

start = time.time()
fens, fes = h8_block(a, b, h, na, nb, nh)
fens, fes = h8_to_h20(fens, fes)
print('Mesh generation', time.time() - start)
geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
u.apply_ebc()
femmk = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
femmk.associate_geometry(geom)
u.numberdofs()
print('Number of degrees of freedom', u.nfreedofs)
start = time.time()
K = femmk.stiffness(geom, u)
K = (K.T + K)/2.0
print('Stiffness assembly', time.time() - start)
start = time.time()
femmm = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=3))
M = femmm.mass(geom, u)
M = (M.T + M)/2.
print('Mass assembly', time.time() - start)

start = time.time()

v = numpy.random.rand(u.nfreedofs, 10)
tol = 1.e-3
maxiter = 20
shift = (2 * math.pi * 0.2) ** 2
lamb, v, converged = gepbinvpwr2(K + shift * M, M, v, tol, maxiter)
if converged:
    lamb -= shift
    ix = numpy.argsort(numpy.abs(lamb))
    lamb = lamb[ix].real
    v = v[:, ix]
    print([math.sqrt(om) / 2.0 / math.pi for om in lamb])
    print('Solution', time.time() - start)
    u.scatter_sysvec(v[:, 6].real)
    vtkexport("unit_cube_h20_results", fes, geom, {"displacements": u})
else:
    print('Not converged!')

print('Done', time.time() - start0)


#

