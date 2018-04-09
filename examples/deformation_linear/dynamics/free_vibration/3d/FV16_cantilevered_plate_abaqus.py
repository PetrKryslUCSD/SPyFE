import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.hexahedra import h8_block, h8_to_h20
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import fenode_select, fe_select
from spyfe.meshing.boxes import bounding_box
import numpy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.femms.femm_defor_linear_h8msgso import FEMMDeforLinearH8MSGSO
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import reverse_cuthill_mckee
import spyfe.bipwr
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

start0 = time.time()
E = 200e9
nu = 0.3
rho = 8000
a = 10.0
b = a
h = 0.05

htol = h / 1000
na, nb, nh = 3,3,1
m = MatDeforTriaxLinearIso(rho=rho, e=E, nu=nu)

start = time.time()
fens, fes = h8_block(a, b, h, na, nb, nh)
fens, fes = h8_to_h20(fens, fes)
print('Mesh generation', time.time() - start)
geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
cn = fenode_select(fens, box=numpy.array([0, 0, 0, b, 0, h]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=0, val=0.0)
    u.set_ebc([j], comp=1, val=0.0)
    u.set_ebc([j], comp=2, val=0.0)
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
M = femmm.lumped_mass(geom, u)
M = (M.T + M)/2.
# for i in range(M.shape[0]):
#     for j in range(M.shape[1]):
#         if i==j:
#             print(i,j,M[i,j])
print('Mass assembly', time.time() - start)

start = time.time()

v = numpy.random.rand(u.nfreedofs, 10)
tol=1.e-3
maxiter=30
lamb, v, converged = spyfe.bipwr.gepbinvpwr2(K,M,v,tol,maxiter)
if converged:
    ix = numpy.argsort(numpy.abs(lamb))
    lamb = lamb[ix].real
    v = v[:, ix]
    print([math.sqrt(om) / 2.0 / math.pi for om in lamb])
    print('Solution', time.time() - start)
else:
    print( 'Not converged!'  )


# start = time.time()
# w, v = eigsh(K, k=10, M=M, which='SM')
# ix = numpy.argsort(numpy.abs(w))
# w = w[ix].real
# v = v[:,ix]
# print('EP solution', time.time() - start)
# print([math.sqrt(om)/2.0/math.pi for om in w])
# print(w, v[:,0])
u.scatter_sysvec(v[:,0].real)

#
print('Done', time.time() - start0)


#
vtkexport("FV16_cantilevered_plate_abaqus_results", fes, geom, {"displacements": u})
