import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.femms.femm_defor_linear_surface_spring import FEMMDeforLinearSurfaceSpring
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import fenode_select, fe_select
from spyfe.meshing.transformation import rotate_mesh
import numpy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from scipy.sparse.linalg import splu, spsolve, minres, lgmres, spilu
from scipy.sparse.csgraph import reverse_cuthill_mckee
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport
from spyfe.meshing.importers import abaqus_importer
from spyfe.materials.mat_defor import OUTPUT_CAUCHY

from spyfe.meshing.generators.hexahedra import h8_block, h8_to_h20

start0 = time.time()

# Reference value is the axial stress at point A
sigma_x_A_ref = -4830e6

E = 210e9
nu = 0.3
alpha = 2.3e-4
m = MatDeforTriaxLinearIso(e=E, nu=nu, alpha=alpha)

start = time.time()
L, H, W = 5.0, 2.0, 1.0
nW, nL, nH = 1,2,1
htol = min(L, H, W) / 1000
fens, fes = h8_block(L, W, H, nL, nW, nH)
fens, fes = h8_to_h20(fens, fes)
print('Mesh import', time.time() - start)

geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
cn = fenode_select(fens, box=numpy.array([0.0, 0.0, 0, W, 0, H]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=0, val=0.0)
    u.set_ebc([j], comp=1, val=0.0)
    u.set_ebc([j], comp=2, val=0.0)
cn = fenode_select(fens, box=numpy.array([L, L, 0, W, 0, H]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=0, val=0.0)

u.apply_ebc()

femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=3))
femm.associate_geometry(geom)
S = femm.connection_matrix(geom)
perm = reverse_cuthill_mckee(S,symmetric_mode=True)
u.numberdofs(node_perm=perm)
print('Number of degrees of freedom', u.nfreedofs)


start = time.time()
K = femm.stiffness(geom, u)
print('Matrix assembly', time.time() - start)

start = time.time()
dT = NodalField(nfens=fens.count(), dim=1)
dT.fun_set_values(fens.xyz, lambda x: 100*x[2])
F = femm.thermal_strain_loads(geom, u, dT)
print(F)
print('Load vector assembly', time.time() - start)

start = time.time()
# U, info = lgmres(K, F)
# print(info)
lu = splu(K)
del K
U = lu.solve(F)
u.scatter_sysvec(U)

print('Solution', time.time() - start)
#
print('Done', time.time() - start0)

stresses = femm.nodal_field_from_integr_points(geom, u, u, dtempn1=dT, output=OUTPUT_CAUCHY, component=[0, 1, 2, 3, 4, 5])
pointA = fenode_select(fens, box=[0,0, 0,0, 0., 0.],
                       inflate=htol)

sigxA = stresses.values[pointA, 0]
print('Stress sigx @ A=', sigxA, ', ', (sigxA / sigma_x_A_ref * 100), ' %')
#
vtkexport("beam_thermal_expansion", fes, geom, {"displacements": u, 'stresses': stresses})
