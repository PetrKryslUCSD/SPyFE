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
import scipy.io

start0 = time.time()

# Reference value is the axial stress at point A (at the spherical part, plane of symmetry, interior)
sigma_z_A_ref = -105e6

E = 210e9
nu = 0.3
alpha = 2.3e-4
m = MatDeforTriaxLinearIso(e=E, nu=nu, alpha=alpha)

start = time.time()
fens, feslist = abaqus_importer.import_mesh('LE11_H20_90deg.inp')
# Account for the instance rotation
fens = rotate_mesh(fens, math.pi*90./180*numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]))
for fes  in feslist:
    print(fes.count())
fes = feslist[0]
scipy.io.savemat('LE11_H20_90deg.mat', {'xyz': fens.xyz, 'conn': fes.conn})
print('Mesh import', time.time() - start)

geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
htol = 1.0/1000
cn = fenode_select(fens, plane=([0,0,0.],[0.,0.,-1.]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=2, val=0.0)
cn = fenode_select(fens, plane=([0,0,1.79],[0.,0.,-1.]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=2, val=0.0)
cn = fenode_select(fens, plane=([0,0,0],[0.,+1.,0.]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=1, val=0.0)
cn = fenode_select(fens, plane=([0,0,0],[+1.,0.,0.]), inflate=htol)
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
dT.fun_set_values(fens.xyz, lambda x: math.sqrt(x[0]**2+x[1]**2)+x[2])
print(numpy.max(dT.values))
print(numpy.min(dT.values))
F = femm.thermal_strain_loads(geom, u, dT)
print('Load vector assembly', time.time() - start)
print(numpy.max(F), numpy.min(F))

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

stresses = femm.nodal_field_from_integr_points_spr(geom, u, u, dTn1=dT, output=OUTPUT_CAUCHY, component=[0, 1, 2, 3, 4, 5])
pointA = fenode_select(fens, box=[707.107E-03,707.107E-03,  -707.107E-03, -707.107E-03, 0., 0.],
                       inflate=htol)
print(pointA)
sigzzA = stresses.values[pointA, 2]
print('Stress sigz @ A=', sigzzA/1.e6, ', ', (sigzzA / sigma_z_A_ref * 100), ' %')
#
stresses.values /= 1.0e6
vtkexport("LE11_H20_90deg_results", fes, geom, {"displacements": u, 'dT': dT, 'stresses': stresses})
