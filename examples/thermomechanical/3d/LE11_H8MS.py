import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear_h8msgso import FEMMDeforLinearH8MSGSO
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

start0 = time.time()

# Reference value is the axial stress at point A (at the spherical part, plane of symmetry, interior)
sigma_z_A_ref = -105e6

E = 210e9
nu = 0.3
alpha = 2.3e-4
m = MatDeforTriaxLinearIso(e=E, nu=nu, alpha=alpha)

start = time.time()
fens, feslist = abaqus_importer.import_mesh('LE11_H8.inp')
# Account for the instance rotation
fens = rotate_mesh(fens, math.pi*90./180*numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]))
for fes  in feslist:
    print(fes.count())
fes = feslist[0]
print('Mesh import', time.time() - start)

geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
u.apply_ebc()

femm = FEMMDeforLinearH8MSGSO(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
femm.associate_geometry(geom)
S = femm.connection_matrix(geom)
perm = reverse_cuthill_mckee(S,symmetric_mode=True)
u.numberdofs(node_perm=perm)
print('Number of degrees of freedom', u.nfreedofs)

bfes = mesh_boundary(fes)
htol = 1.0/1000
feselzm = fe_select(fens, bfes, plane=([1.293431,-535.757E-03,0.],[0.,0.,-1.]), inflate=htol)
feselzp = fe_select(fens, bfes, plane=([653.275E-03,-270.595E-03,1.79],[0.,0.,+1.]), inflate=htol)
feseltp = fe_select(fens, bfes, plane=([1.2,0.,0.],[0.,+1.,0.]), inflate=htol)
feseltm = fe_select(fens, bfes, plane=([707.107E-03,-707.107E-03,1.79],[-1.,-1.,0.]), inflate=htol)
tsfes = bfes.subset(numpy.hstack((feselzm, feselzp, feseltp, feseltm)))

start = time.time()
sfemm = FEMMDeforLinearSurfaceSpring(fes=tsfes, integration_rule=GaussRule(dim=2, order=4),
                                     surface_normal_spring_coefficient=(1. / ((abs(sigma_z_A_ref)/1e12)/E)))
K = femm.stiffness(geom, u) + sfemm.stiffness_normal(geom, u)
print('Matrix assembly', time.time() - start)

start = time.time()
dT = NodalField(nfens=fens.count(), dim=1)
dT.fun_set_values(fens.xyz, lambda x: math.sqrt(x[0]**2+x[1]**2)+x[2])
print(numpy.max(dT.values))
print(numpy.min(dT.values))
F = femm.thermal_strain_loads(geom, u, dT)
print('Load vector assembly', time.time() - start)

start = time.time()
# U, info = lgmres(K, F)
# print(info)
lu = splu(K)
del K
U = lu.solve(F)
#R = cho_factor(K, overwrite_a=True)
#y = spsolve(R, spsolve(R.T, F))
# U = spsolve(K, F)
# import scipy.sparse.linalg as spla
# ilu = spilu(K, diag_pivot_thresh=0.0, fill_factor=1000)
# M = spla.LinearOperator(K.shape, ilu.solve)
# U, info = lgmres(K, F, x0=ilu.solve(F), M=M) #, callback=lambda x: print('iteration', numpy.linalg.norm(x-U)))
# print(info)
u.scatter_sysvec(U)

print('Solution', time.time() - start)
#
print('Done', time.time() - start0)

stresses = femm.nodal_field_from_integr_points(geom, u, u, dtempn1=dT, output=OUTPUT_CAUCHY, component=[0, 1, 2, 3, 4, 5])
pointA = fenode_select(fens, box=[707.107E-03,707.107E-03,  -707.107E-03, -707.107E-03, 0., 0.],
                       inflate=htol)
print(pointA)
sigzzA = stresses.values[pointA, 2]
print('Stress sigz @ A=', sigzzA/1.e6, ', ', (sigzzA / sigma_z_A_ref * 100), ' %')
#
stresses.values /= 1.0e6
vtkexport("LE11_H8MS_results", fes, geom, {"displacements": u, 'dT': dT, 'stresses': stresses})
