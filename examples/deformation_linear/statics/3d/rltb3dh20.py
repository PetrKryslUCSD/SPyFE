import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe

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
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import splu, spsolve, minres, lgmres, spilu
from scipy.sparse.csgraph import reverse_cuthill_mckee
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

start0 = time.time()
E = 1000
nu = 0.4999
W = 2.5;
H = 5;
L = 50;
# nW, nL, nH = 20, 20, 20
nW, nL, nH = 4, 20, 4
htol = min(L, H, W) / 1000;
magn = -0.2 * 12.2334 / 4;
Force = magn * W * H * 2;
Force * L ** 3 / (3 * E * W * H ** 3 * 2 / 12);
uzex = -12.0935378981478
m = MatDeforTriaxLinearIso(e=E, nu=nu)

start = time.time()
fens, fes = h8_block(W, L, H, nW, nL, nH)
fens, fes = h8_to_h20(fens, fes)
print(fes.conn.shape)
print('Mesh generation', time.time() - start)
geom = NodalField(fens=fens)
u = NodalField(nfens=fens.count(), dim=3)
cn = fenode_select(fens, box=numpy.array([0, W, 0, 0, 0, H]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=0, val=0.0)
    u.set_ebc([j], comp=1, val=0.0)
    u.set_ebc([j], comp=2, val=0.0)
cn = fenode_select(fens, box=numpy.array([W, W, 0, L, 0, H]), inflate=htol)
for j in cn:
    u.set_ebc([j], comp=0, val=0.0)
u.apply_ebc()
femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
femm.associate_geometry(geom)
S = femm.connection_matrix(geom)
perm = reverse_cuthill_mckee(S,symmetric_mode=True)
u.numberdofs(node_perm=perm)
#u.numberdofs()
print('Number of degrees of freedom', u.nfreedofs)
start = time.time()
bfes = mesh_boundary(fes)
fesel = fe_select(fens, bfes, box=[0, W, L, L, 0, H], inflate=htol)
fi = ForceIntensity(magn=lambda x, J: numpy.array([0, 0, magn]))
tsfes = bfes.subset(fesel)
sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
F = sfemm.distrib_loads(geom, u, fi, 2)
start = time.time()
K = femm.stiffness(geom, u)
print('Matrix assembly', time.time() - start)
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

tipn = fenode_select(fens, box=[0, W, L, L, 0, H])
uv = u.values[tipn, :]
uz = sum(uv[:, 2]) / len(tipn)
print('Tip displacement uz =', uz, ', ', (uz / uzex * 100), ' %')

#
vtkexport("rltb3dH20_results", fes, geom, {"displacements": u})
