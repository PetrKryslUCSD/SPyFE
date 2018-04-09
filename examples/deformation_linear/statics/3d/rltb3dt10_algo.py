import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe

from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.tetrahedra import t4_block, t4_to_t10
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import fenode_select, fe_select
import numpy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.integ_rules import TetRule, TriRule
from spyfe.force_intensity import ForceIntensity
from spyfe.algorithms import algo_common, algo_defor, algo_defor_linear
import time

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
fens, fes = t4_block(W, L, H, nW, nL, nH, orientation='ca')
fens, fes = t4_to_t10(fens, fes)
print('Mesh generation', time.time() - start)

model_data = {}
model_data['fens'] = fens
model_data['regions'] = [{'femm': FEMMDeforLinear(material=m, fes=fes, integration_rule=TetRule(npts=4))
                          }]

algo_common.plot_mesh(model_data)

model_data['boundary_conditions'] = {}
# Clamped face
cn = fenode_select(fens, box=numpy.array([0, W, 0, 0, 0, H]), inflate=htol)
essential = [{'node_list': cn,
              'comp': 0,
              'value': lambda x: 0.0
              },
             {'node_list': cn,
              'comp': 1,
              'value': lambda x: 0.0
              },
             {'node_list': cn,
              'comp': 2,
              'value': lambda x: 0.0
              },
             ]
# Symmetry plane
cn = fenode_select(fens, box=numpy.array([W, W, 0, L, 0, H]), inflate=htol)
essential.append({'node_list': cn,
                  'comp': 0,
                  'value': lambda x: 0.0
                  })
model_data['boundary_conditions']['essential'] = essential

# Traction on the free end
bfes = mesh_boundary(fes)
fesel = fe_select(fens, bfes, box=[0, W, L, L, 0, H], inflate=htol)
fi = ForceIntensity(magn=lambda x, J: numpy.array([0, 0, magn]))
tsfes = bfes.subset(fesel)
sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
traction = [{'femm': sfemm, 'force_intensity': fi}]
model_data['boundary_conditions']['traction'] = traction

# Call the solver
algo_defor_linear.statics(model_data)
for action, time in model_data['timings']:
    print(action, time, ' sec')

geom = model_data['geom']
u = model_data['u']

tipn = fenode_select(fens, box=[0, W, L, L, 0, H])
uv = u.values[tipn, :]
uz = sum(uv[:, 2]) / len(tipn)
print('Tip displacement uz =', uz, ', ', (uz / uzex * 100), ' %')

model_data['postprocessing']={'file': 'rltb3dt10_algo_results'}
algo_defor.plot_stress(model_data)
