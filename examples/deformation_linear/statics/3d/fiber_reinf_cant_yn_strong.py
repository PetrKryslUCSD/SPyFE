import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.tetrahedra import t4_block, t4_to_t10
from spyfe.meshing.generators.hexahedra import h8_block
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import fenode_select, fe_select
from spyfe.meshing.transformation import rotmat
import numpy
from spyfe.materials.mat_defor_triax_linear_ortho import MatDeforTriaxLinearOrtho
from spyfe.femms.femm_defor_linear_qt10ms import FEMMDeforLinearQT10MS
from spyfe.femms.femm_defor_linear_h8msgso import FEMMDeforLinearH8MSGSO
from spyfe.integ_rules import TetRule, TriRule, GaussRule
from spyfe.force_intensity import ForceIntensity
from spyfe.algorithms import algo_common, algo_defor, algo_defor_linear
from spyfe.csys import CSys

e1, e2, e3 = 1.0e14, 1.0e9, 1.0e9 # Pascal
g12, g13, g23 = 0.2e9, 0.2e9, 0.2e9 # Pascal
nu12, nu13, nu23= 0.25, 0.25, 0.25
aangle =-45
uz_ref = -1.027498445054843e-02 #millimeters
a, b, t = 90.0, 10.0, 20.0 #millimeters
na, nb, nt = 16, 4,4
htol = min(a, b, t) / 1000;
magn = -1000. # Pascal
m = MatDeforTriaxLinearOrtho(e1=e1, e2=e2, e3=e3,
                             g12=g12, g13=g13, g23=g23,
                             nu12=nu12, nu13=nu13, nu23=nu23)
mcsys = CSys(matrix = rotmat( numpy.array([0,aangle/180.*math.pi*1,0])))

def elem_dep_data_t10(na, nb, nt):
    fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = t4_to_t10(fens, fes)
    femm = FEMMDeforLinear(material=m, material_csys=mcsys, fes=fes, integration_rule=TetRule(npts=4))
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
    return fens, femm, sfemm

def elem_dep_data_h8msgso(na, nb, nt):
    fens, fes = h8_block(a, b, t, na, nb, nt)
    femm = FEMMDeforLinearH8MSGSO(material=m, material_csys=mcsys, fes=fes)
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
    return fens, femm, sfemm

def elem_dep_data_qt10ms(na, nb, nt):
    fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = t4_to_t10(fens, fes)
    femm = FEMMDeforLinearQT10MS(material=m, material_csys=mcsys, fes=fes)
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
    return fens, femm, sfemm

models = {'T10': elem_dep_data_t10,
          'QT10MS': elem_dep_data_qt10ms,
          'H8MSGSO': elem_dep_data_h8msgso}
for k, v  in models.items():
    print(k)
    for n  in range(2,7): #range(2,7)
        fens, femm, sfemm = v(n, n, n)
        model_data = {}
        model_data['fens'] = fens
        model_data['regions'] = [{'femm': femm}]

        # algo_common.plot_mesh(model_data)

        model_data['boundary_conditions'] = {}
        # Clamped face
        cn = fenode_select(fens, box=numpy.array([0, 0, 0, b, 0, t]), inflate=htol)
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
        model_data['boundary_conditions']['essential'] = essential

        # Traction on the free end
        fi = ForceIntensity(magn=lambda x, J: numpy.array([0, 0, magn]))
        traction = [{'femm': sfemm, 'force_intensity': fi}]
        model_data['boundary_conditions']['traction'] = traction

        # Call the solver
        algo_defor_linear.statics(model_data)
        # for action, time in model_data['timings']:
        #     print(action, time, ' sec')

        geom = model_data['geom']
        u = model_data['u']

        tipn = fenode_select(fens, box=[a, a, b, b, 0, 0])
        uv = u.values[tipn, :]
        uz = sum(uv[:, 2]) / len(tipn)
        print('Tip displacement uz =', uz, ', ', (uz / uz_ref * 100), ' %')

        algo_defor_linear.plot_displacement(model_data)

        model_data['postprocessing']={'file': 'fiber_reinf_cant_yn_strong_results'}
        algo_defor.plot_stress(model_data)