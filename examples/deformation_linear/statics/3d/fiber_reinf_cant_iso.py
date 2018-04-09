import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.tetrahedra import t4_block, t4_to_t10
from spyfe.meshing.generators.hexahedra import h8_block, h8_to_h20
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import fenode_select, fe_select
from spyfe.meshing.transformation import rotmat
import numpy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.femms.femm_defor_linear_qt10ms import FEMMDeforLinearQT10MS
from spyfe.femms.femm_defor_linear_h8msgso import FEMMDeforLinearH8MSGSO
from spyfe.integ_rules import TetRule, TriRule, GaussRule
from spyfe.force_intensity import ForceIntensity
from spyfe.algorithms import algo_common, algo_defor, algo_defor_linear
from spyfe.csys import CSys
from spyfe import phunits

SI = phunits.SI()
E = 1.0e9*SI.Pa # Pascal
nu = 0.25
uz_ref = -7.516310912734678e-03*SI.mm #millimeters
a, b, t = 90.0*SI.mm, 10.0*SI.mm, 20.0*SI.mm #millimeters
htol = min(a, b, t) / 1000;
magn = -1000.*SI.Pa # Pascal
m = MatDeforTriaxLinearIso(e=E, nu=nu)

def elem_dep_data_t10(na, nb, nt):
    fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = t4_to_t10(fens, fes)
    femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=TetRule(npts=4))
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
    return fens, femm, sfemm

def elem_dep_data_t4(na, nb, nt):
    fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=TetRule(npts=1))
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=1))
    return fens, femm, sfemm

def elem_dep_data_h8msgso(na, nb, nt):
    fens, fes = h8_block(a, b, t, na, nb, nt)
    femm = FEMMDeforLinearH8MSGSO(material=m, fes=fes)
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
    return fens, femm, sfemm

def elem_dep_data_qt10ms(na, nb, nt):
    fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = t4_to_t10(fens, fes)
    femm = FEMMDeforLinearQT10MS(material=m, fes=fes)
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
    return fens, femm, sfemm

def elem_dep_data_h20r(na, nb, nt):
    fens, fes = h8_block(a, b, t, na, nb, nt)
    fens, fes = h8_to_h20(fens, fes)
    femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
    return fens, femm, sfemm

def elem_dep_data_h8(na, nb, nt):
    fens, fes = h8_block(a, b, t, na, nb, nt)
    femm = FEMMDeforLinear(material=m, fes=fes, integration_rule=GaussRule(dim=3, order=2))
    bfes = mesh_boundary(femm.fes)
    fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
    return fens, femm, sfemm

models = {'T10': elem_dep_data_t10,
          'QT10MS': elem_dep_data_qt10ms,
          'H8MSGSO': elem_dep_data_h8msgso,
          'H20R': elem_dep_data_h20r}
# models = {'T4': elem_dep_data_t4}
# models = {'H20R': elem_dep_data_h20r}
# models = {'H8': elem_dep_data_h8}
# models = {'H8MSGSO': elem_dep_data_h8msgso}
for k, v  in models.items():
    print(k)
    for n  in range(10,11): #range(2,7)
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

        geom = model_data['geom']
        u = model_data['u']
        print('nfreedofs =' , u.nfreedofs)
        for action, time in model_data['timings']:
            print(action, time, ' sec')

        tipn = fenode_select(fens, box=[a, a, b, b, 0, 0])
        uv = u.values[tipn, :]
        uz = sum(uv[:, 2]) / len(tipn)
        print('Tip displacement uz =', uz, ', ', (uz / uz_ref * 100), ' %')

        algo_defor_linear.plot_displacement(model_data)

        model_data['postprocessing']={'file': 'fiber_reinf_cant_iso_'+k+'_results'}
        algo_defor.plot_elemental_stress(model_data)