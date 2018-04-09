import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
from spyfe.meshing.generators.tetrahedra import t4_block, t4_to_t10, t4_composite_plate
from spyfe.meshing.generators.hexahedra import h8_block, h8_to_h20, h8_composite_plate
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
from spyfe import phunits

# Three-dimensional Elasticity Solution for Uniformly Loaded Cross-ply
# Laminates and Sandwich Plates
# Ashraf M. Zenkour, Journal of Sandwich Structures and Materials 2007 9: 213-238
# DOI: 10.1177/1099636207065675

SI = phunits.SI()
e1, e2, e3 = 25.0e6 * SI.psi, 1.0e6 * SI.psi, 1.0e6 * SI.psi
g12, g13, g23 = 0.5e6 * SI.psi, 0.5e6 * SI.psi, 0.2e6 * SI.psi
nu12, nu13, nu23 = 0.25, 0.25, 0.25

a, b = 200.0 * SI.mm, 600.0 * SI.mm
q0 = -1. * SI.psi  # transverse load
#  The below values come from Table 2
# h = a / 4.
# wc_analytical = 3.65511 / (100 * e3 * h ** 3 / a ** 4 / q0)
# % h = a / 10;
# wc_analytical = 1.16899 / (100 * e3 * h ** 3 / a ** 4 / q0);
h = a / 50
wc_analytical = 0.66675 / (100 * e3 * h ** 3 / a ** 4 / q0)
# h = a / 100. # Thickness of the plate
# wc_analytical = 0.65071 / (100 * e3 * h ** 3 / a ** 4 / q0)
angles = [0, 90, 0]
nLayers = len(angles)
na, nb, nts = 6, 6, numpy.array([1 for l in range(nLayers)], dtype=int)
ts = numpy.ones((nLayers,)) * (h / nLayers)
htol = min(a, b, sum(ts)) / 1000.

m = MatDeforTriaxLinearOrtho(e1=e1, e2=e2, e3=e3,
                             g12=g12, g13=g13, g23=g23,
                             nu12=nu12, nu13=nu13, nu23=nu23)


# def elem_dep_data_t10(na, nb, nt):
#     fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
#     fens, fes = t4_to_t10(fens, fes)
#     femm = FEMMDeforLinear(material=m, material_csys=mcsys, fes=fes, integration_rule=TetRule(npts=4))
#     bfes = mesh_boundary(femm.fes)
#     fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
#     tsfes = bfes.subset(fesel)
#     sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
#     return fens, femm, sfemm
#
# def elem_dep_data_t4(na, nb, nt):
#     fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
#     femm = FEMMDeforLinear(material=m, material_csys=mcsys, fes=fes, integration_rule=TetRule(npts=1))
#     bfes = mesh_boundary(femm.fes)
#     fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
#     tsfes = bfes.subset(fesel)
#     sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=1))
#     return fens, femm, sfemm
#
# def elem_dep_data_h8msgso(na, nb, nt):
#     fens, fes = h8_block(a, b, t, na, nb, nt)
#     femm = FEMMDeforLinearH8MSGSO(material=m, material_csys=mcsys, fes=fes)
#     bfes = mesh_boundary(femm.fes)
#     fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
#     tsfes = bfes.subset(fesel)
#     sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
#     return fens, femm, sfemm

def elem_dep_data_qt10ms(na, nb, nts):
    print(na, nb, nts)
    # fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = t4_composite_plate(a, b, ts, na, nb, nts)
    fens, fes = t4_to_t10(fens, fes)
    bfes = mesh_boundary(fes)
    t=sum(ts)
    fesel = fe_select(fens, bfes, box=[0, a, 0, b, t, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    femms = []
    for layer in range(len(nts)):
        aangle = angles[layer] / 180. * math.pi
        print(aangle)
        mcsys = CSys(matrix=rotmat(numpy.array([0.0, 0.0, aangle])))
        el = fe_select(fens, fes, bylabel=True, label=layer)
        femms.append(FEMMDeforLinearQT10MS(material=m, material_csys=mcsys, fes=fes.subset(el)))
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=TriRule(npts=3))
    return fens, femms, sfemm


def elem_dep_data_h20r(na, nb, nts):
    print(na, nb, nts)
    # fens, fes = t4_block(a, b, t, na, nb, nt, orientation='a')
    fens, fes = h8_composite_plate(a, b, ts, na, nb, nts)
    fens, fes = h8_to_h20(fens, fes)
    bfes = mesh_boundary(fes)
    t = sum(ts)
    fesel = fe_select(fens, bfes, box=[0, a, 0, b, t, t], inflate=htol)
    tsfes = bfes.subset(fesel)
    femms = []
    for layer in range(len(nts)):
        aangle = angles[layer] / 180. * math.pi
        mcsys = CSys(matrix=rotmat(numpy.array([0.0, 0.0, aangle])))
        el = fe_select(fens, fes, bylabel=True, label=layer)
        femms.append(FEMMDeforLinear(material=m, material_csys=mcsys, fes=fes.subset(el),
                                     integration_rule=GaussRule(dim=3, order=2)))
    sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
    return fens, femms, sfemm

#
# def elem_dep_data_h8(na, nb, nt):
#     fens, fes = h8_block(a, b, t, na, nb, nt)
#     femm = FEMMDeforLinear(material=m, material_csys=mcsys, fes=fes, integration_rule=GaussRule(dim=3, order=2))
#     bfes = mesh_boundary(femm.fes)
#     fesel = fe_select(fens, bfes, box=[a, a, 0, b, 0, t], inflate=htol)
#     tsfes = bfes.subset(fesel)
#     sfemm = FEMMDeforLinear(fes=tsfes, integration_rule=GaussRule(dim=2, order=2))
#     return fens, femm, sfemm

# models = {'T10': elem_dep_data_t10,
#           'QT10MS': elem_dep_data_qt10ms,
#           'H8MSGSO': elem_dep_data_h8msgso,
#           'H20R': elem_dep_data_h20r}
# models = {'T4': elem_dep_data_t4}
# models = {'H20R': elem_dep_data_h20r}
# models = {'H8': elem_dep_data_h8}
# models = {'H8MSGSO': elem_dep_data_h8msgso}
models = {'QT10MS': elem_dep_data_qt10ms}
# models = {'H20R': elem_dep_data_h20r}
for k, elem_dep_data in models.items():
    print(k)
    for n in range(1,3):
        fens, femms, sfemm = elem_dep_data(n * na, n * nb, n * nts)
        model_data = {}
        model_data['fens'] = fens
        model_data['regions'] = [{'femm': femm} for femm in femms]

        # algo_common.plot_mesh(model_data)

        model_data['boundary_conditions'] = {}
        # Simple supports
        essential = []
        t = sum(ts)
        cn1 = fenode_select(fens, box=numpy.array([0, 0, 0, b, 0, t]), inflate=htol)
        cn2 = fenode_select(fens, box=numpy.array([a, a, 0, b, 0, t]), inflate=htol)
        cn = numpy.vstack((cn1,cn2))
        essential.append({'node_list': cn,
                      'comp': 1,
                      'value': lambda x: 0.0
                      })
        essential.append({'node_list': cn,
                      'comp': 2,
                      'value': lambda x: 0.0
                      })
        cn1 = fenode_select(fens, box=numpy.array([0, a, 0, 0, 0, t]), inflate=htol)
        cn2 = fenode_select(fens, box=numpy.array([0, a, b, b, 0, t]), inflate=htol)
        cn = numpy.vstack((cn1, cn2))
        essential.append({'node_list': cn,
                          'comp': 0,
                          'value': lambda x: 0.0
                          })
        essential.append({'node_list': cn,
                          'comp': 2,
                          'value': lambda x: 0.0
                          })
        model_data['boundary_conditions']['essential'] = essential

        # Traction on the free end
        fi = ForceIntensity(magn=lambda x, J: numpy.array([0, 0, q0]))
        traction = [{'femm': sfemm, 'force_intensity': fi}]
        model_data['boundary_conditions']['traction'] = traction

        # Call the solver
        algo_defor_linear.statics(model_data)
        # for action, time in model_data['timings']:
        #     print(action, time, ' sec')

        geom = model_data['geom']
        u = model_data['u']

        bottom_edge = fenode_select(fens, box=[a/2.0, a/2.0, b/2.0, b/2.0, 0, t], inflate=htol)
        uv = u.values[bottom_edge, :]
        uz = numpy.mean(abs(uv[:, 2]))
        print('Center displacement uz =', uz, ', ', (uz / abs(wc_analytical) * 100), ' %')

        algo_defor_linear.plot_displacement(model_data)

        model_data['postprocessing'] = {'file': 'Z_laminate_u_ss_' + k + '_results',
                                        'outcs': CSys()}
        # algo_defor.plot_stress(model_data)
        algo_defor.plot_elemental_stress(model_data)
