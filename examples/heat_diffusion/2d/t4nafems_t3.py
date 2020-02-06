import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy
from context import spyfe
from spyfe.meshing.generators.triangles import t3_ablock
from spyfe.meshing.boxes import bounding_box
from spyfe.meshing.modification import mesh_boundary, merge_meshes
from spyfe.meshing.selection import connected_nodes, fe_select, fenode_select
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.femms.femm_heatdiff_surface import FEMMHeatDiffSurface
from spyfe.integ_rules import TriRule, GaussRule
from spyfe.force_intensity import ForceIntensity
import time
from spyfe.algorithms.algo_heatdiff import steady_state
from spyfe.algorithms.algo_heatdiff import plot_temperature

start0 = time.time()

k = 52.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=numpy.array([[k, 0.0], [0.0, k]]))
start = time.time()
Width, nW = 0.6, 20
Heightb, nHb = 0.2, 20
fens1, fes1 = t3_ablock(Width, Heightb, nW, nHb)
Heightt, nHt = 0.8, 20
fens2, fes2 = t3_ablock(Width, Heightt, nW, nHt)
fens2.xyz[:,1]+=Heightb
tolerance = Heightb/ nHb/100
fens, fes1, fes2 = merge_meshes(fens1, fes1, fens2, fes2, tolerance)
fes = fes1.cat(fes2)
fes.other_dimension = lambda conn, N, x: 1.0
bfes = mesh_boundary(fes)
bottom= fe_select(fens, bfes,
              facing=True, direction=lambda x: numpy.array([0.0, -1.0]), tolerance=0.99)

right= fe_select(fens, bfes,
              facing=True, direction=lambda x: numpy.array([1.0, 0.0]), tolerance=0.99)
top =  fe_select(fens, bfes,
              facing=True, direction=lambda x: numpy.array([0.0, +1.0]), tolerance=0.99)
topright = numpy.concatenate((top, right))
print('Mesh generation', time.time() - start)
model_data = {}
model_data['fens'] = fens
model_data['regions'] \
    = [{'femm': FEMMHeatDiff(material=m, fes=fes,
                             integration_rule=TriRule(npts=1))}
       ]
model_data['boundary_conditions'] \
    = {'essential': [{'node_list': connected_nodes(bfes.subset(bottom)),
                      'temperature': lambda x: 100.0
                      }],
       'surface_transfer': [{'femm': FEMMHeatDiffSurface(fes=bfes.subset(topright),
                                                         integration_rule=GaussRule(dim=1, order=1),
                                                         surface_transfer_coeff=lambda x: 750.0),
                             'ambient_temperature': lambda x: 0.0
                             }]
       }

# Call the solver
steady_state(model_data)
print(model_data['timings'])
nodeA = fenode_select(fens, box=bounding_box([Width, Heightb]), inflate= Width/1e6)
print('Temperature at nodeA (', nodeA, ')=', model_data['temp'].values[nodeA])
# Visualize the temperature
plot_temperature(model_data)

