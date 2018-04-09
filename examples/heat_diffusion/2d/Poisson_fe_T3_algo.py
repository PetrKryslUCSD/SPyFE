import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy
from context import spyfe
from spyfe.meshing.generators.triangles import t3_ablock
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.integ_rules import TriRule
from spyfe.force_intensity import ForceIntensity
import time
from spyfe.algorithms.algo_heatdiff import steady_state
from spyfe.algorithms.algo_heatdiff import plot_temperature

start0 = time.time()

k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=numpy.array([[k, 0.0], [0.0, k]]))
start = time.time()
N = 50
Length, Width, nL, nW = 1.0, 1.0, N, N
fens, fes = t3_ablock(Length, Width, nL, nW)
fes.other_dimension = lambda conn, N, x: 1.0
print('Mesh generation', time.time() -  start)
model_data = {}
model_data['fens'] = fens
model_data['regions'] = [{'femm': FEMMHeatDiff(material=m, fes=fes, integration_rule=TriRule(npts=1)),
'heat_generation': ForceIntensity(magn=lambda x, J: -6.0)}]
bfes = mesh_boundary(fes)
model_data['boundary_conditions'] =  {'essential': [{'node_list': connected_nodes(bfes),
       'value': lambda x: 1.0 + x[0] ** 2 + 2 * x[1] ** 2
    }]
    }

# Call the solver
steady_state(model_data)
for action, time  in model_data['timings']:
    print(action, time, ' sec')

plot_temperature(model_data)

