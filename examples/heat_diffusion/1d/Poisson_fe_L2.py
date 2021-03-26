import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.intervals import l2_blockx
from spyfe.meshing.modification import mesh_boundary
from spyfe.meshing.selection import connected_nodes
import numpy
from numpy import array
from spyfe.materials.mat_heatdiff import MatHeatDiff
from spyfe.femms.femm_heatdiff import FEMMHeatDiff
from spyfe.fields.nodal_field import NodalField
from spyfe.integ_rules import GaussRule
from spyfe.force_intensity import ForceIntensity
from scipy.sparse.linalg import spsolve

import time
from spyfe.meshing.exporters.vtkexporter import vtkexport

# These are the constants in the problem, k is kappa
boundaryf = lambda x: 0.0 
Q = 6  # internal heat generation rate
k = 1.0  # thermal conductivity
m = MatHeatDiff(thermal_conductivity=array([k]).reshape((1, 1)))
CS = 1.0  # cross-section area 

N = 3
xs = numpy.linspace(0.0, 1.0, N+1)
fens, fes = l2_blockx(xs)
fes.other_dimension = lambda conn, N, x: CS

bfes = mesh_boundary(fes)
cn = connected_nodes(bfes)
geom = NodalField(fens=fens)
temp = NodalField(nfens=fens.count(), dim=1)
for index  in cn:
    temp.set_ebc([index], val=boundaryf(fens.xyz[index, 0]))
temp.apply_ebc()
femm = FEMMHeatDiff(material=m, fes=fes, integration_rule=GaussRule(dim=1, order=1))
temp.numberdofs()

fi = ForceIntensity(magn=lambda x, J: Q)
F = femm.distrib_loads(geom, temp, fi, 3)

F += femm.nz_ebc_loads_conductivity(geom, temp)

K = femm.conductivity(geom, temp)

temp.scatter_sysvec(spsolve(K, F))


print('Done')

print(temp.values.T)

# vtkexport("Poisson_fe_Q4_results", fes, geom, {'temperature': temp})
