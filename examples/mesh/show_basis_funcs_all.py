import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
from spyfe.meshing.generators.triangles import t3_ablock
import numpy
from numpy import array, arange
from spyfe.fields.nodal_field import NodalField
from spyfe.fields.gen_field import GenField
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


Length, Width, nL, nW = 10.0, 10.0, 7, 8
fens, fes = t3_ablock(Length, Width, nL, nW)
geom = NodalField(nfens=fens.count(), dim=3)
for index  in range(fens.count()):
    for j  in range(2):
        geom.values[index, j] = fens.xyz[index, j]
#vtkexport("show_basis_funcs-geom", fes, geom)

#bf1 = GenField(data = geom.values)
#bf1.values[0, 2] = 1.0
#vtkexport("show_basis_funcs-bf1", fes, bf1)

# Plotting
#%matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# setup three 1-d arrays for the x-coord, the y-coord, and the z-coordinate
xs = geom.values[:, 0].reshape(fens.count(),)# one value per node
ys = geom.values[:, 1].reshape(fens.count(),)# one value per node
zs = geom.values[:, 2].reshape(fens.count(),)# one value per node
plt.xlabel('x (m)')
plt.ylabel('y (m)')
for index  in numpy.random.permutation(arange(0, len(zs))):
    zs[index] = 1.0
    ax.plot_trisurf(xs, ys, triangles = fes.conn, Z = zs)
    plt.pause(0.1)
    ax.clear()
ax.plot_trisurf(xs, ys, triangles = fes.conn, Z = zs)
plt.pause(2.1)
