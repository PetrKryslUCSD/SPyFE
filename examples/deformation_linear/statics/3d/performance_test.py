import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from context import spyfe
import math
import numpy
import time
from spyfe.meshing.exporters.vtkexporter import vtkexport


from spyfe.cyfuns import three_matrix_product

nr = 24
b = numpy.array(numpy.arange(6*nr), dtype=float).reshape((6,nr))
d = numpy.array(numpy.arange(6*6), dtype=float).reshape((6,6))
k1 = numpy.zeros((nr, nr), dtype=float)
k = numpy.zeros((nr, nr), dtype=float)
nl = 1000000

start = time.time()
for index  in range(nl):
    k[:,:] = numpy.dot(b.T, numpy.dot(d, b))
    #three_matrix_product(k1, b.T, d, b)

print('Done k[:,:]', time.time() - start)

start = time.time()
for index  in range(nl):
    k = numpy.dot(b.T, numpy.dot(d, b))
    #three_matrix_product(k1, b.T, d, b)

print('Done k', time.time() - start)

start = time.time()
for index  in range(nl):
    numpy.dot(b.T, numpy.dot(d, b), out=k)
    #three_matrix_product(k1, b.T, d, b)
    
print('Done out', time.time() - start)

# start = time.time()
# for index  in range(nl):
#     three_matrix_product(k1, b.T, d, b)
#
# print('Done three_matrix_product', time.time() - start)