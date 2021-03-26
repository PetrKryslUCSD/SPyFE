import numpy
from numpy import array
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.curvelike import FESetL2
from spyfe.onebased import OneBased2DArray, range_1based
import copy
import math

def l2_blockx(xs):
    """
    Mesh of L2 elements of an interval.

    Mesh of a 1-D block, L2 finite elements. This allows for
    construction of graded meshes.
    :param xs: array of the X coordinates of the nodes
    :return: fens, fes

    See also:
    """
    nL = len(xs) - 1
    nnodes = (nL + 1) 
    ncells = (nL) 
    X = OneBased2DArray((nnodes, 1))
    f = 1
    for j in range_1based(1, (nL + 1)):
        X[f, 0] = xs[j-1]
        f += 1

    fens = FENodeSet(X.raw_array())

    conns = OneBased2DArray((ncells, 2), dtype=int)
    gc = 1
    for i in range_1based(1, nL):
        conns[gc, 0] = i
        conns[gc, 1] = i+1
        gc += 1

    fes = FESetL2(conn=conns.raw_array()-1)

    return fens, fes


def l2_block(Length, nL):
    """Mesh of a interval, L2 elements

    :param Length:
    :param nL:
    :return:
    """
    xs = numpy.linspace(0.0, Length, nL + 1)
    return l2_blockx(xs)

