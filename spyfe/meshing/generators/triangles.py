from numpy import array

from spyfe.fenode_set import FENodeSet
from spyfe.fesets.surfacelike import FESetT3
from spyfe.onebased import OneBased2DArray, range_1based

def t3_ablock(Length, Width, nL, nW):
    """
    Mesh of T3 elements of a rectangle.  Alternate orientation.

    :param Length:
    :param Width:
    :param nL:
    :param nW:
    :param options:
    :return:

    See also:  T3_blocku, T3_cblock,   T3_crossblock, T3_ablock,
               T3_ablockc, T3_ablockn, T3_block, T3_blockc, T3_blockn
    """
    nnodes = (nL + 1) * (nW + 1)
    ncells = 2 * (nL) * (nW)
    xs = OneBased2DArray((nnodes, 2))
    conns = OneBased2DArray((ncells, 3), dtype=int)
    f = 1
    for j in range_1based(1, (nW + 1)):
        for i in range_1based(1, (nL + 1)):
            xs[f, 0], xs[f, 1] = (i - 1) * Length / nL, (j - 1) * Width / nW
            f += 1

    fens = FENodeSet(xs.raw_array())

    def node_numbers1(i, j, nL, nW):
        f = (j - 1) * (nL + 1) + i
        return array([f, (f + 1), f + (nL + 1) + 1])

    def node_numbers2(i, j, nL, nW):
        f = (j - 1) * (nL + 1) + i
        return array([f, f + (nL + 1) + 1, f + (nL + 1)])

    gc = 1
    for i in range_1based(1, nL):
        for j in range_1based(1, nW):
            nn = node_numbers1(i, j, nL, nW)
            conns[gc, :] = nn[:]
            gc += 1
            nn = node_numbers2(i, j, nL, nW)
            conns[gc, :] = nn[:]
            gc += 1

    fes = FESetT3(conn=conns.raw_array()-1)

    return fens, fes
