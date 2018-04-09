import numpy
from numpy import array
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.surfacelike import FESetQ4, FESetQ8
from spyfe.onebased import OneBased2DArray, range_1based
import copy
import math
from spyfe.meshing.shaping import shape_to_annulus

def q4_blockx(xs, ys):
    """
    Mesh of Q4 elements of a rectangle.

    Mesh of a 2-D block, Q4 finite elements. The nodes are located at the
    Cartesian product of the two intervals on the input.  This allows for
    construction of graded meshes.
    :param xs: array of the X coordinates of the nodes
    :param ys: array of the Y coordinates of the nodes
    :return: fens, fes

    See also:
    """
    nL = len(xs) - 1
    nW = len(ys) - 1
    nnodes = (nL + 1) * (nW + 1)
    ncells = (nL) * (nW)
    X = OneBased2DArray((nnodes, 2))
    f = 1
    for j in range_1based(1, (nW + 1)):
        for i in range_1based(1, (nL + 1)):
            X[f, 0], X[f, 1] = xs[i-1], ys[j-1]
            f += 1

    fens = FENodeSet(X.raw_array())

    def node_numbers(i, j, nL, nW):
        f = (j - 1) * (nL + 1) + i
        return array([f, (f + 1), f + (nL + 1) + 1, f + (nL + 1)]).ravel()

    conns = OneBased2DArray((ncells, 4), dtype=int)
    gc = 1
    for i in range_1based(1, nL):
        for j in range_1based(1, nW):
            nn = node_numbers(i, j, nL, nW)
            conns[gc, :] = nn[:]
            gc += 1

    fes = FESetQ4(conn=conns.raw_array()-1)

    return fens, fes


def q4_block(Length, Width, nL, nW):
    """Mesh of a rectangle, Q4 elements

    :param Length:
    :param Width:
    :param nL:
    :param nW:
    :return:
    """
    xs = numpy.linspace(0.0, Length, nL + 1)
    ys = numpy.linspace(0.0, Width, nW + 1)
    return q4_blockx(xs, ys)


def q4_to_q8(fens, fes):
    """
    
    :param fens:
    :param fes:
    :return:
    """
    # Convert a mesh of quadrilateral Q4 to quadrilateral Q8.
    #
    # function [fens,fes] = Q4_to_Q8(fens,fes,options)
    #
    # options =attributes recognized by the constructor fe_set_Q8
    #
    # Examples:
    #     R=8.5
    #     [fens,fes]=Q4_sphere(R,1,1.0)
    #     [fens,fes] = Q4_to_Q8(fens,fes,[])
    #     fens= onto_sphere(fens,R,[])
    #     drawmesh({fens,fes},'nodes','fes','facecolor','y', 'linewidth',2) hold on
    #
    # See also: fe_set_Q8

    nedges=4
    ec = array([[0, 1],
                [1, 2],
                [2, 3],
                [3, 0]
                ])
    # make a search structure for edges
    edges = {}
    for i in range(fes.conn.shape[0]):
        conn = fes.conn[i, :]
        for J in range(nedges):
            ev = conn[ec[J, :]]
            anchor = numpy.amin(ev)
            otherv = numpy.amax(ev)
            if anchor not in edges:
                edges[anchor] = set([otherv])
            else:
                edges[anchor].add(otherv)

    # now generate new node number for each edge
    nodes = copy.deepcopy(edges)
    n = fens.count()
    for anchor, othervs in edges.items():
        nnew = []
        for index in othervs:
            nnew.append(n)
            n += 1
        nodes[anchor] = nnew

    xyz1 = fens.xyz
    xyz = numpy.zeros((n, xyz1.shape[1]))
    xyz[0:xyz1.shape[0], :] = xyz1[:, :]
    # calculate the locations of the new nodes
    # and construct the new nodes
    for anchor, othervs in edges.items():
        nnew = nodes[anchor]
        othervs = list(othervs)
        for J in range(len(othervs)):
            e = othervs[J]
            xyz[nnew[J], :] = (xyz[anchor, :] + xyz[e,:]) / 2

    # construct new finite elements
    nconns =numpy.zeros((fes.count(),8), dtype=int)
    for i in range(fes.conn.shape[0]):
        conn = fes.conn[i,:]
        econn = numpy.zeros((nedges,))
        for J in range(nedges):
            ev = conn[ec[J, :]]
            anchor = numpy.amin(ev)
            otherv = numpy.amax(ev)
            nnew  = nodes[anchor]
            othervs = list(edges[anchor])
            for k in range(len(othervs)):
                if othervs[k]==otherv:
                    econn[J] = nnew[k]
                    break
        nconns[i, 0:4] = conn
        nconns[i, 4:8] = econn

    fens = FENodeSet(xyz)
    fes =FESetQ8(conn=nconns, label=fes.label)
    return fens, fes

def q4_annulus(rin, rex, nr, nc, Angl):
    """Mesh of an annulus segment.

    Note that if you wish to have an annular region with 360° development
    angle  (closed annulus), the nodes along the slit  need to be fused.
    :param rin: Internal radius
    :param rex: External radius
    :param nr: Number of elements radially
    :param nc: Number of elements circumferentially
    :param Angl: Angle of development.
    :param thickness:
    :return: fens,fes Nodes and elements.
    .. seealso:: fuse_nodes
    """
    fens, fes = q4_block(rex - rin, Angl, nr, nc)
    return shape_to_annulus(fens, fes, rin, rex, Angl)
    # trin = min(rin, rex)
    # trex = max(rin, rex)
    # fens, fes = Q4_block(trex - trin, Angl, nr, nc)
    # for i in range(fens.count()):
    #     r = trin + fens.xyz[i, 0]
    #     a = fens.xyz[i, 1]
    #     fens.xyz[i, 0] = r * math.cos(a)
    #     fens.xyz[i, 1] = r * math.sin(a)
    # return fens, fes
