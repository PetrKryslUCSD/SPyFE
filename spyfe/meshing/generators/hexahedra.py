from numpy import array, hstack
import numpy
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.volumelike import FESetH8, FESetH20
from spyfe.onebased import OneBased2DArray, range_1based
from spyfe.meshing.selection import fe_select
import collections

def h8_blockx(xs, ys, zs):
    """
    Mesh of H8 elements of a rectangular 3D block.

    Mesh of a 3-D block, H8 finite elements. The nodes are located at the
    Cartesian product of the three intervals on the input.  This allows for
    construction of graded meshes.
    :param xs: array of the X coordinates of the nodes
    :param ys: array of the Y coordinates of the nodes
    :param zs: array of the Z coordinates of the nodes
    :return: fens, fes

    See also:
    """
    nL = len(xs) - 1
    nW = len(ys) - 1
    nH = len(zs) - 1
    nnodes = (nL + 1) * (nW + 1) * (nH + 1)
    ncells = (nL) * (nW)*(nH)
    X = OneBased2DArray((nnodes, 3))
    f = 1
    for k in range_1based(1, (nH + 1)):
        for j in range_1based(1, (nW + 1)):
            for i in range_1based(1, (nL + 1)):
                X[f, 0], X[f, 1], X[f, 2] = xs[i-1], ys[j-1], zs[k-1]
                f += 1

    fens = FENodeSet(X.raw_array())

    def node_numbers(i, j, k, nL, nW, nH):
        lf = (k - 1) * ((nL + 1) * (nW + 1)) + (j - 1) * (nL + 1) + i
        an = array([lf, (lf + 1), lf + (nL + 1) + 1, lf + (nL + 1)])
        return hstack((an, an + ((nL + 1) * (nW + 1))))

    conns = OneBased2DArray((ncells, 8), dtype=int)
    gc = 1
    for i in range_1based(1, nL):
        for j in range_1based(1, nW):
            for k in range_1based(1, nH):
                nn = node_numbers(i, j, k, nL, nW, nH)
                conns[gc, :] = nn[:]
                gc += 1

    fes = FESetH8(conn=conns.raw_array()-1)

    return fens, fes


def h8_block(Length, Width, Height, nL, nW, nH):
    """Mesh of a 3 -D block of H8 finite elements

    Length ,Width ,Height =dimensions of the mesh in Cartesian coordinate axes ,
    smallest coordinate in all three directions is 0 (origin )
    nL ,nW ,nH =number of elements in the three directions
    
    Range in xyz =<0 ,Length >x <0 ,Width >x <0 ,Height >
    Divided into elements :nL ,nW ,nH in the first ,second ,and
    third direction (x ,y ,z ).Finite elements of type H8 .

    :param Length:
    :param Width:
    :param Height:
    :param nL:
    :param nW:
    :param nH:
    :return: fens, fes: finite element node set and finite element set
    """
    xs = numpy.linspace(0.0, Length, nL + 1)
    ys = numpy.linspace(0.0, Width, nW + 1)
    zs = numpy.linspace(0.0, Height, nH + 1)
    return h8_blockx(xs, ys, zs)


def h8_to_h20(fens, fes):
    """Convert a mesh of hexahedra H8 to hexahedra H20.

    :param fens: Finite element node set
    :param fes: finite element set
    :return: fens, fes: finite element node set and finite element set
    """
    nedges = 12;
    ec = numpy.array([[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7], \
                      [7, 8], [8, 5], [1, 5], [2, 6], [3, 7], [4, 8]])-1

    # make a search structure for edges
    # The edge is defined by nodes 'anchor' and 'other'. The mid-edge node
    # is stored as 'mid'
    EdgeNodePair = collections.namedtuple('EdgeNodePair', ['other', 'mid'])
    edges = {}
    for i in range(fes.count()):
        for J in range(nedges):
            ev = fes.conn[i, ec[J, :]]
            anchor = numpy.amin(ev)
            other = numpy.amax(ev)
            if anchor not in edges:
                edges[anchor] = set()
            edges[anchor].add(EdgeNodePair(other, -1))

    # now generate new node number for each edge
    n = fens.count()
    for anchor, pairs in edges.items():
        pairl = list(pairs)
        newpairl = []
        for pair in pairl:
            newpairl.append(EdgeNodePair(pair.other, n))
            n = n + 1
        edges[anchor] = newpairl

    nfens = n  # number of nodes in the original mesh plus number of the edge nodes
    xyz = numpy.zeros((nfens, 3))
    xyz[0:fens.count(), :] = fens.xyz[:, :]
    # calculate the locations of the new nodes
    # and construct the new nodes
    for anchor, pairs in edges.items():
        for pair in pairs:
            xyz[pair.mid, :] = (xyz[anchor, :] + xyz[pair.other, :]) / 2.

    fens.xyz = xyz
    # construct new finite elements
    nconns = numpy.zeros((fes.count(), 20), dtype=int)
    for i in range(fes.conn.shape[0]):
        conn = fes.conn[i, :]
        econn = numpy.zeros((nedges,))
        for J in range(nedges):
            ev = conn[ec[J, :]]
            anchor = numpy.amin(ev)
            other = numpy.amax(ev)
            pairs = edges[anchor]
            for pair in pairs:
                if pair.other == other:
                    econn[J] = pair.mid
                    break
        nconns[i, 0:8] = conn
        nconns[i, 8:20] = econn
    # make new set
    fes = FESetH20(conn=nconns, label=fes.label)
    return fens, fes

def h8_composite_plate(L, W, ts, nL, nW, nts, orientation='a'):
    # H8 block mesh for a layered block (composite plate).
    #
    # function [fens,fes] = H8_composite_plate(L,W,ts,nL,nW,nts)
    #
    # L,W= length and width,
    # ts= Array of layer thicknesses,
    # nL,nW= Number of elements per length and width,
    # nts= array of numbers of elements per layer
    #
    # The fes of each layer are labeled with the layer number.
    #
    # Output:
    # fens= finite element node set
    # fes = finite element set
    #
    #
    # Examples:
    #     a=200; b=600; h=50;
    #     angles =[0,90,0];
    #     nLayers =length(angles);
    #     na=4; nb=4;
    #     nts= 1*ones(nLayers,1);# number of elements per layer
    #     ts= h/nLayers*ones(nLayers,1);# layer thicknesses
    #     [fens,fes] = H8_composite_plate(a,b,ts,na,nb,nts);;
    #     gv=drawmesh( {fens,subset(fes,fe_select(fens,fes,struct('label', 1)))},'fes', 'facecolor','r');
    #     gv=drawmesh( {fens,subset(fes,fe_select(fens,fes,struct('label', 2)))},'gv',gv,'fes', 'facecolor','g');
    #     gv=drawmesh( {fens,subset(fes,fe_select(fens,fes,struct('label', 3)))},'gv',gv,'fes', 'facecolor','b');
    #
    #
    # See also: H8_block
    sumnt = numpy.sum(nts)
    fens, fes = h8_block(nL, nW, sumnt, nL, nW, sumnt)
    label = numpy.zeros((fes.count(), 1), dtype=int)
    tnt = 0.0
    for layer, nt  in enumerate(nts):
        box = numpy.array([0, nL, 0, nW, tnt, tnt + nt])
        el = fe_select(fens, fes, box=box, inflate=0.01)
        label[el] = layer
        tnt+=nt
    fes.label = label
    t=numpy.sum(ts)
    fens.xyz[:, 0] *= L / nL
    fens.xyz[:, 1] *= W / nW
    fens.xyz[:, 2] *= t / sumnt
    return fens, fes
