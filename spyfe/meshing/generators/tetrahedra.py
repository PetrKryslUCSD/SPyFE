import numpy
from numpy import array
import scipy
from scipy import random
from scipy import spatial
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.volumelike import FESetT4, FESetT10
from spyfe.onebased import OneBased2DArray, range_1based
from spyfe.meshing.selection import fe_select
import collections


def t4_blockx(xs, ys, zs, orientation='a'):
    """Graded tetrahedral (T4) mesh of a rectangular block.
    
    :param xs: 
    :param ys: 
    :param zs: 
    :param orientation: 'a', 'b', 'ca', 'cb'
    :return: 
    """

    nL = len(xs) - 1
    nW = len(ys) - 1
    nH = len(zs) - 1
    nnodes = (nL + 1) * (nW + 1) * (nH + 1)
    xyzs = OneBased2DArray((nnodes, 3))

    if (orientation == 'a'):
        t4ia = numpy.array([[1, 8, 5, 6],
                            [3, 4, 2, 7],
                            [7, 2, 6, 8],
                            [4, 7, 8, 2],
                            [2, 1, 6, 8],
                            [4, 8, 1, 2]
                            ]) - 1
        t4ib = numpy.array([[1, 8, 5, 6],
                            [3, 4, 2, 7],
                            [7, 2, 6, 8],
                            [4, 7, 8, 2],
                            [2, 1, 6, 8],
                            [4, 8, 1, 2]
                            ]) - 1
    elif (orientation == 'b'):
        t4ia = numpy.array([[2, 7, 5, 6],
                            [1, 8, 5, 7],
                            [1, 3, 4, 8],
                            [2, 1, 5, 7],
                            [1, 2, 3, 7],
                            [3, 7, 8, 1]
                            ]) - 1
        t4ib = numpy.array([[2, 7, 5, 6],
                            [1, 8, 5, 7],
                            [1, 3, 4, 8],
                            [2, 1, 5, 7],
                            [1, 2, 3, 7],
                            [3, 7, 8, 1]
                            ]) - 1
    elif (orientation == 'ca'):
        t4ia = numpy.array([[8, 4, 7, 5],
                            [6, 7, 2, 5],
                            [3, 4, 2, 7],
                            [1, 2, 4, 5],
                            [7, 4, 2, 5]
                            ]) - 1
        t4ib = numpy.array([[7, 3, 6, 8],
                            [5, 8, 6, 1],
                            [2, 3, 1, 6],
                            [4, 1, 3, 8],
                            [6, 3, 1, 8]
                            ]) - 1
    elif (orientation == 'cb'):
        t4ib = numpy.array([[8, 4, 7, 5],
                            [6, 7, 2, 5],
                            [3, 4, 2, 7],
                            [1, 2, 4, 5],
                            [7, 4, 2, 5]
                            ]) - 1
        t4ia = numpy.array([[7, 3, 6, 8],
                            [5, 8, 6, 1],
                            [2, 3, 1, 6],
                            [4, 1, 3, 8],
                            [6, 3, 1, 8]
                            ]) - 1
    else:
        raise Exception('Unknown orientation')

    ncells = t4ia.shape[0] * (nL) * (nW) * (nH)
    conns = OneBased2DArray((ncells, 4))

    f = 1
    for k in range_1based(1, nH + 1):
        for j in range_1based(1, nW + 1):
            for i in range_1based(1, nL + 1):
                xyzs[f, :] = xs[i - 1], ys[j - 1], zs[k - 1]
                f += 1

    fens = FENodeSet(xyzs.raw_array())

    def node_numbers(i, j, k, nL, nW, nH):
        f = (k - 1) * ((nL + 1) * (nW + 1)) + (j - 1) * (nL + 1) + i
        return array([f, (f + 1), f + (nL + 1) + 1, f + (nL + 1),
                      f + ((nL + 1) * (nW + 1)), (f + 1) + ((nL + 1) * (nW + 1)),
                      f + (nL + 1) + 1 + ((nL + 1) * (nW + 1)),
                      f + (nL + 1) + ((nL + 1) * (nW + 1))])

    gc = 1
    for i in range_1based(1, nL):
        for j in range_1based(1, nW):
            for k in range_1based(1, nH):
                nn = node_numbers(i, j, k, nL, nW, nH)
                if ((i + j + k) % 2 == 0):
                    t4i = t4ib
                else:
                    t4i = t4ia
                for r in range(t4i.shape[0]):
                    conns[gc, :] = nn[t4i[r, :]]
                    gc += 1

    c = numpy.array(conns.raw_array() - 1, dtype=int)
    fes = FESetT4(conn=c)
    return fens, fes


def t4_block(L, W, H, nL, nW, nH, orientation='a'):
    """Tetrahedral mesh of a block (structures).
    
    :param L: 
    :param W: 
    :param H: 
    :param nL: 
    :param nW: 
    :param nH: 
    :param orientation: 'a', 'b', 'ca', 'cb'
    :return: 
    """
    xs = numpy.linspace(0.0, L, nL + 1)
    ys = numpy.linspace(0.0, W, nW + 1)
    zs = numpy.linspace(0.0, H, nH + 1)
    return t4_blockx(xs, ys, zs, orientation)


def t4_to_t10(fens, fes):
    """Convert a mesh of Tetrahedron T4 (four-node) to Tetrahedron T10.
    # 
    #
    # function [fens,fes] = T4_to_T10(fens,fes)
    #
    # Examples: 
    # [fens,fes] = T4_sphere(3.1,1)
    # [fens,fes] = T4_to_T10(fens,fes)
    # fens= onto_sphere(fens,3.1,connected_nodes(mesh_boundary(fes,[])))
    # figure drawmesh({fens,fes},'fes','facecolor','y') hold on
    """

    nedges = 6
    ec = numpy.array([[1, 2], [2, 3], [3, 1], [4, 1], [4, 2], [4, 3]])-1

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
    # New definition of the nodes
    fens.xyz = xyz

    # construct new finite elements
    nconns = numpy.zeros((fes.count(), 10), dtype=int)
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
        nconns[i, 0:4] = conn
        nconns[i, 4:10] = econn
    # make new set
    fes = FESetT10(conn=nconns, label=fes.label)
    return fens, fes


def t4_blockdel(Length, Width, Height, nL, nW, nH):
    """Tetrahedral (T4) Delaunay Mesh of a rectangular block.
    
    :param Length: 
    :param Width: 
    :param Height: 
    :param nL: 
    :param nW: 
    :param nH: 
    :return: 
    """

    tol = 1 / 100
    mtol = 80 * tol

    fens, fes = t4_block(nL, nW, nH, nL, nW, nH)
    xs = fens.xyz
    for i in range(0, fens.count()):
        if (abs(xs[i, 0]) > tol) and (abs(xs[i, 0] - nL) > tol):
            xs[i, 0] = xs[i, 0] + mtol * (random.random() - 0.5)
        if (abs(xs[i, 1]) > tol) and (abs(xs[i, 1] - nW) > tol):
            xs[i, 1] = xs[i, 1] + mtol * (random.random() - 0.5)
        if (abs(xs[i, 2]) > tol) and (abs(xs[i, 2] - nH) > tol):
            xs[i, 2] = xs[i, 2] + mtol * (random.random() - 0.5)
    # Compute the triangulation
    tri = scipy.spatial.Delaunay(fens.xyz)
    # Rescale the coordinates to fill the box
    xs[:,0] *= Length / nL
    xs[:,1] *= Width / nW
    xs[:,2] *= Height / nH
    # Check the volumes to make sure they are all positive
    def tetvol(v):
        X = numpy.vstack((v[1,:]-v[0,:], v[2,:]-v[0,:], v[3,:]-v[0,:]))
        return numpy.linalg.det(X)/6.0
    for i  in range(tri.simplices.shape[0]):
        if tetvol(xs[tri.simplices[i,:],:]) < 0:
            tri.simplices[i, :] = tri.simplices[i, (0,2,1,3)]
        assert not tetvol(xs[tri.simplices[i,:],:]) == 0.0

    # Return the generated mesh
    fes = FESetT4(conn=tri.simplices)
    return fens, fes


def t4_composite_plate(L, W, ts, nL, nW, nts, orientation='a'):
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
    fens, fes = t4_block(nL, nW, sumnt, nL, nW, sumnt, orientation=orientation)
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


