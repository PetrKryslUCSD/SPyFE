import numpy
from spyfe.meshing.boxes import inflate_box, bounding_box, in_box
from spyfe.fenode_set import FENodeSet

def mesh_boundary(fes):

    """Extract the boundary finite elements from a mesh.

    Extract the finite elements of manifold dimension (n-1) from the
    supplied list of finite elements of manifold dimension (n).
    options = struct with any attributes that should be passed to the
    construct or of the boundary finite elements

    :param fes: Finite element set.
    :return: Boundary finite element set.
    """
    make = fes.boundary_fe_type()
    # Form all hyperfaces, non-duplicates are boundary hyper faces
    hypf = fes.boundary_conn()
    shypf = numpy.sort(hypf, axis=1)
    rix = numpy.lexsort(shypf[:, ::-1].T)
    shypf = shypf[rix, :]
    n = shypf.shape[0]
    d = shypf[0:(n - 1), :] != shypf[1:n, :]
    adr = numpy.array([True]).reshape((1, 1))
    anyd = numpy.any(d, axis=1).reshape((d.shape[0], 1))
    ad0 = numpy.vstack((adr, anyd))
    ad1 = numpy.vstack((ad0[1:, :], adr))
    iu = (ad0 & ad1).ravel()
    bdryconn = hypf[rix[iu], :]
    return make(conn=bdryconn)

def fuse_nodes(fens1, fens2, tolerance=0.0):
    # Fuse together nodes from to node sets.
    #
    # function [fens,new_indexes_of_fens1_nodes] = fuse_nodes(fens1, fens2, tolerance)
    #
    # Fuse two node sets. If necessary, by gluing together nodes located within tolerance of each other.
    # The two node sets, fens1 and fens2,  are fused together by
    # merging the nodes that fall within a box of size "tolerance".
    # The merged node set, fens, and the new  indexes of the nodes
    # in the set fens1 are returned.
    #
    # The set fens2 will be included unchanged, in the same order,
    # in the node set fens.
    #
    # The indexes of the node set fens1 will have changed.
    #
    # Example: 
    # After the call to this function we have
    # k=new_indexes_of_fens1_nodes(j) is the node in the node set fens which
    # used to be node j in node set fens1.
    # The finite element set connectivity that used to refer to fens1
    # needs to be updated to refer to the same nodes in  the set fens as
    #     fes = update_conn(fes ,new_indexes_of_fens1_nodes)
    #
    # See also: merge_nodes, update_conn
    #

    # I need to have the node number as non-zero to mark the replacement
    # when fused
    xyz1 = fens1.xyz
    id1 = numpy.array(range(1, fens1.count() + 1))
    xyz2 = fens2.xyz
    id2 = numpy.array(range(1, fens2.count() + 1))
    c1 = numpy.ones((xyz2.shape[0], 1))  # column matrix
    # Mark nodes from the first array that are duplicated in the second
    if (tolerance > 0):  # should we attempt to merge nodes?
        Box2 = inflate_box(bounding_box(xyz2), tolerance)
        for i in range(fens1.count()):
            XYZ = xyz1[i, :].reshape(1, xyz1.shape[1])  # row matrix
            # Note  that we are looking for  distances  of this node to nodes in the OTHER node set
            if in_box(Box2, XYZ):  # This check makes this run much faster
                xyzd = abs(xyz2 - c1 * XYZ)  # find the distances along  coordinate directions
                dist = numpy.sum(xyzd, axis=1)
                jx = dist < tolerance
                if numpy.any(jx):
                    minn = numpy.argmin(dist)
                    id1[i] = -id2[minn]

    # Generate  fused arrays of the nodes
    xyzm = numpy.zeros((fens1.count() + fens2.count(), xyz1.shape[1]))
    xyzm[0:fens2.count(), :] = xyz2  # fens2 are there without change
    mid = fens2.count()+1
    for i in range(fens1.count()):  # and then we pick only non-duplicated fens1
        if id1[i] > 0:
            id1[i]  = mid
            xyzm[mid-1, :] = xyz1[i, :]
            mid = mid + 1
        else:
            id1[i] = id2[-id1[i]-1]

    nfens = mid - 1
    xyzm = xyzm[0:nfens, :]

    # Create the fused Node set
    fens = FENodeSet(xyz=xyzm)
    # The Node set 1 numbering will change
    new_indexes_of_fens1_nodes = id1 - 1  # go back to zero-based indexes
    # The node set 2 numbering stays the same, node set 1 will need to be
    # renumbered
    return fens, new_indexes_of_fens1_nodes

def merge_meshes(fens1, fes1, fens2, fes2, tolerance):
    """Merge together two meshes.

        Merge two meshes together by gluing together nodes within tolerance. The
    two meshes, fens1, fes1, and fens2, fes2, are glued together by merging
    the nodes that fall within a box of size "tolerance". If tolerance is set
    to zero, no merging of nodes is performed; the two meshes are simply
    concatenated together.

    The merged node set, fens, and the two arrays of finite elements with
    renumbered  connectivities are returned.

    Important notes: On entry into this function the connectivity of fes1
    point into fens1 and the connectivity of fes2 point into fens2. After
    this function returns the connectivity of both fes1 and fes2 point into
    fens. The order of the nodes of the node set fens1 in the resulting set
    fens will have changed, whereas the order of the nodes of the node set
    fens2 is are guaranteed to be the same. Therefore, the connectivity of
    fes2 will in fact remain the same.
    :param fens1:
    :param fes1:
    :param fens2:
    :param fes2:
    :param tolerance:
    :return:
    """

    # Fuse the nodes
    fens, new_indexes_of_fens1_nodes = fuse_nodes(fens1, fens2, tolerance)

    # Renumber the finite elements
    fes1.update_conn(new_indexes_of_fens1_nodes)
    # Note that now the connectivity of both fes1 and fes2 point into
    # fens.
    return fens, fes1, fes2
