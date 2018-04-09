import numpy


def fenode_to_fe_map(nfens, conn):
    """Create a map from finite element nodes to the connected finite elements.

    Map from finite element nodes to the finite elements connecting them.
    For each  node referenced in the connectivity of
    the finite element set on input, the numbers of the individual
    finite elements that reference that node are stored in a list
    the list of lists femap.
        Example: fes.conn= [7,6,5;
                            4,1,3;
                            3,7,5];
        The map reads
            femap[0] = [];#  note that node number 0 is not referenced by the connectivity
            femap[1] = [2];
            femap[2] = [];#  note that node number 2 is not referenced by the connectivity
            femap[3] = [2,3];
            femap[4] = [2];
            femap[5] = [1,3];
            femap[6] = [1];
            femap[7] = [1,3];
    The individual elements from the connectivity that reference
    node number 5 are 1 and 3, so that fes.conn(femap[5],:) lists all the
    nodes that are connected to node 5 (including node 5 itself).

    :param nfens: Total number of finite element nodes.
    :param conn: Connectivity array
    :return: Map from finite element nodes to finite elements, a list of lists.
    """
    femap = []
    n = nfens  # numpy.amax(numpy.amax(conn, axis=1), axis=0)
    for index in range(n + 1):
        femap.append([])
    for i in range(conn.shape[1]):
        for j in range(conn.shape[0]):
            ni = conn[j, i]
            femap[ni].append(j)
    return femap
