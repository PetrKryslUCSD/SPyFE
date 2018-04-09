import numpy
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.volumelike import FESetT4, FESetT10

def import_mesh(filename):
    """Import tetrahedral (4- and 10-node) NASTRAN mesh.

    Limitations:
    1. only the GRID and CTETRA  sections are read.
    2. Only 4-node and 10-node tetrahedra  are handled.
    3.  The file needs to be free-form (data separated by commas).

    :param filename:
    :return:
    """
    f = open(filename, 'r')

    chunk = 5000
    nnode = 0
    node = numpy.zeros((chunk, 4))
    nelem = 0
    elem = numpy.zeros((chunk, 13), dtype=int)
    ennod = []

    while True:
        temp =f.readline()
        if temp=='':
            f.close()
            break
        temp=temp.strip()
        if temp.upper()[0:4] == 'GRID':
            # Template:
            #   GRID,1,,-1.32846E-017,3.25378E-033,0.216954
            A = temp.replace(',', ' ').split()
            node[nnode, :] = float(A[1]), float(A[2]), float(A[3]), float(A[4])
            nnode = nnode + 1
            if nnode >= node.shape[0]:
                node = numpy.vstack((node, numpy.zeros((chunk, node.shape[1]))))
        elif temp.upper()[0:6] == 'CTETRA':
            # Template:
            #                 CTETRA,1,3,15,14,12,16,8971,4853,8972,4850,8973,4848
            A = temp.replace(',', ' ').split()
            elem[nelem, 0] = int(A[1])
            elem[nelem, 1] = int(A[2])
            if len(A)==7: #  nodes per element  equals  4
                nperel = 4
            else:# nodes per element equals 10
                nperel = 10
                if len(A)<13: # the line is continued: read the next line
                    addtemp = f.readline()
                    addA = addtemp.replace(',', ' ').split()
                    A = A[3:]+addA
            for k  in range(nperel):
                elem[nelem, k + 3] = int(A[k+3])
            elem[nelem, 2] = nperel
            nelem = nelem + 1
            if nelem >= elem.shape[0]:
                elem = numpy.vstack((elem, numpy.zeros((chunk, elem.shape[1]))))


    node=node[0:nnode,:]
    elem=elem[0:nelem,:]


    if numpy.linalg.norm(numpy.array(numpy.arange(1,nnode+1))-node[:,0]) != 0:
        raise Exception('Nodes are not in serial order')


    # Process output arguments
    # Extract coordinates
    xyz=node[:,1:4]

    # Cleanup element connectivities
    ennod=numpy.unique(elem[:,2])
    if  len(ennod)!=1:
        raise Exception('This function cannot treat a mixture of element types at this point')

    if (ennod[0] != 4) and (ennod[0] != 10):
        raise Exception('Unknown element type')

    # Compensate for the Python indexing: make the connectivities zero-based
    conn=elem[:,3:3+ennod[0]] - 1
    label =elem[:,1]

    # Create output arguments. First the nodes
    fens=FENodeSet(xyz=xyz)

    # Now the geometric cells for each element set
    if ennod[0]==4:
        fes = FESetT4(conn=conn, label=label)
    else:
        fes = FESetT10(conn=conn, label=label)

    return fens, [fes]

