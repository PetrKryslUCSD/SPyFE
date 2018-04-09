import numpy
from spyfe.fenode_set import FENodeSet
from spyfe.fesets.volumelike import FESetT4, FESetT10
from spyfe.fesets.volumelike import FESetH8, FESetH20
import copy

def import_mesh(filename):
    """Import Abaqus mesh.
    
    Import tetrahedral (4- and 10-node) or hexahedral (8- and 20-node) ABAQUS Mesh (.INP).
    Limitations: only the *NODE and *ELEMENT  sections are read. Only 3D elements are handled.
    :param filename: 
    :return: fens, feslist
    """

    f = open(filename, 'r')

    chunk = 1000
    node = numpy.zeros((chunk, 4))

    # Find the node section
    while True:
        temp = f.readline()
        if temp == '':
            f.close()
            raise Exception('No nodes in mesh file?')
        temp = temp.strip()
        if temp[0:5].upper() == '*NODE':
            break  # now we process the *NODE section

    # We have the node section, now we need to read the data
    nnode = 0
    while True:
        temp = f.readline()
        temp = temp.strip()
        if temp[0]=='*':
            break # done with reading nodes
        A = temp.replace(',', ' ').split()
        node[nnode, :] = float(A[0]), float(A[1]), float(A[2]), float(A[3])
        nnode = nnode + 1
        if nnode >= node.shape[0]:
            node = numpy.vstack((node, numpy.zeros((node.shape[1] + chunk, node.shape[1]))))

    # Now find and process all *ELEMENT blocks
    More_data=True
    elsets = []
    while More_data:
        # Find the next block
        while temp[0]!='*':
            if len(temp)>=8 and temp.upper()[0:8] == '*ELEMENT':
                break # got it
            temp = f.readline()
            if temp == '':
                f.close()
                More_data=False
                break
            temp = temp.strip()

        Valid, Type, Elset = _Parse_element_line(temp)

        if (Valid):  # Valid element type
            nelem = 0
            ennod = Type
            elem = numpy.zeros((chunk, ennod), dtype=int)
            while True:
                temp = f.readline()
                if temp == '':
                    f.close()
                    More_data = False
                    break
                if temp[0] == '*':
                    elem = elem[0:nelem, :]
                    elsets.append((nelem, copy.deepcopy(elem), Elset))
                    break  # done with reading this element block
                A = temp.replace(',',' ').split()
                A = A[1:]  # get rid of the element serial number
                for k in range(len(A)):
                    elem[nelem, k] = int(A[k])
                if len(A) < ennod: # we need to read a continuation line
                    prevn = len(A)
                    temp = f.readline()
                    if temp == '':
                        f.close()
                        raise Exception('Premature end of element line')
                    A = temp.replace(',',' ').split()
                    for k in range(len(A)):
                        elem[nelem, prevn + k] = int(A[k])
                    if prevn + len(A) != ennod:
                        raise Exception('Wrong number of element nodes')
                nelem += 1
                if nelem >= elem.shape[0]:
                    elem = numpy.vstack((elem, numpy.zeros((chunk, elem.shape[1]))))
        else:
            temp = f.readline()
            if temp == '':
                More_data = False
                break

    # We are done with the file.
    f.close()
    # Some error checks
    node = node[0:nnode, :]
    if numpy.linalg.norm(numpy.array(numpy.arange(1, nnode + 1)) - node[:, 0]) != 0:
        raise Exception('Nodes are not in serial order')
    # Process output arguments
    # Extract coordinates
    fens=FENodeSet(xyz=node[:, 1:4])
    # Now create all element sets
    feslist = []
    for j, e in enumerate(elsets):
        nelem, elem, ElSet = e
        conn = elem - 1
        if elem.shape[1] == 4:
            fes = FESetT4(conn=conn, label=j)
        elif elem.shape[1] == 10:
            fes = FESetT10(conn=conn, label=j)
        elif elem.shape[1] == 8:
            fes = FESetH8(conn=conn, label=j)
        elif elem.shape[1] == 20:
            fes = FESetH20(conn=conn, label=j)
        feslist.append(fes)

    return fens, feslist


def _Parse_element_line(Str):
    Valid = False
    Type = 0
    Elset = ''
    tokens = Str.split(',')
    if tokens[0].upper() == '*ELEMENT':
        if (len(tokens) >= 2):
            tok1 = tokens[1].split('=')
            if tok1[0].upper().strip() == 'TYPE':
                if tok1[1].upper().strip() == 'C3D4':
                    Type = 4
                elif tok1[1].upper().strip()[0:5] == 'C3D10':
                    Type = 10
                elif tok1[1].upper().strip()[0:5] == 'C3D20':
                    Type = 20
                elif tok1[1].upper().strip()[0:4] == 'C3D8':
                    Type = 8
                else:
                    return
        if (len(tokens) >= 3):
            tok1 = tokens[2].split(',')
            if tok1[0] == 'ELSET':
                Elset = tok1[1]
        Valid = True
    return Valid, Type, Elset
