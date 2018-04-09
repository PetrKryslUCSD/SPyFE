import numpy
import copy

ErrorNumberOfNodesSetJustOnce = Exception('Number of nodes may be set just once')
ErrorWrongLabels = Exception('Wrong labels')
ErrorWrongType = Exception('The connectivity is the wrong type (not one-based array)')
ErrorWrongJacobianMatrix = Exception('Wrong Jacobian matrix')
ErrorWrongNumberOfNodes = Exception('Wrong number of nodes')
ErrorWrongDimension = Exception('Wrong dimension')

class FESet:
    """
    Finite element(FE) set

    This class is the base class for FE sets, i.e. every usable FE set has to be
    derived from it.
    """

    def __init__(self, nfens=0, conn=None, label=None):
        self._nfens = nfens
        self._conn = None #Initialize the input connectivity
        self._label = None  # Initialize the input label
        if conn is not None: # If actually supplied
            self.set_conn(conn) # input connectivity
            self.label = label

    def copy(self):
        """Make a copy of the finite element set.

        :return:
        """
        return FESet(self.nfens, conn=self.conn, label=self.label)

    def set_nfens(self, value):
        if self._nfens == 0:
            self._nfens = value
        if value != self._nfens:
            raise ErrorWrongNumberOfNodes
        
    def get_nfens(self):
        return self._nfens
        
    nfens = property(get_nfens, set_nfens)

    def count(self):
        return self._conn.shape[0]

    def set_conn(self, value):
        self._conn = value
        if value is not None:
            self.nfens = value.shape[1]

    def get_conn(self):
        return self._conn

    conn = property(get_conn, set_conn)

    def set_label(self, value):
        if value is None:
            value = 0 # default label is zero
        if (type(value) == int):
            self._label = numpy.zeros((self._conn.shape[0], 1), dtype=int)
            self._label.fill(value)
        else:
            if (value.shape != (self._conn.shape[0],)):
                if (value.shape != (self._conn.shape[0],1)):
                    raise ErrorWrongLabels
            self._label = value

    def get_label(self):
        return self._label
        
    label = property(get_label, set_label)

    def jacmat(self, gradbfun, x):
        """Calculate the Jacobian matrix.
        """
        return numpy.dot(x.T, gradbfun) # to do: speed up w/ cython?
    
    def subset(self, L):
        """Extract a subset of the finite elements.

        :param L: list or array of indexes of elements to keep
        :return: copy of the input object, with the subset of the elements
        """
        selfcopy = copy.deepcopy(self)
        selfcopy.conn = self.conn[L, :]
        selfcopy.label = self.label[L]
        return selfcopy

    def update_conn(self,NewIDs):
        """Update the connectivity after the IDs of nodes changed.

        :param NewIDs: new node IDs, numpy array. Note that indexes in the conn array point
             _into_ the  NewIDs array. After the connectivity was updated
             this is no longer true!
        :return:  The object is modified in place.
        """

        NewIDs = NewIDs.ravel()
        for i in range(self.conn.shape[0]):
            self.conn[i,:]=NewIDs[self.conn[i,:]]
        return

    def cat(self, other):
        """Concatenate two FE sets into one.
        
        :param other: member of the same class as self, descendent of the FESet class
        :return: new instance, comprising  connectivities and labels of both sets
        """
        concatsets = copy.deepcopy(self)
        concatsets.conn = numpy.vstack((concatsets.conn, other.conn))
        concatsets.label = numpy.vstack((concatsets.label, other.label))
        return concatsets