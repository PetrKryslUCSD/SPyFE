import numpy

ErrorMissingLocations = Exception('Locations of nodes must be supplied')
ErrorWrongType = Exception('The location array is the wrong type (not one-based array)')


class FENodeSet:
    """
    Finite element node set class.
    """

    def __init__(self, xyz=None):
        """Constructor.

        :param xyz: Array of node locations.
        """
        self._xyz = xyz
        if xyz is None:
            raise ErrorMissingLocations
        self.set_xyz(xyz)

    def set_xyz(self, value):
        self._xyz = value

    def get_xyz(self):
        return self._xyz

    xyz = property(get_xyz, set_xyz)

    def count(self):
        return self.xyz.shape[0]