import numpy

ErrorAccessAt0Index = Exception('One-based array accessed at row index 0')

class OneBased2DArray:

    def __init__(self, dimensions=None, from_array= None, dtype=numpy.float):
        if from_array is not None:
            self._arr = from_array.copy()
        else:
            self._arr = numpy.zeros(dimensions, dtype=dtype)

    def __getitem__(self, indextuple):
        Ro, Co = indextuple
        if Ro == 0:
            raise ErrorAccessAt0Index
        return self._arr[Ro-1, Co]

    def __setitem__(self, indextuple, value):
        Ro, Co = indextuple
        if Ro == 0:
            raise ErrorAccessAt0Index
        self._arr[Ro-1, Co] = value

    def ncol(self):
        return self._arr.shape[1]

    def nrow(self):
        return self._arr.shape[0]

    def raw_array(self):
        return self._arr.copy()

    def __str__(self):
        return str(self._arr)

def range_1based(*therange):
    """
    Create a range for one based indexing.
    :param therange: The upper value of the range is included!
    :return:
    """
    low, high = therange
    return range(low, high + 1)
