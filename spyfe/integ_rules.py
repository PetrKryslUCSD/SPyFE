import numpy


class GaussRule:
    """
    Class of the Gauss integration rule.

    The rule is applicable for a tensor product of  intervals -1 <=x<= +1.
    """

    ErrorUnknownOrder = Exception('Unknown Gauss quadrature order')
    ErrorInvalidDimension = Exception('Invalid dimension of Gauss quadrature')

    def __init__(self, dim, order=1):
        self.dim = dim
        self.order = order
        self.npts = None
        if self.order == 1:
            pc = [0]
            w = [2]
        elif self.order == 2:
            pc = [-0.577350269189626, 0.577350269189626]
            w = [1, 1]
        elif self.order == 3:
            pc = [-0.774596669241483, 0, 0.774596669241483]
            w = [0.5555555555555556, 0.8888888888888889, 0.5555555555555556]
        elif self.order == 4:
            pc = [-0.86113631159405, -0.33998104358486, 0.33998104358486, 0.86113631159405]
            w = [0.34785484513745, 0.65214515486255, 0.65214515486255, 0.34785484513745]
        else:
            raise GaussRule.ErrorUnknownOrder
        self.param_coords = numpy.zeros((len(pc) ** self.dim, self.dim))
        self.weights = numpy.zeros((len(w) ** self.dim, 1))
        if self.dim == 1:
            n = 0
            for i in range(self.order):
                self.param_coords[n, 0] = pc[i]
                self.weights[n] = w[i]
                n += 1
        elif self.dim == 2:
            n = 0
            for i in range(self.order):
                for j in range(self.order):
                    self.param_coords[n, 0] = pc[i]
                    self.param_coords[n, 1] = pc[j]
                    self.weights[n] = w[i] * w[j]
                    n += 1
        elif self.dim == 3:
            n = 0
            for i in range(self.order):
                for j in range(self.order):
                    for k in range(self.order):
                        self.param_coords[n, 0] = pc[i]
                        self.param_coords[n, 1] = pc[j]
                        self.param_coords[n, 2] = pc[k]
                        self.weights[n] = w[i] * w[j] * w[k]
                        n += 1
        else:
            raise GaussRule.ErrorInvalidDimension
        self.npts = len(self.weights)

class TriRule:
    """
    Class of the triangle integration rule.

    """

    UnknownOrder = Exception('Unknown triangle quadrature order')

    def __init__(self, npts=1):
        self.npts = npts
        if self.npts == 1:
            self.param_coords = numpy.array([1. / 3, 1. / 3]).reshape((1, 2))
            self.weights = numpy.array([1. / 2])
        elif self.npts == 3:
            self.param_coords = numpy.array([[2. / 3, 1. / 6], [1. / 6, 2. / 3], [1. / 6, 1. / 6]]).reshape((3, 2))
            self.weights = numpy.array([1. / 6, 1. / 6, 1. / 6]).reshape((3, 1))
        elif self.npts == 6:
            self.param_coords = numpy.array([[0.816847572980459, 0.091576213509771],
                                             [0.091576213509771, 0.816847572980459],
                                             [0.091576213509771, 0.091576213509771],
                                             [0.108103018168070, 0.445948490915965],
                                             [0.445948490915965, 0.108103018168070],
                                             [0.445948490915965, 0.445948490915965]]).reshape((6, 2))
            self.weights = numpy.array(
                [0.109951743655322, 0.109951743655322, 0.109951743655322,
                 0.223381589678011, 0.223381589678011, 0.223381589678011]).reshape((6, 1)) / 2.0
        elif self.npts == 13:
            self.param_coords = numpy.array([[0.333333333333333, 0.333333333333333],
                                             [0.479308067841923, 0.260345966079038],
                                             [0.260345966079038, 0.479308067841923],
                                             [0.260345966079038, 0.260345966079038],
                                             [0.869739794195568, 0.065130102902216],
                                             [0.065130102902216, 0.869739794195568],
                                             [0.065130102902216, 0.065130102902216],
                                             [0.638444188569809, 0.312865496004875],
                                             [0.638444188569809, 0.048690315425316],
                                             [0.312865496004875, 0.638444188569809],
                                             [0.312865496004875, 0.048690315425316],
                                             [0.048690315425316, 0.638444188569809],
                                             [0.048690315425316, 0.312865496004875]
                                             ]).reshape((13, 2))
            self.weights = numpy.array(
                [-0.074785022233835, 0.087807628716602, 0.087807628716602, 0.087807628716602, 0.0266736178044195,
                 0.0266736178044195, 0.0266736178044195, 0.0385568804451285, 0.0385568804451285, 0.0385568804451285,
                 0.0385568804451285, 0.0385568804451285, 0.0385568804451285]
            ).reshape((13, 1))
        else:
            raise TriRule.UnknownOrder

class TetRule:
    """
        Class of the tetrahedron integration rule.

        """

    UnknownOrder = Exception('Unknown tetrahedron quadrature order')

    def __init__(self, npts=1):
        self.npts = npts
        if self.npts == 1:
            self.param_coords = numpy.array([1. / 4, 1. / 4, 1. / 4]).reshape((1, 3))
            self.weights = numpy.array([1. / 6])
        elif self.npts == 4:
            self.param_coords = numpy.array([[0.1381966, 0.1381966, 0.1381966],
                                             [0.5854102, 0.1381966, 0.1381966],
                                             [0.1381966, 0.5854102, 0.1381966],
                                             [0.1381966, 0.1381966, 0.5854102]
                                             ]).reshape((4, 3))
            w = 1./4./6.
            self.weights = numpy.array([w,w,w,w]).reshape((4, 1))
        elif self.npts == 5: #   Zienkiewicz #3.
            a = 1.0 / 6.0
            b = 0.25
            c = 0.5
            d = - 0.8
            e = 0.45
            self.param_coords = numpy.array([[b,b,b],
                        [c,a,a],
                        [a,c,a],
                        [a,a,c],
                        [a,a,a]]).reshape((5, 3))
            self.weights = (numpy.array([d, e, e, e, e])/6.).reshape((5, 1))
        else:
            raise TetRule.UnknownOrder
