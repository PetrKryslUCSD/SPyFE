import numpy
from spyfe.materials.mat_base import MatBase

OUTPUT_CAUCHY = 0
OUTPUT_PRINCIPAL_CAUCHY = 1
OUTPUT_2ND_PK = 2
OUTPUT_STRAIN_ENERGY = 3
OUTPUT_PRESSURE = 4
OUTPUT_EQUIV_PL_DEF = 5

ID2 = numpy.eye(2, dtype=float)
ID3 = numpy.eye(3, dtype=float)

class MatDefor(MatBase):
    """Base material deformation class.

    """

    def __init__(self, rho=0.0):
        super().__init__(rho=rho)


def strain_3v_from_2x2t(vout, t):
    """Convert a matrix of 2x2 strain components into a 3-component vector .

    Convert a matrix of 2 x2 strain components [tensor-1 ]
    into a 3-component vector.
    :param vout: 3-component vector.
    :param t: matrix of 2 x2 strain components
    :return: Nothing.
    """

    vout[0] = t[0]
    vout[1] = t[1, 1]
    vout[2] = t[0, 1] + t[1, 0]
    return None


def strain_2x2t_from_3v(tout, v):
    """Convert a strain 3-vector to a matrix of 2x2 strain components (tensor )

     Convert a strain 3-vector to a *symmetric *
        matrix of 2 x2 strain components (tensor )
    :param tout: *symmetric *
        matrix of 2 x2 strain components (tensor ), t = numpy.zeros((2, 2))
    :param v: strain 3-vector
    :return: Nothing
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[0, 1] = v[2] / 2.
    tout[1, 0] = v[2] / 2.
    return None


def strain_6v_from_3x3t(vout, t):
    """Convert a matrix of 3 x3 strain components to a 6-component vector .

    convert a matrix of 3 x3 strain components (tensor )
        into a 6-component vector .
    :param vout: 6-component vector of strain. v = numpy.zeros((6,))
    :param t:
    :return: Nothing
    """
    vout[0] = t[0, 0]
    vout[1] = t[1, 1]
    vout[2] = t[2, 2]
    vout[3] = t[0, 1] + t[1, 0]
    vout[4] = t[0, 2] + t[2, 0]
    vout[5] = t[2, 1] + t[1, 2]
    return None


def strain_3x3t_from_6v(tout, v):
    """Convert a strain 6-vector to a matrix of 3 x3 strain components (tensor )

    convert a strain 6-vector to a *symmetric *
        matrix of 3 x3 strain components (tensor )

    :param tout: matrix of 3 x3 strain components (tensor ), numpy.zeros((3, 3))
    :param v: strain 6-vector
    :return: Nothing.
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[2, 2] = v[2]
    tout[0, 1] = v[3] / 2
    tout[1, 0] = v[3] / 2
    tout[0, 2] = v[4] / 2
    tout[2, 0] = v[4] / 2
    tout[2, 1] = v[5] / 2
    tout[1, 2] = v[5] / 2
    return None


def stress_3v_from_2x2t(vout, t):
    """Convert a symmetric matrix of 2 x2 stress components to a 3-component vector .

    Convert a symmetric matrix of 2 x2 stress components (tensor )
    into a 3-component vector .
    :param vout: a 3-component vector, numpy.zeros((3,))
    :param t: symmetric matrix of 2 x2 stress components
    :return: Nothing
    """
    vout[0] = t[0, 0]
    vout[1] = t[1, 1]
    vout[2] = 1. / 2 * (t[0, 1] + t[1, 0])
    return None


def stress_2x2t_from_3v(tout, v):
    """Convert a 3-vector to a matrix of 2 x2 stress components (tensor )

     Convert a 3-vector to a *symmetric *
        matrix of 2 x2 stress components (tensor )
    :param tout: a matrix of 2 x2 stress components (tensor ), numpy.zeros((2, 2))
    :param v: stress 3-vector
    :return: Nothing
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[0, 1] = v[2]
    tout[1, 0] = v[2]
    return None


def stress_3x3t_from_3v(tout, v):
    """Convert a 3-vector to a matrix of 3 x3 stress components (tensor )

    Convert a 3-vector to a *symmetric *
        matrix of 3 x3 stress components (tensor )
    :param tout: a matrix of 3 x3 stress components (tensor ), numpy.zeros((3, 3))
    :param v: stress 3-vector
    :return: Nothing
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[0, 1] = v[2]
    tout[1, 0] = v[2]
    return None


def stress_3x3t_from_4v(tout, v):
    """Convert a 4-vector to a matrix of 3 x3 stress components (tensor ).

    Convert a 4-vector to a *symmetric *
    matrix of 3 x3 stress components (tensor ).This is
    conversion routine that would be useful for plane strain or
    axially symmetric conditions .
    The stress vector components need to be ordered as :
    sigma_x, sigma_y, tau_xy, sigma_z,
    which is the ordering used for the plane -strain model reduction .
    Therefore, for axially symmetric analysis the components need to be
    reordered, as from the constitutive equation they come out
    as sigma_x, sigma_y, sigma_z, tau_xy .
    :param tout: a matrix of 3 x3 stress components (tensor ). numpy.zeros((3, 3))
    :param v: stress 4-vector
    :return: Nothing.
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[0, 1] = v[2]
    tout[1, 0] = v[2]
    tout[2, 2] = v[3]
    return None


def stress_3x3t_from_6v(tout, v):
    """Convert a 6-vector to a matrix of 3 x3 stress components (tensor )

    convert a 6-vector to a *symmetric *
            matrix of 3 x3 stress components (tensor )
    :param tout: a matrix of 3 x3 stress components (tensor ). numpy.zeros((3, 3))
    :param v: stress 6-vector
    :return: Nothing.
    """
    tout[0, 0] = v[0]
    tout[1, 1] = v[1]
    tout[2, 2] = v[2]
    tout[0, 1] = v[3]
    tout[1, 0] = v[3]
    tout[0, 2] = v[4]
    tout[2, 0] = v[4]
    tout[2, 1] = v[5]
    tout[1, 2] = v[5]
    return None


def stress_6v_from_3x3t(vout, t):
    """Convert a matrix of 3 x3 stress components to a 6-component vector .

    Convert a matrix of 3 x3 stress components (tensor )
        into a 6-component vector .
    :param vout: 6-component vector, numpy.zeros((6,)).
    :param t: 3 x 3 matrix of stress components
    :return: Nothing.
    """
    vout[0] = t[0, 0]
    vout[1] = t[1, 1]
    vout[2] = t[2, 2]
    vout[3] = 1. / 2 * (t[0, 1] + t[1, 0])
    vout[4] = 1. / 2 * (t[0, 2] + t[2, 0])
    vout[5] = 1. / 2 * (t[2, 1] + t[1, 2])
    return None


def strain_6v_from_9v(vout, v):
    """Convert a strain 9-vector to a strain 6-vector components (tensor )

    :param vout: strain 6-vector components (tensor), numpy.zeros((6,))
    :param v: a strain 9-vector
    :return: Nothing.
    """
    vout[0] = v[0]
    vout[1] = v[1]
    vout[2] = v[2]
    vout[3] = v[3] + v[4]
    vout[4] = v[7] + v[8]
    vout[5] = v[5] + v[6]
    return None


def strain_9v_from_6v(vout, v):
    """Convert a strain 6-vector to a strain 9-vector components (tensor)

    :param vout: strain 9-vector components (tensor), numpy.zeros((9,))
    :param v: strain 6-vector
    :return: Nothing
    """
    vout[0] = v[0]
    vout[1] = v[1]
    vout[2] = v[2]
    vout[3] = v[3] / 2
    vout[4] = v[3] / 2
    vout[5] = v[5] / 2
    vout[6] = v[5] / 2
    vout[7] = v[4] / 2
    vout[8] = v[4] / 2
    return None


def stress_6v_from_9v(vout, v):
    """Convert a stress 9-vector to a stress 6-vector components (tensor )

    :param vout: stress 6-vector components (symmetric tensor representation), numpy.zeros((6,))
    :param v: stress 9-vector
    :return: Nothing
    """
    vout[0] = v[0]
    vout[1] = v[1]
    vout[2] = v[2]
    vout[3] = v[3]
    vout[4] = v[7]
    vout[5] = v[5]
    return None


def tensor_3x3t_double_contraction(A, B):
    """Compute a tensor double contraction (scalar).

    :param A:
    :param B:
    :return:
    """
    raise Exception('Not implemented yet')
