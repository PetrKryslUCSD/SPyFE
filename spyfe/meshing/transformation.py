import numpy
from spyfe.meshing.boxes import inflate_box, bounding_box, in_box
from spyfe.fenode_set import FENodeSet
import math


def shape_to_annulus(fens, fes, rin, rex, Angl):
    """Shape a 2D mesh block into an annulus.

    The 2D block is reshaped to have the dimension (rex - rin)
    in the first coordinate direction, and the dimension Angl
    in the second coordinate direction.
    Here (rex - rin) is the difference of the external radius and the internal radius,
    and Angl is the development angle in radians.
    The annulus is produced by turning from the first coordinate axis counterclockwise.

    :param fens:
    :param fes:
    :param rin:
    :param rex:
    :param Angl:
    :return:
    """
    minx = numpy.amin(fens.xyz[:, 0])
    miny = numpy.amin(fens.xyz[:, 1])
    maxx = numpy.amax(fens.xyz[:, 0])
    maxy = numpy.amax(fens.xyz[:, 1])
    fens.xyz[:, 0] = fens.xyz[:, 0] * (rex - rin) / (maxx - minx)
    fens.xyz[:, 1] = fens.xyz[:, 1] * (Angl - 0) / (maxy - miny)
    for i in range(fens.count()):
        r = rin + fens.xyz[i, 0]
        a = fens.xyz[i, 1]
        fens.xyz[i, 0] = r * math.cos(a)
        fens.xyz[i, 1] = r * math.sin(a)
    return fens, fes


def rotate_mesh(fens, rotationVector, point):
    """Rotate a mesh around a vector anchored at the given point. 
    
    :param fens: 
    :param rotationVector: 
    :param point: 
    :return: 
    """

    r = rotmat(rotationVector)
    pivot = numpy.ones((fens.count(), 1)) * point.reshape(1, 3)
    fens.xyz = pivot + numpy.dot((fens.xyz - pivot), r.T)
    return fens


_I3 = numpy.identity(3)


def rotmat(theta):
    """Compute rotation matrix from a rotation vector or from the associated
    skew matrix.
    
    :param theta: 3D vector or a 3 x 3 matrix
    :return: 
    """
    if theta.shape == (3, 3):
        a = numpy.array([-theta[1, 2], theta[0, 2], -theta[0, 1]])
        thetatilde = theta
    else:
        thetatilde = skewmat(theta)
        a = numpy.array(theta)

    # R = expm(thetatilde)
    na = numpy.linalg.norm(a)
    if (na == 0.):
        r = _I3
    else:
        a = a / na
        ca = math.cos(na)
        sa = math.sin(na)
        a.shape = (3, 1)
        aa_t = a * a.T
        r = ca * (_I3 - aa_t) + (sa / na) * thetatilde + aa_t
    return r


def skewmat(theta):
    """Compute a skew matrix from its axial vector.
    
    :param theta: 
    :return: 
    """
    theta = numpy.array(theta).reshape(len(theta), 1)
    if theta.shape[0] == 3:
        s = numpy.array([[0, - theta[2], theta[1]],
                         [theta[2], 0, - theta[0]],
                         [- theta[1], theta[0], 0]])
    elif theta.shape[0] == 2:
        s = numpy.array([-theta[1], theta[0]]).reshape(2, 1)
    else:
        raise Exception('Unknown shape of the input')
    return s
