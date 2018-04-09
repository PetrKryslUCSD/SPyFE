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
    fens.xyz[:, 0] = fens.xyz[:, 0]*(rex-rin)/(maxx-minx)
    fens.xyz[:, 1] = fens.xyz[:, 1] * (Angl - 0) / (maxy - miny)
    for i in range(fens.count()):
        r = rin + fens.xyz[i, 0]
        a = fens.xyz[i, 1]
        fens.xyz[i, 0] = r * math.cos(a)
        fens.xyz[i, 1] = r * math.sin(a)
    return fens, fes
