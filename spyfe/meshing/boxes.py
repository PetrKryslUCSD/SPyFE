import numpy
import math


def _inrange(derange, xl):
    return (xl >= derange[0]) and (xl <= derange[1])


def update_box(box=None, x=None):
    """Update a box with another location, or create a new box.

    :param box: either None, in which case a box is created using the
 supplied location x, or an existing box which is expanded to include the
 supplied location x.
    :param xa: The variable array x  can hold multiple points in rows.
    :return: bounding box,
    % box = bounding box
%     for 1-D box=[minx,maxx], or
%     for 2-D box=[minx,maxx,miny,maxy], or
%     for 3-D box=[minx,maxx,miny,maxy,minz,maxz]
.. seealso:: bounding_box, boxes_overlap,  inflate_box,       in_box
    """

    xa = numpy.array(x)
    if xa.ndim < 2:
        xa.shape = (1, len(xa))

    if box is None:
        box = numpy.zeros((2 * xa.shape[1],))
        for i in numpy.arange(xa.shape[1]):
            box[2 * i] = math.inf
            box[2 * i + 1] = -math.inf

    dim = math.ceil(len(box) / 2)
    for j in numpy.arange(xa.shape[0]):
        for i in numpy.arange(dim):
            box[2 * i] = min(box[2 * i], xa[j, i])
            box[2 * i + 1] = max(box[2 * i + 1], xa[j, i])

    return box

def bounding_box(x):
    """Compute bounding box of an array of points.

    :param x: One point per row.
    :return: box
    .. seealso:: update_box, boxes_overlap,  inflate_box,       in_box
    """
    return update_box(box=None, x=x)

def in_box(box, x):
    """Is the given location inside a box?

    :param box: bounding box
%     for 1-D box=[minx,maxx], or
%     for 2-D box=[minx,maxx,miny,maxy], or
%     for 3-D box=[minx,maxx,miny,maxy,minz,maxz]
    :param x: location, as a numpy array
    :return: Boolean
    .. seealso:: bounding_box, boxes_overlap,  inflate_box, update_box
    """
    dim = math.ceil(len(box) / 2)
    x = x.ravel()
    b = _inrange(box[0:2], x[0])
    for i in numpy.arange(1, dim):
        b = (b and _inrange(box[2 * i:2 * (i + 1)], x[i]))
    return b

def inflate_box(box, inflate= 0.0):
    """

    :param box: bounding box
    :param inflate: scalar, amount  by which to increase the box to the left and to the right
    :return: updated box
    .. seealso:: bounding_box, boxes_overlap,  inflate_box, update_box
    """
    abox = numpy.array(box)
    dim = math.ceil(len(box) / 2)
    for i in numpy.arange(dim):
        abox[2 * i] = min(abox[2 * i], abox[2 * i + 1]) - inflate
        abox[2 * i + 1] = max(abox[2 * i], abox[2 * i + 1]) + inflate
    return abox
