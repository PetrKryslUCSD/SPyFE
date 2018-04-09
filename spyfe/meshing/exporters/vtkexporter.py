# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:22:01 2017

@author: PetrKrysl
"""
import numpy
from spyfe.fesets.surfacelike import FESetT3, FESetT6, FESetQ4, FESetQ8
from spyfe.fesets.volumelike import FESetH8, FESetH20, FESetT4, FESetT10
from spyfe.meshing.exporters.pyevtk.hl import unstructuredGridToVTK
from spyfe.meshing.exporters.pyevtk.vtk import VtkTriangle, VtkQuadraticTriangle
from spyfe.meshing.exporters.pyevtk.vtk import VtkQuad, VtkQuadraticQuad
from spyfe.meshing.exporters.pyevtk.vtk import VtkHexahedron, VtkQuadraticHexahedron
from spyfe.meshing.exporters.pyevtk.vtk import VtkTetra, VtkQuadraticTetra

_vtkTypeMap = {FESetT3.__name__: VtkTriangle.tid,
               FESetT6.__name__: VtkQuadraticTriangle.tid,
               FESetQ4.__name__: VtkQuad.tid,
               FESetQ8.__name__: VtkQuadraticQuad.tid,
               FESetT4.__name__: VtkTetra.tid,
               FESetT10.__name__: VtkQuadraticTetra.tid,
               FESetH8.__name__: VtkHexahedron.tid,
               FESetH20.__name__: VtkQuadraticHexahedron.tid}
ErrorTIDNotFound = Exception("The VTK tid of element type not found")

def vtkexport(tofile, fes, geom, flds=None):
    """Export mesh and results to a VTK file.

    :param tofile: file name
    :param fes: finite elements set
    :param geom: geometry field
    :param flds: dictionary of nodal fields to export.  For displacement fields we assume three dimensions.  For scalar
    fields (such as temperature) only one dimension is present.  For fields with larger dimension, for instance
    such as stress field, the field values are exported as multiple point data.
    :return:
    """
    try:
        tid=_vtkTypeMap[type(fes).__name__]
    except  KeyError:
        raise ErrorTIDNotFound

    x = geom.values[:,0].copy()
    y = geom.values[:,1].copy()
    if geom.dim > 2:
        z = geom.values[:,2].copy()
    else:
        z = numpy.zeros_like(x)
    offsets = numpy.array([(i + 1) * fes.nfens for i in range(fes.count())])
    cell_types = numpy.zeros_like(offsets)
    cell_types.fill(tid)

    pointData = None
    cellData = None
    if flds is not None:
        for key, fld in flds.items():
            if fld.nents == geom.nfens: # if the number of nodes is matched it is a nodal field
                pointData = {} if pointData is None else pointData
                if fld.dim == 1:
                    data = fld.values.ravel()
                    pointData[key] = data
                elif fld.dim == 3:
                    data = (fld.values[:, 0].ravel(), fld.values[:, 1].ravel(), fld.values[:, 2].ravel())
                    pointData[key] = data
                else:
                    for index in range(fld.dim):
                        data = fld.values[:, index].ravel()
                        pointData[key + str(index)] = data
            else: # else this is an elemental field
                cellData = {} if cellData is None else cellData
                if fld.dim == 1:
                    data = fld.values.ravel()
                    cellData[key] = data
                elif fld.dim == 3:
                    data = (fld.values[:, 0].ravel(), fld.values[:, 1].ravel(), fld.values[:, 2].ravel())
                    cellData[key] = data
                else:
                    for index in range(fld.dim):
                        data = fld.values[:, index].ravel()
                        cellData[key + str(index)] = data


    unstructuredGridToVTK(tofile, x, y, z,
                          fes.conn.ravel(), offsets, cell_types,
                          pointData=pointData, cellData=cellData)