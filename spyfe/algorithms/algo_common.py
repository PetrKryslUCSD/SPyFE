"""
Module for common operations on models.
"""
import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time
from spyfe.fields.nodal_field import NodalField
from spyfe.meshing.selection import connected_nodes
from spyfe.meshing.exporters.vtkexporter import vtkexport


def plot_mesh(model_data):
    """Generate a VTK file for the plotting of the mesh.
    
    :param model_data: model dictionary, the following keys need to have values:
    model_data['fens']
    model_data['regions']
    :return: Boolean
    """
    file = 'mesh'
    if 'postprocessing' in model_data:
        if 'file' in model_data['postprocessing']:
            file = model_data['postprocessing']['file']

    fens = model_data['fens']
    geom = NodalField(fens=fens)
    for r in range(len(model_data['regions'])):
        region = model_data['regions'][r]
        femm = region['femm']
        vtkexport(file + str(r), femm.fes, geom)
    return True
