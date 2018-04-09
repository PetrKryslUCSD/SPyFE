"""
Module for deformation analysis algorithms.
"""
import numpy
from spyfe.meshing.exporters.vtkexporter import vtkexport
from spyfe.materials.mat_defor import OUTPUT_CAUCHY
from spyfe.csys import CSys

def plot_stress(model_data):
    """Algorithm for plotting stress results.

    :param model_data: Model data dictionary.

    model_data['fens'] = finite element node set (mandatory)

    For each region (connected piece of the domain made of a particular material), mandatory:
    model_data['regions']= list of dictionaries, one for each region
        Each region:
        region['femm'] = finite element set that covers the region (mandatory)
        
    For essential boundary conditions (optional):
    model_data['boundary_conditions']['essential']=list of dictionaries, one for each 
        application of an essential boundary condition.
        For each EBC dictionary ebc:
            ebc['node_list'] =  node list,
            ebc['comp'] = displacement component (zero-based),
            ebc['value'] = function to supply the prescribed value, default is lambda x: 0.0
      

    :return: Success?  True or false.  The model_data object is modified.
    model_data['geom'] =the nodal field that is the geometry
    model_data['temp'] =the nodal field that is the computed temperature
    model_data['timings'] = timing of the individual operations
    """
    file = 'stresses'
    if 'postprocessing' in model_data:
        if 'file' in model_data['postprocessing']:
            file = model_data['postprocessing']['file']
        outcs = model_data['postprocessing']['outcs'] if 'outcs' in model_data['postprocessing']\
            else CSys()

    geom = model_data['geom']
    u = model_data['u']
    dtemp = model_data['dtemp'] if 'dtemp' in model_data else None
    for r in range(len(model_data['regions'])):
        region = model_data['regions'][r]
        femm = region['femm']
        stresses = femm.nodal_field_from_integr_points(geom, u, u,
                                                       dtempn1=dtemp, output=OUTPUT_CAUCHY,
                                                       component=[0, 1, 2, 3, 4, 5],
                                                       outcs=outcs)
        vtkexport(file + str(r), femm.fes, model_data['geom'],
                  {'displacement': u, 'stresses': stresses})
    return True


def plot_elemental_stress(model_data):
    """Algorithm for plotting elementalstress results.

    :param model_data: Model data dictionary.

    model_data['fens'] = finite element node set (mandatory)

    For each region (connected piece of the domain made of a particular material), mandatory:
    model_data['regions']= list of dictionaries, one for each region
        Each region:
        region['femm'] = finite element set that covers the region (mandatory)

    For essential boundary conditions (optional):
    model_data['boundary_conditions']['essential']=list of dictionaries, one for each 
        application of an essential boundary condition.
        For each EBC dictionary ebc:
            ebc['node_list'] =  node list,
            ebc['comp'] = displacement component (zero-based),
            ebc['value'] = function to supply the prescribed value, default is lambda x: 0.0


    :return: Success?  True or false.  The model_data object is modified.
    model_data['geom'] =the nodal field that is the geometry
    model_data['temp'] =the nodal field that is the computed temperature
    model_data['timings'] = timing of the individual operations
    """
    file = 'stresses'
    if 'postprocessing' in model_data:
        if 'file' in model_data['postprocessing']:
            file = model_data['postprocessing']['file']
        outcs = model_data['postprocessing']['outcs'] if 'outcs' in model_data['postprocessing'] \
            else CSys()

    geom = model_data['geom']
    u = model_data['u']
    dtemp = model_data['dtemp'] if 'dtemp' in model_data else None
    for r in range(len(model_data['regions'])):
        region = model_data['regions'][r]
        femm = region['femm']
        stresses = femm.elemental_field_from_integr_points(geom, u, u,
                                                       dtempn1=dtemp, output=OUTPUT_CAUCHY,
                                                       component=[0, 1, 2, 3, 4, 5],
                                                       outcs=outcs)
        vtkexport(file + str(r), femm.fes, model_data['geom'],
                  {'displacement': u, 'stresses': stresses})
    return True


