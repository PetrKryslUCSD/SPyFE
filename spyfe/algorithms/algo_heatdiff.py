import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time
from spyfe.fields.nodal_field import NodalField
from spyfe.meshing.selection import connected_nodes
from spyfe.meshing.exporters.vtkexporter import vtkexport


def steady_state(model_data):
    """Steady-state heat conduction solver.

    :param model_data: Model data dictionary.

    model_data['fens'] = finite element node set (mandatory)

    For each region (connected piece of the domain made of a particular material), mandatory:
    model_data['regions']= list of dictionaries, one for each region
        Each region:
        region['heat_generation'] = material internal heat generation rate (optional), function
        region['femm'] = finite element set that covers the region (mandatory)

    :return: Success?  True or false.  The model_data object is modified.
    model_data['geom'] =the nodal field that is the geometry
    model_data['temp'] =the nodal field that is the computed temperature
    model_data['timings'] = timing of the individual operations
    """

    # For essential boundary conditions (optional):
    # model_data.boundary_conditions.essential = cell array of struct,
    #           each piece of surface with essential boundary condition gets one
    #           element of the array with a struct with the attributes
    #     essential.temperature=fixed (prescribed) temperature (scalar),  or
    #           handle to a function with signature
    #               function T =f(x)
    #     essential.fes = finite element set on the boundary to which
    #                       the condition applies
    #               or alternatively
    #     essential.node_list = list of nodes on the boundary to which
    #                       the condition applies
    #               Only one of essential.fes and essential.node_list needs a given.
    #
    # For convection boundary conditions (optional):
    # model_data.boundary_conditions.convection = cell array of struct,
    #           each piece of surface with convection boundary condition gets one
    #           element of the array with a struct with the attributes
    #     convection.ambient_temperature=ambient temperature (scalar)
    #     convection.surface_transfer_coefficient  = surface heat transfer coefficient
    #     convection.fes = finite element set on the boundary to which
    #                       the condition applies
    #     convection.integration_rule= integration rule
    #
    # For flux boundary conditions (optional):
    # model_data.boundary_conditions.flux = cell array of struct,
    #           each piece of surface with flux boundary condition gets one
    #           element of the array with a struct with the attributes
    #     flux.normal_flux= normal flux component, positive when outward-bound (scalar)
    #     flux.fes = finite element set on the boundary to which
    #                       the condition applies
    #     flux.integration_rule= integration rule
    #
    # Control parameters:
    # model_data.renumber = true or false flag (default is true)
    # model_data.renumbering_method = optionally choose the renumbering
    #       method  (symrcm or symamd)

    # renumber = ~true; # Should we renumber?
    # if (isfield(model_data,'renumber'))
    #     renumber  =model_data.renumber;
    # end
    # Renumbering_options =struct( [] );
    #
    # # Should we renumber the nodes to minimize the cost of the solution of
    # # the coupled linear algebraic equations?
    # if (renumber)
    #     renumbering_method = 'symamd'; # default choice
    #     if ( isfield(model_data,'renumbering_method'))
    #         renumbering_method  =model_data.renumbering_method;;
    #     end
    #     # Run the renumbering algorithm
    #     model_data =renumber_mesh(model_data, renumbering_method);;
    #     # Save  the renumbering  (permutation of the nodes)
    #     clear  Renumbering_options; Renumbering_options.node_perm  =model_data.node_perm;
    # end

    timings = []

    task = 'Preliminaries'
    start = time.time()

    # Extract the nodes
    fens = model_data['fens']

    # Construct the geometry field
    geom = NodalField(fens=fens)

    # Construct the temperature field
    temp = NodalField(dim=1, nfens=geom.nfens)

    # Apply the essential boundary conditions on the temperature field
    if 'boundary_conditions' in model_data:
        if 'essential' in model_data['boundary_conditions']:
            ebcs = model_data['boundary_conditions']['essential']
            for ebc in ebcs:
                fenids = ebc['node_list']
                if 'value' not in ebc:
                    value = lambda xyz: 0.0
                else:
                    value = ebc['value']
                for index in fenids:
                    temp.set_ebc([index], val=value(fens.xyz[index, :]))
            temp.apply_ebc()

    # Number the equations
    temp.numberdofs()

    timings.append((task, time.time() - start))

    # Initialize the heat loads vector
    F = numpy.zeros((temp.nfreedofs,))

    # #  Flux boundary condition:
    # if (isfield(model_data.boundary_conditions, 'flux' ))
    #     for j=1:length(model_data.boundary_conditions.flux)
    #         flux =model_data.boundary_conditions.flux{j};
    #         flux.femm = femm_heat_diffusion (struct (...
    #             'fes',flux.fes,...
    #             'integration_rule',flux.integration_rule));
    #         model_data.boundary_conditions.flux{j}=flux;
    #     end
    #     clear flux fi  femm
    # end
    #

    task = 'Conductivity matrix'
    start = time.time()

    # Construct the system conductivity matrix
    K = csr_matrix((temp.nfreedofs, temp.nfreedofs))  # (all zeros, for the moment)
    for region in model_data['regions']:
        # Add up all the conductivity matrices for all the regions
        K += region['femm'].conductivity(geom, temp)

    timings.append((task, time.time() - start))

    task = 'Heat generation load'
    start = time.time()

    for region in model_data['regions']:
        Q = None  # default is no internal heat generation rate
        if 'heat_generation' in region:  # Was it defined?
            Q = region['heat_generation']
        if Q is not None:  # If it was supplied, and it is nonzero, compute its contribution.
            F = F + region['femm'].distrib_loads(geom, temp, Q, 3)

    timings.append((task, time.time() - start))

    task = 'NZEBC load'
    start = time.time()

    if 'boundary_conditions' in model_data:
        if 'essential' in model_data['boundary_conditions']:
            for region in model_data['regions']:
                # Loads due to the essential boundary conditions on the temperature field
                F += region['femm'].nz_ebc_loads_conductivity(geom, temp)

    timings.append((task, time.time() - start))

    # Convection boundary conditions:
    if 'boundary_conditions' in model_data:
        if 'surface_transfer' in model_data['boundary_conditions']:
            amb = NodalField(dim=1, nfens=geom.nfens)# create the ambient temperature field
            sbcs = model_data['boundary_conditions']['surface_transfer']
            for sbc in sbcs:
                if 'ambient_temperature' not in sbc:
                    ambient_temperature = lambda xyz: 0.0
                else:
                    ambient_temperature = sbc['ambient_temperature']
                femm = sbc['femm']
                for index in connected_nodes(femm.fes):
                    amb.set_ebc([index], val=ambient_temperature(fens.xyz[index, :]))
            amb.apply_ebc()
            for sbc in sbcs:
                femm = sbc['femm']
                K += femm.surface_transfer(geom, temp)
                F += femm.surface_transfer_loads(geom, temp, amb)
                F += femm.nz_ebc_loads_surface_transfer(geom, temp)

    # # Process the flux boundary condition
    # if (isfield(model_data.boundary_conditions, 'flux' ))
    #     for j=1:length(model_data.boundary_conditions.flux)
    #         flux =model_data.boundary_conditions.flux{j};
    #         fi= force_intensity(struct('magn',flux.normal_flux));
    #         # Note the sign  which reflects the formula (negative sign
    #         # in front of the integral)
    #         F = F - distrib_loads(flux.femm, sysvec_assembler, geom, temp, fi, 2);
    #     end
    #     clear flux fi
    # end


    task = 'System solution'
    start = time.time()

    # Solve for the temperatures
    temp.scatter_sysvec(spsolve(K, F));

    timings.append((task, time.time() - start))

    # Update the model data
    model_data['geom'] = geom
    model_data['temp'] = temp
    model_data['timings'] = timings
    return True


def plot_temperature(model_data):
    file = 'temp'
    if 'postprocessing' in model_data:
        if 'file' in model_data['postprocessing']:
            file = model_data['postprocessing']['file']
    for r in range(len(model_data['regions'])):
        region = model_data['regions'][r]
        femm = region['femm']
    vtkexport(file + str(r), femm.fes, model_data['geom'],
              {'temperature': model_data['temp']})
    return True
