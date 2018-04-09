import numpy
import math
from spyfe.meshing.boxes import in_box


def connected_nodes(fes):
    """Extract the node numbers of the nodes connected by the finite element set.

    Extract the list of unique node numbers for the nodes that
    are connected by the finite element set fes. Note that it is assumed
    that all the FEs are of the same type (the same number of connected nodes
    by each cell).

    :param fes: Finite element set.
    :return: Array of node indexes.
    """
    return numpy.unique(fes.conn.ravel())


def fenode_select(fens, box=None, inflate=None, invert=False, plane=None):
    return v_select(fens.xyz, box=box, inflate=inflate, plane=plane, invert=invert)


def v_select(v, box=None, inflate=None, invert=False,
             plane=None):
    """Select locations.

    :param v: Array of locations, one per row.
    :return: Array of indices of locations matching the criteria.
    """
    sel = None
    if box is not None:
        sel = v_select_box(v, box, inflate)

    if plane is not None:
        sel = v_select_plane(v, plane, inflate)

    if invert:
        sel = invert_selection(v.shape[0], sel)

    return sel


def v_select_box(v, box, inflate):
    """

    :param v:
    :param box: Selection box.
    :param inflate: Increase the box by this much in either direction.
    :Example:
    Locations are selected when they are inside a box or on its surface.
v_select_box(v, box=numpy.array([1 -1 2 0 4 3])),
selects locations which are strictly inside the box
 -1<= x <=1     0<= y <=2     3<= z <=4
    :return: Array of indices of locations matching the criteria.
    """
    if inflate is None:
        inflate = 0.0
    dim = math.ceil(len(box) / 2)
    assert dim == v.shape[1]
    abox = numpy.zeros_like(box)
    for j in range(dim):
        abox[2 * j] = min(box[2 * j], box[2 * j + 1]) - inflate
        abox[2 * j + 1] = max(box[2 * j], box[2 * j + 1]) + inflate
    vlist = list()
    for i in numpy.arange(v.shape[0]):
        if in_box(abox, v[i, :]):
            vlist.append(i)
    return numpy.array(vlist)


def v_select_plane(v, plane, inflate):
    """

    :param v:
    :param plate: Selection plane: tuple of a point on the plane
    and the normal vector (not necessarily normalized).
    :param inflate: Select points on the plane and within this distance
     in either direction.
    :Example:
    
    :return: Array of indices of locations matching the criteria.
    """
    if inflate is None:
        inflate = 0.0
    point, normal = plane
    thepoint = numpy.array(point)
    thenormal = numpy.array(normal)
    assert thepoint.shape[0]==v.shape[1]
    assert thenormal.shape[0] == v.shape[1]
    thenormal = thenormal / numpy.linalg.norm(thenormal)
    vlist = list()
    for i in numpy.arange(v.shape[0]):
        a = v[i, :] - thepoint
        an = numpy.dot(a, thenormal)
        if abs(an) <= inflate:
            vlist.append(i)
    return numpy.array(vlist)

def invert_selection(n, sel):
    """
    :param n: Number of entities.
    :param sel: Array of indexes: which entities were selected.
    :return: Inverted selection.
    """
    alli = numpy.zeros((n,), dtype=int)
    alli[:] = numpy.arange(0, n)
    return numpy.setdiff1d(alli, sel)


def fe_select(fens, fes,
              bylabel=False, label=0,
              box=None, inflate=None,
              anynode=False, inside=True,
              facing=False, direction=lambda x: numpy.array([1.0, 0.0, 0.0]), tolerance=0.99,
              plane=None):
    """Select finite elements.

    Select finite elements (FEs) using some criterion, for instance a finite element is
    selected because all its nodes are inside a box.

    Examples of selection criteria:

    box

    Select all FEs with all nodes inside the box:
        fe_select(fens,fes,struct ('box',[1 -1 2 0 4 3]))
    Select all FEs with at least one node inside the box:
        fe_select(fens,fes,struct ('box',[1 -1 2 0 4 3],'anynode',true))

    label

    Select all FEs with given label:
        fe_select(fens,fes,struct ('label', 13))

    flood

    Select all FEs connected together (Starting from node 13):
        fe_select(fens,fes,struct ('flood', true, 'startfen', 13))

    facing

    Select all FEs "facing" in the direction [1,0,0] (along the x-axis):
        fe_select(fens,fes,struct ('facing', true, 'direction', [1,0,0]))
    Select all FEs "facing" in the direction [x(1),x(2),0] (away from the z-axis):
        fe_select(fens,fes,struct ('facing', true, 'direction', @(x)(+[x(1:2),0])))
    Here x is the centroid of the nodes of each selected fe.
    Select all FEs "facing" in the direction [x(1),x(2),0] (away from the z-axis):
        fe_select(fens,fes,struct ('facing',1,'direction',@(x)(+[x(1:2),0]),'tolerance',0.01))
    Here the fe is considered facing in the given direction if the dot
    product of its normal and the direction vector is greater than tolerance.

    The option 'inflate' may be used to increase or decrease the extent of
    the box (or the distance) to make sure some nodes which would be on the
    boundary are either excluded or included.

    nearestto

    Select the geometric cell nearest to the given location.
    nearest=fe_select(fens,fes,struct('nearestto',[2.3,-2]))

    distance

    Select the finite elements within a given distance from the center.
    Example: fe_select(fens,fes,struct ('distance',0.5, 'from',[1 -1])), selects
    elements whose nodes are Less than 0.5 units removed from the point [1 -1].

    cylinder

    locations are selected because they are inside a cylinder or on its
    surface. Called as :

      fe_select(fens,fes,struct ('cylinder',[x, y, R, h]))  for 2D
      fe_select(fens,fes,struct ('cylinder',[x, y, z, R, h])) for 3D

    the orientation of the cylinder can be changed by supplying the option
    'orientation'. x,y,z specify the centre of the cylinder, and R and h the
    radius and height.

    Example: v_select(v,struct ('cylinder',[0 0 0, 1, 2])), selects locations
    which are strictly inside the cylinder with centre located at (0,0,0),
    having radius 1 and height 2.

    smoothpatch

    Select all FEs that are part of a smooth surface.
    For instance, starting from the finite element number 13, select all finite elements
    whose normals differ from the normal of the Neighbor element by less than
    0.05 in the sense that dot(n1,n2)>sqrt(1-0.05^2).
        fe_select(fens,fes,struct ('flood', true, 'startfe', 13,  'normaldelta',0.05))
    # Select all FEs "facing" in the direction [x(1),x(2),0] (away from the z-axis):
        fe_select(fens,fes,struct ('facing', true, 'direction', @(x)(+[x(1:2),0])))
    Here x is the centroid of the nodes of each selected fe.
    # Select all FEs "facing" in the direction [x(1),x(2),0] (away from the z-axis):
        fe_select(fens,fes,struct ('facing',1,'direction',@(x)(+[x(1:2),0]),'tolerance',0.01))
    Here the fe is considered facing in the given direction if the dot
    product of its normal and the direction vector is greater than tolerance.


    Output:
    felist= list of finite elements from the set that satisfy the criteria

    Examples:
        topl =fe_select (fens,bfes,struct('box', [0,L,-Inf,Inf,b/2,b/2],...
                                'inflate',tolerance))
        #  The subset  of the finite elements that are in the box is
        traction.fes= subset(bfes,topl)

    Note: This function uses the node selection function fenode_select() to search
    the nodes.

    See also: fenode_select

    :param fens:
    :param fes:
    :param label:
    :param labels:
    :return:
    """

    if anynode:
        inside = False
        anynode = True

    # if isfield(options,'flood')
    #     flood = true
    #     if isfield(options,'startfen')
    #         startfen = options.startfen
    #     else
    #         error('Need the identifier of the Starting node, startfen')
    #     end
    # end

    # if isfield(options,'smoothpatch')
    #     smoothpatch = true
    #     if isfield(options,'normaldelta')
    #         normaldelta = options.normaldelta
    #     else
    #         normaldelta= 0.05
    #     end
    #     if isfield(options,'startfe')
    #         startfe = options.startfe
    #     else
    #         error('Need the number of the starting finite element')
    #     end
    # end

    # if isfield(options,'overlapping_box')
    #     overlapping_box = true
    #     box=options.overlapping_box
    #     bounding_boxes =[]# precomputed bounding boxes of fes can be passed in
    #     if isfield(options,'bounding_boxes')
    #         bounding_boxes = options.bounding_boxes
    #     end
    #     inflate =0
    #     if isfield(options,'inflate')
    #         inflate = (options.inflate)
    #     end
    #     box = inflate_box (box, inflate)
    # end
    # if isfield(options, 'nearestto')
    #     nearestto = true
    #     locations = options.nearestto
    # end
    # if isfield(options,'cylinder')
    #     cylinder = true
    #     orientation='z'
    #     cylinderdata=options.cylinder
    #     inflate =0
    #     if isfield(options,'inflate')
    #         inflate = (options.inflate)
    #     end
    # end

    # Select based on fe label
    if bylabel:
        felist = []
        for i in numpy.arange(fes.conn.shape[0]):
            if label == fes.label[i]:
                felist.append(i)

        return felist

    # Select by flooding
    # if (flood)
    #    fen2fe_map=fenode_to_fe_map (struct ('fes',fes))
    #    gmap=fen2fe_map.map
    #    conn=fes.conn
    #    felist= zeros(1, size(conn,1))
    #    felist(gmap{startfen})=1
    #    while true
    #        pfelist=felist
    #        markedl=find(felist~=0)
    #        for j=markedl
    #            for k=conn(j,:)
    #                felist(gmap{k})=1
    #            end
    #        end
    #        if sum(pfelist-felist)==0, break end
    #    end
    #    felist =find(felist~=0)
    #    return
    # end

    # Select by in which direction the normal of the fes face
    if facing is not None and facing:
        def normal(Tangents, sdim, mdim):
            if mdim == 1:  # 1-D fe
                N = numpy.array([Tangents[1, 0], -Tangents[0, 0]])
            elif mdim == 2:  # 2-D fe
                N = numpy.cross(Tangents[:, 0], Tangents[:, 1])
            else:
                raise Exception('Got an incorrect size of tangents')
            return N / numpy.linalg.norm(N)

        if (fes.dim != fens.xyz.shape[1] - 1):
            raise Exception('"Facing" selection of fes make sense only for Manifold dimension ==Space dimension-1')
        param_coords = numpy.zeros((1, fes.dim))
        x = numpy.zeros((fes.nfens, fens.xyz.shape[1]))
        gradNpar = fes.gradbfunpar(param_coords)
        fesel = set()
        for i in range(fes.conn.shape[0]):
            x[:, :] = fens.xyz[fes.conn[i, :], :]
            tangents = numpy.dot(x.T, gradNpar)
            n = normal(tangents, fens.xyz.shape[1], fes.dim)
            d = direction(numpy.mean(x, axis=0))
            if (numpy.dot(n, d) > tolerance):
                fesel.add(i)

        return numpy.array(list(fesel), dtype=int)

        # Select by the change in normal
    # if (smoothpatch)
    #    mincos=sqrt(1-normaldelta^2)
    #    xs=fens.xyz
    #    sdim =size(xs,2)
    #    mdim=fes.dim
    #    if (mdim~=sdim-1)
    #        error ('"Smoothpatch" selection of fes make sense only for Manifold dimension ==Space dimension-1')
    #    end
    #    fen2fe_map=fenode_to_fe_map (struct ('fes',fes))
    #    gmap=fen2fe_map.map
    #    conn=fes.conn
    #    param_coords =zeros(1,mdim)# This is a hack: this may not be the proper location for all element types
    #    Nder = bfundpar (fes, param_coords)
    #    felist= zeros(1, size(conn,1))
    #    felist(startfe)=1
    #    while true
    #        pfelist=felist
    #        markedl=find(felist~=0)
    #        for j=markedl
    #            xyz =xs(conn(j,:),:)
    #            Tangents =xyz'*Nder
    #            Nj = normal (Tangents,sdim, mdim)
    #            for k=conn(j,:)
    #                for ke=gmap{k}
    #                    xyz =xs(conn(ke,:),:)
    #                    Tangents =xyz'*Nder
    #                    Nk = normal (Tangents,sdim, mdim)
    #                    if (dot(Nj,Nk)>mincos)
    #                        felist(ke)=1
    #                    end
    #                end
    #            end
    #        end
    #        if sum(pfelist-felist)==0, break end
    #    end
    #    felist =find(felist~=0)
    #    return
    # end

    # Select all FEs whose bounding box overlaps given box
    # if (overlapping_box)
    #    felist= []
    #    xs =fens.xyz
    #    conns=fes.conn
    #    if (isempty(bounding_boxes))
    #        bounding_boxes =zeros(size(conns,1),length(box))
    #        for i=1: size(conns,1)
    #            conn=conns(i,:)
    #            xyz =xs(conn,:)
    #            bounding_boxes(i,:)= bounding_box(xyz)
    #        end
    #    end
    #    for i=1: size(conns,1)
    #        if (boxes_overlap (box,bounding_boxes(i,:)))
    #            felist(end+1)=i
    #        end
    #    end
    #    felist =unique(felist)
    #    return
    # end

    # get the fe nearest to the supplied point
    # if (nearestto)
    #
    #    felist = []
    #
    #    # get the locations of all the nodes
    #    xs = fens.xyz
    #    #     Disqualify all the nodes that are not connected to the finite
    #    #     elements on input
    #    cn = connected_nodes(fes)
    #    xs(setdiff(1:size(xs,1),cn),:) =Inf
    #
    #    # get the connectivity of all the FEs
    #    conns = fes.conn
    #
    #    for i = 1:size(locations,1)
    #
    #        # Get the smallest distances between the node locations and the
    #        # desired location using ipdm (by John D'Errico)
    #        distance = ipdm(xs, locations(i,:))
    #
    #        # Get array index of smallest distance to location from
    #        # all nodes
    #        [junk,IX] = min(distance)
    #
    #        # Find the fes connected to the nearest node
    #        [crows, ccols] = find(conns == IX(1))
    #
    #        if numel(crows) == 1
    #            # if there's only one, this is superb, we can add it to the
    #            # list and continue to the next location
    #            felist(end+1) = crows
    #        else
    #
    #            # otherwise we must determine which cell is closest based on
    #            # the other connected nodes
    #            nearconns = conns(crows, :)
    #
    #            nearconndist = zeros(size(nearconns))
    #
    #            # Get the distances between all the nodes in the nearest
    #            # FEs and the nearest node to the location
    #            for j = 1:size(nearconns, 1)
    #                for k = 1:size(nearconns, 2)
    #                    nearconndist(j,k) = ipdm(xs(nearconns(j,k), :), locations(i,:))
    #                end
    #            end
    #
    #            # set the distance from the nearest node to in the cell to
    #            # the location to infinity
    #            nearconndist(nearconns == IX(1)) = inf
    #
    #            # order the FEs by the smallest distance of any connected
    #            # node from the nearest node
    #            [junk,IX] = sort(min(nearconndist,[],2))
    #
    #            # return the index in conns to the first of these ordered
    #            # FEs
    #            if (~isempty(crows))
    #                felist(end+1) = crows(IX(1))
    #            end
    #
    #        end
    #
    #    end
    #
    #    return
    #
    # end

    # get the finite elements within the supplied cylinder
    # if (cylinder)
    #
    #    felist = []
    #
    #    #     Select  the nodes that satisfy this criteria
    #    vl=v_select(fens.xyz,struct ('cylinder',cylinderdata,'inflate',inflate))
    #
    #    # get the connectivity of all the FEs
    #    conns = fes.conn
    #
    #    for i = 1:size(conns,1)
    #        ix=intersect(vl,conns(i,:))
    #        # If all the nodes of the element are  captured by the cylinder, or
    #        # if any node will do
    #        if (length(ix)==size(conns,2)) || (anynode && (length(ix)>0))
    #            felist(end+1) =i
    #        end
    #    end
    #
    #    return
    #
    # end

    # Select by distance from a plane
    if plane is not None:
        nodeset = set(fenode_select(fens, plane=plane, inflate=inflate))
        fesel = []
        for i in numpy.arange(fes.conn.shape[0]):
            match = 0
            for j in numpy.arange(fes.conn.shape[1]):
                match += (fes.conn[i, j] in nodeset)
            if inside and match == fes.conn.shape[1]:
                    fesel.append(i)
            elif anynode and match >= 1:
                fesel.append(i)

        return numpy.array(list(fesel), dtype=int)

    #  Default (implicit) selection method:   Select based on location of nodes
    nodeset = set(fenode_select(fens, box=box, inflate=inflate))
    fesel = []
    for i in numpy.arange(fes.conn.shape[0]):
        match = 0
        for j in numpy.arange(fes.conn.shape[1]):
            match += (fes.conn[i, j] in nodeset)
        if inside:
            if match == fes.conn.shape[1]:
                fesel.append(i)
        elif anynode:
            if match >= 1:
                fesel.append(i)

    return numpy.array(fesel, dtype=int)
