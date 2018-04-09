import numpy
from numpy import dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_defor import FEMMDefor
from spyfe.materials.mat_defor import OUTPUT_CAUCHY
from spyfe.csys import CSys


class FEMMDeforLinear(FEMMDefor):
    """
    The class is for operations on models of mechanical deformation.

    """

    def __init__(self, material=None, fes=None, integration_rule=None,
                 material_csys=CSys(), assoc_geom=None):
        """Constructor.

        :param material: Material object.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        super().__init__(fes=fes, integration_rule=integration_rule,
                         material_csys=material_csys, assoc_geom=assoc_geom)
        self.material = material

    def stiffness(self, geom, u):
        """Compute the stiffness matrix.

        :param geom: Geometry field.
        :param u: Displacement field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        md = self.material.modulidata()  # material stiffness in material CS
        d = numpy.zeros_like(md)  # material stiffness in global CS
        transfd = numpy.zeros_like(d)
        self.material.stress_vector_rotation_matrix(transfd, None)  # generate identity
        if mcs.isconstant and not mcs.isidentity:
            mcsmtx = mcs.eval_matrix()  # constant
            self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
        d_constant = self.material.moduli_are_constant()
        if d_constant:
            self.material.tangent_moduli(md)
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        assm = SysmatAssemblerSparseFixedSymm(fes, u)
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                c = dot(bfuns[j].T, x)
                if not d_constant:
                    self.material.tangent_moduli(md, xyz=c)
                numpy.copyto(d, md)  # moduli in material CS
                if not mcs.isidentity:  # Transformation is required
                    if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                        mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                        self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
                    self.material.rotate_stiffness(d, transfd)  # moduli in global CS
                bmatfun(b, bfuns[j], gradbfun, c)
                assm.elmtx[i, :, :] += dot(b.T, dot((jac * w[j]) * d, b))
        return assm.make_matrix()

    def nz_ebc_loads(self, geom, u):
        """Compute the stiffness matrix.

        :param geom: Geometry field.
        :param u: Displacement field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        md = self.material.modulidata()  # material stiffness in material CS
        d = numpy.zeros_like(md)  # material stiffness in global CS
        transfd = numpy.zeros_like(d)
        self.material.stress_vector_rotation_matrix(transfd, None)  # generate identity
        if mcs.isconstant and not mcs.isidentity:
            mcsmtx = mcs.eval_matrix()  # constant
            self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
        d_constant = self.material.moduli_are_constant()
        if d_constant:
            self.material.tangent_moduli(md)
        pu = numpy.zeros((self.fes.conn.shape[1] * u.dim,), dtype=numpy.float64)
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        assm = SysvecAssembler(fes, u)
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            u.gather_fixed_values_vec(fes.conn[i, :], pu)
            if any(pu != 0.0):
                ke = numpy.zeros((b.shape[1], b.shape[1]))
                for j in range(npts):
                    jacmat = dot(x.T, gradbfunpars[j])
                    jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                    fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                    c = dot(bfuns[j].T, x)
                    if not d_constant:
                        self.material.tangent_moduli(md, xyz=c)
                    numpy.copyto(d, md)  # moduli in material CS
                    if not mcs.isidentity:  # Transformation is required
                        if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                            mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                            self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
                        self.material.rotate_stiffness(d, transfd)  # moduli in global CS
                    bmatfun(b, bfuns[j], gradbfun, c)
                    ke[:, :] += dot(b.T, dot((jac * w[j]) * d, b))
                assm.elvec[i, :] = dot(ke, pu).ravel()
        return assm.make_vector()

    def inspect_integration_points(self, fe_list, inspector, idat,
                                   geom, un1, un, dt=0.0, dtempn1=None,
                                   outcs=CSys(), output=OUTPUT_CAUCHY):
        """Inspect integration point quantities.

        :param fe_list: indexes of the finite elements that are to be inspected:
               The fes to be included are: fes.conn[fe_list, :].
        :param inspector: inspector function,
        :param idat: data for inspector function,
        :param geom: Geometry field.
        :param un1: Displacement field at the time t_n+1.
        :param un: Displacement field at time t_n.
        :param dt: Time step from t_n to t_n+1.
        :param dtempn1: Temperature increment field or None.
        :param outcs: Output coordinate system.
        :return: idat: data for inspector function
        """
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        if mcs.isconstant:
            mcsmtx = mcs.eval_matrix()  # constant
        if outcs is not None and outcs.isconstant:
            outcsmtx = outcs.eval_matrix()  # constant
        if dtempn1 is None:
            dtemps = numpy.zeros((geom.nfens, 1))
        else:
            dtemps = dtempn1.values
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        elemu = numpy.zeros((fes.nfens * un1.dim,))
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        quantityout = None
        vout = None
        for i in fe_list:
            x = geom.values[fes.conn[i, :], :]
            un1.gather_values_vec(fes.conn[i, :], elemu)
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)  # derivatives WRT global CS
                c = dot(bfuns[j].T, x)  # Model location of the quadrature point
                u_c = dot(bfuns[j].T, un1.values[fes.conn[i, :], :])  # Displacement of the quad point
                bmatfun(b, bfuns[j], gradbfun, c)
                strain = dot(b, elemu) # the strains are in the global Cartesian coordinate system
                mstrain = numpy.copy(strain) #strain in material coordinate
                # If necessary, transform the strain from the global CS to the material CS
                if not mcs.isidentity:  # Transformation is required
                    if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                        mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                    self.material.rotate_strain_vector(mstrain, mcsmtx, strain)
                dtemp = dot(bfuns[j].T, dtemps[fes.conn[i, :]])
                quantityout = self.material.state(None, strain=mstrain, dtemp=dtemp,
                                                  output=output, quantityout=quantityout)
                if output == OUTPUT_CAUCHY:  # vector quantity: transformation may be required
                    if vout is None:
                        vout = numpy.zeros_like(quantityout)
                    if not mcs.isidentity:  # Transformation is required
                        self.material.rotate_stress_vector(vout, mcsmtx.T, quantityout)
                        quantityout[:] = vout[:]
                    if not outcs.isidentity:  # Transformation is required
                        if not outcs.isconstant:  # Do I need to evaluate the local mat orient?
                            outcsmtx = outcs.eval_matrix(c, jacmat, fes.label[i])
                            self.material.rotate_stress_vector(vout, outcsmtx, quantityout)
                        quantityout[:] = vout[:]
                else:  #
                    pass
                if inspector is not None:
                    inspector(idat, quantityout, c, u_c, pc[j, :])
        return idat

    def thermal_strain_loads(self, geom, u, dtemp):
        # Compute the thermal-strain load vectors of the finite element set.
        #
        # function F = thermal_strain_loads(self, assembler, geom, u, dT)
        #
        # Return the assembled system vector F.
        #    Arguments
        #     assembler =  descendent of sysvec_assembler
        #     geom=geometry field
        #     u=displacement field
        #     dT=temperature difference field (current temperature minus the
        #         reference temperature at which the solid experiences no strains)
        #
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        if mcs.isconstant:
            mcsmtx = mcs.eval_matrix()  # constant
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        assm = SysvecAssembler(fes, u)
        gradbfun = numpy.zeros((fes.nfens, geom.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfun, gradbfunpars[j], jacmat)
                c = dot(bfuns[j].T, x)
                cdtemp = dot(bfuns[j].T, dtemp.values[fes.conn[i, :], :])
                msigth = self.material.thermal_stress(cdtemp) # stress in material coordinates
                sigth = numpy.copy(msigth)
                if not mcs.isidentity:  # Transformation is required
                    if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                        mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                    self.material.rotate_stress_vector(sigth, mcsmtx.T, msigth)
                bmatfun(b, bfuns[j], gradbfun, c) #strain-displacement and global Cartesian CS
                assm.elvec[i, :] += dot(b.T, sigth * ((-1.) * jac * w[j])).ravel()
        return assm.make_vector()
