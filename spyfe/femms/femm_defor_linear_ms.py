import numpy
from numpy import dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
import copy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso, ID3
from spyfe.materials.mat_defor import OUTPUT_CAUCHY
from spyfe.csys import CSys


class FEMMDeforLinearMS(FEMMDeforLinear):
    """
    Class for small-strain linear deformation based on the mean-strain
    technology with stabilization by full quadrature (energy-sampling stabilization).

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
        self._phis = None
        e = getattr(self.material, 'e', None)
        if e is not None:
            e = self.material.e
            if self.material.nu < 0.3:
                nu = self.material.nu
            else:
                nu = 0.3 + (self.material.nu - 0.3) / 2.0
        else:
            e1 = getattr(self.material, 'e1', None)
            if e1 is not None:
                e = min(self.material.e1, self.material.e2, self.material.e3)
                nu = min(self.material.nu12, self.material.nu13, self.material.nu23)
            else:
                raise Exception('No clues on how to construct the stabilization material')
        self.stabilization_material = MatDeforTriaxLinearIso(e=e, nu=nu)
        # Now try to figure out which Poisson ratio to use in the optimal scaling
        # factor(to account for geometry)
        nu = getattr(self.material, "nu", None)
        if nu is not None:
            self.nu = self.material.nu
        else:
            nu12 = getattr(self.material, "nu12", None)
            if nu12 is not None:
                self.nu = max(self.material.nu12, self.material.nu13, self.material.nu23)
            else:
                raise Exception('No clues on how to construct the stabilization material')

    def associate_geometry(self, geom):
        """Associate geometry.

        This method needs to be called before any computation with the FEMM is performed.
        It calculates the stabilization factors dependent on the geometry.
        :param geom: Geometry field.
        :return: Nothing.  The object is modified.
        """
        raise Exception('Needs to be overridden')

    def stiffness(self, geom, u):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        md = self.material.modulidata()  # material stiffness in material CS
        d = numpy.zeros_like(md)  # material stiffness in global CS
        dstab = numpy.zeros_like(d)
        transfd = numpy.zeros_like(d)
        self.material.stress_vector_rotation_matrix(transfd, None) #generate identity
        if mcs.isconstant and not mcs.isidentity:
            mcsmtx = mcs.eval_matrix()  # constant
            self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
        d_constant = self.material.moduli_are_constant()
        if d_constant:
            self.material.tangent_moduli(md)
        self.stabilization_material.tangent_moduli(dstab)  # the stabilization material assumed to have constant moduli
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        bbar = copy.copy(b)
        assm = SysmatAssemblerSparseFixedSymm(fes, u)
        gradbfun = list()
        jac = list()
        gradbfunmean = numpy.zeros((fes.nfens, geom.dim))
        for j in range(npts):
            gradbfun.append(numpy.zeros((fes.nfens, geom.dim)))
            jac.append(0.0)
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            # Calculate mean basis function gradient + volume of the element
            vol = 0
            gradbfunmean.fill(0.0)
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac[j] = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfun[j], gradbfunpars[j], jacmat)
                dvol = (jac[j] * w[j])
                gradbfunmean += gradbfun[j] * dvol
                vol = vol + dvol
            gradbfunmean /= vol
            bmatfun(bbar, None, gradbfunmean, None)  # strain-displacement matrix in global CS
            if not d_constant:
                self.material.tangent_moduli(md)
            numpy.copyto(d, md) # moduli in material CS
            if not mcs.isidentity:  # Transformation is required
                if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                    mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                    self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
                self.material.rotate_stiffness(d, transfd)  # moduli in global CS
            # These matrices are in the global Cartesian coordinate system
            assm.elmtx[i, :, :] = dot(bbar.T, dot(vol * (d - self._phis[i] * dstab), bbar))
            for j in range(npts):
                bmatfun(b, None, gradbfun[j], None)  # strain-displacement matrix in global CS
                assm.elmtx[i, :, :] += dot(b.T, dot((self._phis[i] * jac[j] * w[j]) * dstab, b))
        return assm.make_matrix()

    def nz_ebc_loads(self, geom, u):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        md = self.material.modulidata()  # material stiffness in material CS
        d = numpy.zeros_like(md)  # material stiffness in global CS
        dstab = numpy.zeros_like(d)
        transfd = numpy.zeros_like(d)
        self.material.stress_vector_rotation_matrix(transfd, None)  # generate identity
        if mcs.isconstant and not mcs.isidentity:
            mcsmtx = mcs.eval_matrix()  # constant
            self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
        d_constant = self.material.moduli_are_constant()
        if d_constant:
            self.material.tangent_moduli(md)
        self.stabilization_material.tangent_moduli(dstab)  # the stabilization material assumed to have constant moduli
        pu = numpy.zeros((self.fes.conn.shape[1] * u.dim,), dtype=numpy.float64)
        bmatfun, b = self.fes.bmatdata(self.fes.nfens)
        bbar = copy.copy(b)
        assm = SysvecAssembler(fes, u)
        gradbfun = list()
        jac = list()
        gradbfunmean = numpy.zeros((fes.nfens, geom.dim))
        for j in range(npts):
            gradbfun.append(numpy.zeros((fes.nfens, geom.dim)))
            jac.append(0.0)
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            u.gather_fixed_values_vec(fes.conn[i, :], pu)
            if any(pu != 0.0):
                # Calculate mean basis function gradient + volume of the element
                vol = 0
                gradbfunmean.fill(0.0)
                for j in range(npts):
                    jacmat = dot(x.T, gradbfunpars[j])
                    jac[j] = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                    fes.gradbfun(gradbfun[j], gradbfunpars[j], jacmat)
                    dvol = (jac[j] * w[j])
                    gradbfunmean += gradbfun[j] * dvol
                    vol = vol + dvol
                gradbfunmean /= vol
                bmatfun(bbar, None, gradbfunmean, None)  # strain-displacement matrix in global CS
                if not d_constant:
                    self.material.tangent_moduli(md)
                numpy.copyto(d, md)  # moduli in material CS
                if not mcs.isidentity:  # Transformation is required
                    if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                        mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                        self.material.stress_vector_rotation_matrix(transfd, mcsmtx.T)
                    self.material.rotate_stiffness(d, transfd)  # moduli in global CS
                # These matrices are in the global Cartesian coordinate system
                ke = dot(bbar.T, dot(vol * (d - self._phis[i] * dstab), bbar))
                for j in range(npts):
                    bmatfun(b, None, gradbfun[j], None)  # strain-displacement matrix in global CS
                    ke += dot(b.T, dot((self._phis[i] * jac[j] * w[j]) * dstab, b))
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
        bmatfun, bbar = self.fes.bmatdata(self.fes.nfens)
        elemu = numpy.zeros((fes.nfens * un1.dim,))
        gradbfun = list()
        jac = list()
        gradbfunmean = numpy.zeros((fes.nfens, geom.dim))
        for j in range(npts):
            gradbfun.append(numpy.zeros((fes.nfens, geom.dim)))
            jac.append(0.0)
        quantityout = None
        vout = None
        for i in fe_list:
            x = geom.values[fes.conn[i, :], :]
            # Calculate mean basis function gradient + volume of the element
            vol = 0
            gradbfunmean.fill(0.0)
            for j in range(npts):
                jacmat = dot(x.T, gradbfunpars[j])
                jac[j] = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfun[j], gradbfunpars[j], jacmat)
                dvol = jac[j] * w[j]
                gradbfunmean += gradbfun[j] * dvol
                vol = vol + dvol
            gradbfunmean /= vol
            bmatfun(bbar, None, gradbfunmean, None)
            un1.gather_values_vec(fes.conn[i, :], elemu)
            c = dot(bfuns[j].T, x)  # Model location of the quadrature point
            u_c = dot(bfuns[j].T, un1.values[fes.conn[i, :], :])  # Displacement of the quad point
            strain = dot(bbar, elemu) # strain in global Cartesian coordinate system
            mstrain = numpy.copy(strain) #strain in material coordinate
            # If necessary, transform the strain from the global CS to the material CS
            if not mcs.isidentity:  # Transformation is required
                if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                    mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                self.material.rotate_strain_vector(mstrain, mcsmtx, strain)
            dtemp = dot(bfuns[j].T, dtemps[fes.conn[i, :]])
            quantityout = self.material.state(None, strain=strain, dtemp=dtemp,
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
