import numpy
from numpy import dot
from numpy.linalg import inv
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_defor_linear import FEMMDeforLinear
import copy
from spyfe.materials.mat_defor_triax_linear_iso import MatDeforTriaxLinearIso
from spyfe.materials.mat_defor_triax import strain_vector_rotation_matrix
from spyfe.materials.mat_defor import OUTPUT_CAUCHY
from spyfe.csys import CSys


class FEMMDeforNonlinearQT10MS(FEMMDeforLinear):
    """
    Class for large-strain nonlinear deformation based on the mean-strain
     quadratic tetrahedron and stabilization by full four-point quadrature.

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
        self._gamma = 2.6 #one of the stabilization parameters
        self._C = 1.e4 #the other stabilization parameter
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

        :param geom: Geometry field.
        :return: Nothing.  The object is modified.
        """
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        jacmat = numpy.zeros((geom.dim, self.fes.dim))
        gradbfun = numpy.zeros((self.fes.nfens, geom.dim))
        self._phis = numpy.zeros((self.fes.conn.shape[0],))
        for i in range(self.fes.conn.shape[0]):
            x = geom.values[self.fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                condjacmat = numpy.linalg.cond(jacmat)
                cap_phi = self._C * (1. / condjacmat)** (self._gamma)
                phi = cap_phi / (1. + cap_phi)
                self._phis[i] = max(self._phis[i], phi)

    def stiffness(self, geom, un1, un, dt):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        mcs = self.material_csys
        if mcs.isconstant:
            mcsmtx = mcs.eval_matrix()  # constant
        bmatfun, b = self.fes.bmatdata()
        d = self.material.modulidata()
        dstab = numpy.zeros_like(d)
        bbar = copy.copy(b) #buffer
        jacmat = numpy.zeros((geom.dim, self.fes.dim)) #buffer
        fnbar  = numpy.zeros((geom.dim, geom.dim))  # buffer
        fn1bar = numpy.zeros((geom.dim, geom.dim))  # buffer
        fn1 = numpy.zeros((geom.dim, geom.dim))  # buffer
        gradbfun = list()
        jac = list()
        gradbfunmean = numpy.zeros((fes.nfens, geom.dim))
        for j in range(npts):
            gradbfun.append(numpy.zeros((fes.nfens, geom.dim)))
            jac.append(0.0)
        assm = SysmatAssemblerSparseFixedSymm(fes, un1)
        # Now we loop over all finite elements
        for i in range(fes.conn.shape[0]): # For all elements
            X = geom.values[fes.conn[i, :], :]
            xn = X + un.values[fes.conn[i, :], :]
            xn1 = X + un1.values[fes.conn[i, :], :]
            # Calculate mean basis function gradient + volume of the element
            vol = 0
            gradbfunmean.fill(0.0)
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                jac[j] = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfunpars[j], jacmat, gradbfun[j])
                dvol = (jac[j] * w[j])
                gradbfunmean += gradbfun[j] * dvol
                vol = vol + dvol
            gradbfunmean /= vol
            fnbar[:,:] = dot(xn.T, gradbfunmean)
            fn1bar[:,:] = dot(xn1.T, gradbfunmean)
            if not mcs.isidentity:  # Transformation is required
                if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                    mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                    mcsmtxT = mcsmtx.T
                fnbar[:, :] = dot(mcsmtxT, dot(fnbar, mcsmtx))
                fn1bar[:, :] = dot(mcsmtxT, dot(fn1bar, mcsmtx))
            bmatfun(None, dot(gradbfunmean, inv(fn1bar)), None, None, bbar)
            self.material.tangent_moduli(fn1=fn1bar, fn=fnbar, dt=dt, ms=self.matstates[i], d)
            self.stabilization_material.tangent_moduli(fn1=fn1bar, dstab)
            if not mcs.isidentity:  # Transformation is required
                d = self.material.rotate_stiffness(d, mcsmtxT)
                dstab = self.stabilization_material.rotate_stiffness(dstab, mcsmtxT)
            assm.elmtx[i, :, :] = dot(bbar.T, dot(vol * (d - self._phis[i] * dstab), bbar))
            for j in range(npts):
                fn1[:, :] = dot(xn1.T, gradbfun[j])
                bmatfun(None, dot(gradbfun[j], inv(fn1)), None, None, b)
                self.stabilization_material.tangent_moduli(fn1=fn1, dstab)
                assm.elmtx[i, :, :] += dot(b.T, dot((self._phis[i] * jac[j] * w[j]) * dstab, b))
        return assm.make_matrix()

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
        bmatfun, bbar = self.fes.bmatdata()
        elemu = numpy.zeros((fes.nfens * un1.dim,))
        jacmat = numpy.zeros((geom.dim, self.fes.dim))
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
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                jac[j] = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                fes.gradbfun(gradbfunpars[j], jacmat, gradbfun[j])
                dvol = jac[j] * w[j]
                gradbfunmean += gradbfun[j] * dvol
                vol = vol + dvol
            gradbfunmean /= vol
            bmatfun(None, gradbfunmean, None, None, bbar)
            un1.gather_values_vec(fes.conn[i, :], elemu)
            c = dot(bfuns[j].T, x)  # Model location of the quadrature point
            u_c = dot(bfuns[j].T, un1.values[fes.conn[i, :], :])  # Displacement of the quad point
            strain = dot(bbar, elemu)
            if not mcs.isidentity:  # Transformation is required
                if not mcs.isconstant:  # Do I need to evaluate the local mat orient?
                    mcsmtx = mcs.eval_matrix(c, jacmat, fes.label[i])
                strain = dot(self.material.strain_vector_rotation_matrix(mcsmtx), strain)
            dtemp = dot(bfuns[j].T, dtemps[fes.conn[i, :]])
            quantityout = self.material.state(None, strain=strain, dtemp=dtemp,
                                              output=output, quantityout=quantityout)
            if output == OUTPUT_CAUCHY:  # vector quantity: transformation may be required
                if vout is None:
                    vout = numpy.zeros_like(quantityout)
                if not mcs.isidentity:  # Transformation is required
                    self.material.rotate_stress_vector(mcsmtx.T, quantityout, vout)
                    quantityout[:] = vout[:]
                if not outcs.isidentity:  # Transformation is required
                    if not outcs.isconstant:  # Do I need to evaluate the local mat orient?
                        outcsmtx = outcs.eval_matrix(c, jacmat, fes.label[i])
                    self.material.rotate_stress_vector(outcsmtx, quantityout, vout)
                    quantityout[:] = vout[:]
            else:  #
                pass
            if inspector is not None:
                inspector(idat, quantityout, c, u_c, pc[j, :])
        return idat
