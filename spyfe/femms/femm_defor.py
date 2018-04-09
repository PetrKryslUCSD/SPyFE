from spyfe.csys import CSys
import numpy
from numpy import linalg, dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_base import FEMMBase
from spyfe.materials.mat_defor import OUTPUT_CAUCHY
from spyfe.csys import CSys
from spyfe.fenode_to_fe_map import fenode_to_fe_map
from spyfe.fields.nodal_field import NodalField
from spyfe.fields.elemental_field import ElementalField


class FEMMDefor(FEMMBase):
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

    def associate_geometry(self, geom):
        """Associate geometry.

        :param geom: Geometry field.
        :return: Nothing.  The object is modified.
        """
        pass  # do nothing by default

    def distrib_loads(self, geom, u, fi, m):
        fes = self.fes
        bfuns, gradbfunpars, npts, pc, w = self.integration_data()
        bfunst = list()
        for index in range(len(bfuns)):
            bfunst.append(bfuns[index].T)
            bfunst[index] = bfunst[index].ravel()
            bfuns[index].shape = (bfuns[index].shape[0], 1)
        jacmat = numpy.zeros((geom.dim, fes.dim))
        assm = SysvecAssembler(fes, u)
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradbfunpars[j])
                jac = fes.jac_mdim(fes.conn[i, :], bfuns[j], jacmat, x, m)
                f = fi.get_magn(dot(bfunst[j], x), jacmat)
                fn = (bfuns[j] * f.T)
                fn.shape = (assm.elem_mat_nrowcol,)
                assm.elvec[i, :] += fn * (jac * w[j])
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
        :param outcs: output coordinate system
        :param output: type of output
        :return: idat: data for inspector function
        """
        raise Exception(
            'This method needs to be overridden in the derived class')  # need to override in the child class

    def nodal_field_from_integr_points(self, geom, un1, un, dt=0.0, dtempn1=None,
                                       outcs=CSys(), output=OUTPUT_CAUCHY, component=(0,)):
        """Create a nodal field from quantities at integration points.

        The procedure is the universe-distance interpolation.
        :param geom: Geometry field.
        :param un1: Displacement field at the time t_n+1.
        :param un: Displacement field at time t_n.
        :param dt: Time step from t_n to t_n+1.
        :param dtempn1: Temperature increment field or None.
        :param outcs: Output coordinate system.
        :param output: Output quantity (enumeration).
        :param component: Which component of the output quantity?
        :return: nodal field
        """
        # Container of intermediate results
        sum_inv_dist = numpy.zeros((geom.nfens,))
        sum_quant_inv_dist = numpy.zeros((geom.nfens, len(component)))
        fld = NodalField(nfens=geom.nfens, dim=len(component))

        # This is an inverse-distance interpolation inspector.
        def idi(idat, out, xyz, u, pc):
            x, conn = idat
            da = x - numpy.ones((x.shape[0], 1)) * xyz
            d = numpy.sum(da ** 2, axis=1)
            zi = d == 0
            d[zi] = min(d[~zi]) / 1.e9
            invd = numpy.reshape(1. / d, (x.shape[0], 1))
            quant = numpy.reshape(out[component], (1, len(component)))
            sum_quant_inv_dist[conn, :] += invd * quant
            sum_inv_dist[conn] += invd.ravel()
            return

        # Loop over cells to interpolate to nodes
        for i in range(self.fes.conn.shape[0]):
            x1 = geom.values[self.fes.conn[i, :], :]
            idat1 = (x1, self.fes.conn[i, :])
            self.inspect_integration_points([i], idi, idat1,
                                            geom, un1, un, dt, dtempn1,
                                            outcs, output)

        # compute the field data array
        nzi = ~(sum_inv_dist == 0)
        for j in range(len(component)):
            fld.values[nzi, j] = sum_quant_inv_dist[nzi, j] / sum_inv_dist[nzi]
        return fld

    def nodal_field_from_integr_points_spr(self, geom, un1, un, dt=0.0, dtempn1=None,
                                           outcs=CSys(), output=OUTPUT_CAUCHY, component=(0,)):
        """Create a nodal field from quantities at integration points.

        The procedure is the Super-convergent Patch Recovery.
        :param geom: Geometry field.
        :param un1: Displacement field at the time t_n+1.
        :param un: Displacement field at time t_n.
        :param dt: Time step from t_n to t_n+1.
        :param dtempn1: Temperature increment field or None.
        :param outcs: Output coordinate system.
        :param output: Output quantity (enumeration).
        :param component: Which component of the output quantity?
        :return: nodal field
        """
        fes = self.fes
        # Make the inverse map from finite element nodes to finite elements
        femap = fenode_to_fe_map(geom.nfens, fes.conn)
        fld = NodalField(nfens=geom.nfens, dim=len(component))

        # This is an inverse-distance interpolation inspector.
        def idi(idat, out, xyz, u, pc):
            idat.append((numpy.array(out), xyz))

        def spr(idat, xc):
            """Super-convergent Patch Recovery.

            :param idat: List of integration point data.
            :param xc: Location at which the quantities to be recovered.
            :return: Recovered quantity.
            """

            n = len(idat)
            if n == 0:  # nothing can be done
                raise Exception('No data for SPR')
            elif n == 1:  # we got a single value: return it
                out, xyz = idat[0]
                sproutput = out.ravel() / n
            else:  # attempt to solve the least-squares problem
                out, xyz = idat[0]
                dim = xyz.size  # the number of modeling dimensions (1, 2, 3)
                nc = out.size
                na = dim + 1
                if (n >= na):
                    A = numpy.zeros((na, na))
                    b = numpy.zeros((na, nc))
                    pk = numpy.zeros((na, 1))
                    pk[0] = 1.0
                    for k in range(n):
                        out, xyz = idat[k]
                        out.shape = (1, nc)
                        xk = xyz - xc
                        pk[1:] = xk[:].reshape(dim, 1)
                        A += pk * pk.T
                        b += pk * out
                    try:
                        a = numpy.linalg.solve(A, b)
                        # xk = xc - xc
                        # p = [1, xk]
                        sproutput = a[0, :].ravel()
                    except:  # not enough to solve the least-squares problem: compute simple average
                        out, xyz = idat[0]
                        sproutput = out / n
                        for k in range(1, n):
                            out, xyz = idat[k]
                            sproutput += out / n
                        sproutput = sproutput.ravel()
                else:  # not enough to solve the least-squares problem: compute simple average
                    out, xyz = idat[0]
                    sproutput = out / n
                    for k in range(1, n):
                        out, xyz = idat[k]
                        sproutput += out / n
                    sproutput = sproutput.ravel()
            return sproutput

        # Loop over nodes, and for each visit the connected FEs
        for i in range(geom.nfens):
            idat = []
            xc = geom.values[i, :]  # location of the current node
            # construct the list of the relevant integr points
            # and the values at these points
            idat = self.inspect_integration_points(femap[i], idi, idat,
                                                   geom, un1, un, dt, dtempn1,
                                                   outcs, output)
            out1 = spr(idat, xc)
            fld.values[i, :] = out1[:]

        return fld

    def mass(self, geom, u):
        """Compute the mass matrix.

        :param geom: Geometry field.
        :param u: Displacement field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradfunpars, npts, pc, w = self.integration_data()
        assm = SysmatAssemblerSparseFixedSymm(fes, u)
        nexp_nexp = []  # Precomputed for efficiency
        for j in range(npts):
            nexp = numpy.zeros((u.dim, assm.elem_mat_nrowcol))
            for m in range(fes.nfens):
                nexp[:, m * u.dim:(m + 1) * u.dim] = numpy.identity(u.dim) * bfuns[j][m]
            nexp_nexp.append(dot(nexp.T, nexp))
        rho = self.material.rho
        jacmat = numpy.zeros((geom.dim, fes.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                assm.elmtx[i, :, :] += nexp_nexp[j] * (rho * jac * w[j])
        return assm.make_matrix()

    def lumped_mass(self, geom, u):
        """Compute the lumped mass matrix.

        :param geom: Geometry field.
        :param u: Displacement field.
        :return: Sparse matrix.
        """
        fes = self.fes
        bfuns, gradfunpars, npts, pc, w = self.integration_data()
        assm = SysmatAssemblerSparseFixedSymm(fes, u)
        nexp_nexp = []  # Precomputed for efficiency
        for j in range(npts):
            nexp = numpy.zeros((u.dim, assm.elem_mat_nrowcol))
            for m in range(fes.nfens):
                nexp[:, m * u.dim:(m + 1) * u.dim] = numpy.identity(u.dim) * bfuns[j][m]
            nexp_nexp.append(dot(nexp.T, nexp))
        rho = self.material.rho
        jacmat = numpy.zeros((geom.dim, fes.dim))
        me = numpy.zeros((assm.elem_mat_nrowcol, assm.elem_mat_nrowcol))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            me.fill(0.0)
            for j in range(npts):
                jacmat[:, :] = dot(x.T, gradfunpars[j])
                jac = fes.jac_volume(fes.conn[i, :], bfuns[j], jacmat, x)
                me += nexp_nexp[j] * (rho * jac * w[j])
            # Hinton et al. lumping technique
            em2 = numpy.sum(numpy.sum(me, axis=0))
            dem2 = numpy.sum(numpy.diag(me), axis=0)
            me = numpy.diag(numpy.diag(me) / dem2 * em2)
            assm.elmtx[i, :, :] = me
        return assm.make_matrix()

    def elemental_field_from_integr_points(self, geom, un1, un, dt=0.0, dtempn1=None,
                                           outcs=CSys(), output=OUTPUT_CAUCHY, component=(0,)):
        """Create a elemental field from quantities at integration points.

        :param geom: Geometry field.
        :param un1: Displacement field at the time t_n+1.
        :param un: Displacement field at time t_n.
        :param dt: Time step from t_n to t_n+1.
        :param dtempn1: Temperature increment field or None.
        :param outcs: Output coordinate system.
        :param output: Output quantity (enumeration).
        :param component: Which component of the output quantity?
        :return: nodal field
        """
        # Container of intermediate results
        nelems = self.fes.conn.shape[0]
        n_q = numpy.zeros((nelems,))
        sum_q = numpy.zeros((nelems, len(component)))
        fld = ElementalField(nelems=nelems, dim=len(component))

        # This is an inverse-distance interpolation inspector.
        def idi(idat, out, xyz, u, pc):
            i = idat
            quant = numpy.reshape(out[component], (1, len(component)))
            sum_q[i, :] += quant.ravel()
            n_q[i] += 1
            return

        # Loop over cells to interpolate to nodes
        for i in range(self.fes.conn.shape[0]):
            x1 = geom.values[self.fes.conn[i, :], :]
            idat1 = i
            self.inspect_integration_points([i], idi, idat1,
                                            geom, un1, un, dt, dtempn1,
                                            outcs, output)

        # compute the field data array
        for i in range(self.fes.conn.shape[0]):
            for j in range(len(component)):
                fld.values[i, j] = sum_q[i, j] / n_q[i]
        return fld
