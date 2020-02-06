import numpy
from numpy import linalg, dot
from spyfe.assemblers import SysmatAssemblerSparseFixedSymm, SysvecAssembler
from spyfe.femms.femm_base import FEMMBase


class FEMMHeatDiffSurface(FEMMBase):
    """
    The class is for operations on models of heat conduction that operate
    on surfaces (namely convection boundary conditions).

    """

    def __init__(self, fes=None,
                 integration_rule=None,
                 surface_transfer_coeff=lambda x: 0.0):
        """Constructor.

        :param surface_transfer_coeff: Surface heat transfer coefficient function.
        :param fes: Finite element set object.
        :param integration_rule: Integration rule object.
        """
        super().__init__(fes=fes, integration_rule= integration_rule)
        self.surface_transfer_coeff = surface_transfer_coeff

    def surface_transfer(self, geom, temp):
        fes = self.fes
        Ns, gradNpars, npts, pc, w = self.integration_data()
        Hedim = temp.dim * fes.nfens
        Assm = SysmatAssemblerSparseFixedSymm(fes, temp)
        temp.gather_all_dofnums(fes.conn, Assm.dofnums)
        J = numpy.zeros((geom.dim, geom.dim))
        for i in range(fes.conn.shape[0]):
            x = geom.values[fes.conn[i, :], :]
            for j in range(npts):
                c = dot(Ns[j].T, x)
                J = dot(x.T, gradNpars[j])
                Jac = fes.jac_volume(fes.conn[i, :], Ns[j], J, x)
                h = self.surface_transfer_coeff(c)
                Assm.elmtx[i, :, :] += dot((h * Jac * w[j]) * Ns[j], Ns[j].T)
        return Assm.make_matrix()

    def surface_transfer_loads(self, geom, temp, amb):
        fes = self.fes
        Ns, gradNpars, npts, pc, w = self.integration_data()
        conns = fes.conn  # input connectivity
        Hedim = temp.dim * fes.nfens
        Assm = SysvecAssembler(fes, temp)
        He = numpy.zeros((Hedim, Hedim))
        temp.gather_all_dofnums(fes.conn, Assm.dofnums)
        pT = numpy.zeros((self.fes.conn.shape[1] * temp.dim,), dtype=numpy.float64)
        for i in range(conns.shape[0]):
            conn = conns[i, :]
            amb.gather_fixed_values_vec(conn, pT)
            if any(pT != 0.0):
                x = geom.values[conn, :]
                He.fill(0.0)
                for j in range(npts):
                    c = dot(Ns[j].T, x)
                    J = dot(x.T, gradNpars[j])
                    Jac = fes.jac_volume(conn, Ns[j], J, x)
                    h = self.surface_transfer_coeff(c)
                    He += dot((h * Jac * w[j]) * Ns[j], Ns[j].T)
                Assm.elvec[i, :] = dot(He, pT).ravel()
        return Assm.make_vector()

    def nz_ebc_loads_surface_transfer(self, geom, temp):
        fes = self.fes
        Ns, gradNpars, npts, pc, w = self.integration_data()
        conns = fes.conn  # input connectivity
        Hedim = temp.dim * fes.nfens
        Assm = SysvecAssembler(fes, temp)
        He = numpy.zeros((Hedim, Hedim))
        temp.gather_all_dofnums(fes.conn, Assm.dofnums)
        pT = numpy.zeros((self.fes.conn.shape[1] * temp.dim,), dtype=numpy.float64)
        for i in range(conns.shape[0]):
            conn = conns[i, :]
            temp.gather_fixed_values_vec(conn, pT)
            if any(pT != 0.0):
                x = geom.values[conn, :]
                He.fill(0.0)
                for j in range(npts):
                    c = dot(Ns[j].T, x)
                    J = dot(x.T, gradNpars[j])
                    Jac = fes.jac_volume(conn, Ns[j], J, x)
                    h = self.surface_transfer_coeff(c)
                    He += dot((h * Jac * w[j]) * Ns[j], Ns[j].T)
                Assm.elvec[i, :] = -dot(He, pT).ravel()
        return Assm.make_vector()
