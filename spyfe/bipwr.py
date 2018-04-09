import numpy
from scipy.sparse.linalg import splu
from numpy import dot
from numpy.linalg import norm, qr

def gepbinvpwr2(K, M, v, tol, maxiter):
    """Block inverse power method.

     Block inverse power method for k smallest eigenvalues of the generalized
     eigenvalue problem
               K*v= lambda*M*v

    (C) 2008-2016, Petr Krysl
    :param K: square stiffness matrix,
    :param M: square mass matrix,
    :param v: initial guess of the eigenvectors (for instance random),
    :param tol: relative tolerance on the eigenvalue, expressed in terms of norms of the
          change of the eigenvalue estimates from iteration to iteration.
    :param maxiter: maximum number of allowed iterations
    :return: lamb, v, converged
    lambda = computed eigenvalue,
    v= computed eigenvector,
    converged= Boolean flag, converged or not?
    """
    nvecs = v.shape[1]  # How many eigenvalues?
    plamb = numpy.zeros((nvecs,))  # previous eigenvalue
    lamb = numpy.zeros((nvecs,))
    lu = splu(K.tocsc())
    converged = False  # not yet
    for i in range(maxiter):
        u = lu.solve(M.dot(v))
        v, r = qr(u, mode='reduced')  # economy factorization
        for j in range(nvecs):
            lamb[j] = dot(v[:, j].T, K.dot(v[:, j])) / dot(v[:, j].T, M.dot(v[:, j]))
        if norm(lamb - plamb) / norm(lamb) < tol:
            converged = True
            break
        plamb[:] = lamb[:]
        print(lamb[:])
    return lamb, v, converged
