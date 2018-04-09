# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:50:05 2017

@author: PetrKrysl
"""
import numpy
cimport numpy
cimport libc
import cython
from cython cimport boundscheck, wraparound

@cython.boundscheck(False)
def gradbfun2dcy(double[:, :] gradbfunpars, double[:, :] redjacmat, double[:, :] gradbfunout):
    # Compute the gradient of the basis functions with the respect to the "reduced" spatial coordinates.
    # 
    # gradN= output, matrix of gradients, one per row
    # gradbfunpars= matrix of gradients with respect to parametric coordinates, one per row
    # redjacmat= reduced Jacobian matrix redjacmat=Rm'*J
    # 
    # This is the unrolled version that avoids allocation of a 2 x 2 matr
    cdef double r00 = redjacmat[0, 0]
    cdef double r01 = redjacmat[0, 1]
    cdef double r10 = redjacmat[1, 0]
    cdef double r11 = redjacmat[1, 1]
    cdef double invdet = 1.0 / (r00 * r11 - r01 * r10)
    cdef double invredjacmat11 = (r11) * invdet
    cdef double invredjacmat12 = -(r01) * invdet
    cdef double invredjacmat21 = -(r10) * invdet
    cdef double invredjacmat22 = (r00) * invdet
    cdef double Temp0    
    cdef double Temp1
    cdef int nr = gradbfunpars.shape[0]
    cdef int r
    for r in range(nr):
        Temp0 = gradbfunpars[r, 0]
        Temp1 = gradbfunpars[r, 1]
        gradbfunout[r, 0] = Temp0 * invredjacmat11 + Temp1 * invredjacmat21
        gradbfunout[r, 1] = Temp0 * invredjacmat12 + Temp1 * invredjacmat22
                
@cython.boundscheck(False)
def gradbfun3dcy(double[:, :] gradbfunpars, double[:, :] redjacmat, double[:, :] gradbfunout):
    # Compute the gradient of the basis functions with the respect to the "reduced" spatial coordinates.
    # 
    # gradN= output, matrix of gradients, one per row
    # gradbfunpars= matrix of gradients with respect to parametric coordinates, one per row
    # redjacmat= reduced Jacobian matrix redjacmat=Rm'*J
    # 
    cdef double invdet = 1.0 / (+redjacmat[0, 0] * (redjacmat[1, 1] * redjacmat[2, 2] - redjacmat[2, 1] * redjacmat[1, 2])
                       - redjacmat[0, 1] * (redjacmat[1, 0] * redjacmat[2, 2] - redjacmat[1, 2] * redjacmat[2, 0])
                       + redjacmat[0, 2] * (redjacmat[1, 0] * redjacmat[2, 1] - redjacmat[1, 1] * redjacmat[2, 0]))
    cdef double     invredjacmat11 = (redjacmat[1, 1] * redjacmat[2, 2] - redjacmat[2, 1] * redjacmat[1, 2]) * invdet
    cdef double     invredjacmat12 = -(redjacmat[0, 1] * redjacmat[2, 2] - redjacmat[0, 2] * redjacmat[2, 1]) * invdet
    cdef double     invredjacmat13 = (redjacmat[0, 1] * redjacmat[1, 2] - redjacmat[0, 2] * redjacmat[1, 1]) * invdet
    cdef double     invredjacmat21 = -(redjacmat[1, 0] * redjacmat[2, 2] - redjacmat[1, 2] * redjacmat[2, 0]) * invdet
    cdef double     invredjacmat22 = (redjacmat[0, 0] * redjacmat[2, 2] - redjacmat[0, 2] * redjacmat[2, 0]) * invdet
    cdef double     invredjacmat23 = -(redjacmat[0, 0] * redjacmat[1, 2] - redjacmat[1, 0] * redjacmat[0, 2]) * invdet
    cdef double     invredjacmat31 = (redjacmat[1, 0] * redjacmat[2, 1] - redjacmat[2, 0] * redjacmat[1, 1]) * invdet
    cdef double     invredjacmat32 = -(redjacmat[0, 0] * redjacmat[2, 1] - redjacmat[2, 0] * redjacmat[0, 1]) * invdet
    cdef double     invredjacmat33 = (redjacmat[0, 0] * redjacmat[1, 1] - redjacmat[1, 0] * redjacmat[0, 1]) * invdet
    cdef int r, n=gradbfunout.shape[0]
    for r in range(n):
        gradbfunout[r, 0] = gradbfunpars[r, 0] * invredjacmat11 + gradbfunpars[r, 1] * invredjacmat21 + gradbfunpars[r, 2] * invredjacmat31
        gradbfunout[r, 1] = gradbfunpars[r, 0] * invredjacmat12 + gradbfunpars[r, 1] * invredjacmat22 + gradbfunpars[r, 2] * invredjacmat32
        gradbfunout[r, 2] = gradbfunpars[r, 0] * invredjacmat13 + gradbfunpars[r, 1] * invredjacmat23 + gradbfunpars[r, 2] * invredjacmat33

 
#def three_matrix_product(double[:, :] out, double[:, :] bt, double[:, :] d, double[:, :] b):
#    cdef int r, c, k, m, nr=bt.shape[0], nc=b.shape[1], ni=d.shape[0]
#    with boundscheck(False), wraparound(False):
#        #cdef int r, c, k, m, nr=bt.shape[0], nc=b.shape[1], ni=d.shape[0]
#        #cdef double accumulator 
#        for r in range(nr):
#            for c in range(nc):
#                accumulator = 0.0
#                for k in range(ni):
#                    for m in range(ni):
#                        accumulator += bt[r,k]*d[k,m]*b[m,c]
#                out[r,c]   = accumulator     

@cython.boundscheck(False)     
@cython.wraparound(False)   
def three_matrix_product(double[:, :] out, double[:, :] bt, double[:, :] d, double[:, :] b):
    cdef:
        int r, c, k, m, nr=bt.shape[0], nc=b.shape[1], ni=d.shape[0]
        double accumulator = 0.0
    for r in range(nr):
        for c in range(nc):
            accumulator = 0.0
            for k in range(ni):
                for m in range(ni):
                    accumulator += bt[r,k]*d[k,m]*b[m,c]
            out[r,c]   = accumulator  
               

               
@cython.boundscheck(False)              
def matrix_2_x_2_det(double[:, :] jacmat):
    cdef double result = jacmat[0, 0] * jacmat[1, 1] - jacmat[0, 1] * jacmat[1, 0]
    return result