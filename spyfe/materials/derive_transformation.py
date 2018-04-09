import sympy
import numpy

T, alpha, R = sympy.symbols('T, alpha, R')
a11, a12, a13, a21, a22, a23, a31, a32, a33 = sympy.symbols('a11, a12, a13, a21, a22, a23, a31, a32, a33')

a = sympy.Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
a = a.transpose()  # a = a'; # his matrix "a"  is the transpose of the featbox matrix "Rm".
print(a)
#   We also use the featbox numbering of the strains.
Numbering = sympy.Matrix([[1, 4, 5], [4, 2, 6], [5, 6, 3]])
Numbering -= sympy.ones(3, 3)
print(Numbering)
T = sympy.Matrix(numpy.zeros((6, 6)))

for i in range(3):
    for j in range(3):
        # if i==j:
        #     alpha = j
        # else:
        #     alpha = 9 - i - j
        alpha = Numbering[i, j]
        for p in range(3):
            for q in range(3):
                beta = Numbering[p, q]
                if alpha < 3 and beta < 3:
                    T[alpha, beta] = a[i, p] * a[i, p]
                if alpha >= 3 and beta < 3:
                    T[alpha, beta] = a[i, p] * a[j, p]
                if alpha < 3 and beta >= 3:
                    T[alpha, beta] = a[i, q] * a[i, p] + a[i, p] * a[i, q]
                if alpha >= 3 and beta >= 3:
                    T[alpha, beta] = a[i, p] * a[j, q] + a[i, q] * a[j, p]
print(T)

R = sympy.eye(6) #Reuter matrix
R[3,3] = R[4,4] = R[5,5] = 2
Tbar = R*T*R**-1
print(Tbar)

v = sympy.MatrixSymbol('v',6,1).as_explicit()
print(Tbar*v)
