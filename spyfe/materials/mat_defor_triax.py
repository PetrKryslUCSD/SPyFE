import numpy
from spyfe.materials.mat_defor import MatDefor
from spyfe.materials.mat_defor import ID3

M1 = numpy.reshape(numpy.array([1., 1., 1., 0., 0., 0.]), (6, 1))
M1M1T = M1 * M1.T
MI = numpy.diag([1, 1, 1, 0.5, 0.5, 0.5])


class MatDeforTriax(MatDefor):
    """
    Triaxial deformation material class.


    """
    def __init__(self, rho=0.0):
        super().__init__(rho=rho)

    def modulidata(self):
        """Return an array buffer for the matrix of elastic moduli.

        :return: an appropriately-sized buffer (array) for the matrix of tangent moduli.
        """

        d = numpy.zeros((6, 6))
        return d

    def Lagrangean_to_Eulerian(self, C, F):
        """Convert a Lagrangean constitutive matrix to an Eulerian one .
    
        Convert a Lagrangean constitutive matrix to an Eulerian one .
                NOTE :the Lagrangean matrix is presumed symmetric .
        :param C: Lagrangean constitutive matrix ,6 x6 ,symmetric
        :param F: current deformation gradient ,F_ij = \partial x_i / \partial X_j
        :return:
        """
        cout = numpy.zeros((6,6))
        raise Exception('Not implemented yet')
        return cout

    def stress_vector_rotation_matrix(self, Tout, Rm=None):
        """Calculate the 6 x 6 rotation matrix for a stress vector.
    
             Rm = 3 x 3 orthogonal matrix; columns are components of 'bar' basis vectors on the 'plain'
                  basis vectors
            
             Calculate the rotation of the 'plain' stress vector to the
             'bar' coordinate system given by the columns of the rotation matrix Rm.
            
             Example:
             The stress vector "stress" is given in the material coordinate
             system defined by the orientation matrix Rm. The following two
             transformations are equivalent:
             (i)
                    mat.stress_3x3t_from_6v(t, stress)
                    t = dot(Rm, dot(t, Rm.T)) # in global coordinate system
                    t = dot(outputRm.T, dot(t, outputRm)) # in output coordinate system
                    mat.stress_6v_from_3x3t(t, stress) # in output coordinate system
             (ii)
                    stress = dot(mat.stress_vector_rotation_matrix(outputRm),
                              dot(mat.stress_vector_rotation_matrix(Rm.T), stress)) # in output coordinate system
    
    
             Derivation of the transformation matrix [T]
             This is from Barbero''s  book Finite element analysis of composite
             materials  using Abaqus.  Note that his matrix "a"  is the transpose of
             the featbox matrix "Rm".
             Note: We use the featbox numbering of the strains.
            
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
        :return:
        """
        if Rm is None:
            Rm = ID3

        a11 = Rm[0, 0]
        a12 = Rm[0, 1]
        a13 = Rm[0, 2]
        a21 = Rm[1, 0]
        a22 = Rm[1, 1]
        a23 = Rm[1, 2]
        a31 = Rm[2, 0]
        a32 = Rm[2, 1]
        a33 = Rm[2, 2]
        Tout[:, :] = numpy.array([
            [a11 ** 2, a21 ** 2, a31 ** 2, 2 * a11 * a21, 2 * a11 * a31, 2 * a21 * a31],
            [a12 ** 2, a22 ** 2, a32 ** 2, 2 * a12 * a22, 2 * a12 * a32, 2 * a22 * a32],
            [a13 ** 2, a23 ** 2, a33 ** 2, 2 * a13 * a23, 2 * a13 * a33, 2 * a23 * a33],
            [a11 * a12, a21 * a22, a31 * a32, a11 * a22 + a12 * a21, a11 * a32 + a12 * a31, a21 * a32 + a22 * a31],
            [a11 * a13, a21 * a23, a31 * a33, a11 * a23 + a13 * a21, a11 * a33 + a13 * a31, a21 * a33 + a23 * a31],
            [a12 * a13, a22 * a23, a32 * a33, a12 * a23 + a13 * a22, a12 * a33 + a13 * a32, a22 * a33 + a23 * a32]])

    def rotate_stress_vector(self, vout, Rm, v):
        """Calculate the stress vector rotated by the supplied rotation matrix.
    
        Calculate the rotation of the 'plain' stress vector to the
        'bar' coordinate system given by the columns of the rotation matrix Rm.
        
        For details refer to: stress_vector_rotation_matrix
    
        :param vout: output stress vector
        :param Rm: columns are components of 'bar' basis vectors on the 'plain'  basis vectors
        :param v: input stress vector
        :return: Nothing
        """
        a11 = Rm[0, 0]
        a12 = Rm[0, 1]
        a13 = Rm[0, 2]
        a21 = Rm[1, 0]
        a22 = Rm[1, 1]
        a23 = Rm[1, 2]
        a31 = Rm[2, 0]
        a32 = Rm[2, 1]
        a33 = Rm[2, 2]
        vout[0] = v[0] * a11 ** 2 + 2 * v[3] * a11 * a21 \
                               + 2 * v[4] * a11 * a31 + v[1] * a21 ** 2 + 2 * v[5] * a21 * a31 + v[2] * a31 ** 2
        vout[1] = v[0] * a12 ** 2 + 2 * v[3] * a12 * a22 \
                               + 2 * v[4] * a12 * a32 + v[1] * a22 ** 2 + 2 * v[5] * a22 * a32 + v[2] * a32 ** 2
        vout[2] = v[0] * a13 ** 2 + 2 * v[3] * a13 * a23 \
                               + 2 * v[4] * a13 * a33 + v[1] * a23 ** 2 + 2 * v[5] * a23 * a33 + v[2] * a33 ** 2
        vout[3] = v[3] * (a11 * a22 + a12 * a21) + v[4] * (a11 * a32 + a12 * a31) \
                  + v[5] * (a21 * a32 + a22 * a31) + a11 * a12 *    v[0] + a21 * a22 * v[1] + a31 * a32 * v[2]
        vout[4] = v[3] * (a11 * a23 + a13 * a21) + v[4] * (a11 * a33 + a13 * a31) \
                  + v[5] * (a21 * a33 + a23 * a31) + a11 * a13 *v[0] + a21 * a23 * v[1] + a31 * a33 * v[2]
        vout[5] = v[3] * (a12 * a23 + a13 * a22) + v[4] * (a12 * a33 + a13 * a32) \
                  + v[5] * (a22 * a33 + a23 * a32) + a12 * a13 *v[0] + a22 * a23 * v[1] + a32 * a33 * v[2]

    def strain_vector_rotation_matrix(self, Tout, Rm=None):
        """Calculate the 6 x 6 rotation matrix for a strain vector.
    
            Calculate the rotation of the 'plain' strain vector to the
            'bar' coordinate system given by the columns of the rotation matrix Rm .

        :param Rm: =columns are components of 'bar' basis vectors on the 'plain'
            basis vectors

        :return: 6 x 6 rotation matrix
        """
        if Rm is None:
            Rm = ID3

        a11 = Rm[0, 0]
        a12 = Rm[0, 1]
        a13 = Rm[0, 2]
        a21 = Rm[1, 0]
        a22 = Rm[1, 1]
        a23 = Rm[1, 2]
        a31 = Rm[2, 0]
        a32 = Rm[2, 1]
        a33 = Rm[2, 2]
        Tout[:, :] = numpy.array([
            [a11 ** 2, a21 ** 2, a31 ** 2, a11 * a21, a11 * a31, a21 * a31],
            [a12 ** 2, a22 ** 2, a32 ** 2, a12 * a22, a12 * a32, a22 * a32],
            [a13 ** 2, a23 ** 2, a33 ** 2, a13 * a23, a13 * a33, a23 * a33],
            [2 * a11 * a12, 2 * a21 * a22, 2 * a31 * a32, a11 * a22 + a12 * a21, a11 * a32 + a12 * a31, a21 * a32 + a22 * a31],
            [2 * a11 * a13, 2 * a21 * a23, 2 * a31 * a33, a11 * a23 + a13 * a21, a11 * a33 + a13 * a31, a21 * a33 + a23 * a31],
            [2 * a12 * a13, 2 * a22 * a23, 2 * a32 * a33, a12 * a23 + a13 * a22, a12 * a33 + a13 * a32, a22 * a33 + a23 * a32]])

    def rotate_strain_vector(self, vout, Rm, v):
        """Calculate the strain vector rotated by the supplied rotation matrix.

        Calculate the rotation of the 'plain' strain vector to the
        'bar' coordinate system given by the columns of the rotation matrix Rm.

        For details refer to: strain_vector_rotation_matrix

        :param vout: output strain vector
        :param Rm: columns are components of 'bar' basis vectors on the 'plain'  basis vectors
        :param v: input strain vector
        :return: Nothing
        """
        a11 = Rm[0, 0]
        a12 = Rm[0, 1]
        a13 = Rm[0, 2]
        a21 = Rm[1, 0]
        a22 = Rm[1, 1]
        a23 = Rm[1, 2]
        a31 = Rm[2, 0]
        a32 = Rm[2, 1]
        a33 = Rm[2, 2]
        vout[0] = a11 ** 2 * v[0] + a11 * a21 * v[3] + a11 * a31 * v[4] + a21 ** 2 * v[1] \
                  + a21 * a31 * v[5] + a31 ** 2 * v[2]
        vout[1] = a12 ** 2 * v[0] + a12 * a22 * v[3] + a12 * a32 * v[4] + a22 ** 2 * v[1] \
                  + a22 * a32 * v[5] + a32 ** 2 * v[2]
        vout[2] = a13 ** 2 * v[0] + a13 * a23 * v[3] + a13 * a33 * v[4] + a23 ** 2 * v[1] \
                  + a23 * a33 * v[5] + a33 ** 2 * v[2]
        vout[3] = 2 * a11 * a12 * v[0] + 2 * a21 * a22 * v[1] + 2 * a31 * a32 * v[2] \
                  + (a11 * a22 + a12 * a21) * v[3] + (a11 * a32 + a12 * a31) * v[4] + (a21 * a32 + a22 * a31) * v[5]
        vout[4] = 2 * a11 * a13 * v[0] + 2 * a21 * a23 * v[1] + 2 * a31 * a33 * v[2] \
                  + (a11 * a23 + a13 * a21) * v[3] + (a11 * a33 + a13 * a31) * v[4] + (a21 * a33 + a23 * a31) * v[5]
        vout[5] = 2 * a12 * a13 * v[0] + 2 * a22 * a23 * v[1] + 2 * a32 * a33 * v[2] \
                  + (a12 * a23 + a13 * a22) * v[3] + (a12 * a33 + a13 * a32) * v[4] + (a22 * a33 + a23 * a32) * v[5]

    def rotate_stiffness(self, d_inout, t):
        """Rotate constitutive stiffness matrix of the material.
    
        :param d_inout: matrix of the tangent moduli of the material, 6 x 6, both as input and as output
        :param t: stress-vector rotation matrix, 6 x 6
        :return:
        """
        # t = stress_vector_rotation_matrix(Rm)
        d_inout[:, :] = numpy.dot(t, numpy.dot(d_inout, t.T))

    def rotate_compliance(self, c_inout, tbar):
        """Rotate constitutive compliance matrix of the material.
    
        :param D:
        :param tbar: strain vector rotation matrix, 6 x 6
        :return:
        """
        # Tbar = strain_vector_rotation_matrix(Rm)
        c_inout[:, :] = numpy.dot(tbar, numpy.dot(c_inout, tbar.T))


