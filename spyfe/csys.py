import numpy

SuspectTransformation = Exception('Suspect transformation matrix')


class CSys:
    """
    Class for coordinate system transformations.

            In the columns of the csys orientation (rotation) matrix are the
            basis vectors expressed in the global Cartesian basis. If the
            orientation matrix is not supplied, then
            (i) if the space dimension is equal to the manifold dimension
                of the elements, then matrix being an identity is assumed.
            (ii) otherwise, the transformation matrix is computed as
                appropriate for  an isotropic material for an element
                embedded in a higher dimensional space using the columns
                of the Jacobian matrix.

            Parameters:
             either of the following:
                 matrix= csys orientation (rotation) matrix, of appropriate
                    dimension.
                 fun = function to compute the csys orientation matrix.

                       The function to compute the orientation matrix must have
                    the signature
                          function Matrix = SampleFunction(XYZ, ts, label)
                    The orientation matrix can be computed based on any of the three
                    arguments.
                    XYZ= global Cartesian location of the point at which the orientation
                            matrix is desired,
                    ts= the Jacobian matrix with the tangents to the parametric coordinates
                            as columns,
                    label= the label of the finite element in which the point XYZ resides

    """

    def __init__(self, matrix=None, fun=None):
        self.isidentity = True
        self.isconstant = True
        self.matrix = matrix
        self.fun = fun
        if not self.fun is None:
            self.isidentity = False
            self.isconstant = False
        elif not self.matrix is None:
            pId = numpy.dot(self.matrix.T, self.matrix)  # this should be an identity
            Id = numpy.eye(pId.shape[0])
            if numpy.linalg.norm(Id - pId) > 1.e-6:
                raise SuspectTransformation
            if numpy.linalg.norm(self.matrix - Id) > 1.e-6:
                self.isidentity = False

    def eval_matrix(self, XYZ=None, tangents=None, fe_label=None):
        if self.isconstant:
            return self.matrix
        else:
            return self.fun(XYZ, tangents, fe_label)
