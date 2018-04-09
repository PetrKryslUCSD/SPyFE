import numpy

ErrorUnknownOutput = Exception('Unknown requested output')

class MatBase:
    """
    Base class for all other materials.

    rho = Mass density of the material.
    """

    def __init__(self, rho=0.0):
        self.rho = rho # mass density
