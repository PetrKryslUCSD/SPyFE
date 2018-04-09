import numpy

class SI():
    """
    """

    def __init__(self):
        self.m = 1.0
        self.kg = 1.0
        self.sec = 1.0
        self.K = 1.0
        self.mm = self.m / 1000.
        self.N = self.kg * self.m / (self.sec ** 2)
        self.Pa = self.N / (self.m ** 2)
        self.MPa = self.Pa * 1.e6
        self.GPa = self.Pa * 1.e9
        self.psi = 6.8948e+03 * self.Pa
