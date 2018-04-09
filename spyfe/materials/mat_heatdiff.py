import numpy
from spyfe.materials.mat_base import MatBase

Flux=0

class MatHeatDiff(MatBase):
    """
    Heat diffusion material class.

        thermal_conductivity = 0.0 # Thermal conductivity
        specific_heat= 0.0 # Specific heat per unit volume

    """

    def __init__(self, thermal_conductivity=None, specific_heat=0.0, rho=0.0):
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        super().__init__(rho=rho)

    def state(self, ms, gradtheta=None, output=Flux):
        if output == Flux:
            flux = - numpy.dot(self.thermal_conductivity, gradtheta)
            out = flux
        return out, ms

    def newmatstate(self):
        return 0