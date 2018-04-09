import numpy
from spyfe.fields.gen_field import GenField
from spyfe.fenode_set import FENodeSet

class NodalField(GenField):
    """
    Class to represent general fields.

    Class that represents general fields: geometry, displacement, 
    incremental rotation,  temperature,... defined by values associated
    with nodes or elements.

    """

    def __init__(self, nfens=0, dim=0, fens=None):
        if fens is not None:
            nfens=fens.count()
            super().__init__(nents=nfens, data=fens.xyz)
        else:
            super().__init__(nents=nfens, dim=dim)

    def get_nfens(self):
        return super().nents

    def set_nfens(self, value):
        super().set_nents(value)

    nfens = property(get_nfens, set_nfens)


