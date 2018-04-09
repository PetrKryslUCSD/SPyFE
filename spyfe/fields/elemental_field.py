import numpy
from spyfe.fields.gen_field import GenField
from spyfe.fenode_set import FENodeSet

class ElementalField(GenField):
    """
    Class to represent elemental fields.

    Class that represents elemental fields: geometry, displacement, 
    incremental rotation,  temperature,... defined by values associated
    with elements.

    """

    def __init__(self, nelems=0, dim=0, fes=None):
        if fes is not None:
            nelems=fes.conn.shape[0]
            super().__init__(nents=nelems, data=numpy.zeros((nelems,1)))
        else:
            super().__init__(nents=nelems, dim=dim)

    def get_nelems(self):
        return super().nents

    def set_nelems(self, value):
        super().set_nents(value)

        nelems = property(get_nelems, set_nelems)


