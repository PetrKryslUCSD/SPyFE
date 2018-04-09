import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from context import spyfe

class NewTests(unittest.TestCase):

    def test_nastran_importer(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.importers import nastran_importer
        from spyfe.fesets.volumelike import FESetT4, FESetT10
        from spyfe.meshing.exporters.vtkexporter import vtkexport

        fens, fes = nastran_importer.import_mesh('Slot-coarser.nas')
        print(fes.count())
        geom = NodalField(fens=fens)
        vtkexport("test_nastran_importer", fes, geom)
        print( 'Done' )

    def method(self):
        from spyfe.fields.nodal_field import NodalField
        from spyfe.meshing.importers import abaqus_importer
        from spyfe.meshing.exporters.vtkexporter import vtkexport

        fens, feslist = abaqus_importer.import_mesh('LE11_H20.inp')
        for fes in feslist:
            print(fes.count())
        fes = feslist[0]
        geom = NodalField(fens=fens)
        vtkexport("test_Abaqus_importer", fes, geom)
        print('Done')

if __name__ == "__main__":
    unittest.main()