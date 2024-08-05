import numpy as np
import pytest
import dpm_tools

from numpy.testing import assert_allclose
import pathlib
import porespy as ps

class TestIO:
    def setup_class(self):
        self.path = pathlib.Path(__file__).parent
        self.blobs = ps.generators.blobs([256, 125, 512], porosity=0.3)

    def test_scalar_dataclass_from_numpy(self):
        test_img = dpm_tools.io.Image(scalar=self.blobs)

        assert test_img.nz == 256
        assert test_img.nx == 125
        assert test_img.ny == 512

    def test_vector_dataclass_from_numpy(self):
        pass

    def test_dataclass_from_numpy(self):
        pass

    def test_scalar_dataclass_from_raw(self):
        pass

    def test_scalar_dataclass_from_tiff(self):
        pass

    def test_vector_dataclass_from_raw(self):
        pass

    def test_vector_dataclass_from_tiff(self):
        pass

    def test_dataclass_from_raw(self):
        pass

    def test_dataclass_from_tiff(self):
        pass

    def test_find_files_with_ext(self):
        pass

    def test_get_tiff_metadata(self):
        pass

    def test_natural_sort(self):
        pass

    def test_combine_slices(self):
        pass

    def test_convert_filetype(self):
        pass




# import sys
# sys.path.append('../..') #Add custom filepath here
# import unittest
# from numpy.testing import assert_allclose
# import dpm_tools.io as dpm
#
# class TestDPMTools(unittest.TestCase):
#   def test_find_files(self):
#     actual_tif = dpm.io_utils._find_files("..\data",".tif")
#     expected_tif = [('..\\data\\3_fractures.tif', '1022300 bytes')]
#     self.assertEqual(actual_tif, expected_tif)
#
#     actual_tiff = dpm.io_utils._find_files("..\data",".tiff")
#     expected_tiff = [('..\\data\\35_1.tiff', '1975838 bytes'), ('..\\data\\combined_test.tiff', '16400736 bytes'), ('..\\data\\Initial_1_00000.tiff', '4100464 bytes'), ('..\\data\\Initial_1_00001.tiff', '4100464 bytes')]
#     self.assertEqual(actual_tiff, expected_tiff)
#
#     actual_raw = dpm.io_utils._find_files("..\data",".raw")
#     expected_raw = [('..\\data\\3_fractures.raw', '1000000 bytes'), ('..\\data\\berea_pore.raw', '1331000 bytes')]
#     self.assertEqual(actual_raw, expected_raw)
#
#     #actual_nc = dpm.io_utils._find_files("..\data",".nc")
#     #expected_nc = [('C:\\Users\\Frieda\\Downloads\\io_test\\block00000000.nc', '285838100 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\block00000001.nc', '512000672 bytes')]
#     #self.assertEqual(actual_nc, expected_nc)
#
#   def test_find_tiff_files(self):
#     actual = str(dpm.io_utils._find_tiff_files("..\data"))
#     expected = "[('35_1.tiff', '1975838 bytes', '\\\\..\\\\data', 1, 1008, 980, dtype('uint16'), '='), ('combined_test.tiff', '16400736 bytes', '\\\\..\\\\data', 4, 1024, 1001, dtype('float32'), '='), ('Initial_1_00000.tiff', '4100464 bytes', '\\\\..\\\\data', 1, 1024, 1001, dtype('float32'), '='), ('Initial_1_00001.tiff', '4100464 bytes', '\\\\..\\\\data', 1, 1024, 1001, dtype('float32'), '='), ('3_fractures.tif', '1022300 bytes', '\\\\..\\\\data', 100, 100, 100, dtype('uint8'), '|')]"
#     self.assertEqual(actual, expected)
#
#   def test_evaluate_dimensions(self):
#     actual_2D = dpm.io_utils._evaluate_dimensions("..\data","3_fractures.tif")
#     expected_2D = 100
#     self.assertEqual(actual_2D, expected_2D)
#
#   def test_sort_files(self):
#     actual = dpm.io_utils._sort_files("..\data",".tiff","Initial_1_00000.tiff",2)
#     #expected = 100
#     #self.assertEqual(actual, expected)
#
#
# test = TestDPMTools()
#
# test.test_find_files()
# test.test_find_tiff_files()
# test.test_evaluate_dimensions()
# test.test_sort_files()

if __name__ == "__main__":
    tests = TestIO()
    tests.setup_class()
    for item in tests.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            tests.__getattribute__(item)()