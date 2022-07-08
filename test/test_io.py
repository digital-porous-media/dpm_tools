import sys
sys.path.append('..') #Add custom filepath here
import unittest
from numpy.testing import assert_allclose
import dpm_tools.io as dpm
import tifffile as tiff
import numpy as np
import pandas as pd
import glob
import os
from PIL import Image
import exifread

class TestDPMTools(unittest.TestCase):
  def test_find_files(self):
    actual_tif = dpm.io_utils._find_files("C:\\Users\Frieda\Downloads\io_test",".tif")
    expected_tif = [('C:\\Users\\Frieda\\Downloads\\io_test\\12.8_bar.tif', '1894867213 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\5_bar.tif', '1894867226 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\RLFeSO4_8bit_C1.tif', '763019050 bytes')]
    self.assertEqual(actual_tif, expected_tif)
    
    actual_tiff = dpm.io_utils._find_files("C:\\Users\Frieda\Downloads\io_test",".tiff")
    expected_tiff = [('C:\\Users\\Frieda\\Downloads\\io_test\\combined_test.tiff', '16400736 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\Initial_1_00000.tiff', '4100464 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\Initial_1_00001.tiff', '4100464 bytes')]
    self.assertEqual(actual_tiff, expected_tiff)
    
    actual_raw = dpm.io_utils._find_files("C:\\Users\Frieda\Downloads\io_test",".raw")
    expected_raw = [('C:\\Users\\Frieda\\Downloads\\io_test\\CQ_0p25_A_xy350_z450_int16.raw', '110250000 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\Frac01_Dry_Image_Slices1_800.raw', '838860800 bytes')]
    self.assertEqual(actual_raw, expected_raw)
    
    actual_nc = dpm.io_utils._find_files("C:\\Users\Frieda\Downloads\io_test",".nc")
    expected_nc = [('C:\\Users\\Frieda\\Downloads\\io_test\\block00000000.nc', '285838100 bytes'), ('C:\\Users\\Frieda\\Downloads\\io_test\\block00000001.nc', '512000672 bytes')]
    self.assertEqual(actual_nc, expected_nc)
    
  def test_find_tiff_files(self):
    actual = str(dpm.io_utils._find_tiff_files("C:\\Users\Frieda\Downloads\io_test"))
    expected = "[('combined_test.tiff', '16400736 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 4, 1024, 1001, dtype('float32'), '='), ('Initial_1_00000.tiff', '4100464 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 1, 1024, 1001, dtype('float32'), '='), ('Initial_1_00001.tiff', '4100464 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 1, 1024, 1001, dtype('float32'), '='), ('12.8_bar.tif', '1894867213 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 1094, 1316, 1316, dtype('uint8'), '|'), ('5_bar.tif', '1894867226 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 1094, 1316, 1316, dtype('uint8'), '|'), ('RLFeSO4_8bit_C1.tif', '763019050 bytes', '\\\\C:\\\\Users\\\\Frieda\\\\Downloads\\\\io_test', 566, 1164, 1158, dtype('uint8'), '|')]"
    print(actual)
    print(expected)
    self.assertEqual(actual, expected)
    
  def test_evaluate_dimensions(self):
    actual_2D = dpm.io_utils._evaluate_dimensions("C:\\Users\Frieda\Downloads\io_test","12.8_bar.tif")
    expected_2D = 1094
    self.assertEqual(actual_2D, expected_2D)
    
test = TestDPMTools()
    
test.test_find_files()
test.test_find_tiff_files()
test.test_evaluate_dimensions()
