from piscat.Localization import directional_intensity

import unittest
import os
import pickle

current_path = os.path.abspath(os.path.join('.'))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


class DirectionalIntensity(unittest.TestCase):
    def setUp(self):
        self.directory_path = os.path.join(current_path, 'Test Data/Video/')
        file_name_save = os.path.join(self.directory_path, 'test_localization_input_video.pck')
        self.video = load_fixture(file_name_save)
        self.frame = self.video[0, :, :]

    def test_interpolate_pixels_along_line(self):
        test_obj = directional_intensity.DirectionalIntensity()
        pixels = test_obj.interpolate_pixels_along_line(12, 12, 24, 24)
        self.assertTrue(len(pixels) == 26)
        self.assertTrue(pixels[18] == (21, 21))

    def test_interpolate_pixels_along_line_steep(self):
        test_obj = directional_intensity.DirectionalIntensity()
        pixels = test_obj.interpolate_pixels_along_line(12, 12, 24, 34)
        self.assertTrue(len(pixels) == 46)
        self.assertTrue(pixels[18] == (16, 21))

    def test_interpolate_pixels_along_line_x0(self):
        test_obj = directional_intensity.DirectionalIntensity()
        pixels = test_obj.interpolate_pixels_along_line(36, 12, 24, 24)
        self.assertTrue(len(pixels) == 26)
        self.assertTrue(pixels[18] == (33, 15))

