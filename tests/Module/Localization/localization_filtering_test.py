import pandas as pd

from piscat.Localization import localization_filtering
from piscat.Trajectory.particle_linking import Linking
import numpy as np

import unittest
import os
import pickle

current_path = os.path.abspath(os.path.join('.'))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


class SpatialFilter(unittest.TestCase):
    def setUp(self):
        self.directory_path = os.path.join(current_path, 'TestData/Video/')
        file_name_save = os.path.join(self.directory_path, 'test_fit_Gaussian2D_wrapper.pck')
        psf_dataframe = load_fixture(file_name_save)
        linking_ = Linking()
        self.linked_PSFs = linking_.create_link(psf_position=psf_dataframe, search_range=2, memory=10)
        self.linked_PSFs = self.linked_PSFs.reset_index()
        self.test_obj = localization_filtering.SpatialFilter()

    def test_outlier_frames(self):
        filtered_psf = self.test_obj.outlier_frames(self.linked_PSFs, threshold=20)
        file_name = os.path.join(self.directory_path, 'test_outlier_frames.pck')
        loaded_data_frame = load_fixture(file_name)
        self.assertTrue(np.all(np.nan_to_num(filtered_psf - loaded_data_frame) < 1e-6))

    def test_outlier_frames_value_error(self):
        with self.assertRaises(ValueError):
            self.test_obj.outlier_frames(pd.DataFrame(), threshold=20)

    def test_dense_PSFs(self):
        filtered_psf = self.test_obj.dense_PSFs(self.linked_PSFs, threshold=1)
        file_name = os.path.join(self.directory_path, 'test_dense_PSFs.pck')
        loaded_data_frame = load_fixture(file_name)
        self.assertTrue(np.all(np.nan_to_num(filtered_psf - loaded_data_frame) < 1e-6))

    def test_symmetric_PSFs(self):
        filtered_psf = self.test_obj.symmetric_PSFs(self.linked_PSFs, threshold=0.7)
        file_name = os.path.join(self.directory_path, 'symmetric_PSFs.pck')
        loaded_data_frame = load_fixture(file_name)
        self.assertTrue(np.all(np.nan_to_num(filtered_psf - loaded_data_frame) < 1e-6))

    def test_symmetric_PSFs_value_error(self):
        with self.assertRaises(Exception):
            self.test_obj.symmetric_PSFs(pd.DataFrame(), threshold=20)

    def test_symmetric_PSFs_value_error_empty(self):
        with self.assertRaises(Exception):
            self.test_obj.symmetric_PSFs(None, threshold=20)

    def test_remove_side_lobes_artifact(self):
        filtered_psf = self.test_obj.remove_side_lobes_artifact(self.linked_PSFs, threshold=0)
        self.assertTrue(np.all(np.nan_to_num(filtered_psf - self.linked_PSFs) < 1e-6))



