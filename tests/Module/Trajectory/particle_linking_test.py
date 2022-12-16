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


class LinkingTest(unittest.TestCase):
    def setUp(self):
        self.directory_path = os.path.join(current_path, 'TestData/Video/')
        file_name_save = os.path.join(self.directory_path, 'test_fit_Gaussian2D_wrapper.pck')
        self.psf_dataframe = load_fixture(file_name_save)
        self.test_obj = Linking()

    def test_create_link(self):
        linked_psf = self.test_obj.create_link(psf_position=self.psf_dataframe, search_range=2, memory=10)
        self.assertTrue(linked_psf.shape[1] == 19)
        file_name_save = os.path.join(self.directory_path, 'test_create_link.pck')
        loaded_data_frame = load_fixture(file_name_save)
        self.assertTrue(np.all(np.nan_to_num(linked_psf - self.psf_dataframe) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(linked_psf - loaded_data_frame) < 1e-6))

    def test_sort_linking(self):
        linked_psf = self.test_obj.create_link(psf_position=self.psf_dataframe, search_range=2, memory=10)
        sorted_linking_df = self.test_obj.sorting_linking(linked_psf)
        file_name_save = os.path.join(self.directory_path, 'test_sort_linking.pck')
        loaded_data_frame = load_fixture(file_name_save)
        self.assertTrue(np.all(np.nan_to_num(sorted_linking_df - loaded_data_frame) < 1e-6))

    def test_trajectory_counter(self):
        linked_psf = self.test_obj.create_link(psf_position=self.psf_dataframe, search_range=2, memory=10)
        len_linked_psf = self.test_obj.trajectory_counter(linked_psf)
        self.assertTrue(len_linked_psf == 89)

