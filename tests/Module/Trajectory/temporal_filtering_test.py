from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter
import numpy as np

import unittest
import os
import pickle

current_path = os.path.abspath(os.path.join('.'))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def check_list(calculated_list, loaded_list):
    try:
        for index in range(len(calculated_list[0])):
            if isinstance(calculated_list[0][index], list):
                for item, load_item in zip(calculated_list[0][index], loaded_list[0][index]):
                    if np.all(item == load_item):
                        pass
                    else:
                        if np.isnan(item) and np.isnan(load_item):
                            pass
                        else:
                            return False
            else:
                if np.all(calculated_list[0][index] == loaded_list[0][index]):
                    pass
                else:
                    return False
        return True
    except:
        return False


class TemporalFilterTest(unittest.TestCase):
    def setUp(self):
        self.directory_path = os.path.join(current_path, 'Test Data/Video/')
        file_name_save = os.path.join(self.directory_path, 'test_fit_Gaussian2D_wrapper.pck')
        psf_dataframe = load_fixture(file_name_save)
        linking_ = Linking()
        linked_psf = linking_.create_link(psf_position=psf_dataframe, search_range=2, memory=10)
        spatial_filters = localization_filtering.SpatialFilter()
        psf_filtered = spatial_filters.outlier_frames(linked_psf, threshold=20)
        psf_filtered = spatial_filters.dense_PSFs(psf_filtered, threshold=0)
        self.psf_filtered = spatial_filters.symmetric_PSFs(psf_filtered, threshold=0.7)
        file_name_save = os.path.join(self.directory_path, 'test_localization_input_video.pck')
        video = load_fixture(file_name_save)
        self.batch_size = 3
        self.test_obj = TemporalFilter(video=video, batchSize=self.batch_size)

    def test_v_trajectory(self):
        all_trajectories, linked_PSFs_filter, his_all_particles = self.test_obj.v_trajectory(df_PSFs=self.psf_filtered,
                                                                                             threshold_min=2,
                                                                                             threshold_max=2 * self.batch_size)
        file_name_save = os.path.join(self.directory_path, 'test_v_trajectory_all_trajectories.pck')
        loaded_file = load_fixture(file_name_save)
        check_result = check_list(all_trajectories, loaded_file)
        self.assertTrue(check_result)
        file_name_save = os.path.join(self.directory_path, 'test_v_trajectory_linked_PSFs_filter.pck')
        loaded_file = load_fixture(file_name_save)
        self.assertTrue(linked_PSFs_filter.equals(loaded_file))
        file_name_save = os.path.join(self.directory_path, 'his_all_particles.pck')
        loaded_file = load_fixture(file_name_save)
        self.assertTrue(his_all_particles.equals(loaded_file))
