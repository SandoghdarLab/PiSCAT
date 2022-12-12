from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter, protein_trajectories_list2dic
from piscat.Trajectory.data_handeling import fixed_length
import numpy as np

import unittest
import os
import pickle

current_path = os.path.abspath(os.path.join('.'))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def check_dict(calculated_list, loaded_dict):
    try:
        for key, value in calculated_list['#0'].items():
            if np.all(value == loaded_dict['#0'][key]):
                pass
            else:
                if np.all(np.isnan(value)) and np.all(np.isnan(value)):
                    pass
                else:
                    return False
        return True
    except:
        return False


def list_to_array(input_list):
    for index in range(len(input_list[0])):
        if index < 2:
            input_list[0][index] = np.asarray(fixed_length(input_list[0][index]))
        else:
            input_list[0][index] = np.asarray(input_list[0][index])
    return input_list


class DataHandlingTest(unittest.TestCase):
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

    def test_protein_trajectories_list2dic_list(self):
        all_trajectories, linked_PSFs_filter, his_all_particles = self.test_obj.v_trajectory(df_PSFs=self.psf_filtered,
                                                                                             threshold_min=2,
                                                                                             threshold_max=2 * self.batch_size)
        file_name_save = os.path.join(self.directory_path, 'test_v_trajectory_all_trajectories.pck')
        loaded_file = load_fixture(file_name_save)
        loaded_dict = protein_trajectories_list2dic(loaded_file)
        all_trajectories_dict = protein_trajectories_list2dic(all_trajectories)
        check_result = check_dict(all_trajectories_dict, loaded_dict)
        self.assertTrue(check_result)

    def test_protein_trajectories_list2dic(self):
        all_trajectories, linked_PSFs_filter, his_all_particles = self.test_obj.v_trajectory(df_PSFs=self.psf_filtered,
                                                                                             threshold_min=2,
                                                                                             threshold_max=2 * self.batch_size)
        file_name_save = os.path.join(self.directory_path, 'test_v_trajectory_all_trajectories.pck')
        loaded_file = load_fixture(file_name_save)
        loaded_file_array = list_to_array(loaded_file)
        all_trajectories_array = list_to_array(all_trajectories)
        loaded_dict = protein_trajectories_list2dic(loaded_file_array)
        all_trajectories_dict = protein_trajectories_list2dic(all_trajectories_array)
        check_result = check_dict(all_trajectories_dict, loaded_dict)
        self.assertTrue(check_result)

