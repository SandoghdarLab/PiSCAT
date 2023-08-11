import os
import pickle
import unittest

import numpy as np

from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter
from piscat.Trajectory.data_handling import fixed_length
from piscat.Trajectory.particle_linking import Linking

current_path = os.path.abspath(os.path.join("."))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def check_dict(calculated_list, loaded_dict):
    try:
        for key, value in calculated_list["#0"].items():
            if np.all(value == loaded_dict["#0"][key]):
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
        self.directory_path = os.path.join(current_path, "TestData/Video/")
        file_name_save = os.path.join(self.directory_path, "test_fit_Gaussian2D_wrapper.pck")
        psf_dataframe = load_fixture(file_name_save)
        linking_ = Linking()
        linked_psf = linking_.create_link(psf_position=psf_dataframe, search_range=2, memory=10)
        spatial_filters = localization_filtering.SpatialFilter()
        psf_filtered = spatial_filters.outlier_frames(linked_psf, threshold=20)
        psf_filtered = spatial_filters.dense_PSFs(psf_filtered, threshold=0)
        self.psf_filtered = spatial_filters.symmetric_PSFs(psf_filtered, threshold=0.7)
        file_name_save = os.path.join(self.directory_path, "test_localization_input_video.pck")
        video = load_fixture(file_name_save)
        self.batch_size = 3
        self.test_obj = TemporalFilter(video=video, batchSize=self.batch_size)
