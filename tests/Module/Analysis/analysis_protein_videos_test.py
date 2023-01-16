from piscat.Analysis import protein_analysis
from piscat.InputOutput import reading_videos, read_write_data
import unittest
import os
import pickle
import numpy as np
import pandas as pd
import shutil


def check_trajectory(calculated, loaded):
    for data, loaded_data in zip(calculated, loaded):
        for sub_data, sub_loaded_data in zip(data, loaded_data):
            if np.all(sub_data - sub_loaded_data < 1e-5):
                pass
            else:
                return False
    return True


def delete_directory(path):
    try:
        shutil.rmtree(path)
        print("\nDirectory ", path, " deleted")
    except FileNotFoundError:
        print("\nDirectory ", path, " dose not exist")


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


current_path = os.path.abspath(os.path.join('.'))


class ProteinAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.directory_path = [os.path.join(current_path, 'TestData/Video/')]
        self.file_name = ['5nm_GNPs_128x128_uint16_3333fps_10Acc.raw']
        self.hyperparameters = {'function': 'dog', 'batch_size': 10, 'min_V_shape_width': 3, 'max_V_shape_width': 30,
                                'search_range': 2, 'memory': 2, 'min_sigma': 1.6, 'max_sigma': 1.8, 'sigma_ratio': 1.1,
                                'PSF_detection_thr': 8e-4, 'overlap': 0, 'outlier_frames_thr': 20,
                                'Mode_PSF_Segmentation': 'BOTH',
                                'symmetric_PSFs_thr': 0.7, 'mode_FPN': 'mFPN', 'select_correction_axis': 1,
                                'im_size_x': 128,
                                'im_size_y': 128, 'image_format': '<u2', 'start_fr': 9500, 'end_fr': 9700}
        self.flag = {'PN': True, 'FPNc': True, 'outlier_frames_filter': True, 'Dense_Filter': True, 'symmetric_PSFs_Filter': True, 'FFT_flag': False, 'filter_hotPixels':True}

    def test_protein_analysis(self):
        protein_analysis(self.directory_path, self.file_name, self.hyperparameters, self.flag, 'mFPN')
        df_histogram = reading_videos.DirectoryType(os.path.join(self.directory_path[0], 'mFPN'), type_file='json').return_df()
        paths = df_histogram['Directory'].tolist()
        file_names = df_histogram['File'].tolist()
        for path, file_name in zip(paths, file_names):
            if 'hyperparameters' in file_name:
                hyperparameters = read_write_data.read_json2dic(path, file_name)
                self.assertTrue(self.hyperparameters == hyperparameters)
            elif 'flags' in file_name:
                flags = read_write_data.read_json2dic(path, file_name)
                self.assertTrue(self.flag == flags)
            elif 'PSFs_Particels_num' in file_name:
                particles_loaded = read_write_data.read_json2dic(self.directory_path[0], 'PSFs_Particels_num.json')
                particles = read_write_data.read_json2dic(path, file_name)
                self.assertTrue(particles == particles_loaded)
            else:
                self.assertTrue(False)

        df_histogram = reading_videos.DirectoryType(os.path.join(self.directory_path[0], 'mFPN'), type_file='mat').return_df()
        paths = df_histogram['Directory'].tolist()
        file_names = df_histogram['File'].tolist()
        for path, file_name in zip(paths, file_names):
            if 'all_trajectories' in file_name:
                all_trajectories = read_write_data.read_mat(path, file_name)
                loaded_all_trajectories = read_write_data.read_mat(self.directory_path[0], 'all_trajectories.mat')
                self.assertTrue(check_trajectory(all_trajectories, loaded_all_trajectories))

        df_histogram = reading_videos.DirectoryType(os.path.join(self.directory_path[0], 'mFPN'), type_file='csv').return_df()
        paths = df_histogram['Directory'].tolist()
        file_names = df_histogram['File'].tolist()
        for path, file_name in zip(paths, file_names):
            if 'position_PSFs' in file_name:
                positions = pd.read_csv(os.path.join(self.directory_path[0], 'position_PSFs.csv'))
                loaded_positions = pd.read_csv(os.path.join(path, file_name))
                self.assertTrue(positions.equals(loaded_positions))

        delete_directory(os.path.join(self.directory_path[0], 'mFPN'))



