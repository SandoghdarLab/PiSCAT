from piscat.BackgroundCorrection.noise_floor import *
from piscat.InputOutput.reading_videos import video_reader
import unittest
import os
from piscat.InputOutput import read_status_line
from unittest.mock import patch

current_path = os.path.abspath(os.path.join('.'))


class TestNoiseFloor(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(current_path, 'TestData/Video/')
        self.file_name = 'control_4999_128_128_uint16_2.33FPS.raw'
        file_path = os.path.join(self.path, self.file_name)
        video = video_reader(file_name=file_path, type='binary', img_width=128, img_height=128,
                             image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)
        video = video[0:50, :, :]
        status_ = read_status_line.StatusLine(video)  # Reading the status line
        self.video_remove_status, status_information = status_.find_status_line()

    def test_noise_floor_mode_temporal(self):
        l_range = list(range(1, 10, 2))
        noise_floor = NoiseFloor(self.video_remove_status, l_range, FPN_flag=False)
        correct_resutlt = [0.001464, 0.000856, 0.000665, 0.000560, 0.000489]
        np.testing.assert_almost_equal(correct_resutlt, noise_floor.mean, 6)


    def test_noise_floor_mode_spatial(self):
        l_range = list(range(1, 10, 2))
        noise_floor = NoiseFloor(self.video_remove_status, l_range, FPN_flag=False, mode='mode_spatial')
        correct_resutlt = [0.001472, 0.000857, 0.000671, 0.000573, 0.000512]
        np.testing.assert_almost_equal(correct_resutlt, noise_floor.mean, 6)

    @patch("matplotlib.pyplot.show")
    def test_plot_fn(self, mock_show):
        l_range = list(range(1, 10, 2))
        noise_floor = NoiseFloor(self.video_remove_status, l_range, FPN_flag=False, mode='mode_spatial')
        noise_floor.plot_result()
        noise_floor.plot_result(flag_log=False)