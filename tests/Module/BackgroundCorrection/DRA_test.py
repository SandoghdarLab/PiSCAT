import os
import pickle
import unittest

import numpy as np

from piscat.BackgroundCorrection.DRA import DifferentialRollingAverage
from piscat.InputOutput.reading_videos import video_reader

current_path = os.path.abspath(os.path.join("."))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


class TestDifferentialRollingAverage(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(current_path, "TestData/Video/")
        self.file_name = "control_4999_128_128_uint16_2.33FPS.raw"
        file_path = os.path.join(self.path, self.file_name)
        video = video_reader(
            file_name=file_path,
            type="binary",
            img_width=128,
            img_height=128,
            image_type=np.dtype("<u2"),
            s_frame=0,
            e_frame=-1,
        )
        video = video[0:50, :, :]
        self.test_obj = DifferentialRollingAverage(video, batchSize=10)

    def test_differential_rolling(self):
        output_video, _ = self.test_obj.differential_rolling(FFT_flag=False)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(os.path.join(self.path, "test_differential_rolling.pck"))
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_mFPN_column(self):
        self.test_obj.mode_FPN = "mFPN"
        output_video, _ = self.test_obj.differential_rolling(FPN_flag=True)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_mFPN_column.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_mFPN_row(self):
        self.test_obj.mode_FPN = "mFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis=0
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_mFPN_row.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_mFPN_both(self):
        self.test_obj.mode_FPN = "mFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis="Both"
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_mFPN_both.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_cpFPN_column(self):
        self.test_obj.mode_FPN = "cpFPN"
        output_video, _ = self.test_obj.differential_rolling(FPN_flag=True)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_cpFPN_column.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_cpFPN_row(self):
        self.test_obj.mode_FPN = "cpFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis=0
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_cpFPN_row.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_cpFPN_both(self):
        self.test_obj.mode_FPN = "cpFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis="Both"
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_cpFPN_both.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_fFPN_column(self):
        self.test_obj.mode_FPN = "fFPN"
        output_video, _ = self.test_obj.differential_rolling(FPN_flag=True)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_fFPN_column.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_fFPN_row(self):
        self.test_obj.mode_FPN = "fFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis=0
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_fFPN_row.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_fFPN_both(self):
        self.test_obj.mode_FPN = "fFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis="Both"
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_fFPN_both.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_wFPN_column(self):
        self.test_obj.mode_FPN = "wFPN"
        output_video, _ = self.test_obj.differential_rolling(FPN_flag=True)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_wFPN_column.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_wFPN_row(self):
        self.test_obj.mode_FPN = "wFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis=0
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_wFPN_row.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_wFPN_both(self):
        self.test_obj.mode_FPN = "wFPN"
        output_video, _ = self.test_obj.differential_rolling(
            FPN_flag=True, select_correction_axis="Both"
        )
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_wFPN_both.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))

    def test_differential_rolling_FFT_flag(self):
        output_video, _ = self.test_obj.differential_rolling(FFT_flag=True)
        expected_shape = (
            self.test_obj.video.shape[0] - 2 * self.test_obj.batchSize,
            self.test_obj.video.shape[1],
            self.test_obj.video.shape[2],
        )
        loaded_data = load_fixture(
            os.path.join(self.path, "test_differential_rolling_FFT_flag.pck")
        )
        self.assertTrue(output_video.shape == expected_shape)
        self.assertTrue(np.all(np.nan_to_num(output_video) - np.nan_to_num(loaded_data) < 1e-6))
