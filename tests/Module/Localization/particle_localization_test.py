import os
import pickle
import unittest

import numpy as np

from piscat.Localization import data_handling, difference_of_gaussian, particle_localization

current_path = os.path.abspath(os.path.join("."))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def set_obj_parameter(obj, function, mode):
    obj.function = function
    obj.mode = mode
    obj.min_sigma = 1.6
    obj.max_sigma = 1.8
    obj.sigma_ratio = 1.1
    obj.threshold = 8.5e-4
    obj.overlap = 0


def psf_detection_preview(psf_obj, function_name: str, frame_number, mode):
    psf_dataframe_list_of_frame = psf_obj.psf_detection_preview(
        function=function_name,
        min_sigma=1.6,
        max_sigma=1.8,
        sigma_ratio=1.1,
        threshold=8.5e-4,
        overlap=0,
        mode=mode,
        frame_number=frame_number,
    )
    return psf_dataframe_list_of_frame


def psf_detection(psf_obj, function_name: str, mode):
    psf_dataframe_list_of_frame = psf_obj.psf_detection(
        function=function_name,
        min_sigma=1.6,
        max_sigma=1.8,
        sigma_ratio=1.1,
        threshold=8.5e-4,
        overlap=0,
        mode=mode,
    )
    return psf_dataframe_list_of_frame


class TestPSFsExtraction(unittest.TestCase):
    def setUp(self):
        self.directory_path = os.path.join(current_path, "TestData/Video/")
        file_name_save = os.path.join(self.directory_path, "test_localization_input_video.pck")
        self.video = load_fixture(file_name_save)
        self.frame = self.video[0, :, :]
        self.test_obj = particle_localization.PSFsExtraction(video=self.video)
        file_name_save = os.path.join(
            self.directory_path, "test_localization_input_video_doh.pck"
        )
        video = load_fixture(file_name_save)
        self.test_obj_doh = particle_localization.PSFsExtraction(video=video)

    def test_psf_detection_preview_dog_video_BOTH(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_dog_video_both.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_dog_video_Bright(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=[0], mode="Bright"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=0, mode="Bright"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_dog_video_Bright.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_dog_video_Dark(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=[0], mode="Dark"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "dog", frame_number=0, mode="Dark"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_dog_video_Dark.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_doh_video(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj_doh, "doh", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj_doh, "doh", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_doh_video.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_log_video_BOTH(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_log_video_both.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_log_video_Bright(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=[0], mode="Bright"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=0, mode="Bright"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_log_video_Bright.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_log_video_Dark(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=[0], mode="Dark"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "log", frame_number=0, mode="Dark"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_log_video_Dark.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_frst_one_psf_video(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "frst_one_psf", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "frst_one_psf", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_frst_one_psf_video.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

        self.test_obj.video = np.repeat(self.test_obj.video, 2, axis=1)
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj, "frst_one_psf", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj, "frst_one_psf", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(
                self.directory_path, "test_psf_detection_preview_frst_one_psf_repeat_video.pck"
            )
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_preview_RVT_video(self):
        psf_dataframe_list_of_frame = psf_detection_preview(
            self.test_obj_doh, "RVT", frame_number=[0], mode="BOTH"
        )
        psf_dataframe_one_frame = psf_detection_preview(
            self.test_obj_doh, "RVT", frame_number=0, mode="BOTH"
        )
        self.assertTrue(psf_dataframe_list_of_frame.shape == psf_dataframe_one_frame.shape)
        self.assertTrue(psf_dataframe_list_of_frame.equals(psf_dataframe_list_of_frame))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_preview_RVT_video.pck")
        )
        self.assertTrue(np.all((psf_dataframe_one_frame - loaded_data) < 1e-6))
        self.assertTrue(np.all(np.nan_to_num(psf_dataframe_list_of_frame - loaded_data) < 1e-6))

    def test_psf_detection_dog_video_BOTH(self):
        psf_dataframe_p = psf_detection(self.test_obj, "dog", mode="BOTH")
        self.assertTrue(psf_dataframe_p.shape == (222, 5))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_dog_video_both.pck")
        )
        self.assertTrue(np.all((psf_dataframe_p - loaded_data) < 1e-6))
        self.test_obj.cpu.parallel_active = False
        psf_dataframe = psf_detection(self.test_obj, "dog", mode="BOTH")
        self.assertTrue(psf_dataframe.shape == (222, 5))
        self.assertTrue(np.all(psf_dataframe_p == psf_dataframe))

    def test_psf_detection_dog_video_Bright(self):
        psf_dataframe = psf_detection(self.test_obj, "dog", mode="Bright")
        self.assertTrue(psf_dataframe.shape == (117, 5))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_dog_video_Bright.pck")
        )
        self.assertTrue(np.all((psf_dataframe - loaded_data) < 1e-6))

    def test_psf_detection_dog_video_Dark(self):
        psf_dataframe = psf_detection(self.test_obj, "dog", mode="Dark")
        self.assertTrue(psf_dataframe.shape == (105, 5))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_psf_detection_dog_video_Dark.pck")
        )
        self.assertTrue(np.all((psf_dataframe - loaded_data) < 1e-6))

    def test_fit_Gaussian2D_wrapper(self):
        psf_dataframe_p = psf_detection(self.test_obj, "dog", mode="Bright")
        psf_dataframe_gaussian_p = self.test_obj.fit_Gaussian2D_wrapper(
            PSF_List=psf_dataframe_p, scale=5, internal_parallel_flag=True
        )
        self.assertTrue(psf_dataframe_p.shape[0] == psf_dataframe_gaussian_p.shape[0])
        self.assertTrue(psf_dataframe_gaussian_p.shape[1] == 18)
        self.test_obj.cpu.parallel_active = False
        psf_dataframe_gaussian = self.test_obj.fit_Gaussian2D_wrapper(
            PSF_List=psf_dataframe_p, scale=5, internal_parallel_flag=True
        )
        self.assertTrue(psf_dataframe_p.shape[0] == psf_dataframe_gaussian_p.shape[0])
        self.assertTrue(psf_dataframe_gaussian.shape[1] == 18)
        self.assertTrue(psf_dataframe_gaussian_p.equals(psf_dataframe_gaussian))
        loaded_data = load_fixture(
            os.path.join(self.directory_path, "test_fit_Gaussian2D_wrapper.pck")
        )
        self.assertTrue(
            np.all(
                np.nan_to_num(
                    psf_dataframe_gaussian_p[["y", "x", "frame"]]
                    - loaded_data[["y", "x", "frame"]]
                )
                < 1e-6
            )
        )

    def test_psf_detection_kernel_dog(self):
        self.test_obj.video = self.test_obj.video[0, :, :]
        set_obj_parameter(self.test_obj, "dog", "BOTH")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (13, 4))
        set_obj_parameter(self.test_obj, "dog", "Bright")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (6, 4))
        set_obj_parameter(self.test_obj, "dog", "Dark")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (7, 4))

    def test_psf_detection_kernel_doh(self):
        self.test_obj_doh.video = self.test_obj_doh.video[0, :, :]
        set_obj_parameter(self.test_obj_doh, "doh", "BOTH")
        psf_list = self.test_obj_doh.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (308, 4))

    def test_psf_detection_kernel_log(self):
        self.test_obj.video = self.test_obj.video[0, :, :]
        set_obj_parameter(self.test_obj, "log", "BOTH")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (25, 4))
        set_obj_parameter(self.test_obj, "log", "Bright")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (14, 4))
        set_obj_parameter(self.test_obj, "log", "Dark")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (11, 4))

    def test_psf_detection_kernel_frst_one_psf(self):
        self.test_obj.video = self.test_obj.video[0, :, :]
        set_obj_parameter(self.test_obj, "frst_one_psf", "BOTH")
        psf_list = self.test_obj.psf_detection_kernel(0)
        self.assertTrue(psf_list.shape == (1, 4))

    def test_improve_localization_with_frst(self):
        psf_dataframe_p = psf_detection(self.test_obj, "dog", mode="BOTH")
        psf_df = self.test_obj.improve_localization_with_frst(psf_dataframe_p, 4)
        self.assertTrue(psf_df.shape == (145, 5))
        self.test_obj.cpu.parallel_active = False
        psf_df_p = self.test_obj.improve_localization_with_frst(psf_dataframe_p, 4)
        self.assertTrue(psf_df_p.shape == (145, 5))
        self.assertTrue(psf_df.equals(psf_df_p))

    def test_list2dataframe(self):
        set_obj_parameter(self.test_obj, "dog", "BOTH")
        psf_list = self.test_obj.psf_detection_kernel(0)
        data_frame = data_handling.list2dataframe(psf_list, self.test_obj.video)
        self.assertTrue(data_frame.shape == (13, 5))
        append_list = np.zeros(shape=(data_frame.shape[0], 1))
        appended_list = np.append(psf_list, append_list, axis=1)
        data_frame = data_handling.list2dataframe(appended_list, self.test_obj.video)
        self.assertTrue(data_frame.shape == (13, 7))

    def test_difference_of_gaussian(self):
        min_range, max_range = difference_of_gaussian.dog_preview(
            self.video, min_sigma=1.6, max_sigma=1.8, sigma_ratio=1.1
        )
        self.assertAlmostEqual(max_range, 0.0002561082, 6)
        self.assertAlmostEqual(min_range, -0.000269173, 6)

    def test_fit_2D_Gaussian_varAmp(self):
        psf_dataframe_p = psf_detection(self.test_obj, "dog", mode="Bright")
        psf_dataframe_gaussian_p = self.test_obj.fit_Gaussian2D_wrapper(
            PSF_List=psf_dataframe_p, scale=5, internal_parallel_flag=True
        )
        kernel_outputs = self.test_obj.fit_2D_gussian_kernel(0, scale=5, display_flag=False)
        self.assertTrue(len(kernel_outputs) == 18)
