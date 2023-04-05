import os
import unittest

from read_write_data_test import delete_directory

from piscat.InputOutput.gpu_configurations import *

current_path = os.path.abspath(os.path.join("."))


class TestGPUConfigurations(unittest.TestCase):
    def setUp(self):
        current_dir_name = os.path.dirname(current_path)
        self.dir_name = os.path.join(current_dir_name, "piscat_configuration")
        self.file_name = os.path.join(self.dir_name, "gpu_configurations.json")

    def test_save_read_gpu_setting(self):
        test_obj_save = GPUConfigurations()
        self.assertTrue(os.path.exists(self.dir_name), "directory is not created")
        self.assertTrue(os.path.exists(self.file_name), "file is not saved")
        test_obj_load = GPUConfigurations(flag_report=True)
        self.assertTrue(
            test_obj_save.gpu_active_flag == test_obj_load.gpu_active_flag,
            "configuration file is saved incorrectly",
        )
        # delete_directory(self.dir_name)

    def test_print_all_available_gpu(self):
        test_obj = GPUConfigurations()
        test_obj.print_all_available_gpu()


if __name__ == "__main__":
    unittest.main()
