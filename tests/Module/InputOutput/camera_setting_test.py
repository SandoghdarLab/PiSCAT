import os
import unittest

from piscat.InputOutput.camera_setting import CameraParameters

current_path = os.path.abspath(os.path.join("."))


class TestCameraParameters(unittest.TestCase):
    def setUp(self):
        current_dir_name = os.path.dirname(current_path)
        self.dir_name = os.path.join(current_dir_name, "piscat_configuration")
        self.file_path = os.path.join(self.dir_name, "PhotonFocus.json")
        self.file_name = "PhotonFocus.json"

    def test_save_read_camera_setting(self):
        test_obj_save = CameraParameters(name=self.file_name)
        self.assertTrue(os.path.exists(self.dir_name), "directory is not created")
        self.assertTrue(os.path.exists(self.file_path), "file is not saved")
        test_obj_read = CameraParameters(name=self.file_name)
        self.assertTrue(
            test_obj_save.quantum_efficiency == test_obj_read.quantum_efficiency,
            "configuration file is saved incorrectly",
        )
        # delete_directory(self.dir_name)


if __name__ == "__main__":
    unittest.main()
