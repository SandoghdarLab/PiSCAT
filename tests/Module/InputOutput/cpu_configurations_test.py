from piscat.InputOutput.cpu_configurations import *
from read_write_data_test import delete_directory
import os
import unittest

current_path = os.path.abspath(os.path.join('.'))


class TestCPUConfigurations(unittest.TestCase):
    def setUp(self):
        current_dir_name = os.path.dirname(current_path)
        self.dir_name = os.path.join(current_dir_name, 'piscat_configuration')
        self.file_name = os.path.join(self.dir_name, 'cpu_configurations.json')

    def test_save_read_cpu_setting(self):
        CPUConfigurations(n_jobs=2)
        self.assertTrue(os.path.exists(self.dir_name), 'directory is not created')
        self.assertTrue(os.path.exists(self.file_name), 'file is not saved')
        test_obj = CPUConfigurations()
        self.assertTrue(test_obj.n_jobs == 2, 'configuration file is saved incorrectly')
        delete_directory(self.dir_name)


if __name__ == '__main__':
    unittest.main()
