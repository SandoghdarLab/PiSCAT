from piscat.InputOutput.image_to_video import *
from read_write_data_test import delete_directory
import os
import unittest


current_path = os.path.abspath(os.path.join('.'))


class TestImage2Video(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(current_path, 'TestData/Images/')
        current_dir_name = os.path.dirname(current_path)
        self.dir_name = os.path.join(current_dir_name, 'piscat_configuration')
        file_format = '*.raw'
        self.test_obj = Image2Video(path=self.path, file_format=file_format,
                                    width_size=64,
                                    height_size=64,
                                    image_type='f8',
                                    reader_type='binary')

    def test_image_to_video(self):
        video = self.test_obj()
        self.assertTrue(video.dtype == 'float64', 'created data type is incorrect')
        self.assertTrue(video.shape == (101, 64, 64), 'shape of created video is incorrect')
        delete_directory(self.dir_name)

    def test_parallel_read_img(self):
        temp = self.test_obj .parallel_read_img(self.test_obj .path_list[0])
        self.assertTrue(len(temp.shape) > 1, 'data inserted incorrectly')
