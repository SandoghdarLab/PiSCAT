import numpy as np

from piscat.InputOutput.read_status_line import *
from piscat.InputOutput.reading_videos import video_reader
import os
import unittest

current_path = os.path.abspath(os.path.join('.'))


class TestStatusLine(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(current_path, 'Test Data/Video/')
        self.file_name = 'control_4999_128_128_uint16_2.33FPS.raw'
        file_path = os.path.join(self.path, self.file_name)
        self.video = video_reader(file_name=file_path, type='binary', img_width=128, img_height=128,
                                  image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)

    def test_read_status_line(self):
        test_obj = StatusLine(self.video)
        out_video, camera_info = test_obj.find_status_line()
        self.assertTrue(camera_info['status_line_position'] == 'column')
        self.assertTrue(len(camera_info) == 21)

    def test_read_status_line_tra(self):
        tra_video = np.transpose(self.video, (0, 2, 1))
        test_obj = StatusLine(tra_video)
        out_video, camera_info = test_obj.find_status_line()
        self.assertTrue(camera_info['status_line_position'] == 'row')
        self.assertTrue(len(camera_info) == 21)