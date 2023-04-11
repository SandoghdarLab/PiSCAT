import unittest

import numpy as np

from piscat.Preproccessing.FPNc import ColumnProjectionFPNc, FrequencyFPNc, MedianProjectionFPNc


class TestMedianProjectionFPNc(unittest.TestCase):
    def setUp(self):
        sample_video = np.ones((4, 5, 5))
        sample_video[:, 2, 2] = 2
        self.test_obj = MedianProjectionFPNc(sample_video, 0)

    def test_mFPNc_first_axis(self):
        out_put = self.test_obj.mFPNc(0)
        self.assertTrue(out_put.shape == self.test_obj.video.shape)
        self.assertTrue(out_put[0][2, 2] == 1)

    def test_mFPNc_second_axis(self):
        out_put = self.test_obj.mFPNc(1)
        self.assertTrue(out_put.shape == self.test_obj.video.shape)
        self.assertTrue(out_put[0][2, 2] == 1)


class TestColumnProjectionFPNc(unittest.TestCase):
    def setUp(self):
        sample_video = np.ones((4, 5, 5))
        sample_video[:, 2, 2] = 2
        self.test_obj = ColumnProjectionFPNc(sample_video, 0)

    def test_cpFPNc_first_axis(self):
        out_put = self.test_obj.cpFPNc(0)
        self.assertTrue(out_put.shape == self.test_obj.video.shape)
        self.assertTrue(out_put[0][2, 2] == 1)

    def test_cpFPNc_second_axis(self):
        out_put = self.test_obj.cpFPNc(1)
        self.assertTrue(out_put.shape == self.test_obj.video.shape)
        self.assertTrue(out_put[0][2, 2] == 1)


class TestFrequencyFPNc(unittest.TestCase):
    def setUp(self):
        sample_video = np.ones((4, 5, 5))
        sample_video[:, 2, 2] = 2
        self.test_obj = FrequencyFPNc(sample_video)

    def test_update_fFPN_first_axis(self):
        out_put = self.test_obj.update_fFPN(direction="Horizontal")
        self.assertTrue(out_put.shape == self.test_obj.video.shape)

    def test_update_fFPN_second_axis(self):
        out_put = self.test_obj.update_fFPN(direction="Vertical")
        self.assertTrue(out_put.shape == self.test_obj.video.shape)

    def test_update_fFPN_without_parallel(self):
        self.test_obj.inter_flag_parallel_active = False
        out_put_h = self.test_obj.update_fFPN(direction="Horizontal")
        self.assertTrue(out_put_h.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = False
        out_put_v = self.test_obj.update_fFPN(direction="Vertical")
        self.assertTrue(out_put_v.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = True
        out_put_parallel = self.test_obj.update_fFPN(direction="Horizontal")
        self.assertTrue(np.all(out_put_parallel == out_put_h))

    def test_update_wFPN_first_axis(self):
        out_put = self.test_obj.update_wFPN(direction="Horizontal")
        self.assertTrue(out_put.shape == self.test_obj.video.shape)

    def test_update_wFPN_second_axis(self):
        out_put = self.test_obj.update_wFPN(direction="Vertical")
        self.assertTrue(out_put.shape == self.test_obj.video.shape)

    def test_update_wFPN_first_axis_without_parallel_first_axis(self):
        self.test_obj.inter_flag_parallel_active = False
        out_put_h = self.test_obj.update_wFPN(direction="Horizontal")
        self.assertTrue(out_put_h.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = True
        out_put_parallel = self.test_obj.update_wFPN(direction="Horizontal")
        self.assertTrue(out_put_parallel.shape == self.test_obj.video.shape)
        self.assertTrue(np.all(out_put_parallel == out_put_h))

    def test_update_wFPN_first_axis_without_parallel_second_axis(self):
        self.test_obj.inter_flag_parallel_active = False
        out_put_v = self.test_obj.update_wFPN(direction="Vertical")
        self.assertTrue(out_put_v.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = True
        out_put_parallel = self.test_obj.update_wFPN(direction="Vertical")
        self.assertTrue(out_put_parallel.shape == self.test_obj.video.shape)
        self.assertTrue(np.all((out_put_parallel - out_put_v) < 1e-5))
