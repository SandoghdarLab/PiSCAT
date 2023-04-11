import unittest

import numpy as np

from piscat.Preproccessing.applying_mask import Mask2Video


class TestMask2Video(unittest.TestCase):
    def setUp(self):
        sample_video = np.ones((4, 5, 5))
        self.test_obj = Mask2Video(sample_video, None)

    def test_mask_generating_circle(self):
        filtered_image = self.test_obj.mask_generating_circle(center=(2, 2))
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)
        self.assertTrue(filtered_image[0][2, 2] == 1)

    def test_apply_mask_parallel(self):
        filtered_image = self.test_obj.mask_generating_circle(center=(2, 2))
        masked_video = self.test_obj.apply_mask()
        self.assertTrue(masked_video.shape == self.test_obj.video.shape)
        self.assertTrue(masked_video[0][2, 2] == 1)

    def test_apply_mask(self):
        filtered_image = self.test_obj.mask_generating_circle(center=(2, 2))
        self.test_obj.inter_flag_parallel_active = False
        masked_video = self.test_obj.apply_mask()
        self.assertTrue(masked_video.shape == self.test_obj.video.shape)
        self.assertTrue(masked_video[0][2, 2] == 1)

    def test_apply_mask_nan(self):
        filtered_image = self.test_obj.mask_generating_circle(center=(2, 2))
        masked_video = self.test_obj.apply_mask(flag_nan=False)
        self.assertTrue(masked_video.shape == self.test_obj.video.shape)
        self.assertTrue(masked_video[0][2, 2] == 1)
