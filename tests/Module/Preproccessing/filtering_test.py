from piscat.Preproccessing.filtering import *
import unittest


class TestFilters(unittest.TestCase):
    def setUp(self):
        sample_video = np.arange(0, 10, 0.1, dtype=float).reshape((4, 5, 5))
        self.test_obj = Filters(sample_video)

    def test_temporal_median(self):
        filtered_image = self.test_obj.temporal_median()
        self.assertTrue(filtered_image.dtype == self.test_obj.video.dtype)
        self.assertAlmostEqual(filtered_image[-1, 0, 0], 1.0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)

    def test_flat_field(self):
        self.test_obj.inter_flag_parallel_active = False
        filtered_image = self.test_obj.flat_field(1.0)
        self.assertTrue(filtered_image.dtype == self.test_obj.video.dtype)
        self.assertAlmostEqual(filtered_image[0, 0, 0], -1.0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)

    def test_flat_field_parallel(self):
        self.test_obj.inter_flag_parallel_active = True
        filtered_image = self.test_obj.flat_field(1.0)
        self.assertTrue(filtered_image.dtype == self.test_obj.video.dtype)
        self.assertAlmostEqual(filtered_image[0, 0, 0], -1.0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = False

    def test_median(self):
        self.test_obj.inter_flag_parallel_active = False
        filtered_image = self.test_obj.median(2)
        self.assertTrue(filtered_image.dtype == self.test_obj.video.dtype)
        self.assertAlmostEqual(filtered_image[-1, -1, -1], 9.8)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)

    def test_median_parallel(self):
        self.test_obj.inter_flag_parallel_active = True
        filtered_image = self.test_obj.median(2)
        self.assertTrue(filtered_image.dtype == self.test_obj.video.dtype)
        self.assertAlmostEqual(filtered_image[-1, -1, -1], 9.8)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)
        self.test_obj.inter_flag_parallel_active = False


class TestFFT2D(unittest.TestCase):
    def setUp(self):
        sample_video = np.arange(0, 10, 0.1, dtype=float).reshape((4, 5, 5))
        self.test_obj = FFT2D(sample_video)

    def test_fft2D(self):
        filtered_image = self.test_obj.fft2D()
        self.assertTrue(filtered_image.dtype == "complex")
        self.assertAlmostEqual(filtered_image[0, 0, 0], 0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)
        log_image = self.test_obj.log2_scale(filtered_image)
        self.assertTrue(log_image.dtype == "float64")
        self.assertTrue(filtered_image[0, 0, 0]< 0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)

if __name__ == '__main__':
    unittest.main()
