from piscat.Preproccessing.normalization import *
import unittest


class TestNormalization(unittest.TestCase):
    def setUp(self):
        sample_video = np.arange(0, 10, 0.1, dtype=float).reshape((4, 5, 5))
        self.test_obj = Normalization(sample_video)

    def test_normalized_image_global(self):
        normalized_image = self.test_obj.normalized_image_global()
        self.assertTrue(normalized_image.dtype == self.test_obj.video.dtype)
        self.assertTrue(normalized_image.max() == 1.0)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)
        normalized_image = self.test_obj.normalized_image_global(new_max=2, new_min=0)
        self.assertTrue(normalized_image.dtype == self.test_obj.video.dtype)
        self.assertTrue(normalized_image.max() == 2.0)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)

    def test_normalized_image_specific(self):
        normalized_image = self.test_obj.normalized_image_specific()
        self.assertTrue(normalized_image.dtype == 'uint8')
        self.assertTrue(normalized_image.max() == 255)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)
        normalized_image = self.test_obj.normalized_image_specific(scale=64, format='float')
        self.assertTrue(normalized_image.dtype == 'float')
        self.assertTrue(normalized_image.max() == 64)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)

    def test_normalized_image_specific_2D(self):
        sample_image = np.arange(0, 10, 0.1, dtype=float).reshape((10, 10))
        test_obj = Normalization(sample_image)
        normalized_image = test_obj.normalized_image_specific()
        self.assertTrue(normalized_image.dtype == 'uint8')
        self.assertTrue(normalized_image.max() == 255)
        self.assertTrue(normalized_image.shape == test_obj.video.shape)
        normalized_image = test_obj.normalized_image_specific(scale=64, format='float')
        self.assertTrue(normalized_image.dtype == 'float')
        self.assertTrue(normalized_image.max() == 64)
        self.assertTrue(normalized_image.shape == test_obj.video.shape)

    def test_normalized_image_specific_by_max(self):
        normalized_image = self.test_obj.normalized_image_specific_by_max()
        self.assertTrue(normalized_image.dtype == 'float64')
        self.assertTrue(normalized_image.max() == 1.0)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)

    def test_normalized_image(self):
        normalized_image = self.test_obj.normalized_image()
        self.assertTrue(normalized_image.dtype == self.test_obj.video.dtype)
        self.assertTrue(normalized_image.max() == 1.0)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)

    def test_power_normalized(self):
        normalized_image, power_fluctuation_percentage = self.test_obj.power_normalized()
        self.assertTrue(normalized_image.dtype == self.test_obj.video.dtype)
        self.assertTrue(normalized_image.shape == self.test_obj.video.shape)
        self.assertAlmostEqual(normalized_image[0, -1, -1], self.test_obj.video[-1, -1, -1])

    def test_power_normalized_parallel(self):
        # normalized_image, power_fluctuation_percentage = self.test_obj.power_normalized(inter_flag_parallel_active=True)
        # self.assertTrue(normalized_image.dtype == self.test_obj.video.dtype)
        pass


if __name__ == '__main__':
    unittest.main()
