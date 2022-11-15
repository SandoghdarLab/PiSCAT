import numpy as np

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
        self.assertTrue(filtered_image[0, 0, 0] < 0)
        self.assertTrue(filtered_image.shape == self.test_obj.video.shape)


class TestRadialVarianceTransform(unittest.TestCase):
    def setUp(self):
        self.test_obj = RadialVarianceTransform()

    def test_gen_r_kernel(self):
        kernel = self.test_obj.gen_r_kernel(3, 5)
        self.assertTrue(np.shape(kernel) == (11, 11))
        self.assertTrue(kernel[3, 2] == 0.05)

    def test_generate_all_kernels(self):
        kernels = self.test_obj.generate_all_kernels(3, 5, coarse_factor=1, coarse_mode="add")
        self.assertTrue(len(kernels) == 3)
        self.assertTrue(kernels[0][3, 2] == 0.05)
        kernels = self.test_obj.generate_all_kernels(3, 5, coarse_factor=2, coarse_mode="add")
        self.assertTrue(len(kernels) == 2)
        self.assertTrue(kernels[0][1, 3] != 0)
        kernels = self.test_obj.generate_all_kernels(3, 5, coarse_factor=2, coarse_mode="skip")
        self.assertTrue(len(kernels) == 2)
        self.assertTrue(kernels[0][3, 2] == 0.05)

    def test__check_core_args(self):
        self.assertRaisesWithMessage('radius should be non-negative', self.test_obj._check_core_args,
                                     {'rmin': -1, 'rmax': 2, 'kind': 'basic', 'coarse_mode': 'add'})
        self.assertRaisesWithMessage('radius should be non-negative', self.test_obj._check_core_args,
                                     {'rmin': 1, 'rmax': -2, 'kind': 'basic', 'coarse_mode': 'add'})
        self.assertRaisesWithMessage("unrecognized kind: {}; can be either 'basic' or 'normalized'",
                                     self.test_obj._check_core_args, {'rmin': 1, 'rmax': 2, 'kind': 'basi',
                                                                      'coarse_mode': 'add'})
        self.assertRaisesWithMessage("unrecognized coarse mode: {}; can be either 'add' or 'skip'",
                                     self.test_obj._check_core_args, {'rmin': 1, 'rmax': 2, 'kind': 'basic',
                                                                      'coarse_mode': 'ad'})

    def test__check_args(self):
        self.assertRaisesWithMessage("upsampling factor should be positive", self.test_obj._check_args,
                                     {'rmin': 1, 'rmax': 2, 'kind': 'basic', 'coarse_mode': 'add',
                                      'highpass_size': 0.4, 'upsample': 0})
        self.assertRaisesWithMessage("high-pass filter size should be >= 0.3", self.test_obj._check_args,
                                     {'rmin': 1, 'rmax': 2, 'kind': 'basic', 'coarse_mode': 'add',
                                      'highpass_size': 0.2, 'upsample': 1})

    def test_get_fshape(self):
        shape = self.test_obj.get_fshape(np.array((10, 10)), np.array((3, 3)), fast_mode=False)
        self.assertTrue(shape == (12, 12))
        shape = self.test_obj.get_fshape(np.array((10, 10)), np.array((3, 3)), fast_mode=True)
        self.assertTrue(shape == (10, 10))

    def test_prepare_fft(self):
        kernels = self.test_obj.generate_all_kernels(3, 5, coarse_factor=1, coarse_mode="add")
        shape = self.test_obj.get_fshape(np.array((10, 10)), np.array((11, 11)), fast_mode=False)
        fft_kernel = self.test_obj.prepare_fft(kernels[0], shape, pad_mode="fast")
        self.assertTrue(fft_kernel.dtype == "complex")
        self.assertTrue(fft_kernel.shape == (20, 11))
        fft_kernel_constant = self.test_obj.prepare_fft(kernels[0], shape)
        self.assertTrue(fft_kernel_constant.dtype == "complex")
        self.assertTrue(fft_kernel_constant.shape == (20, 11))

    def test_convolve_fft(self):
        img = np.full((12, 12), 1)
        shape = self.test_obj.get_fshape(np.array((12, 12)), np.array((11, 11)), fast_mode=False)
        img_fft = self.test_obj.prepare_fft(img, shape, pad_mode='fast')
        kernels = self.test_obj.generate_all_kernels(3, 5, coarse_factor=1, coarse_mode="add")
        fft_kernel = self.test_obj.prepare_fft(kernels[0], shape, pad_mode="fast")
        img_fft = self.test_obj.convolve_fft(img_fft, fft_kernel, np.array((10, 10)), np.array((11, 11)), shape,
                                             fast_mode=False)
        self.assertTrue(img_fft.dtype == 'float64')
        self.assertTrue(img_fft.shape == (10, 10))
        shape = self.test_obj.get_fshape(np.array((12, 12)), np.array((11, 11)), fast_mode=True)
        img_fft = self.test_obj.prepare_fft(img, shape, pad_mode='fast')
        fft_kernel = self.test_obj.prepare_fft(kernels[0], shape, pad_mode="fast")
        img_fft = self.test_obj.convolve_fft(img_fft, fft_kernel, np.array((10, 10)), np.array((11, 11)), shape,
                                             fast_mode=True)
        self.assertTrue(img_fft.dtype == 'float64')
        self.assertTrue(img_fft.shape == (10, 10))

    def test_rvt_core(self):
        img = np.full((12, 12), 1)
        result = self.test_obj.rvt_core(img, 3, 5, kind="basic", rweights=None, coarse_factor=1,
                                        coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result == np.zeros((12, 12))).all())
        result = self.test_obj.rvt_core(img, 3, 5, kind="basic", rweights=np.full((3,), 1), coarse_factor=1,
                                        coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result == np.zeros((12, 12))).all())
        img = np.full((12, 12), 2)
        img[6, 6] = 0
        result = self.test_obj.rvt_core(img, 3, 5, kind="normalized", rweights=None, coarse_factor=1,
                                        coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result <= np.ones((12, 12))).all())
        result = self.test_obj.rvt_core(img, 3, 5, kind="normalized", rweights=np.full((3,), 1), coarse_factor=1,
                                        coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result <= np.ones((12, 12))).all())

    def test_high_pass(self):
        img = np.full((12, 12), 1)
        img_out = self.test_obj.high_pass(img, 1)
        self.assertTrue((img_out == np.zeros((12, 12))).all())

    def test_rvt(self):
        img = np.full((12, 12), 1)
        result = self.test_obj.rvt(img, 3, 5, kind="basic", highpass_size=None, upsample=1, rweights=None,
                                   coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result == np.zeros((12, 12))).all())
        result = self.test_obj.rvt(img, 3, 5, kind="basic", highpass_size=1, upsample=1, rweights=None,
                                   coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (12, 12))
        self.assertTrue((result == np.zeros((12, 12))).all())
        result = self.test_obj.rvt(img, 3, 5, kind="basic", highpass_size=None, upsample=2, rweights=None,
                                   coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (24, 24))
        self.assertTrue((result == np.zeros((24, 24))).all())

    def test_rvt_video(self):
        video = np.full((10, 12, 12), 1)
        result = self.test_obj.rvt_video(video, 3, 5, kind="basic", highpass_size=None, upsample=1, rweights=None,
                                         coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (10, 12, 12))
        self.assertTrue((result == np.zeros((10, 12, 12))).all())
        self.test_obj_s = RadialVarianceTransform()
        self.test_obj_s.inter_flag_parallel_active = False
        result_s = self.test_obj_s.rvt_video(video, 3, 5, kind="basic", highpass_size=None, upsample=1, rweights=None,
                                             coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result_s.shape == (10, 12, 12))
        self.assertTrue((result_s == np.zeros((10, 12, 12))).all())

    def test_rvt_video_highpass(self):
        video = np.full((10, 12, 12), 1)
        self.test_obj.inter_flag_parallel_active = False
        result = self.test_obj.rvt_video(video, 3, 5, kind="basic", highpass_size=3, upsample=1, rweights=None,
                                         coarse_factor=1, coarse_mode="add", pad_mode="constant")
        self.assertTrue(result.shape == (10, 12, 12))
        self.assertTrue((result == np.zeros((10, 12, 12))).all())

    def assertRaisesWithMessage(self, msg, func, kwargs):
        try:
            func(**kwargs)
        except Exception as inst:
            self.assertEqual(str(inst), msg)


class TestFastRadialSymmetryTransform(unittest.TestCase):
    def setUp(self):
        self.img = np.full((12, 12), 1)
        self.test_obj = FastRadialSymmetryTransform()

    def test_gradx(self):
        filtered_image = self.test_obj.gradx(self.img)
        self.assertTrue(filtered_image.shape == (12, 12))
        self.assertTrue((filtered_image == np.zeros((12, 12))).all())

    def test_grady(self):
        filtered_image = self.test_obj.grady(self.img)
        self.assertTrue(filtered_image.shape == (12, 12))
        self.assertTrue((filtered_image == np.zeros((12, 12))).all())

    def test__frst(self):
        self.img[6, 6] = 0
        filtered_image = self.test_obj._frst(self.img, 3, 2, 0.5, 3, mode='BOTH')
        self.assertTrue(filtered_image.shape == (12, 12))
        self.assertTrue((filtered_image <= 1).all())


class TestGuidedFilter(unittest.TestCase):
    def setUp(self):
        self.flt_img = np.full((12, 12), 2)
        self.test_obj_img = GuidedFilter(self.flt_img, [3, 3], 0.1)
        self.flt_video = np.full((10, 12, 12), 2)
        self.test_obj_video = GuidedFilter(self.flt_video, [3, 3, 3], 0.1)

    def test_filter(self):
        self.img = np.full((12, 12), 1)
        filtered_image = self.test_obj_img.filter(self.img)
        self.assertTrue(filtered_image.shape == (12, 12))
        self.assertTrue((filtered_image == np.ones((12, 12))).all())


if __name__ == '__main__':
    unittest.main()
