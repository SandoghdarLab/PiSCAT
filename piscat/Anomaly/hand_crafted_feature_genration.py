"""
__author__ = "Houman Mirzaalian D."
"""
import numpy as np
import pywt

from piscat.Preproccessing.patch_genrator import ImagePatching
from piscat.InputOutput.cpu_configurations import CPUConfigurations

from skimage.filters import difference_of_gaussians
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_minimum
from skimage.filters import threshold_niblack
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from scipy import stats, ndimage



class CreateFeatures():

    def __init__(self, video):
        """
        This class is used to generate various spatiotemporal features from video.

        Parameters
        ----------
        video: NDArray
            Numpy 3D video array.
        """
        self.cpu = CPUConfigurations()

        self.video = video
        self.video_dog = None
        self.time_gaussian_low = np.empty_like(video)
        self.time_gaussian_high = np.empty_like(video)
        self.dog_2D = None

        self.low_sigma = None
        self.high_sigma = None

        self.batchSize = None
        self.size_A_diff = None
        self.A_diff = None
        self.mean_batch_1 = None
        self.mean_batch_2 = None
        self.std_batch_1 = None
        self.std_batch_2 = None
        self.t_test = None

    def _dog_2Dfeatures(self, f_):
        dog_ = difference_of_gaussians(self.video[f_, :, :], low_sigma=self.low_sigma, high_sigma=self.high_sigma)
        return dog_

    def dog_3Dfeatures(self, low_sigma, high_sigma):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        return difference_of_gaussians(self.video, low_sigma=self.low_sigma, high_sigma=self.high_sigma)

    def original_time_blure(self, r_, c_, sigma_low, sigma_high):
        self.time_gaussian_low[:, r_, c_] = gaussian_filter1d(self.video[:, r_, c_], sigma=sigma_low, axis=0)
        self.time_gaussian_high[:, r_, c_] = gaussian_filter1d(self.video[:, r_, c_], sigma=sigma_high, axis=0)

    def dog_time_blure(self, r_, c_, sigma_low, sigma_high):
        self.time_gaussian_low[:, r_, c_] = gaussian_filter1d(self.dog_2D[:, r_, c_], sigma=sigma_low, axis=0)
        self.time_gaussian_high[:, r_, c_] = gaussian_filter1d(self.dog_2D[:, r_, c_], sigma=sigma_high, axis=0)

    def dog2D_creater(self, low_sigma, high_sigma, internal_parallel_flag=True):
        """
        Calculate the difference between the Gaussian transforms.

        Parameters
        ----------
        low_sigma: float
            The sigma minimum of a convalved Gaussian kernel.

        high_sigma: float
            The sigma maximum of a convalved Gaussian kernel.

        Returns
        -------
        The video after applying this transformation.

        """
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

        if self.cpu.parallel_active and internal_parallel_flag:
            print("\n---start DOG feature with parallel loop---")
            result0 = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                delayed(self._dog_2Dfeatures)(f_) for f_ in tqdm(range(self.video.shape[0])))
        else:
            print("\n---start DOG feature without parallel loop---")
            result0 = [self._dog_2Dfeatures(f_) for f_ in tqdm(range(self.video.shape[0]))]

        return np.asarray(result0)

    def dog3D_creater(self, low_sigma, high_sigma):
        self.dog_2D = self.dog2D_creater(low_sigma[1:], high_sigma[1:])
        Parallel(n_jobs=self.cpu.n_jobs, backend='threading', verbose=self.cpu.verbose)(
            delayed(self.dog_time_blure)(r_, c_, sigma_low=low_sigma[0], sigma_high=high_sigma[0])
            for r_ in tqdm(range(self.dog_2D.shape[1]))
            for c_ in range(self.dog_2D.shape[2]))

        return (self.time_gaussian_high - self.time_gaussian_low), np.abs(self.time_gaussian_high - self.time_gaussian_low)

    def threshold_kernel(self, frame_):
        im_ = self.video_dog[frame_, ...]

        try:
            thresh = threshold_minimum(im_)
            binary_min = im_ > thresh
        except:
            binary_min = np.full_like(im_, True)

        q = 1
        threshold_image = threshold_niblack(im_, window_size=3, k=2) * q

        return binary_min, threshold_image

    def threshold_feature(self, video_dog):

        self.video_dog = video_dog

        if self.cpu.parallel_active:
            print("\n---start threshold feature with parallel loop---")

            thr_features = Parallel(n_jobs=self.cpu.n_jobs, backend='multiprocessing', verbose=self.cpu.verbose)(
                delayed(self.threshold_kernel)(f_) for f_ in tqdm(range(self.video.shape[0])))
        else:
            print("\n---start threshold feature without parallel loop---")
            thr_features = [self.threshold_kernel(f_) for f_ in tqdm(range(self.video.shape[0]))]

        thr_minimum = [r_[0] for r_ in thr_features]
        thr_niblack = [r_[1] for r_ in thr_features]
        arr_thr_minimum = np.array(thr_minimum)
        arr_thr_niblack = np.array(thr_niblack)
        print(arr_thr_minimum.shape)
        print(arr_thr_niblack.shape)
        return arr_thr_minimum, arr_thr_niblack

    def patch_genrator(self, depth=5, width=10, height=0):

        self.patch_gen = ImagePatching(depth=depth, width=width, height=height, depth_overlap=1, width_overlap=width, height_overlap=height)
        self.patch = self.patch_gen.split_video(self.video)

    def gradient3D_features(self):
        self.patch3Dgradient = []
        for p_ in self.patch:
            tmp_ = np.gradient(p_)
            tmp_ = normalize(tmp_, norm='l2')
            self.patch3Dgradient.append(tmp_)

    def cwt_features(self, video, scale=50, stride=2):
        self.video_1D = video.reshape(-1, video.shape[1]*video.shape[2])

        if self.cpu.parallel_active:
            print("\n---start CWT feature with parallel loop---")

            cwt_features = Parallel(n_jobs=self.cpu.n_jobs, backend='multiprocessing', verbose=self.cpu.verbose)(
                delayed(self.threshold_kernel)(f_, stride) for f_ in tqdm(range(self.video.shape[0])))
        else:
            print("\n---start CWT feature without parallel loop---")

            stack_cwt = []
            for idx_ in tqdm(range(self.video_1D.shape[0])):
                coef, freqs = pywt.cwt(self.video_1D[idx_, ...], np.arange(1, scale, stride), 'mexh')
                stack_cwt.append(coef)

        stack_cwt = np.asarray(stack_cwt)

    def cwt_kernel(self, idx_, stride, scale):
        coef, freqs = pywt.cwt(self.video_1D[idx_, ...], np.arange(1, scale, stride), 'mexh')
        return coef

    def temporal_features(self, batchSize, flag_dc=False):
        """
        Creating temporal features.

        Parameters
        ----------
        batchSize: int
            The total number of frames used in each batch.

        flag_dc: bool
           The activation of this flag removes the uncolibrate mean from the data; otherwise, we presume that the data has zero mean.

        Returns
        -------
        features_list: list
            The list of extracted features.
        """

        self.batchSize = batchSize
        self.size_A_diff = (self.video.shape[0] - 2 * batchSize, self.video.shape[1], self.video.shape[2])

        self.mean_batch_1 = np.empty(self.size_A_diff)
        self.mean_batch_2 = np.empty(self.size_A_diff)
        self.diff_abs = np.empty(self.size_A_diff)

        self.std_batch_1 = np.empty(self.size_A_diff)
        self.std_batch_2 = np.empty(self.size_A_diff)
        self.t_test = np.empty(self.size_A_diff)

        print("\n---create temporal feature map ---")
        self.temporal_moving_average(flag_dc=flag_dc)
        features_list = [self.mean_batch_1, self.mean_batch_2, self.std_batch_1, self.std_batch_2, self.diff_abs]
        return features_list

    def _temporal_features_kernel(self, i_, flag_dc):
        batch1_ = self.video[i_:(i_ + self.batchSize), :, :]
        batch2_ = self.video[(i_ + self.batchSize):(i_ + 2 * self.batchSize), :, :]
        if flag_dc:
            bias_ = 0
        else:
            bias_ = np.mean(self.video[i_:(i_ + 2 * self.batchSize), :, :], axis=0)

        m_b1_ = np.mean(batch1_ - bias_, axis=0)
        m_b2_ = np.mean(batch2_ - bias_, axis=0)

        std_b1_ = np.std(batch1_ - bias_, axis=0)
        std_b2_ = np.std(batch2_ - bias_, axis=0)

        if flag_t_test:
            statisticfloat, pvalue = stats.ttest_ind(batch1_ - bias_, batch2_ - bias_)
        else:
            pvalue = None

        return [m_b1_, m_b2_, std_b1_, std_b2_, pvalue]

    def temporal_moving_average(self, flag_dc):

        batch_sum_1 = np.sum(self.video[0:self.batchSize, :, :], axis=0)
        batch_sum_2 = np.sum(self.video[self.batchSize:2 * self.batchSize, :, :], axis=0)

        batch_mean_1_ = np.divide(batch_sum_1, self.batchSize)
        batch_mean_2_ = np.divide(batch_sum_2, self.batchSize)

        if flag_dc:
            bias_ = 0
        else:
            bias_ = (batch_sum_1 + batch_sum_2)/(2*self.batchSize)

        self.mean_batch_1[0, :, :] = batch_mean_1_ - bias_
        self.mean_batch_2[0, :, :] = batch_mean_2_ - bias_

        batch_std_1_ = np.power((self.video[0:self.batchSize, :, :] - batch_mean_1_), 2)
        batch_std_1_ = np.sum(batch_std_1_, axis=0)
        self.std_batch_1[0, :, :] = np.divide(batch_std_1_, self.batchSize)

        batch_std_2_ = np.power((self.video[self.batchSize:2 * self.batchSize, :, :] - batch_mean_2_), 2)
        batch_std_2_ = np.sum(batch_std_2_, axis=0)
        self.std_batch_2[0, :, :] = np.divide(batch_std_2_, self.batchSize)

        self.diff_abs[0, :, :] = np.abs(batch_mean_2_ - batch_mean_1_)

        for i_ in tqdm(range(1, self.video.shape[0] - 2 * self.batchSize)):

            batch_sum_1 = batch_sum_1 - self.video[i_ - 1, :, :] + self.video[self.batchSize + i_ - 1, :, :]
            batch_sum_2 = batch_sum_2 - self.video[self.batchSize + i_ - 1, :, :] + self.video[(2 * self.batchSize) + i_ - 1, :, :]
            batch_mean_1_ = np.divide(batch_sum_1, self.batchSize)
            batch_mean_2_ = np.divide(batch_sum_2, self.batchSize)

            if flag_dc:
                bias_ = 0
            else:
                bias_ = (batch_sum_1 + batch_sum_2) / (2 * self.batchSize)

            self.mean_batch_1[i_, :, :] = batch_mean_1_ - bias_
            self.mean_batch_2[i_, :, :] = batch_mean_2_ - bias_

            batch_std_1_ = batch_std_1_ - np.power((self.video[i_ - 1, :, :]-batch_mean_1_), 2) + np.power((self.video[self.batchSize + i_ - 1, :, :]-batch_mean_1_), 2)
            self.std_batch_1[i_, :, :] = np.divide(batch_std_1_, self.batchSize)

            batch_std_2_ = batch_std_2_ - np.power((self.video[self.batchSize + i_ - 1, :, :]-batch_mean_2_), 2) + \
                           np.power((self.video[(2 * self.batchSize) + i_ - 1, :, :]-batch_mean_2_), 2)

            self.std_batch_2[i_, :, :] = np.divide(batch_std_2_, self.batchSize)

            self.diff_abs[i_, :, :] = np.abs(batch_mean_2_ - batch_mean_1_)












