from __future__ import print_function

import numpy as np
import cv2
import scipy.signal
import scipy.ndimage
import scipy.fftpack

from PySide2.QtCore import *
from tqdm.autonotebook import tqdm
from skimage import filters
from scipy.ndimage.filters import median_filter
from joblib import Parallel, delayed
from piscat.InputOutput.cpu_configurations import CPUConfigurations


class WorkerSignals(QObject):

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Filters():

    def __init__(self, video, inter_flag_parallel_active=True):
        """
       This class generates a list of video/image filters.
       To improve performance on large video files, some of them have a parallel implementation.

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        inter_flag_parallel_active: bool
            If the user wants to enable general parallel tasks in the CPU configuration, he or she can only use this flag to enable or disable this process.
        """
        self.cpu = CPUConfigurations()
        self.inter_flag_parallel_active = inter_flag_parallel_active

        self.video = video
        self.filtered_video = None

    def temporal_median(self):
        """
        By extracting the temporal median from pixels, the background is corrected.

        Returns
        -------
        @returns: NDArray
            The background corrected video as 3D-numpy
        """
        video_med = np.median(self.video, axis=0)
        video_med_ = np.expand_dims(video_med, axis=0)
        return np.divide(self.video, video_med_) - 1

    def flat_field(self, sigma):
        """
        This method corrects the video background by creating a synthetic flat fielding version of the background.

        Parameters
        ----------
        sigma: float
            Sigma of Gaussian filter that use to create blur video.

        Returns
        -------
        flat_field_video: NDArray
            The background corrected video as 3D-numpy

        """
        blur_video = self.gaussian(sigma)
        flat_field_video = np.divide(self.video, blur_video) - 1
        return flat_field_video

    def median(self, size):
        """
        This function applies a 2D median filter on each frame.

        Parameters
        ----------
        size: int
            Kernel size of the median filter.

        Returns
        -------
        self.blur_video: NDArray
            The filter video as 3D-numpy.

        """
        if self.cpu.parallel_active is True and self.inter_flag_parallel_active is True:
            print("\n---start median filter with Parallel---")

            result = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(
                self.median_kernel)(x, size) for x in tqdm(range(self.video.shape[0])))

            arry_result = np.asarray(result)
            self.filtered_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))
        else:
            print("\n---start median filter without Parallel---")

            result = [median_filter(self.video[i_, :, :], size) for i_ in range(self.video.shape[0])]
            arry_result = np.asarray(result)
            self.filtered_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))
        return self.filtered_video

    def median_kernel(self, i_, size):
        return median_filter(self.video[i_, :, :], size)

    def gaussian(self, sigma):
        """
        This function applies a 2D Gaussian filter on each frame.

        Parameters
        ----------
        sigma: float
            Sigma of Gaussian filter that use to create blur video.

        Returns
        -------
        self.blur_video: NDArray
            The filter video as 3D-numpy.

        """

        if self.cpu.parallel_active is True and self.inter_flag_parallel_active is True:
            print("\n---start gaussian filter with Parallel---")

            result = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(
                self.gaussian_kernel)(x, sigma) for x in tqdm(range(self.video.shape[0])))

            arry_result = np.asarray(result)
            self.filtered_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))
        else:
            print("\n---start gaussian filter without Parallel---")

            result = [filters.gaussian(self.video[i_, :, :], sigma=sigma) for i_ in range(self.video.shape[0])]
            arry_result = np.asarray(result)
            self.filtered_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))

        return self.filtered_video

    def gaussian_kernel(self, i_, sigma):
        return filters.gaussian(self.video[i_, :, :], sigma=sigma, preserve_range=True)


class FFT2D(QRunnable):

    def __init__(self, video):
        """
        This class computes the 2D spectrum of video.

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).
        """
        super(FFT2D, self).__init__()
        self.signals = WorkerSignals()
        self.video = video

    @Slot()
    def run(self):
        result_ = self.fft2D()
        result = self.log2_scale(result_)
        self.signals.result.emit(result)

    def fft2D(self):
        im_fft = np.ndarray(self.video.shape, complex)
        for i in range(0, self.video.shape[0]):
            im_fft[i, :, :] = np.fft.fftshift(np.fft.fft2(self.video[i, :, :]))
        return im_fft

    def log2_scale(self, video):
        return np.log2(np.abs(video))


class RadialVarianceTransform():

    def __init__(self, inter_flag_parallel_active=True):
        """
        Efficient Python implementation of Radial Variance Transform.

        The main function is :func:`rvt` in the bottom of the file, which applies the transform to a single image (2D numpy array).


        Compared to the vanilla convolution implementation, there are two speed-ups:
        1) Pre-calculating and caching kernel FFT; this way so only one inverse FFT is calculated per convolution + one direct fft of the image is used for all convolutions
        2) When finding MoV, calculate ``np.mean(rsqmeans)`` in a single convolution by averaging all kernels first

        Parameters
        ----------
        inter_flag_parallel_active: bool
            In case the user wants to active general parallel tasks in CPU configuration,
            the user can only active or deactivate this method by this flag.

        References
        ----------
        [1] Kashkanova, Anna D., et al. "Precision single-particle localization using radial variance transform." Optics Express 29.7 (2021): 11070-11083.
        """
        self.cpu = CPUConfigurations()
        self._kernels_fft_cache = {}
        self.video = None
        self.inter_flag_parallel_active = inter_flag_parallel_active

    def gen_r_kernel(self, r, rmax):
        """Generate a ring kernel with radius `r` and size ``2*rmax+1``"""
        a = rmax * 2 + 1
        k = np.zeros((a, a))
        for i in range(a):
            for j in range(a):
                rij = ((i - rmax) ** 2. + (j - rmax) ** 2) ** .5
                if int(rij) == r:
                    k[i][j] = 1.

        tmp = k / np.sum(k)
        return tmp

    def generate_all_kernels(self, rmin, rmax, coarse_factor=1, coarse_mode="add"):
        """
        Generate a set of kernels with radii between `rmin` and `rmax` and sizes ``2*rmax+1``.

        ``coarse_factor`` and ``coarse_mode`` determine if the number of those kernels is reduced by either skipping or adding them
        (see :func:`rvt` for a more detail explanation).
        """
        kernels = [self.gen_r_kernel(r, rmax) for r in range(rmin, rmax + 1)]
        if coarse_factor > 1:
            if coarse_mode == "skip":
                kernels = kernels[::coarse_factor]
            else:
                kernels = [np.sum(kernels[i:i + coarse_factor], axis=0) / coarse_factor for i in
                           range(0, len(kernels), coarse_factor)]
        return kernels

    def _check_core_args(self, rmin, rmax, kind, coarse_mode="add"):
        """Check validity of the core algorithm arguments"""
        if rmin < 0 or rmax < 0:
            raise ValueError("radius should be non-negative")
        if rmin >= rmax:
            raise ValueError("rmax should be strictly greater than rmin")
        if kind not in {"basic", "normalized"}:
            raise ValueError("unrecognized kind: {}; can be either 'basic' or 'normalized'")
        if coarse_mode not in {"add", "skip"}:
            raise ValueError("unrecognized coarse mode: {}; can be either 'add' or 'skip'")

    def _check_args(self, rmin, rmax, kind, coarse_mode, highpass_size, upsample):
        """Check validity of all the algorithm arguments"""
        self._check_core_args(rmin, rmax, kind, coarse_mode)
        if upsample < 1:
            raise ValueError("upsampling factor should be positive")
        if highpass_size is not None and highpass_size < 0.3:
            raise ValueError("high-pass filter size should be >= 0.3")

    ## Prepare auxiliary parameters
    def get_fshape(self, s1, s2, fast_mode=False):
        """Get the required shape of the transformed image given the shape of the original image and the kernel"""
        shape = s1 if fast_mode else s1 + s2 - 1
        fshape = [scipy.fftpack.next_fast_len(int(d)) for d in shape]

        tmp = tuple(fshape)
        return tmp

    ## Preparing FFTs of arrays
    def prepare_fft(self, inp, fshape, pad_mode="constant"):
        """Prepare the image for a convolution by taking its Fourier transform, applying padding if necessary"""
        if pad_mode == "fast":
            return np.fft.rfftn(inp, fshape)
        else:
            pad = [((td - d) // 2, (td - d + 1) // 2) for td, d in zip(fshape, inp.shape)]
            inp = np.pad(inp, pad, mode=pad_mode)
            tmp = np.fft.rfftn(inp)
            return tmp

    ## Shortcut of SciPy fftconvolve, which takes already fft'd arrays on the input
    def convolve_fft(self, sp1, sp2, s1, s2, fshape, fast_mode=False):
        """Calculate the convolution from the Fourier transforms of the original image and the kernel, trimming the result if necessary"""
        ret = np.fft.irfftn(sp1 * sp2, fshape)
        if fast_mode:
            return np.roll(ret, (-(s2[0] // 2), -(s2[1] // 2)), (0, 1))[:s1[0], :s1[1]].copy()
        else:
            off = (fshape[0] - s1[0]) // 2 + s2[0] // 2, (fshape[1] - s1[1]) // 2 + s2[1] // 2
            tmp = ret[off[0]:off[0] + s1[0], off[1]:off[1] + s1[1]].copy()
            return tmp

    def rvt_core(self, img, rmin, rmax, kind="basic", rweights=None, coarse_factor=1, coarse_mode="add", pad_mode="constant"):
        """
        Perform core part of Radial Variance Transform (RVT) of an image.

        Parameters
        ----------
        img: NDArray
            source image (2D numpy array)

        rmin: float
            minimal radius (inclusive)

        rmax: float
            maximal radius (inclusive)

        kind: str, ("basic", "normalized")
            either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR

        rweights:
            relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``

        coarse_factor:
            the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

        coarse_mode: str, ("add", "skip")
            The reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        pad_mode: str, ("constant", "reflect", "edge", "fast")
            Edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean
        """
        self._check_core_args(rmin, rmax, kind, coarse_mode)  # check arguments validity
        s1 = np.array(img.shape)
        s2 = np.array([rmax * 2 + 1, rmax * 2 + 1])
        fast_mode = pad_mode == "fast"
        fshape = self.get_fshape(s1, s2,
                            fast_mode=fast_mode)  # calculate the padded image shape (add padding and get to the next "good" FFT size)
        cache_k = (rmin, rmax, coarse_factor, coarse_mode) + fshape
        if cache_k not in self._kernels_fft_cache:  # generate convolution kernels, if they are not in cache yet
            kernels = self.generate_all_kernels(rmin, rmax, coarse_factor=coarse_factor, coarse_mode=coarse_mode)
            self._kernels_fft_cache[cache_k] = [self.prepare_fft(k, fshape, pad_mode="fast") for k in
                                           kernels]  # pad_mode="fast" corresponds to the default zero-padding here
        kernels_fft = self._kernels_fft_cache[cache_k]  # get the convolution kernels (either newely generated, or cached)
        if rweights is not None:
            rweights = np.asarray(rweights)
            if len(rweights) != len(kernels_fft):
                raise ValueError(
                    "the number of kernel weights {} is different from the number of kernels {}".format(len(rweights),
                                                                                                        len(
                                                                                                            kernels_fft)))
            rweights = rweights / rweights.sum()  # normalize weights by their sum
        img = img - img.mean()  # subtract mean (makes VoM calculation more stable and zero-padding more meaningful)
        img_fft = self.prepare_fft(img, fshape, pad_mode=pad_mode)  # prepare image FFT (only needs to be done once)
        rmeans = np.array([self.convolve_fft(img_fft, k_fft, s1, s2, fshape, fast_mode=fast_mode) for k_fft in
                           kernels_fft])  # calculate M_r for all radii
        if rweights is None:
            vom = np.var(rmeans, axis=0)  # calculate VoM as a standard variance of M_r along the radius axis
        else:
            vom = np.sum(rmeans ** 2 * rweights[:, None, None], axis=0) - np.sum(rmeans * rweights[:, None, None],
                                                                                 axis=0) ** 2  # calculate VoM as a weighted variance of M_r along the radius axis
        if kind == "basic":
            return vom
        else:  # calculate MoV for normalization
            imgsq_fft = self.prepare_fft(img ** 2, fshape, pad_mode=pad_mode)  # prepare image FFT
            if rweights is None:
                sumk_fft = np.mean(kernels_fft, axis=0)  # find combined kernel as a standard mean
                mov = self.convolve_fft(imgsq_fft, sumk_fft, s1, s2, fshape, fast_mode=fast_mode) - np.mean(rmeans ** 2,
                                                                                                       axis=0)  # use the combined kernel to find MoV in one convolution
            else:
                sumk_fft = np.sum(kernels_fft * rweights[:, None, None],
                                  axis=0)  # find combined kernel as a weighted mean
                mov = self.convolve_fft(imgsq_fft, sumk_fft, s1, s2, fshape, fast_mode=fast_mode) - np.sum(
                    rmeans ** 2 * rweights[:, None, None],
                    axis=0)  # use the combined kernel to find MoV in one convolution
            tmp = vom / mov
            return tmp

    def high_pass(self, img, size):
        """Perform Gaussian high-pass filter on the image"""
        img = img.astype(float)
        tmp = img - scipy.ndimage.gaussian_filter(img, size)
        return tmp

    def rvt(self, img, rmin, rmax, kind="basic", highpass_size=None, upsample=1, rweights=None, coarse_factor=1,
            coarse_mode="add", pad_mode="constant"):
        """
        Perform Radial Variance Transform (RVT) of an image.

        Parameters
        ----------
        img: NDArray
            source image (2D numpy array)

        rmin:
            minimal radius (inclusive)

        rmax:
            maximal radius (inclusive)

        kind:
            either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR

        highpass_size:
            size of the high-pass filter; ``None`` means no filter (effectively, infinite size)

        upsample: int
            integer image upsampling factor;
            `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
            if ``upsample>1``, the resulting image size is multiplied by ``upsample``

        rweights:
            relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
            coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
            coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        coarse_factor:
            the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

        coarse_mode:
            the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        pad_mode:
            edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

        Returns
        -------
            Returns transform source image

        """
        upsample = int(upsample)
        self._check_args(rmin, rmax, kind, coarse_mode, highpass_size, upsample)
        if highpass_size is not None:
            img = self.high_pass(img, highpass_size)
        if upsample > 1:
            img = img.repeat(upsample, axis=-2).repeat(upsample, axis=-1)  # nearest-neighbor upsampling on both axes
            if rweights is not None:
                rweights = np.asarray(rweights).repeat(upsample)  # upsample radial weights as well
            rmin *= upsample  # increase minimal and maximal radii
            rmax *= upsample

        tmp = self.rvt_core(img, rmin, rmax, kind=kind, rweights=rweights, coarse_factor=coarse_factor,
                                coarse_mode=coarse_mode, pad_mode=pad_mode)
        return tmp

    def rvt_video(self, video, rmin, rmax, kind="basic", highpass_size=None, upsample=1, rweights=None, coarse_factor=1,
                coarse_mode="add", pad_mode="constant"):
        """
        This is an RVT wrapper that allows you to get video in parallel.

        Parameters
        ----------
        video: NDArray
             The video is 3D-numpy (number of frames, width, height).

        rmin:
            minimal radius (inclusive)

        rmax:
            maximal radius (inclusive)

        kind:
            either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR

        highpass_size:
            size of the high-pass filter; ``None`` means no filter (effectively, infinite size)

        upsample: int
            integer image upsampling factor;
            `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
            if ``upsample>1``, the resulting image size is multiplied by ``upsample``

        rweights:
            relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
            coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
            coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        coarse_factor:
            the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

        coarse_mode:
            the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        pad_mode:
            edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

        Returns
        -------
            Returns transform source video.
        """
        self.video = video

        if self.cpu.parallel_active is True and self.inter_flag_parallel_active is True:
            print("\n---start RVT with Parallel---")

            result = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(self.rvt_kernel)(f_,
                                                                                                                        rmin=rmin,
                                                                                                                        rmax=rmax,
                                                                                                                        kind=kind,
                                                                                                                        highpass_size=highpass_size,
                                                                                                                        upsample=upsample,
                                                                                                                        rweights=rweights,
                                                                                                                        coarse_factor=coarse_factor,
                                                                                                                        coarse_mode=coarse_mode,
                                                                                                                        pad_mode=pad_mode) for f_ in tqdm(range(video.shape[0])))
            arry_result = np.asarray(result)
            self.rvt_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))
        else:
            print("\n---start RVT without Parallel---")

            result = [self.rvt_kernel(i_, rmin=rmin, rmax=rmax, kind=kind, highpass_size=highpass_size,
                                               upsample=upsample, rweights=rweights, coarse_factor=coarse_factor,
                                            coarse_mode=coarse_mode, pad_mode=pad_mode) for i_ in tqdm(range(self.video.shape[0]))]
            arry_result = np.asarray(result)
            self.rvt_video = np.reshape(arry_result, (len(result), self.video.shape[1], self.video.shape[2]))
        return self.rvt_video

    def rvt_kernel(self, frame_num, rmin, rmax, kind="basic", highpass_size=None, upsample=1, rweights=None, coarse_factor=1,
            coarse_mode="add", pad_mode="constant"):
        """
        Perform Radial Variance Transform (RVT) of an image.

        Parameters
        ----------
        frame_num: int
            frame number

        rmin:
            minimal radius (inclusive)

        rmax:
            maximal radius (inclusive)

        kind:
            either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR

        highpass_size:
            size of the high-pass filter; ``None`` means no filter (effectively, infinite size)

        upsample: int
            integer image upsampling factor;
            `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
            if ``upsample>1``, the resulting image size is multiplied by ``upsample``

        rweights:
            relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
            coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
            coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        coarse_factor:
            the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

        coarse_mode:
            the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

        pad_mode:
            edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

        Returns
        -------
            Returns transform source image

        """
        img = self.video[frame_num, :, :]
        upsample = int(upsample)
        self._check_args(rmin, rmax, kind, coarse_mode, highpass_size, upsample)
        if highpass_size is not None:
            img = self.high_pass(img, highpass_size)
        if upsample > 1:
            img = img.repeat(upsample, axis=-2).repeat(upsample, axis=-1)  # nearest-neighbor upsampling on both axes
            if rweights is not None:
                rweights = np.asarray(rweights).repeat(upsample)  # upsample radial weights as well
            rmin *= upsample  # increase minimal and maximal radii
            rmax *= upsample
        return self.rvt_core(img, rmin, rmax, kind=kind, rweights=rweights, coarse_factor=coarse_factor,
                             coarse_mode=coarse_mode, pad_mode=pad_mode)


class FastRadialSymmetryTransform():
    def __init__(self):
        '''
        Implementation of fast radial symmetry transform in pure Python using OpenCV and numpy.

        References
        ----------
        [1] Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for detecting points of interest. Computer Vision, ECCV 2002.
        [2] https://github.com/Xonxt/frst
        '''
        pass

    def gradx(self, img):

      rows, cols = img.shape
      return np.hstack((np.zeros((rows, 1)), (img[:, 2:] - img[:, :-2])/2.0, np.zeros((rows, 1))))


    def grady(self, img):
        # img = img.astype('int')
        rows, cols = img.shape
        # Use vstack to add back the rows that were dropped as zeros
        return np.vstack( (np.zeros((1, cols)), (img[2:, :] - img[:-2, :])/2.0, np.zeros((1, cols))) )


    def frst(self, img, radii, alpha, beta, stdFactor, mode='BOTH'):
        """
        Performs fast radial symmetry transform

        Parameters
        ----------
        img: NDArray
           Input_video image, grayscale.

        radii: int
           Integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel

        alpha: float
           Strictness of symmetry transform (higher=more strict; 2 is good place to start)

        beta: float
            Gradient threshold parameter, float in [0,1]

        stdFactor: float
           Standard deviation factor for gaussian kernel

        mode: str
           BRIGHT, DARK, or BOTH
        """
        mode = mode.upper()
        assert mode in ['BRIGHT', 'DARK', 'BOTH']
        dark = (mode == 'DARK' or mode == 'BOTH')
        bright = (mode == 'BRIGHT' or mode == 'BOTH')

        workingDims = tuple((e + 2*radii) for e in img.shape)

        output = np.zeros(img.shape, np.float64)
        O_n = np.zeros(workingDims, np.float64)
        M_n = np.zeros(workingDims, np.float64)

        #Calculate gradients
        gx = self.gradx(img)
        gy = self.grady(img)

        #Find gradient vector magnitude
        gnorms = np.sqrt(np.add(np.multiply(gx, gx), np.multiply(gy, gy)))

        #Use beta to set threshold - speeds up transform significantly
        gthresh = np.amax(gnorms) * beta

        #Find x/y distance to affected pixels
        gpx = np.multiply(np.divide(gx, gnorms, out=np.zeros(gx.shape), where=gnorms!=0), radii).round().astype(int)
        gpy = np.multiply(np.divide(gy, gnorms, out=np.zeros(gy.shape), where=gnorms!=0), radii).round().astype(int)

        #Iterate over all pixels (w/ gradient above threshold)
        for coords, gnorm in np.ndenumerate(gnorms):
            if gnorm > gthresh:
              i, j = coords
              #Positively affected pixel
              if bright:
                ppve = (i+gpx[i, j], j+gpy[i, j])
                O_n[ppve] += 1
                M_n[ppve] += gnorm
              #Negatively affected pixel
              if dark:
                pnve = (i-gpx[i, j], j-gpy[i, j])
                O_n[pnve] -= 1
                M_n[pnve] -= gnorm

        #Abs and normalize O matrix
        O_n = np.abs(O_n)
        O_n = O_n / float(np.amax(O_n))

        #Normalize M matrix
        M_max = float(np.amax(np.abs(M_n)))
        M_n = M_n / M_max

        #Elementwise multiplication
        F_n = np.multiply(np.power(O_n, alpha), M_n)

        #Gaussian blur
        kSize = int(np.ceil(radii / 2))
        kSize = kSize + 1 if kSize % 2 == 0 else kSize

        S = cv2.GaussianBlur(F_n, (kSize, kSize), int(radii * stdFactor ))
        S = np.fliplr(S)

        return S[radii:-radii, radii:-radii]




