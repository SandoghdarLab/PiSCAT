from __future__ import print_function

import numpy as np
import pywt
from joblib import Parallel, delayed
from PySide6.QtCore import *
from skimage import filters
from skimage.morphology import rectangle
from tqdm.autonotebook import tqdm
from tqdm.notebook import tqdm

from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Preproccessing.filtering import GuidedFilter
from piscat.Visualization.print_colors import PrintColors


class WorkerSignals(QObject):
    updateProgress_FPNC = Signal(int)
    result_FPNc = Signal(object)
    finished_FPNc = Signal()


class MedianProjectionFPNc(QRunnable):
    def __init__(self, video, select_correction_axis, flag_GUI=False):
        """
        This class uses a heuristic procedure called Median Projection FPN (mFPN) to reduce fixed pattern noise (FPN).

        References
        ----------
        [1] Mirzaalian Dastjerdi, Houman, et al. "Optimized analysis for sensitive detection and analysis of single
        proteins via interferometric scattering microscopy." Journal of Physics D: Applied Physics (2021).
        (http://iopscience.iop.org/article/10.1088/1361-6463/ac2f68)

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        select_correction_axis: int (0/1)
            * `0`: FPN will be applied row-wise.
            * `1`: FPN will be applied column-wise.
        """

        super(MedianProjectionFPNc, self).__init__()
        self.cpu = CPUConfigurations()
        self.video = video
        self.select_correction_axis = select_correction_axis
        self.flag_GUI = flag_GUI
        self.threadpool = QThreadPool()
        self.signals = WorkerSignals()

    @Slot()
    def run(self, *args, **kwargs):
        result = self.mFPNc(self.select_correction_axis)
        self.signals.result_FPNc.emit(result)

    def mFPNc(self, select_correction_axis):
        """
        Using the mPN approach on video.

        Parameters
        ----------
        select_correction_axis: int (0/1)

           * `0`: FPN will be applied row-wise.
           * `1`: FPN will be applied column-wise.

        Returns
        -------
        output: NDArray
            Video after using the mFPNc technique.
        """
        if False:
            print("\nFPN correction with parallel loop --->", end=" ")

            result = Parallel(
                n_jobs=self.cpu.n_jobs, backend="multiprocessing", verbose=self.cpu.verbose
            )(
                delayed(self.FPNc_kernel)(self.video[f_, :, :], select_correction_axis)
                for f_ in tqdm(range(0, self.video.shape[0]))
            )
            output = np.asarray(result)

        else:
            print("\nmedian FPN correction without parallel loop --->", end=" ")
            output = np.zeros_like(self.video)

            for i_ in tqdm(range(0, self.video.shape[0])):
                if self.flag_GUI is True:
                    self.signals.updateProgress_FPNC.emit(i_)

                diff = self.video[i_, :, :]

                gainMap1D_median = np.median(diff, axis=select_correction_axis)

                if select_correction_axis == 1:
                    FPN2D_median = np.repeat(
                        gainMap1D_median[:, np.newaxis],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )
                elif select_correction_axis == 0:
                    FPN2D_median = np.repeat(
                        gainMap1D_median[np.newaxis, :],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )

                output[i_, :, :] = diff - FPN2D_median

        if self.flag_GUI is True:
            self.signals.finished_FPNc.emit()
        print("Done")

        return output

    def FPNc_kernel(self, diff, select_correction_axis):
        gainMap1D_median = np.median(diff, axis=select_correction_axis)

        if select_correction_axis == 1:
            FPN2D_median = np.repeat(
                gainMap1D_median[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            FPN2D_median = np.repeat(
                gainMap1D_median[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        AD_absDev = np.abs(diff - FPN2D_median)
        MAD_1D_MedianAD = np.median(AD_absDev, axis=select_correction_axis)

        if select_correction_axis == 1:
            MAD_2D = np.repeat(
                MAD_1D_MedianAD[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            MAD_2D = np.repeat(
                MAD_1D_MedianAD[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        diff_f1f2_nan = diff.copy()
        diff_f1f2_nan[MAD_2D < AD_absDev] = np.nan
        gainMap1D_mean = np.nanmean(diff_f1f2_nan, axis=select_correction_axis)

        if select_correction_axis == 1:
            FPN2D = np.repeat(
                gainMap1D_mean[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            FPN2D = np.repeat(
                gainMap1D_mean[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        return diff - FPN2D


class ColumnProjectionFPNc(QRunnable):
    def __init__(self, video, select_correction_axis, flag_GUI=False):
        """
        This class uses a heuristic procedure called Column Projection FPN (cpFPN) to reduce fixed pattern noise (FPN).

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        select_correction_axis: int (0/1)
            * `0`: FPN will be applied row-wise.
            * `1`: FPN will be applied column-wise.
        """
        super(ColumnProjectionFPNc, self).__init__()
        self.cpu = CPUConfigurations()
        self.video = video
        self.select_correction_axis = select_correction_axis
        self.flag_GUI = flag_GUI
        self.threadpool = QThreadPool()
        self.signals = WorkerSignals()

    @Slot()
    def run(self, *args, **kwargs):
        result = self.cpFPNc(self.select_correction_axis)
        self.signals.result_FPNc.emit(result)

    def cpFPNc(self, select_correction_axis):
        """
        Using the cpFPN approach on video.

        Parameters
        ----------
        select_correction_axis: int (0/1)

           * `0`: FPN will be applied row-wise.
           * `1`: FPN will be applied column-wise.

        Returns
        -------
        output: NDArray
            Video after using the cpFPNc technique.
        """
        if False:
            print("\nFPN correction with parallel loop --->", end=" ")

            result = Parallel(
                n_jobs=self.cpu.n_jobs, backend="multiprocessing", verbose=self.cpu.verbose
            )(
                delayed(self.FPNc_kernel)(self.video[f_, :, :], select_correction_axis)
                for f_ in tqdm(range(0, self.video.shape[0]))
            )
            output = np.asarray(result)

        else:
            print("\ncpFPN correction without parallel loop --->", end=" ")
            output = np.zeros_like(self.video)

            for i_ in tqdm(range(0, self.video.shape[0])):
                if self.flag_GUI is True:
                    self.signals.updateProgress_FPNC.emit(i_)

                diff = self.video[i_, :, :]

                gainMap1D_median = np.median(diff, axis=select_correction_axis)

                if select_correction_axis == 1:
                    FPN2D_median = np.repeat(
                        gainMap1D_median[:, np.newaxis],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )
                elif select_correction_axis == 0:
                    FPN2D_median = np.repeat(
                        gainMap1D_median[np.newaxis, :],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )

                AD_absDev = np.abs(diff - FPN2D_median)
                MAD_1D_MedianAD = np.median(
                    AD_absDev, axis=select_correction_axis
                )  # convert to median from here

                if select_correction_axis == 1:
                    MAD_2D = np.repeat(
                        MAD_1D_MedianAD[:, np.newaxis],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )
                elif select_correction_axis == 0:
                    MAD_2D = np.repeat(
                        MAD_1D_MedianAD[np.newaxis, :],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )

                diff_f1f2_nan = diff.copy()
                diff_f1f2_nan[MAD_2D < AD_absDev] = np.nan
                gainMap1D_mean = np.nanmean(diff_f1f2_nan, axis=select_correction_axis)

                if select_correction_axis == 1:
                    FPN2D = np.repeat(
                        gainMap1D_mean[:, np.newaxis],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )
                elif select_correction_axis == 0:
                    FPN2D = np.repeat(
                        gainMap1D_mean[np.newaxis, :],
                        diff.shape[select_correction_axis],
                        axis=select_correction_axis,
                    )

                output[i_, :, :] = diff - FPN2D

        if self.flag_GUI is True:
            self.signals.finished_FPNc.emit()
        print("Done")

        return output

    def FPNc_kernel(self, diff, select_correction_axis):
        gainMap1D_median = np.median(diff, axis=select_correction_axis)

        if select_correction_axis == 1:
            FPN2D_median = np.repeat(
                gainMap1D_median[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            FPN2D_median = np.repeat(
                gainMap1D_median[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        AD_absDev = np.abs(diff - FPN2D_median)
        MAD_1D_MedianAD = np.median(AD_absDev, axis=select_correction_axis)

        if select_correction_axis == 1:
            MAD_2D = np.repeat(
                MAD_1D_MedianAD[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            MAD_2D = np.repeat(
                MAD_1D_MedianAD[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        diff_f1f2_nan = diff.copy()
        diff_f1f2_nan[MAD_2D < AD_absDev] = np.nan
        gainMap1D_mean = np.nanmean(diff_f1f2_nan, axis=select_correction_axis)

        if select_correction_axis == 1:
            FPN2D = np.repeat(
                gainMap1D_mean[:, np.newaxis],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )
        elif select_correction_axis == 0:
            FPN2D = np.repeat(
                gainMap1D_mean[np.newaxis, :],
                diff.shape[select_correction_axis],
                axis=select_correction_axis,
            )

        return diff - FPN2D


class FrequencyFPNc:
    def __init__(self, video, inter_flag_parallel_active=True):
        """
        This class corrects FPN using two well-known frequency domain techniques from the literature.

        References
        ----------
        [1] Cao, Yanlong, et al. "A multi-scale non-uniformity correction method based on wavelet decomposition and
        guided filtering for uncooled long wave infrared camera." Signal Processing: Image Communication 60 (2018): 13-21.

        [2] Zeng, Qingjie, et al. "Single infrared image-based stripe non-uniformity correction via a two-stage
        filtering method." Sensors 18.12 (2018): 4299.

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        inter_flag_parallel_active: bool
            If the user wants to enable generic parallel tasks in CPU configuration, this flag is used to disable parallel execution of this function.
        """
        super(FrequencyFPNc, self).__init__()
        self.cpu = CPUConfigurations()
        self.inter_flag_parallel_active = inter_flag_parallel_active

        self.coeffs = None
        self.arr = None
        self.coeff_slices = None
        self.video_out = None
        self.im_fft_noiseless = None
        self.im_fft_noise = None
        self.im_recon = None
        self.coeffs_result = None
        self.print_color = PrintColors()
        self.video = video

    def update_fFPN(self, direction="Horizontal", max_iterations=10, width=1):
        """
        This method corrects the FPNc by using FFT [2].

        Parameters
        ----------
        direction: str
            Axis that FPN correction should apply on it.

            * ``'Horizontal'``
            * ``'Vertical'``

        max_iterations: int
            Total number of filtering iterations.

        width: int
            The frequency mask's width.

        Returns
        -------
        n_video: NDArray
            Video after using the fFPNc technique.
        """
        self.max_iterations = max_iterations
        self.video_out = np.ndarray(self.video.shape)
        self.direction = direction

        w = width
        if self.direction == "Horizontal":
            self.mask = np.ones(self.video[0, :, :].shape)
            center_freq_line_H = round(1 + (0.5 * self.mask.shape[1]))
            self.mask[center_freq_line_H - w : center_freq_line_H + w, :] = 0

        elif self.direction == "Vertical":
            self.mask = np.ones(self.video[0, :, :].shape)
            center_freq_line_v = round(1 + (0.5 * self.mask.shape[1]))
            self.mask[:, center_freq_line_v - w : center_freq_line_v + w] = 0

        if self.cpu.parallel_active is True and self.inter_flag_parallel_active is True:
            print("\n---start fFPNc with Parallel---")
            result = Parallel(
                n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose
            )(delayed(self.fft_kernel)(f_) for f_ in tqdm(range(self.video.shape[0])))

            self.video_out = np.asarray(result)
        else:
            print("\n---start fFPNc without Parallel---")
            result = [self.fft_kernel(f_) for f_ in tqdm(range(0, self.video.shape[0]))]
            self.video_out = np.asarray(result)

        return self.video_out

    def fft_kernel(self, f_):
        selected_frame = self.video[f_, :, :]
        im_fft = np.fft.fftshift(np.fft.fft2(selected_frame))
        amplitude = np.abs(im_fft)
        phase = np.angle(im_fft)

        amplitude_noiseless = np.multiply(self.mask, amplitude)
        im_fft_noiseless = np.multiply(amplitude_noiseless, np.exp(phase * 1j))
        im_fft_noiseless = np.fft.fftshift(im_fft_noiseless)
        im_noiseless = np.fft.ifft2(im_fft_noiseless)

        im_noise = self.video[f_, :, :] - np.real(im_noiseless)

        if self.direction == "Horizontal":
            selem = rectangle(nrows=1, ncols=5)

        elif self.direction == "Vertical":
            selem = rectangle(nrows=5, ncols=1)

        filter_img = im_noise.copy()
        for i_ in range(self.max_iterations):
            try:
                filter_img = filters.gaussian(filter_img, sigma=1.2, preserve_range=True)
                # filter_img = rank.mean(filter_img, selem=selem)

            except:
                print(f"{self.WARNING}\nThe Gaussian filter can not work!{self.ENDC}")

        img_FPNc = np.real(im_noiseless) + np.real(filter_img)

        return img_FPNc

    def update_wFPN(self, direction="Horizontal"):
        """
        This method corrects the FPNc by using wavelet[1].

        Parameters
        ----------
        direction: str
            Axis that FPN correction should apply on it.

            * ``'Horizontal'``
            * ``'Vertical'``

        Returns
        -------
        n_video: NDArray
            Video after using the wFPNc technique.
        """
        self.video_out = np.empty_like(self.video)

        if self.cpu.parallel_active is True and self.inter_flag_parallel_active is True:
            print("\n---start wFPNc with Parallel---")
            result = Parallel(
                n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose
            )(
                delayed(self.wavelet_denoise)(f_, direction)
                for f_ in tqdm(range(self.video.shape[0]))
            )

            self.video_out = np.asarray(result)
        else:
            print("\n---start wFPNc without Parallel---")
            for f_ in tqdm(range(0, self.video.shape[0])):
                self.video_out[f_, :, :] = self.wavelet_denoise(f_, direction=direction)

        return self.video_out

    def wavelet_denoise(self, f_, direction="Horizontal"):
        coeffs_1 = pywt.dwt2(self.video[f_, :, :], wavelet="sym8")
        cA_1, (cH_1, cV_1, cD_1) = coeffs_1

        coeffs_2 = pywt.dwt2(cA_1, wavelet="sym8")
        cA_2, (cH_2, cV_2, cD_2) = coeffs_2

        coeffs_3 = pywt.dwt2(cA_2, wavelet="sym8")
        cA_3, (cH_3, cV_3, cD_3) = coeffs_3

        if direction == "Horizontal":
            cH_3_corr = self.guided_filter(
                cH_3, cA_3, win_size=(1, np.int(0.5 * cA_3.shape[0])), eps=0.3**2
            )
            coeffs_3_corr = cA_3, (cH_3_corr, cV_3, cD_3)
            cA_2_new_ = pywt.idwt2(coeffs_3_corr, "sym8")
            cA_2_new = self.size_handling(cH_2, cA_2_new_)

            cH_2_corr = self.guided_filter(
                cH_2, cA_2_new, win_size=(1, np.int(0.25 * cA_3.shape[0])), eps=0.2**2
            )
            coeffs_2_corr = cA_2_new, (cH_2_corr, cV_2, cD_2)
            cA_1_new_ = pywt.idwt2(coeffs_2_corr, "sym8")
            cA_1_new = self.size_handling(cH_1, cA_1_new_)

            cH_1_corr = self.guided_filter(
                cH_1, cA_1_new, win_size=(1, np.int(0.1 * cA_3.shape[0])), eps=0.1**2
            )
            coeffs_1_corr = cA_1_new, (cH_1_corr, cV_1, cD_1)
            im_corr = pywt.idwt2(coeffs_1_corr, "sym8")
            return self.size_handling(self.video[f_, :, :], im_corr)

        elif direction == "Vertical":
            cV_3_corr = self.guided_filter(
                cV_3, cA_3, win_size=(np.int(0.5 * cA_3.shape[0]), 1), eps=0.3**2
            )
            coeffs_3_corr = cA_3, (cH_3, cV_3_corr, cD_3)
            cA_2_new_ = pywt.idwt2(coeffs_3_corr, "sym8")
            cA_2_new = self.size_handling(cV_2, cA_2_new_)

            cV_2_corr = self.guided_filter(
                cV_2, cA_2_new, win_size=(np.int(0.25 * cA_3.shape[0]), 1), eps=0.2**2
            )
            coeffs_2_corr = cA_2_new, (cH_2, cV_2_corr, cD_2)
            cA_1_new_ = pywt.idwt2(coeffs_2_corr, "sym8")
            cA_1_new = self.size_handling(cV_1, cA_1_new_)

            cV_1_corr = self.guided_filter(
                cV_1, cA_1_new, win_size=(np.int(0.1 * cA_3.shape[0]), 1), eps=0.1**2
            )
            coeffs_1_corr = cA_1_new, (cH_1, cV_1_corr, cD_1)
            im_corr = pywt.idwt2(coeffs_1_corr, "sym8")
            return self.size_handling(self.video[f_, :, :], im_corr)

    def guided_filter(self, input_image, guided_image, win_size, eps):
        GF_ = GuidedFilter(guided_image, radius=win_size, eps=eps)
        return GF_.filter(input_image)

    def size_handling(self, im_1, im_2):
        if im_1.shape[0] == im_2.shape[0] and im_1.shape[1] == im_2.shape[1]:
            im_out = im_2
        elif im_1.shape[0] == im_2.shape[0]:
            num_cut = np.abs(im_1.shape[1] - im_2.shape[1])
            im_out = im_2[:, 0:-num_cut]
        elif im_1.shape[1] == im_2.shape[1]:
            num_cut = np.abs(im_1.shape[0] - im_2.shape[0])
            im_out = im_2[0:-num_cut, :]
        else:
            num_cut_0 = np.abs(im_1.shape[0] - im_2.shape[0])
            num_cut_1 = np.abs(im_1.shape[1] - im_2.shape[1])
            im_out = im_2[0:-num_cut_0, 0:-num_cut_1]
        return im_out
