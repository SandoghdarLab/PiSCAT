import numpy as np
from joblib import Parallel, delayed
from PySide2 import QtGui
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2.QtCore import *
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from tqdm.autonotebook import tqdm


class WorkerSignals(QObject):

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Normalization(QRunnable):

    def __init__(self, video, flag_pn=False, flag_global=False, flag_image_specific=False):
        """
        This class contains a different version of video/image normalization methods.

        Parameters
        ----------
        video: NDArray
            It is Numpy 3D video array.

        optional_1: GUI
            * `flag_pn`: bool
                If it is True, the power_normalized method is applied on input video on GUI.

            * `flag_global`: bool
                If it is True, the normalized_image_global method is applied on input video on GUI.

            * `flag_image_specific`: bool
                If it is True, the normalized_image_specific method is applied on input video on GUI.

        """
        super(Normalization, self).__init__()
        self.cpu = CPUConfigurations()

        self.video = video
        self.signals = WorkerSignals()

        self.flag_pn = flag_pn
        self.flag_global = flag_global
        self.flag_image_specific = flag_image_specific

    @Slot()
    def run(self, *args, **kwargs):
        if self.flag_pn:
            result = self.power_normalized()
        if self.flag_global:
            result = self.normalized_image_global()
        if self.flag_image_specific:
            result = self.normalized_image_specific()

        self.signals.result.emit(result)

    def normalized_image_global(self, new_max=1, new_min=0):
        """
        Based on the global min and max in the video, this method normalizes all pixels in the video between``new_min`` and ``new_max``.

        Parameters
        ----------
        new_max: float
            Video's new global maximum pixel intensity.

        new_min: float
            Video's new global minimum pixel intensity.

        Returns
        -------
        img2: NDArray
            Normalize video (3D-Numpy array).
        """
        mins = self.video.min(axis=(0, 1, 2), keepdims=True)
        maxs = self.video.max(axis=(0, 1, 2), keepdims=True)
        img2 = ((self.video - mins) * (new_max - new_min)) / (maxs - mins) + new_min
        return img2

    def normalized_image_specific(self, scale=255, format='uint8'):
        """
        This approach normalizes all pixels in the image between 0 and 1 based on the image min and max in the video frame.

        Parameters
        ----------
        scale: float
            Video's new global maximum pixel intensity.

        format: str
            It describes how the bytes in the fixed-size block of memory corresponding to an array item should be interpreted.

        Returns
        -------
        n_video: NDArray
            Normalize video (3D-Numpy array).
        """
        if len(self.video.shape) == 3:
            print('\nconverting video bin_type to ' + format + '--->', end=" ")

            tmp_0 = np.reshape(self.video, (self.video.shape[0], -1))

            min_tmp1 = np.expand_dims(tmp_0.min(axis=1), axis=1)

            ptp_tmp1 = np.expand_dims(tmp_0.ptp(axis=1), axis=1)

            tmp2 = np.subtract(tmp_0, min_tmp1)
            tmp3 = np.divide(tmp2, ptp_tmp1)
            n_video_ = scale * np.reshape(tmp3, (self.video.shape[0], self.video.shape[1], self.video.shape[2]))
            n_video = n_video_.astype(format)
            print('Done')

        elif len(self.video.shape) == 2:

            min_tmp1 = self.video.min()

            ptp_tmp1 = self.video.ptp()

            tmp2 = np.subtract(self.video, min_tmp1)
            tmp3 = np.divide(tmp2, ptp_tmp1)
            n_video_ = scale * tmp3
            n_video = n_video_.astype(format)

        return n_video

    def normalized_image_specific_by_max(self):
        print('\nnormalize image by max --->', end=" ")
        n_video = np.empty_like(self.video, dtype=np.float64)
        for i in range(self.video.shape[0]):
            img = self.video[i, :, :]
            n_video[i, :, :] = np.divide(img, img.max())
        print('Done')
        return n_video

    def normalized_image(self):
        """
        Normalization of video between 0 and 1.

        Returns
        -------
        n_video: NDArray
            Normalize video (3D-Numpy array).
        """
        mins = self.video.min()
        maxs = self.video.max()
        img2 = (self.video - mins) / (maxs - mins)
        return img2

    def power_normalized(self, inter_flag_parallel_active=False):
        """
        This function corrects the fluctuations in the laser light intensity
        by dividing each pixel in an image by the sum of all pixels on the same frames.

        Parameters
        ----------
        inter_flag_parallel_active: bool
            Internal flag for activating parallel computation. Default is False!

        Returns
        -------
        normalized_power: NDArray
            Normalize video (3D-Numpy array).

        power_fluctuation_percentage: NDArray
            Temporal fluctuations of all pixels after power normalization.
        """
        temp0 = np.sum(self.video, axis=1)
        sum_pixels = np.sum(temp0, axis=1)
        if inter_flag_parallel_active is True and self.cpu.parallel_active is True:
            print("\n---start power_normalized with parallel loop---")
            result = Parallel(n_jobs=self.cpu.n_jobs, backend="threading", verbose=self.cpu.verbose)(
                delayed(self.power_normalized_kernel)(f_) for f_ in tqdm(range(self.video.shape[0])))
            normalized_power = np.asarray(result)
            normalized_power = normalized_power * np.mean(sum_pixels)

            power_fluctuation_percentage = np.divide(sum_pixels, np.mean(sum_pixels)) - 1

        else:
            print("\nstart power_normalized without parallel loop--->", end=" ")
            tmp_ = np.repeat(sum_pixels, self.video.shape[1] * self.video.shape[2])
            temp2 = np.reshape(tmp_, (self.video.shape[0], self.video.shape[1], self.video.shape[2]))
            normalized_power = np.divide(self.video, temp2) * np.mean(sum_pixels)
            power_fluctuation_percentage = np.divide(sum_pixels, np.mean(sum_pixels)) - 1
            print("Done")
        return (normalized_power, power_fluctuation_percentage)

    def power_normalized_kernel(self, f_):
        sum_img_pixels = np.sum(self.video[f_])
        return np.divide(self.video[f_], sum_img_pixels)