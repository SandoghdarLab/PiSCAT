import numpy as np
from numba import njit
import time

from scipy.ndimage import uniform_filter1d
from tqdm.autonotebook import tqdm
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from joblib import Parallel, delayed

from PySide2 import QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore

from PySide2.QtCore import *


@njit(parallel=True)
def numba_divide(ndarray_1, ndarray_2):
    divide_result = np.divide(ndarray_1, ndarray_2)
    return divide_result

@njit(parallel=True)
def numba_sum(video, axis):
    sum_result = np.sum(video, axis=axis)
    return sum_result

@njit(parallel=True)
def numba_diff(ndarray_1, ndarray_2):
    diff_result = ndarray_1 -ndarray_2
    return diff_result


class WorkerSignals(QObject):

    updateProgress_DRA = Signal(int)
    result_final = Signal(object)
    finished_DRA = Signal()
    DRA_complete_signal = Signal(bool)


class DifferentialRollingAverage(QRunnable):

    def __init__(self, video=None, batchSize=500, flag_GUI=False, object_update_progressBar=None):
        """
        Differential Rolling Average (DRA).

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        batchSize: int
            The number of frames in each batch.

        optional_1: GUI
            These flags are used when GUI calls this method.

            * `flag_GUI`: bool
                This flag is defined as True when GUI calls this method.

            * `object_update_progressBar`: object
                Object that updates the progress bar in GUI.
        """
        super(DifferentialRollingAverage, self).__init__()

        self.cpu = CPUConfigurations()

        self.flag_GUI = flag_GUI
        self.DRA_flag_run = True

        self.size_A_diff = (video.shape[0] - 2 * batchSize, video.shape[1], video.shape[2])

        self.video = video.astype(np.float64)
        self.batchSize = batchSize

        self.moving_avg = np.empty_like(self.video)
        self.output_diff = np.empty(self.size_A_diff)

        self.output_batch_1 = np.empty(self.size_A_diff)
        self.fft_output_len = self.video.shape[0] - self.batchSize

        self.object_update_progressBar = object_update_progressBar
        self.threadpool = QThreadPool()
        self.signals = WorkerSignals()

    @Slot()
    def check_DRA_finish(self, flag):
        self.DRA_flag_run = flag

    @Slot()
    def run(self, *args, **kwargs):
        self.signals.DRA_complete_signal.connect(self.check_DRA_finish)
        video_DRA = self.differential_rolling(FFT_flag=False)

        self.signals.result_final.emit(video_DRA)

    def differential_rolling(self, FFT_flag=False):
        """
        To use DRA, you'll need to call the "differential rolling" process.
        This system is fed by the following four inputs:

        Parameters
        ----------
        FFT_flag: bool
            In case it is True, DRA will be performed in parallel to improve the time performance.

        Returns
        -------
        output: NDArray
            Returns DRA video.

        """
        print("\n--- start DRA ---")

        self._apply_moving_average(FFT_flag)

        video = self.output_diff

        output = numba_divide(video, self.output_batch_1)

        return output

    def _apply_moving_average(self, FFT_flag):
        if FFT_flag:
            self.movingAvg_FFT_based()
        else:
            if self.flag_GUI:
                self.DRA_flag_run = True
                self.object_update_progressBar.setLabel('DRA')
            self.temporal_moving_average()

    def temporal_moving_average(self):

        batch_1 = np.sum(self.video[0:self.batchSize, :, :], axis=0)
        batch_2 = np.sum(self.video[self.batchSize:2 * self.batchSize, :, :], axis=0)

        batch_1_ = np.divide(batch_1, self.batchSize)
        batch_2_ = np.divide(batch_2, self.batchSize)

        self.output_diff[0, :, :] = batch_2_ - batch_1_
        self.output_batch_1[0, :, :] = batch_1_

        for i_ in tqdm(range(1, self.video.shape[0] - 2 * self.batchSize)):
            if self.flag_GUI is True:
                self.signals.updateProgress_DRA.emit(i_)
            batch_1 = batch_1 - self.video[i_ - 1, :, :] + self.video[self.batchSize + i_ - 1, :, :]
            batch_2 = batch_2 - self.video[self.batchSize + i_ - 1, :, :] + self.video[(2 * self.batchSize) + i_ - 1, :, :]
            batch_1_ = np.divide(batch_1, self.batchSize)
            batch_2_ = np.divide(batch_2, self.batchSize)

            self.output_diff[i_, :, :] = (batch_2_ - batch_1_)
            self.output_batch_1[i_, :, :] = batch_1_

        if self.flag_GUI is True:
            self.signals.finished_DRA.emit()

    def numba_temporal_moving_average(self):

        batch_1 = numba_sum(self.video[0:self.batchSize, :, :], axis=0)
        batch_2 = numba_sum(self.video[self.batchSize:2 * self.batchSize, :, :], axis=0)

        batch_1_ = numba_divide(batch_1, self.batchSize)
        batch_2_ = numba_divide(batch_2, self.batchSize)

        self.output_diff[0, :, :] = numba_diff(batch_2_, batch_1_)
        self.output_batch_1[0, :, :] = batch_1_

        for i_ in tqdm(range(1, self.video.shape[0] - 2 * self.batchSize)):
            if self.flag_GUI is True:
                self.signals.updateProgress_DRA.emit(i_)
            batch_1 = batch_1 - self.video[i_ - 1, :, :] + self.video[self.batchSize + i_ - 1, :, :]
            batch_2 = batch_2 - self.video[self.batchSize + i_ - 1, :, :] + self.video[(2 * self.batchSize) + i_ - 1, :, :]

            batch_1_ = numba_divide(batch_1, self.batchSize)
            batch_2_ = numba_divide(batch_2, self.batchSize)

            self.output_diff[0, :, :] = numba_diff(batch_2_, batch_1_)
            self.output_batch_1[0, :, :] = batch_1_

        if self.flag_GUI is True:
            self.signals.finished_DRA.emit()

    def movingAvg_FFT_based(self):

        Parallel(n_jobs=self.cpu.n_jobs, backend='threading')(delayed(self.uniform_filter1d_kernel)(r_, c_)
                            for r_ in range(self.video.shape[1])
                            for c_ in range(self.video.shape[2]))

        self.output_diff = self.moving_avg[self.batchSize:self.fft_output_len, ...] - self.moving_avg[0:self.fft_output_len-self.batchSize, ...]

        self.output_batch_1 = self.moving_avg[0:self.fft_output_len - self.batchSize, ...]

    def uniform_filter1d_kernel(self, r_, c_):
        self.moving_avg[:, r_, c_] = uniform_filter1d(self.video[:, r_, c_], size=self.batchSize,
                                                      mode='constant', cval=0.0,
                                                      origin=-(self.batchSize // 2), axis=0)










