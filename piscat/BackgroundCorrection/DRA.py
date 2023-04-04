import time

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from PySide6 import QtCore
#from PySide6.QtCore import *
from scipy.ndimage import uniform_filter1d
from tqdm.autonotebook import tqdm

from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Preproccessing import FPNc


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
    diff_result = ndarray_1 - ndarray_2
    return diff_result


class WorkerSignals(QtCore.QObject):
    updateProgress_DRA = QtCore.Signal(int)
    result_final = QtCore.Signal(object)
    finished_DRA = QtCore.Signal()
    DRA_complete_signal = QtCore.Signal(bool)
    updateProgress_FPNC = QtCore.Signal(int)


class DifferentialRollingAverage(QtCore.QRunnable):
    def __init__(
        self,
        video=None,
        batchSize=500,
        flag_GUI=False,
        object_update_progressBar=None,
        mode_FPN="mFPN",
        FPN_flag_GUI=False,
        gui_select_correction_axis=1,
    ):
        """
        Differential Rolling Average (DRA).

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        batchSize: int
            The number of frames in each batch.

        mode_FPN: {‘cpFPN’, ‘mFPN’, ‘wFPN’, 'fFPN'}, optional
            Flag that defines method of FPNc.

                    * `mFPN`: Median fixed pattern noise correction
                    * `cpFPN`: Median fixed pattern noise correction
                    * `wFPN`: Wavelet FPNc
                    * `fFPN`: FFT2D_Wavelet FPNc

        optional_1: GUI
            These flags are used when GUI calls this method.

            * `flag_GUI`: bool
                This flag is defined as True when GUI calls this method.

            * `FPN_flag_GUI`: bool
                This flag is defined as True when GUI calls this method while we want activate FPNc.

            * `gui_select_correction_axis`: int (0/1), 'Both'
                This parameter is used only when FPN_flag_GUI is True, otherwise it will be ignored.

            * `object_update_progressBar`: object
                Object that updates the progress bar in GUI.
        """
        super(DifferentialRollingAverage, self).__init__()

        self.cpu = CPUConfigurations()

        self.flag_GUI = flag_GUI
        self.mode_FPN = mode_FPN
        self.FPN_flag_run = FPN_flag_GUI
        self.DRA_flag_run = True
        self.select_correction_axis_run = gui_select_correction_axis

        self.size_A_diff = (video.shape[0] - 2 * batchSize, video.shape[1], video.shape[2])

        self.video = video.astype(np.float64)
        self.batchSize = batchSize

        self.moving_avg = np.empty_like(self.video)
        self.output_diff = np.empty(self.size_A_diff)

        self.output_batch_1 = np.empty(self.size_A_diff)
        self.fft_output_len = self.video.shape[0] - self.batchSize

        self.FPNc_video = None
        self.flag_thread_FPNc = True
        self.object_update_progressBar = object_update_progressBar
        self.threadpool = QtCore.QThreadPool()
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def check_DRA_finish(self, flag):
        self.DRA_flag_run = flag

    @QtCore.Slot()
    def run(self, *args, **kwargs):
        self.signals.DRA_complete_signal.connect(self.check_DRA_finish)
        video_DRA = self.differential_rolling(
            FPN_flag=self.FPN_flag_run,
            select_correction_axis=self.select_correction_axis_run,
            FFT_flag=False,
            inter_flag_parallel_active=True,
        )

        self.signals.result_final.emit(video_DRA)

    @QtCore.Slot()
    def thread_FPNc_complete(self):
        self.object_update_progressBar.setRange(self.p_max)
        self.object_update_progressBar.setLabel("")
        print("THREAD FPNc COMPLETE!")

    @QtCore.Slot()
    def result_tread_FPNc(self, result):
        self.FPNc_video = result
        self.flag_thread_FPNc = False
        print("FPNc video update!")

    @QtCore.Slot()
    def startProgressBarFPNc(self, instance, **kwargs):
        self.thread_FPNc = instance(**kwargs)
        self.thread_FPNc.signals.updateProgress_FPNC.connect(
            self.object_update_progressBar.setProgress
        )
        self.thread_FPNc.signals.finished_FPNc.connect(self.thread_FPNc_complete)
        self.thread_FPNc.signals.result_FPNc.connect(self.result_tread_FPNc)

        self.threadpool.start(self.thread_FPNc)

    def differential_rolling(
        self,
        FPN_flag=False,
        select_correction_axis=1,
        FFT_flag=False,
        inter_flag_parallel_active=True,
        max_iterations=10,
        FFT_widith=1,
    ):
        """
        To use DRA, you'll need to call the "differential rolling" process.

        Parameters
        ----------
        FPN_flag: bool
            This flag activates the fixed pattern noise correction function in case define as true.

        select_correction_axis: int (0/1), 'Both'
            This parameter is used only when FPN_flag is True, otherwise it will be ignored.

            * 0: FPN will be applied row-wise.
            * 1: FPN will be applied column-wise.
            * 'Both': FPN will be applied on two axis.

        FFT_flag: bool
            In case it is True, DRA will be performed in parallel to improve the time performance.

        inter_flag_parallel_active: bool
            This flag actives/inactives parallel computation of wFPNc.

        max_iterations: int
            This parameter is used when fFPT is selected that defines the total number of filtering iterations.

        FFT_widith: int
            This parameter is used when fFPT is selected that defines the frequency mask's width.

        Returns
        -------
        output: NDArray
            Returns DRA video.

        gainMap1D_: NDArray
            Returns projection on each frame based on the correction axis
        """
        if FPN_flag and self.mode_FPN == "cpFPN":
            print("\n--- start DRA + cpFPN_axis: " + str(select_correction_axis) + "---")
        elif FPN_flag and self.mode_FPN == "wFPN":
            print("\n--- start DRA + wFPN_axis: " + str(select_correction_axis) + "---")
        elif FPN_flag and self.mode_FPN == "fFPN":
            print("\n--- start DRA + fFPN_axis: " + str(select_correction_axis) + "---")
        elif FPN_flag and self.mode_FPN == "mFPN":
            print("\n--- start DRA + mFPN_axis: " + str(select_correction_axis) + "---")
        else:
            print("\n--- start DRA ---")

        self._apply_moving_average(FFT_flag)

        if FPN_flag:
            if self.mode_FPN == "mFPN" or self.mode_FPN == "cpFPN":
                if self.flag_GUI is True:
                    self.DRA_flag_run = True

                    while self.DRA_flag_run:
                        QtCore.QCoreApplication.processEvents()

                    time.sleep(2)

                    video = self._FPNc_GUI_axis(
                        self.output_diff, select_correction_axis, self.mode_FPN
                    )

                    time.sleep(2)

                else:
                    out_diff = self.output_diff
                    video = self._FPNc_axis(
                        out_diff,
                        select_correction_axis,
                        self.mode_FPN,
                        inter_flag_parallel_active,
                    )
            else:
                video = self.output_diff
        else:
            video = self.output_diff

        output = numba_divide(video, self.output_batch_1)

        if FPN_flag:
            if self.mode_FPN == "wFPN" or self.mode_FPN == "fFPN":
                output = self._FPNc_axis(
                    output,
                    select_correction_axis,
                    self.mode_FPN,
                    inter_flag_parallel_active,
                    max_iterations,
                    FFT_widith,
                )

        if select_correction_axis == "Both":
            gainMap1D_1 = np.mean(output, axis=1)
            gainMap1D_2 = np.mean(output, axis=2)
            gainMap1D_ = [gainMap1D_1, gainMap1D_2]

        elif select_correction_axis is not None:
            gainMap1D_ = np.mean(output, axis=select_correction_axis + 1)

        else:
            gainMap1D_ = None

        return output, gainMap1D_

    def _apply_moving_average(self, FFT_flag):
        if FFT_flag:
            self.movingAvg_FFT_based()
        else:
            if self.flag_GUI:
                self.DRA_flag_run = True
                self.object_update_progressBar.setLabel("DRA")
            self.temporal_moving_average()

    def temporal_moving_average(self):
        batch_1 = np.sum(self.video[0 : self.batchSize, :, :], axis=0)
        batch_2 = np.sum(self.video[self.batchSize : 2 * self.batchSize, :, :], axis=0)

        batch_1_ = np.divide(batch_1, self.batchSize)
        batch_2_ = np.divide(batch_2, self.batchSize)

        self.output_diff[0, :, :] = batch_2_ - batch_1_
        self.output_batch_1[0, :, :] = batch_1_

        for i_ in tqdm(range(1, self.video.shape[0] - 2 * self.batchSize)):
            if self.flag_GUI is True:
                self.signals.updateProgress_DRA.emit(i_)
            batch_1 = (
                batch_1 - self.video[i_ - 1, :, :] + self.video[self.batchSize + i_ - 1, :, :]
            )
            batch_2 = (
                batch_2
                - self.video[self.batchSize + i_ - 1, :, :]
                + self.video[(2 * self.batchSize) + i_ - 1, :, :]
            )
            batch_1_ = np.divide(batch_1, self.batchSize)
            batch_2_ = np.divide(batch_2, self.batchSize)

            self.output_diff[i_, :, :] = batch_2_ - batch_1_
            self.output_batch_1[i_, :, :] = batch_1_

        if self.flag_GUI is True:
            self.signals.finished_DRA.emit()

    def numba_temporal_moving_average(self):
        batch_1 = numba_sum(self.video[0 : self.batchSize, :, :], axis=0)
        batch_2 = numba_sum(self.video[self.batchSize : 2 * self.batchSize, :, :], axis=0)

        batch_1_ = numba_divide(batch_1, self.batchSize)
        batch_2_ = numba_divide(batch_2, self.batchSize)

        self.output_diff[0, :, :] = numba_diff(batch_2_, batch_1_)
        self.output_batch_1[0, :, :] = batch_1_

        for i_ in tqdm(range(1, self.video.shape[0] - 2 * self.batchSize)):
            if self.flag_GUI is True:
                self.signals.updateProgress_DRA.emit(i_)
            batch_1 = (
                batch_1 - self.video[i_ - 1, :, :] + self.video[self.batchSize + i_ - 1, :, :]
            )
            batch_2 = (
                batch_2
                - self.video[self.batchSize + i_ - 1, :, :]
                + self.video[(2 * self.batchSize) + i_ - 1, :, :]
            )

            batch_1_ = numba_divide(batch_1, self.batchSize)
            batch_2_ = numba_divide(batch_2, self.batchSize)

            self.output_diff[0, :, :] = numba_diff(batch_2_, batch_1_)
            self.output_batch_1[0, :, :] = batch_1_

        if self.flag_GUI is True:
            self.signals.finished_DRA.emit()

    def movingAvg_FFT_based(self):
        Parallel(n_jobs=self.cpu.n_jobs, backend="threading")(
            delayed(self.uniform_filter1d_kernel)(r_, c_)
            for r_ in range(self.video.shape[1])
            for c_ in range(self.video.shape[2])
        )

        self.output_diff = (
            self.moving_avg[self.batchSize : self.fft_output_len, ...]
            - self.moving_avg[0 : self.fft_output_len - self.batchSize, ...]
        )

        self.output_batch_1 = self.moving_avg[0 : self.fft_output_len - self.batchSize, ...]

    def uniform_filter1d_kernel(self, r_, c_):
        self.moving_avg[:, r_, c_] = uniform_filter1d(
            self.video[:, r_, c_],
            size=self.batchSize,
            mode="constant",
            cval=0.0,
            origin=-(self.batchSize // 2),
            axis=0,
        )

    def _FPNc_axis(
        self,
        video,
        select_correction_axis,
        mode_FPN,
        inter_flag_parallel_active=True,
        max_iterations=10,
        FFT_widith=1,
    ):
        if mode_FPN == "cpFPN" or mode_FPN == "mFPN":
            if select_correction_axis == "Both":
                vid_FPNc_axis_1 = self._FPN(
                    video, 1, mode_FPN, inter_flag_parallel_active, max_iterations, FFT_widith
                )
                vid_FPNc = self._FPN(
                    vid_FPNc_axis_1,
                    0,
                    mode_FPN,
                    inter_flag_parallel_active,
                    max_iterations,
                    FFT_widith,
                )

            else:
                vid_FPNc = self._FPN(
                    video,
                    select_correction_axis,
                    mode_FPN,
                    inter_flag_parallel_active,
                    max_iterations,
                    FFT_widith,
                )
        else:
            vid_FPNc = self._FPN(
                video,
                select_correction_axis,
                mode_FPN,
                inter_flag_parallel_active,
                max_iterations,
                FFT_widith,
            )

        return vid_FPNc

    def _FPN(
        self,
        video,
        select_correction_axis,
        mode_FPN,
        inter_flag_parallel_active=True,
        max_iterations=10,
        FFT_widith=1,
    ):
        if mode_FPN == "mFPN":
            noise_c = FPNc.MedianProjectionFPNc(
                video=video, select_correction_axis=select_correction_axis, flag_GUI=self.flag_GUI
            )
            video_FPNc = noise_c.mFPNc(select_correction_axis)

        elif mode_FPN == "cpFPN":
            noise_c = FPNc.ColumnProjectionFPNc(
                video=video, select_correction_axis=select_correction_axis, flag_GUI=self.flag_GUI
            )
            video_FPNc = noise_c.cpFPNc(select_correction_axis)

        elif mode_FPN == "wFPN":
            if select_correction_axis == 0:
                wFPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wFPN.update_wFPN(direction="Horizontal")

            elif select_correction_axis == 1:
                wFPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wFPN.update_wFPN(direction="Vertical")

            elif select_correction_axis == "Both":
                wFPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                output = wFPN.update_wFPN(direction="Horizontal")

                wFPN = FPNc.FrequencyFPNc(
                    output, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wFPN.update_wFPN(direction="Vertical")

        elif mode_FPN == "fFPN":
            if select_correction_axis == 0:
                wf_FPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wf_FPN.update_fFPN(
                    direction="Horizontal", max_iterations=max_iterations, width=FFT_widith
                )

            elif select_correction_axis == 1:
                wf_FPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wf_FPN.update_fFPN(
                    direction="Vertical", max_iterations=max_iterations, width=FFT_widith
                )

            elif select_correction_axis == "Both":
                wf_FPN = FPNc.FrequencyFPNc(
                    video, inter_flag_parallel_active=inter_flag_parallel_active
                )
                output = wf_FPN.update_fFPN(
                    direction="Horizontal", max_iterations=max_iterations, width=FFT_widith
                )

                wf_FPN = FPNc.FrequencyFPNc(
                    output, inter_flag_parallel_active=inter_flag_parallel_active
                )
                video_FPNc = wf_FPN.update_fFPN(
                    direction="Vertical", max_iterations=max_iterations, width=FFT_widith
                )

        return video_FPNc

    def _FPNc_GUI_axis(self, video, select_correction_axis, mode_FPN):
        if select_correction_axis == "Both":
            self.flag_thread_FPNc = True
            vid_FPNc_axis_1 = self._FPN_GUI(video, 1, mode_FPN)

            while self.flag_thread_FPNc:
                QtCore.QCoreApplication.processEvents()

            time.sleep(2)

            self.flag_thread_FPNc = True
            vid_FPNc = self._FPN_GUI(vid_FPNc_axis_1, 0, mode_FPN)

            while self.flag_thread_FPNc:
                QtCore.QCoreApplication.processEvents()

            time.sleep(2)

        else:
            self.flag_thread_FPNc = True
            vid_FPNc = self._FPN_GUI(video, select_correction_axis, mode_FPN)

        return vid_FPNc

    def _FPN_GUI(self, video, select_correction_axis, mode_FPN):
        if mode_FPN == "mFPN":
            d_arg = {
                "video": video,
                "flag_GUI": True,
                "instance": FPNc.MedianProjectionFPNc,
                "select_correction_axis": select_correction_axis,
            }
        elif mode_FPN == "cpFPN":
            d_arg = {
                "video": video,
                "flag_GUI": True,
                "instance": FPNc.ColumnProjectionFPNc,
                "select_correction_axis": select_correction_axis,
            }

        self.p_max = video.shape[0] - (2 * self.batchSize) - 1
        self.object_update_progressBar.setRange(self.p_max)
        self.object_update_progressBar.setProgress(0)
        self.object_update_progressBar.setLabel("FPNc")

        self.flag_thread_FPNc = True
        self.startProgressBarFPNc(**d_arg)

        while self.flag_thread_FPNc:
            QtCore.QCoreApplication.processEvents()

        time.sleep(2)

        return self.FPNc_video
