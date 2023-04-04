import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from piscat.BackgroundCorrection import DRA
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Visualization.print_colors import PrintColors


class NoiseFloor(CPUConfigurations, PrintColors):
    def __init__(
        self,
        video,
        list_range,
        FPN_flag=False,
        mode_FPN="mFPN",
        select_correction_axis=1,
        n_jobs=None,
        inter_flag_parallel_active=False,
        max_iterations=10,
        FFT_widith=1,
        mode="mode_temporal",
    ):
        """
        This class measures the noise floor for various batch sizes.

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        list_range: list
            list os all batch size that DRA should be calculated for them.

        FPN_flag: bool
            This flag activates the fixed pattern noise correction function in case define as true.

        mode_FPN: {‘cpFPN’, ‘mFPN’, ‘wFPN’, 'fFPN'}, optional
            Flag that defines method of FPNc.

            * `mFPN`: Median fixed pattern noise correction
            * `cpFPN`: Median fixed pattern noise correction
            * `wFPN`: Wavelet FPNc
            * `fFPN`: FFT2D_Wavelet FPNc

        select_correction_axis: int (0/1), 'Both'
            This parameter is used only when FPN_flag is True, otherwise it will be ignored.

            * `0`: FPN will be applied row-wise.
            * `1`: FPN will be applied column-wise.
            * `'Both'`: FPN will be applied on two axis.

        max_iterations: int
            This parameter is used when fFPT is selected that defines the total number of filtering iterations.

        FFT_widith: int
            This parameter is used when fFPT is selected that defines the frequency mask's width.
        """

        CPUConfigurations.__init__(self)
        PrintColors.__init__(self)

        self.video = video
        self.max_iterations = max_iterations
        self.FFT_widith = FFT_widith
        self.FPN_flag = FPN_flag
        self.select_correction_axis = select_correction_axis
        self.inter_flag_parallel_active = inter_flag_parallel_active
        self.mode_FPN = mode_FPN
        self.thr_shot_noise = None
        self.list_range = list_range
        if self.parallel_active and self.inter_flag_parallel_active:
            if n_jobs is not None:
                self.n_jobs = n_jobs
                print("\nThe number of usage CPU cores are {}!".format(n_jobs))

            self.mean = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=0)(
                delayed(self.best_radius_kernel)(i_, flag_parallel=True, mode=mode)
                for i_ in range(len(self.list_range))
            )
        else:
            print(f"{self.WARNING}\nThe noise floor is running without parallel loop!{self.ENDC}")

            self.mean = []
            for i_ in range(len(self.list_range)):
                self.mean.append(self.best_radius_kernel(i_, flag_parallel=True, mode=mode))

    def best_radius_kernel(self, i_, flag_parallel, mode):
        DRA_ = DRA.DifferentialRollingAverage(
            video=self.video, batchSize=self.list_range[i_], mode_FPN=self.mode_FPN
        )
        video_DRA, _ = DRA_.differential_rolling(
            FPN_flag=self.FPN_flag,
            select_correction_axis=self.select_correction_axis,
            FFT_flag=False,
            inter_flag_parallel_active=flag_parallel,
            max_iterations=self.max_iterations,
            FFT_widith=self.FFT_widith,
        )

        if mode == "mode_temporal":
            noise_floor = np.mean(np.std(video_DRA, axis=0))
            # noise_floor = np.std(video_DRA, axis=0)
        elif mode == "mode_spatial":
            list_frame_std = []
            for f_ in range(video_DRA.shape[0]):
                frame_ = video_DRA[f_, ...]
                std_frame = np.std(np.ravel(frame_))
                list_frame_std.append(std_frame)

            noise_floor = np.mean(list_frame_std)

        return noise_floor

    def plot_result(self, flag_log=True):
        """
        The result of the noise floor is plotted when this function is called.

        Parameters
        ----------
        flag_log: bool
            The log-log plot style is enabled by this parameter.
        """
        list_batch = [1] + self.list_range[1:]
        shot_noise_ = np.divide(self.mean[0], np.sqrt(list_batch))
        if flag_log is False:
            fig, ax = plt.subplots()
            ax.plot(self.list_range, self.mean, "ro", label="Experimental result")
            ax.plot(self.list_range, shot_noise_, "b-", label="shot noise")

            ax.set_xlabel("Batch size")
            ax.set_ylabel("Noise floor")
            ax.legend()
            plt.show()

        if flag_log is True:
            fig, ax = plt.subplots()
            ax.loglog(self.list_range, self.mean, "ro", label="Experimental result")
            ax.loglog(self.list_range, shot_noise_, "b-", label="shot noise")

            ax.set_xlabel("Batch size")
            ax.set_ylabel("Noise floor")
            ax.legend()
            plt.show()
