import math

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from piscat.BackgroundCorrection import DRA
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Visualization.print_colors import PrintColors
from piscat.Preproccessing.filtering import Filters

class NoiseFloor(CPUConfigurations, PrintColors):

    def __init__(self, video, list_range, n_jobs=None, inter_flag_parallel_active=True):
        """
        This class calculates the noise floor for the different batch sizes on DRA.

        Parameters
        ----------
        video: NDArray
            Video should be Numpy array with the following shape (number of frame, width, height).

        list_range: list
            list os all batch size that DRA should be calculated for them.
        """

        CPUConfigurations.__init__(self)
        PrintColors.__init__(self)

        self.video = video
        self.inter_flag_parallel_active = inter_flag_parallel_active
        self.thr_shot_noise = None
        self.list_range = list_range
        if self.parallel_active and self.inter_flag_parallel_active:
            if n_jobs is not None:
                self.n_jobs = n_jobs
                print("\nThe number of usage CPU cores are {}!".format(n_jobs))

            self.mean = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=0)(delayed(self.best_radius_kernel)(i_, flag_parallel=True) for i_ in range(len(self.list_range)))
        else:
            print(f"{self.WARNING}\nThe noise floor is running without parallel loop!{self.ENDC}")

            self.mean = []
            for i_ in range(len(self.list_range)):
                self.mean.append(self.best_radius_kernel(i_, flag_parallel=True))

    def best_radius_kernel(self, i_, flag_parallel):
        DRA_ = DRA.DifferentialRollingAverage(video=self.video, batchSize=self.list_range[i_])
        video_DRA = DRA_.differential_rolling(FFT_flag=False)

        return np.mean(np.std(video_DRA, axis=0))

    def plot_result(self):
        """
        By calling this function the result of the noise floor is plotted.
        """
        fig, ax = plt.subplots()
        ax.plot(self.list_range, self.mean, '--r', label='Experimental result')
        ax.plot(self.list_range, self.mean, 'ro')

        if self.thr_shot_noise is not None:
            ax.plot(self.list_range, self.thr_shot_noise, '--g', label='Theoretical result')
            ax.plot(self.list_range, self.thr_shot_noise, 'go')

        ax.set_xlabel("Batch size")
        ax.set_ylabel("Noise floor")
        ax.legend()
        plt.show()