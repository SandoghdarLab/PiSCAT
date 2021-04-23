from __future__ import print_function

from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Localization import data_handeling

from tqdm.autonotebook import tqdm
from PySide2.QtCore import *

import trackpy as tp
import pandas as pd

class WorkerSignals(QObject):
    updateProgress = Signal(int)
    result = Signal(object)
    finished = Signal()


class Linking():


    def __init__(self):
        """
        To obtain the temporal activity of each iPSF, we use the Trackpy packages' algorithm.

       References
       ----------
       [1] http://soft-matter.github.io/trackpy/v0.4.2/
       """

        self.cpu = CPUConfigurations()

    def create_link(self, psf_position, search_range=50, memory=1):
        """
        Each iPSF temporal activity is obtained.

        Parameters
        ----------
        psf_position: pandas data frame
            The data frame contains PSFs locations( x, y, frame, sigma, ...)

        search_range: float or tuple
            The maximum distance features can move between frames, optionally per dimension.

        memory: int
            The maximum number of frames during which a feature can vanish, then reappear nearby, and be considered the same particle. 0 by default.

        Returns
        -------
        df_PSF: pandas dataframe
            To the input data frame, append the 'particle' ID column. ( x, y, frame, sigma, particle, ...).

        """
        df_PSF = tp.link_df(psf_position, search_range=search_range, memory=memory)

        return df_PSF

    def sorting_linking(self, df_PSFs):
        """
        This function uses trajectory lengths to sort particles in a dataframe.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations(x, y, frame, sigma, particle, ...)

        Returns
        -------
        total_sort_df_PSFs: pandas dataframe
            The sort version of data frame contains PSFs locations(x, y, frame, sigma, particle, ...)

        """
        his_all_particles = df_PSFs['particle'].value_counts()
        index_particles = his_all_particles.index
        total_sort_df_PSFs = None

        for index_ in tqdm(index_particles):
            sort_df_PSFs = df_PSFs.loc[df_PSFs['particle'] == index_]
            if total_sort_df_PSFs is None:
                total_sort_df_PSFs = sort_df_PSFs
            else:
                total_sort_df_PSFs = total_sort_df_PSFs.append(sort_df_PSFs)

        return total_sort_df_PSFs

    def trajectory_counter(self, df_PSFs):
        """
        This function counts the number of unique particles in the data frame.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations(x, y, frame, sigma, particle,...)

        Returns
        -------
        unique_list: int
            Returns the number of particles in data frame
        """
        if df_PSFs.shape[0] != 0:
            particles_label = df_PSFs['particle'].tolist()
            unique_list = set(particles_label)
            return len(unique_list)
        else:
            return 0

