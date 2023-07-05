import math

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from piscat.InputOutput.cpu_configurations import CPUConfigurations


class SpatialFilter:
    def __init__(self):
        """We have a `SpatialFilter` class in PiSCAT that allows users to
        filter `outlier_frames` that have a strong vibration or a particle
        flying by, `dense_PSFs`, and non-symmetric PSFs that may not properly
        resemble the iPSF expected from the experimental setup.  The
        threshold_min parameter in each of these filters determines the
        filter's sensitivity.

        """
        self.cpu = CPUConfigurations()

    def list_frames(self, df_PSFs):
        list_frames = df_PSFs["frame"].tolist()
        list_frames = np.sort(np.unique(list_frames))
        list_frames = list_frames.tolist()
        return list_frames

    def outlier_frames(self, df_PSFs, threshold=20):
        """This function eliminates all detected PSFs in the frame that are
        greater than the threshold_min value.  PSFs that were detected in
        unstable frames are reduced using this method.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations( x, y, frame, sigma)

        threshold: int
            Maximum number of PSFs in one frame.

        Returns
        -------
        filter_df_PSFs: pandas dataframe
            The filter data frame contains PSFs locations( x, y, frame, sigma)
        """
        if df_PSFs.shape[0] == 0 or df_PSFs is None:
            raise ValueError("---data frames is empty!---")

        print("\nstart removing crappy frames --->", end=" ")
        his_all_particles = df_PSFs["frame"].value_counts()
        temp = his_all_particles.where(his_all_particles <= threshold)
        select_particles = temp[~temp.isnull()]
        index_particles = select_particles.index
        filter_df_PSFs = df_PSFs.loc[df_PSFs["frame"].isin(index_particles)]
        print("Done!")
        return filter_df_PSFs

    def dense_PSFs(self, df_PSFs, threshold=0):
        """
        Remove PSFs from the dataframe that have an overlap greater than the specified portion.

        Parameters
        ----------
        df_PSFs : pandas DataFrame
            The dataframe containing PSF locations (x, y, frame, sigma).

        threshold : float
            The threshold specifies the maximum allowable portion of overlap between two PSFs.
            It should be a value between 0 and 1.

        Returns
        -------
        filter_df_PSFs : pandas DataFrame
            The filtered dataframe containing PSF locations (x, y, frame, sigma).

        Equation:
        ---------
        - The radius of PSF1 and PSF2 are calculated as sqrt(2) * sigma1 and sqrt(2) * sigma2, respectively.
        - The distance between two PSFs is calculated as d = sqrt((x1 - x2)^2 + (y1 - y2)^2).
        - The minimum acceptable distance (without overlap) is calculated as l = sqrt(2) * (sigma1 + sigma2).
        - PSFs are removed if d <= l * (1 - threshold).
        """
        if df_PSFs.shape[0] == 0 or df_PSFs is None:
            print("---data frames is empty!---")
            return df_PSFs

        if type(df_PSFs) is pd.core.frame.DataFrame:
            df_PSFs = df_PSFs
        else:
            raise ValueError(
                "Input does not have correct type! This function needs panda data frames."
            )

        list_frames = self.list_frames(df_PSFs)

        self.particles_after_closeFilter = df_PSFs
        self.index_particles_filter = self.particles_after_closeFilter.index

        num_particles = self.particles_after_closeFilter.shape[0]

        print("\n---Cleaning the df_PSFs that have drift without parallel loop---")
        self.point_1 = np.zeros((1, 2), dtype=np.float64)
        self.point_2 = np.zeros((1, 2), dtype=np.float64)
        self.remove_list_close = []

        for frame_num in tqdm(list_frames):
            particle = self.particles_after_closeFilter.loc[
                self.particles_after_closeFilter["frame"] == frame_num
            ]
            index_list = [index for index in particle.index]
            particle_X = particle["x"].tolist()
            particle_Y = particle["y"].tolist()
            particle_sigma = particle["sigma"].tolist()

            if len(index_list) != 1:
                for i_ in range(len(particle_X)):
                    self.point_1[0, 0] = particle_X[i_]
                    self.point_1[0, 1] = particle_Y[i_]
                    sigma_1 = particle_sigma[i_]

                    count_ = i_ + 1
                    while count_ <= (len(particle_X) - 1):
                        self.point_2[0, 0] = particle_X[count_]
                        self.point_2[0, 1] = particle_Y[count_]
                        sigma_2 = particle_sigma[count_]

                        distance = math.sqrt(
                            ((self.point_1[0, 0] - self.point_2[0, 0]) ** 2)
                            + ((self.point_1[0, 1] - self.point_2[0, 1]) ** 2)
                        )
                        min_d = math.sqrt(2) * (sigma_1 + sigma_2)
                        if distance <= (min_d*(1 - threshold)):
                            self.remove_list_close.append(index_list[i_])
                            self.remove_list_close.append(index_list[count_])

                        count_ = count_ + 1

        remove_list = list(set(self.remove_list_close))
        self.particles_after_closeFilter = self.particles_after_closeFilter.drop(
            remove_list, axis=0, errors="ignore"
        )
        print("\nNumber of PSFs before filters = {}".format(num_particles))
        print(
            "\nNumber of PSFs after filters = {}".format(
                self.particles_after_closeFilter.shape[0]
            )
        )

        return self.particles_after_closeFilter.reset_index(drop=True)

    def symmetric_PSFs(self, df_PSFs, threshold=0.7):
        """
        Remove astigmatism-affected PSFs with this filter.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations( x, y, frame, sigma)

        threshold: float
            The smallest sigma ratio that is acceptable (sigma max/sigma min).

        Returns
        -------
        df_PSF_thr: pandas dataframe
            The filter data frame contains PSFs locations( x, y, frame, sigma)

        """

        if df_PSFs is not None and df_PSFs.shape[0] > 0:
            if "Sigma_ratio" in df_PSFs.keys():
                idx_thr_sigma = df_PSFs.Sigma_ratio > threshold
                idx_nan_sigma = df_PSFs.Sigma_ratio.isnull()

                df_PSF_thr = df_PSFs[idx_thr_sigma.tolist() or idx_nan_sigma.tolist()]

                return df_PSF_thr

            else:
                raise RuntimeError("---Sigma ratio does not exist in data frames---")

        else:
            raise RuntimeError("---data frames is empty!---")

    def remove_side_lobes_artifact(self, df_PSFs, threshold=0):
        """This filter removes false detections on side lobes of PSFs caused
        by the localization algorithm by comparing center intensity contrast.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations( x, y, frame, sigma, center_intensity)

        threshold: float
            It specifies the portion of the overlay that two PSFs must have to
            remove from the list.

        Returns
        -------
        filter_df_PSFs: pandas dataframe
            The filter data frame contains PSFs locations( x, y, frame, sigma, center_intensity)

        """
        if df_PSFs.shape[0] == 0 or df_PSFs is None:
            print("---data frames is empty!---")
            return df_PSFs

        if type(df_PSFs) is pd.core.frame.DataFrame:
            df_PSFs = df_PSFs
        else:
            raise ValueError(
                "Input does not have correct type! This function needs panda data frames."
            )

        list_frames = self.list_frames(df_PSFs)

        self.particles_after_closeFilter = df_PSFs
        self.index_particles_filter = self.particles_after_closeFilter.index

        num_particles = self.particles_after_closeFilter.shape[0]

        print("\n---Cleaning the df_PSFs that have side lobs without parallel loop---")
        self.point_1 = np.zeros((1, 2), dtype=np.float64)
        self.point_2 = np.zeros((1, 2), dtype=np.float64)
        self.remove_list_close = []

        for frame_num in tqdm(list_frames):
            particle = self.particles_after_closeFilter.loc[
                self.particles_after_closeFilter["frame"] == frame_num
            ]
            index_list = [index for index in particle.index]
            particle_X = particle["x"].tolist()
            particle_Y = particle["y"].tolist()
            particle_sigma = particle["sigma"].tolist()
            particle_center_intensity = particle["center_intensity"].tolist()

            if len(index_list) != 1:
                for i_ in range(len(particle_X)):
                    self.point_1[0, 0] = particle_X[i_]
                    self.point_1[0, 1] = particle_Y[i_]
                    sigma_1 = particle_sigma[i_]

                    count_ = i_ + 1
                    while count_ <= (len(particle_X) - 1):
                        self.point_2[0, 0] = particle_X[count_]
                        self.point_2[0, 1] = particle_Y[count_]
                        sigma_2 = particle_sigma[count_]

                        distance = math.sqrt(
                            ((self.point_1[0, 0] - self.point_2[0, 0]) ** 2)
                            + ((self.point_1[0, 1] - self.point_2[0, 1]) ** 2)
                        )
                        tmp = math.sqrt(2) * (sigma_1 + sigma_2)
                        if distance <= ((math.sqrt(2) * (sigma_1 + sigma_2)) - (threshold * tmp)):
                            intesity_1 = particle_center_intensity[i_]
                            intesity_2 = particle_center_intensity[count_]

                            if np.abs(intesity_1) == np.abs(intesity_2):
                                self.remove_list_close.append(index_list[i_])
                                self.remove_list_close.append(index_list[count_])
                            if np.abs(intesity_1) > np.abs(intesity_2):
                                self.remove_list_close.append(index_list[count_])
                            if intesity_1 < intesity_2:
                                self.remove_list_close.append(index_list[i_])
                        count_ = count_ + 1

        remove_list = list(set(self.remove_list_close))
        self.particles_after_closeFilter = self.particles_after_closeFilter.drop(
            remove_list, axis=0, errors="ignore"
        )
        print("\nNumber of PSFs before filters = {}".format(num_particles))
        print(
            "\nNumber of PSFs after filters = {}".format(
                self.particles_after_closeFilter.shape[0]
            )
        )

        return self.particles_after_closeFilter.reset_index(drop=True)
