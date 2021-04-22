from __future__ import print_function

import numpy as np
import math
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from piscat.Trajectory.data_handeling import protein_trajectories_list2dic
from piscat.InputOutput import read_write_data
from piscat.Visualization.print_colors import PrintColors


class PlotProteinHistogram(PrintColors):

    def __init__(self, intersection_display_flag=False, imgSizex=5, imgSizey=5):

        """
        This class use video analysis data ('HDF5', 'Matlab') to plot histograms.

        Parameters
        ----------
        intersection_display_flag: bool
            This flag can be used when we want to see the result of intersection on
            top of v-shaped (Please read tutorials 4).

        imgSizex: int
            The width of the histogram figure.

        imgSizey: int
            The height of the histogram figure.

        """
        PrintColors.__init__(self)

        self.imgSizex = imgSizex
        self.imgSizey = imgSizey

        self.intersection_display_flag = intersection_display_flag
        self.folder_name = []

        self.t_particle_center_intensity = []
        self.t_particle_center_intensity_follow = []

        self.tSmooth_particle_center_intensity = []
        self.tSmooth_particle_center_intensity_follow = []

        self.t_contrast_peaks = []
        self.t_contrast_peaks_index = []

        self.t_contrast_intersection = []
        self.t_contrast_intersection_index = []

        self.t_contrast_Prominence = []
        self.t_contrast_Prominence_index = []

        self.t_iPSFCentroidSigmas_ = []

        self.t_particle_center_fit_intensity = []

        self.tSmooth_particle_center_intensity = []

        self.t_contrast_fit_peaks = []
        self.t_contrast_fit_peaks_index = []

        self.t_contrast_fit_intersection = []
        self.t_contrast_fit_intersection_index = []

        self.t_iPSFCentroidSigmas_fit_x = []
        self.t_iPSFCentroidSigmas_fit_y = []

    def __call__(self, folder_name, particles, batch_size, video_frame_num, MinPeakWidth, MinPeakProminence,
                 pixel_size=0.66):
        """
        By calling the object of class this function tries to read the result of the corresponding video it is defined
        with ``folder_name``. These results concatenated with previous results to use for plotting histogram.

        Parameters
        ----------
        folder_name: str
            Name of a folder that data read from it. This saves next to data for convinces tracking the analysis.

        particles: list/dict

            | [intensity_horizontal, intensity_vertical, particle_center_intensity,
                            particle_center_intensity_follow, particle_frame, particle_sigma, particle_X, particle_Y, particle_ID,
                            optional(fit_intensity, fit_x, fit_y, fit_X_sigma, fit_Y_sigma, fit_Bias, fit_intensity_error,
                            fit_x_error, fit_y_error, fit_X_sigma_error, fit_Y_sigma_error, fit_Bias_error)]

            | {"#0": {'intensity_horizontal': ..., 'intensity_vertical': ..., ..., 'particle_ID': ...},
                        "#1": {}, ...}

        batch_size: int
            The size of batch that was used on the DRA step.

        video_frame_num: int
            The number of frames for the corresponding video.

        MinPeakWidth: int
            This is defined as the minimum V-shaped mouth that will use for prominence.

        MinPeakProminence: int
            This is defined as the minimum V-shape height that will use for prominence.

        pixel_size: float
            The size of the camera pixel.

        """

        if type(particles) is list:
            particles = np.asarray(particles)

        if type(particles) is dict:
            for key, item in particles.items():
                intensity_horizontal = item['intensity_horizontal']
                intensity_vertical = item['intensity_vertical']
                center_int = item['center_int']
                center_int_flow = item['center_int_flow']
                frame_number = item['frame_number']
                sigma = item['sigma']
                x_center = item['x_center']
                y_center = item['y_center']
                particle_ID = item['particle_ID']

                num_parameters = len(item)

                if num_parameters == 21:
                    fit_intensity = item['fit_intensity']
                    fit_x = item['fit_x'] * pixel_size
                    fit_y = item['fit_y'] * pixel_size
                    fit_X_sigma = item['fit_X_sigma'] * pixel_size
                    fit_Y_sigma = item['fit_Y_sigma'] * pixel_size
                    fit_Bias = item['fit_Bias']
                    fit_intensity_error = item['fit_intensity_error']
                    fit_x_error = item['fit_x_error'] * pixel_size
                    fit_y_error = item['fit_y_error'] * pixel_size
                    fit_X_sigma_error = item['fit_X_sigma_error'] * pixel_size
                    fit_Y_sigma_error = item['fit_Y_sigma_error'] * pixel_size
                    fit_Bias_error = item['fit_Bias_error']

                else:
                    fit_intensity = None
                    fit_X_sigma = None
                    fit_Y_sigma = None

                self.data_handling(center_int, center_int_flow, folder_name, sigma, num_parameters, fit_intensity,
                                   fit_X_sigma, fit_Y_sigma, frame_number, batch_size, video_frame_num,
                                   MinPeakWidth=MinPeakWidth, MinPeakProminence=MinPeakProminence)

        if type(particles) is np.ndarray:

            for n_particle in range(particles.shape[0]):
                if type(particles[n_particle][0]) is list:
                    intensity_horizontal = np.asarray(particles[n_particle][0])
                    intensity_vertical = np.asarray(particles[n_particle][1])
                    center_int = np.asarray(particles[n_particle][2])
                    center_int_flow = np.asarray(particles[n_particle][3])
                    frame_number = np.asarray(particles[n_particle][4])
                    sigma = np.asarray(particles[n_particle][5])
                    x_center = np.asarray(particles[n_particle][6])
                    y_center = np.asarray(particles[n_particle][7])
                    particle_ID = np.asarray(particles[n_particle][8])
                else:
                    intensity_horizontal = particles[n_particle][0].ravel()
                    intensity_vertical = particles[n_particle][1].ravel()
                    center_int = particles[n_particle][2].ravel()
                    center_int_flow = particles[n_particle][3].ravel()
                    frame_number = particles[n_particle][4].ravel()
                    sigma = particles[n_particle][5].ravel()
                    x_center = particles[n_particle][6].ravel()
                    y_center = particles[n_particle][7].ravel()
                    particle_ID = particles[n_particle][8].ravel()

                num_parameters = len(particles[n_particle])

                if num_parameters == 21:
                    if type(particles[n_particle][0]) is list:
                        fit_intensity = np.asarray(particles[n_particle][9])
                        fit_x = np.asarray(particles[n_particle][10]) * pixel_size
                        fit_y = np.asarray(particles[n_particle][11]) * pixel_size
                        fit_X_sigma = np.asarray(particles[n_particle][12]) * pixel_size
                        fit_Y_sigma = np.asarray(particles[n_particle][13]) * pixel_size
                        fit_Bias = np.asarray(particles[n_particle][14])
                        fit_intensity_error = np.asarray(particles[n_particle][15])
                        fit_x_error = np.asarray(particles[n_particle][16]) * pixel_size
                        fit_y_error = np.asarray(particles[n_particle][17]) * pixel_size
                        fit_X_sigma_error = np.asarray(particles[n_particle][18]) * pixel_size
                        fit_Y_sigma_error = np.asarray(particles[n_particle][19]) * pixel_size
                        fit_Bias_error = np.asarray(particles[n_particle][20])

                    else:
                        fit_intensity = particles[n_particle][9].ravel()
                        fit_x = particles[n_particle][10].ravel() * pixel_size
                        fit_y = particles[n_particle][11].ravel() * pixel_size
                        fit_X_sigma = particles[n_particle][12].ravel() * pixel_size
                        fit_Y_sigma = particles[n_particle][13].ravel() * pixel_size
                        fit_Bias = particles[n_particle][14].ravel()
                        fit_intensity_error = particles[n_particle][15].ravel()
                        fit_x_error = particles[n_particle][16].ravel() * pixel_size
                        fit_y_error = particles[n_particle][17].ravel() * pixel_size
                        fit_X_sigma_error = particles[n_particle][18].ravel() * pixel_size
                        fit_Y_sigma_error = particles[n_particle][19].ravel() * pixel_size
                        fit_Bias_error = particles[n_particle][20].ravel()

                else:
                    fit_intensity = None
                    fit_X_sigma = None
                    fit_Y_sigma = None

                self.data_handling(center_int, center_int_flow, folder_name, sigma, num_parameters, fit_intensity,
                                   fit_X_sigma, fit_Y_sigma, frame_number, batch_size, video_frame_num,
                                   MinPeakWidth=MinPeakWidth, MinPeakProminence=MinPeakProminence)

    def data_handling(self, center_int, center_int_flow, folder_name, sigma, num_parameters, fit_intensity,
                      fit_X_sigma, fit_Y_sigma, frame_number, batch_size, video_frame_num, MinPeakWidth, MinPeakProminence):
        if len(center_int) != 0:
            win_size = self.determine_windows_size(center_int)
            if win_size > 3:
                V_smooth = savgol_filter(center_int, win_size, 3)

                win_size = self.determine_windows_size(center_int_flow)
                V_smooth_follow = savgol_filter(center_int_flow, win_size, 3)

                peak, index_peak, idx_ = self.find_peack_contrast(frame_number, V_smooth)
                xi, yi = self.intersection_point(data_index=frame_number, data=V_smooth, index=index_peak,
                                                 display=self.intersection_display_flag)

                tprofile_frameNo_DRA_longer, start_FN_earlyV, end_FN_earlyV = self.frameNumber_longerProfile_2_Vshape(batch_size,
                                                                                                                      video_frame_num,
                                                                                                                      frame_number,
                                                                                                                      V_smooth_follow)

                prom, prom_idx, p_idx_, pro_ = self.find_Prominence_contrast(tprofile_DRA_smooth=V_smooth, tprofile_DRA_tail_smooth=V_smooth_follow,
                                                               tprofile_frameNo_DRA_longer=tprofile_frameNo_DRA_longer,
                                                               start_FN_earlyV=start_FN_earlyV, end_FN_earlyV=end_FN_earlyV,
                                                               MinPeakWidth=MinPeakWidth, MinPeakProminence=MinPeakProminence)

                properties = [V_smooth, V_smooth_follow, [peak, index_peak], [yi, xi], [prom, prom_idx, p_idx_, pro_],
                              tprofile_frameNo_DRA_longer, start_FN_earlyV, end_FN_earlyV]

                self.folder_name.append(folder_name)

                self.t_particle_center_intensity.append(center_int)
                self.t_particle_center_intensity_follow.append(center_int_flow)

                self.tSmooth_particle_center_intensity.append(V_smooth)
                self.tSmooth_particle_center_intensity_follow.append(V_smooth_follow)

                self.t_contrast_peaks.append(peak)
                self.t_contrast_peaks_index.append(index_peak)

                self.t_contrast_intersection.append(yi)
                self.t_contrast_intersection_index.append(xi)

                self.t_contrast_Prominence.append(prom)
                self.t_contrast_Prominence_index.append(prom_idx)

                self.t_iPSFCentroidSigmas_.append(sigma[idx_])
            else:
                properties = None
        else:
            properties = None

        if num_parameters == 21:
            nan_array = np.isnan(fit_intensity)
            not_nan_array = ~ nan_array
            fit_intensity_ = fit_intensity[not_nan_array]

            win_size = self.determine_windows_size(fit_intensity_)
            if len(fit_intensity_) != 0:
                if win_size > 3:
                    V_smooth = savgol_filter(fit_intensity_, win_size, 3)

                    peak, index_peak, idx_ = self.find_peack_contrast(frame_number, V_smooth)
                    xi, yi = self.intersection_point(data_index=frame_number, data=V_smooth, index=index_peak,
                                                     display=False)

                    self.t_particle_center_fit_intensity.append(fit_intensity_)
                    self.tSmooth_particle_center_intensity.append(V_smooth)

                    self.t_contrast_fit_peaks.append(peak)
                    self.t_contrast_fit_peaks_index.append(index_peak)

                    self.t_contrast_fit_intersection.append(yi)
                    self.t_contrast_fit_intersection_index.append(xi)

                    fit_X_sigma_ = fit_X_sigma[not_nan_array]
                    fit_Y_sigma_ = fit_Y_sigma[not_nan_array]

                    self.t_iPSFCentroidSigmas_fit_x.append(fit_X_sigma_[idx_])
                    self.t_iPSFCentroidSigmas_fit_y.append(fit_Y_sigma_[idx_])
                    fit_properties = [V_smooth, V_smooth_follow, [peak, index_peak], [yi, xi], [prom, prom_idx]]

                    return properties, fit_properties

                else:
                    return properties, None

            else:
                return properties, None

        else:
            return properties, None

    def determine_windows_size(self, signal):

        window_size = round(0.05 * len(signal))
        if (window_size % 2) == 0:
            window_size = window_size + 1
        return window_size

    def find_peack_contrast(self, tprofile_DRA_index, tprofile_DRA):
        if np.mean(tprofile_DRA) >= 0:
            max_center_intensity = np.max(tprofile_DRA)
            index_max_ = np.argmax(tprofile_DRA)
            index_max_center_intensity = tprofile_DRA_index[index_max_]
            return max_center_intensity, index_max_center_intensity, index_max_

        elif np.mean(tprofile_DRA) < 0:
            min_center_intensity = np.min(tprofile_DRA)
            index_min_ = np.argmin(tprofile_DRA)
            index_min_center_intensity = tprofile_DRA_index[index_min_]
            return min_center_intensity, index_min_center_intensity, index_min_

    def find_Prominence_contrast(self, tprofile_DRA_smooth, tprofile_DRA_tail_smooth, tprofile_frameNo_DRA_longer,
                                 start_FN_earlyV, end_FN_earlyV, MinPeakWidth, MinPeakProminence):

        if np.mean(tprofile_DRA_smooth) >= 0:
            peaks_index, properties = find_peaks(x=tprofile_DRA_tail_smooth, width=[MinPeakWidth, np.inf],
                                                 prominence=[MinPeakProminence, np.inf])
            peaks_index = peaks_index.tolist()
            frame_index_ = [tprofile_frameNo_DRA_longer[peaks_index[i_]] for i_ in range(len(peaks_index))]

            idx_ = []
            frame_idx_ = []
            pro_properties = []
            for i_ in range(len(frame_index_)):
                f_idx = frame_index_[i_]
                if f_idx <= end_FN_earlyV and f_idx >= start_FN_earlyV:
                    idx_.append(peaks_index[i_])
                    frame_idx_.append(f_idx)
                    pro_properties.append(properties["prominences"][i_])

            if len(idx_) != 0:
                prom_ = [tprofile_DRA_tail_smooth[i_] for i_ in idx_]
                prom = np.min(prom_)
                prom_idx = np.argmin(prom_)
                prom_index = frame_idx_[prom_idx]
                pro = pro_properties[prom_idx]

            else:
                prom = np.nan
                prom_index = np.nan
                prom_idx = np.nan
                pro = np.nan

        elif np.mean(tprofile_DRA_smooth) < 0:
            peaks_index, properties = find_peaks(x=-1*tprofile_DRA_tail_smooth, width=[MinPeakWidth, np.inf],
                                                 prominence=[MinPeakProminence, np.inf])
            peaks_index = peaks_index.tolist()
            frame_index_ = [tprofile_frameNo_DRA_longer[peaks_index[i_]] for i_ in range(len(peaks_index))]

            idx_ = []
            frame_idx_ = []
            pro_properties = []
            for i_ in range(len(frame_index_)):
                f_idx = frame_index_[i_]
                if f_idx <= end_FN_earlyV and f_idx >= start_FN_earlyV:
                    idx_.append(peaks_index[i_])
                    frame_idx_.append(f_idx)
                    pro_properties.append(properties["prominences"][i_])

            if len(idx_) != 0:
                prom_ = [tprofile_DRA_tail_smooth[i_] for i_ in idx_]
                prom = -1 * np.min(np.abs(prom_))
                prom_idx = np.argmin(prom_)
                prom_index = frame_idx_[prom_idx]
                pro = pro_properties[prom_idx]
            else:
                prom = np.nan
                prom_index = np.nan
                prom_idx = np.nan
                pro = np.nan

        return prom, prom_index, prom_idx, pro

    def intersection_point(self, data_index, data, index, display=False):
        try:
            data = data.ravel()
            idx_ = np.where(data_index == index)
            right_side = data[0:idx_[0][0]]
            left_side = data[idx_[0][0]:]

            data_index = np.asarray(data_index)
            x_right = data_index[data_index < index]
            x_left = data_index[data_index >= index]

            z_right = np.polyfit(x_right, right_side, 1, full=True)
            p_l_right = np.poly1d(z_right[0])
            y_right = p_l_right[0] + p_l_right[1] * x_right

            z_left = np.polyfit(x_left, left_side, 1, full=True)
            p_l_left = np.poly1d(z_left[0])
            y_left = p_l_left[0] + p_l_left[1] * x_left

            if p_l_right[1] == p_l_left[1]:
                raise Exception('---lines do not intersect!---')
            else:
                xi = (p_l_left[0] - p_l_right[0]) / (p_l_right[1] - p_l_left[1])
                yi = p_l_left[1] * xi + p_l_left[0]

            if display:
                self.fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
                self.axs = self.fig.add_axes([0, 0, 1, 1])
                self.axs.plot(x_right, y_right, 'r', label='Fitted line at right', zorder=1)
                self.axs.plot(x_left, y_left, 'r', label='Fitted line at left', zorder=2)
                self.axs.scatter(xi, yi, s=80, facecolors='none', edgecolors='k', zorder=3, label='Intersection of the lines')
                self.axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            return int(xi), yi

        except:
            xi = np.nan
            yi = np.nan

            return xi, yi

    def mean_std(self, input):
        input = np.abs(input)
        if input.shape[0] != 0:
            std_ = np.nanstd(input)
            mean_ = np.nanmean(input)
        else:
            std_ = 0
            mean_ = 0
        return mean_, std_

    def extract_hist_information(self, con_intersections, con_peaks, con_proms, upper_limitation=1, lower_limitation=-1, max_n_components=3, Flag_GMM_fit=True):
        t_contrast_intersection = np.asarray(con_intersections)
        t_contrast_intersection = t_contrast_intersection[~np.isnan(t_contrast_intersection)]

        t_bright_contrast_intersection = t_contrast_intersection[t_contrast_intersection >= 0]
        t_black_contrast_intersection = t_contrast_intersection[t_contrast_intersection < 0]

        t_contrast_peaks = np.asarray(con_peaks)
        t_contrast_peaks = t_contrast_peaks[~np.isnan(t_contrast_peaks)]

        t_bright_contrast_peaks = t_contrast_peaks[t_contrast_peaks >= 0]
        t_black_contrast_peaks = t_contrast_peaks[t_contrast_peaks < 0]
        if con_proms is not None:
            t_contrast_proms = np.asarray(con_proms)
            t_contrast_proms = t_contrast_proms[~np.isnan(t_contrast_proms)]

            t_bright_contrast_proms = t_contrast_proms[t_contrast_proms >= 0]
            t_black_contrast_proms = t_contrast_proms[t_contrast_proms < 0]

        # lower limit
        t_bright_contrast_intersection = t_bright_contrast_intersection[t_bright_contrast_intersection >= lower_limitation]
        t_black_contrast_intersection = t_black_contrast_intersection[t_black_contrast_intersection >= lower_limitation]

        t_bright_contrast_peaks = t_bright_contrast_peaks[t_bright_contrast_peaks >= lower_limitation]
        t_black_contrast_peaks = t_black_contrast_peaks[t_black_contrast_peaks >= lower_limitation]

        if con_proms is not None:
            t_bright_contrast_proms = t_bright_contrast_proms[t_bright_contrast_proms >= lower_limitation]
            t_black_contrast_proms = t_black_contrast_proms[t_black_contrast_proms >= lower_limitation]
            t_contrast_proms = t_contrast_proms[t_contrast_proms >= lower_limitation]

        t_contrast_intersection = t_contrast_intersection[t_contrast_intersection >= lower_limitation]
        t_contrast_peaks = t_contrast_peaks[t_contrast_peaks >= lower_limitation]

        # upper limit
        t_bright_contrast_intersection = t_bright_contrast_intersection[t_bright_contrast_intersection < upper_limitation]
        t_black_contrast_intersection = t_black_contrast_intersection[t_black_contrast_intersection < upper_limitation]

        t_bright_contrast_peaks = t_bright_contrast_peaks[t_bright_contrast_peaks < upper_limitation]
        t_black_contrast_peaks = t_black_contrast_peaks[t_black_contrast_peaks < upper_limitation]

        if con_proms is not None:
            t_bright_contrast_proms = t_bright_contrast_proms[t_bright_contrast_proms < upper_limitation]
            t_black_contrast_proms = t_black_contrast_proms[t_black_contrast_proms < upper_limitation]
            t_contrast_proms = t_contrast_proms[t_contrast_proms < upper_limitation]

        t_contrast_intersection = t_contrast_intersection[t_contrast_intersection < upper_limitation]
        t_contrast_peaks = t_contrast_peaks[t_contrast_peaks < upper_limitation]

        # std & mean
        mean_bright_intersection, std_bright_intersection = self.mean_std(t_bright_contrast_intersection)
        mean_black_intersection, std_black_intersection = self.mean_std(t_black_contrast_intersection)

        mean_bright_peak, std_bright_peak = self.mean_std(t_bright_contrast_peaks)
        mean_black_peak, std_black_peak = self.mean_std(t_black_contrast_peaks)

        if con_proms is not None:
            mean_bright_prom, std_bright_prom = self.mean_std(t_bright_contrast_proms)
            mean_black_prom, std_black_prom = self.mean_std(t_black_contrast_proms)
            mean_prom, std_prom = self.mean_std(t_contrast_proms)

        mean_intersection, std_intersection = self.mean_std(t_contrast_intersection)
        mean_peak, std_peak = self.mean_std(t_contrast_peaks)

        sci_num = lambda x: "{:.2e}".format(x)

        if con_proms is not None:

            list_data = [t_bright_contrast_peaks, t_bright_contrast_intersection, t_bright_contrast_proms,
                         t_black_contrast_peaks, t_black_contrast_intersection, t_black_contrast_proms,
                         t_contrast_peaks, t_contrast_intersection, t_contrast_proms]

            title = ['Peak bright', 'Intersection bright', 'Prominence bright',
                     'Peak black', 'Intersection black', 'Prominence black',
                     'Total peak', 'Total Intersection', 'Total Prominence']

            dic = {'Peak bright': [sci_num(mean_bright_peak), sci_num(std_bright_peak), int(t_bright_contrast_peaks.shape[0])],
                   'Peak black': [sci_num(mean_black_peak), sci_num(std_black_peak), int(t_black_contrast_peaks.shape[0])],
                   'Total peak': [sci_num(mean_peak), sci_num(std_peak), int(t_contrast_peaks.shape[0])],
                   'Intersection bright': [sci_num(mean_bright_intersection), sci_num(std_bright_intersection), int(t_bright_contrast_intersection.shape[0])],
                   'Intersection black': [sci_num(mean_black_intersection), sci_num(std_black_intersection), int(t_black_contrast_intersection.shape[0])],
                   'Total Intersection': [sci_num(mean_intersection), sci_num(std_intersection), int(t_contrast_intersection.shape[0])],
                   'Prominence bright': [sci_num(mean_bright_prom), sci_num(std_bright_prom), int(t_bright_contrast_proms.shape[0])],
                   'Prominence black': [sci_num(mean_black_prom), sci_num(std_black_prom), int(t_black_contrast_proms.shape[0])],
                   'Total Prominence': [sci_num(mean_prom), sci_num(std_prom), int(t_contrast_proms.shape[0])]
                   }
        else:
            list_data = [t_bright_contrast_peaks, t_bright_contrast_intersection,
                         t_black_contrast_peaks, t_black_contrast_intersection,
                         t_contrast_peaks, t_contrast_intersection]

            title = ['Peak bright', 'Intersection bright',
                     'Peak black', 'Intersection black',
                     'Total peak', 'Total Intersection']

            dic = {'Peak bright': [sci_num(mean_bright_peak), sci_num(std_bright_peak),
                                   int(t_bright_contrast_peaks.shape[0])],
                   'Peak black': [sci_num(mean_black_peak), sci_num(std_black_peak),
                                  int(t_black_contrast_peaks.shape[0])],
                   'Total peak': [sci_num(mean_peak), sci_num(std_peak), int(t_contrast_peaks.shape[0])],
                   'Intersection bright': [sci_num(mean_bright_intersection), sci_num(std_bright_intersection),
                                           int(t_bright_contrast_intersection.shape[0])],
                   'Intersection black': [sci_num(mean_black_intersection), sci_num(std_black_intersection),
                                          int(t_black_contrast_intersection.shape[0])],
                   'Total Intersection': [sci_num(mean_intersection), sci_num(std_intersection),
                                          int(t_contrast_intersection.shape[0])]
                    }

        num_gmm_mean = None
        list_means = []
        list_stds = []
        list_weights = []
        list_keys = []
        list_num_GMM = []
        if Flag_GMM_fit:
            for data, key in zip(list_data, title):
                try:
                    means, stdevs, weights = self.GMM(np.abs(data), max_n_components)
                    list_means.append(means)
                    list_stds.append(stdevs)
                    list_weights.append(weights)
                    list_keys.append(key)
                    list_num_GMM.append(len(means))

                except:
                    print('---GMM did not extract for' + key, '---')
                    list_means.append(None)
                    list_stds.append(None)
                    list_weights.append(None)
                    list_keys.append(key)
                    list_num_GMM.append(None)

            max_num_gmm_mean = np.max(list_num_GMM)
            for means, stdevs, weights, key in zip(list_means, list_stds, list_weights, list_keys):
                if means is not None:
                    if len(means) == max_num_gmm_mean:
                        for m_, s_, w_ in zip(means, stdevs, weights):
                            dic[key].append(m_)
                            dic[key].append(s_)
                            dic[key].append(w_)
                    else:
                        diff_ = max_num_gmm_mean - len(means)
                        for m_, s_, w_ in zip(means, stdevs, weights):
                            dic[key].append(m_)
                            dic[key].append(s_)
                            dic[key].append(w_)

                        for i_ in range(diff_):
                            dic[key].append(None)
                            dic[key].append(None)
                            dic[key].append(None)
                else:
                    for i_ in range(max_num_gmm_mean):
                        dic[key].append(None)
                        dic[key].append(None)
                        dic[key].append(None)

            index_ = ['Mean', 'Std', '#Particles']
            for i_ in range(max_num_gmm_mean):
                index_.append('GMM_mean' + str(i_+1))
                index_.append('GMM_std' + str(i_+1))
                index_.append('GMM_weight' + str(i_+1))
            df = pd.DataFrame(dic, index=index_)

        else:
            df = pd.DataFrame(dic, index=['Mean', 'Std', '#Particles'])

        return df, list_data, title

    def GMM(self, data, max_n_components):
        data = data.reshape(-1, 1)
        nan_array = np.isnan(data)
        not_nan_array = ~ nan_array
        data = data[not_nan_array]
        data = data.reshape(-1, 1)

        N = np.arange(1, max_n_components+1)
        models = [None for i in range(len(N))]
        for i in range(len(N)):
            try:
                models[i] = GaussianMixture(N[i], covariance_type='full', reg_covar=1e-10).fit(data)
            except:
                models[i] = None

        # compute the AIC and the BIC
        AIC = []
        BIC = []
        for m in models:
            if m is not None:
                AIC.append(m.aic(data))
                BIC.append(m.bic(data))
            else:
                AIC.append(np.nan)
                BIC.append(np.nan)

        AIC = [m.aic(data) for m in models if m is not None]
        BIC = [m.bic(data) for m in models if m is not None]

        try:
            M_best = models[np.nanargmin(AIC)]

            means = M_best.means_
            stdevs = np.sqrt(M_best.covariances_)
            weights = M_best.weights_

            means_ = means.tolist()
            stdevs_ = stdevs.tolist()
            weights_ = weights.tolist()
            sci_num = lambda x: "{:.2e}".format(x)

            means = [sci_num(m_[0]) for m_ in means_]
            stdevs = [sci_num(s_[0][0]) for s_ in stdevs_]
            weights = [sci_num(w_) for w_ in weights_]

            return means, stdevs, weights
        except:
            return np.nan, np.nan, np.nan
            print('---Data is not enough for GMM!---')

    def frameNumber_longerProfile_2_Vshape(self, batch_size, video_shape_0, particle_frame_number, V_smooth_follow):
        BS2 = 2 * batch_size
        start_FN_earlyV = particle_frame_number[0]
        end_FN_earlyV = particle_frame_number[-1]
        end_extended_DRA = np.min([video_shape_0, end_FN_earlyV + BS2])
        start_extended_DRA = np.max([0, start_FN_earlyV - BS2])

        tprofile_frameNo_DRA_longer = []
        tprofile_frameNo_DRA_longer.append(list(range(start_extended_DRA, start_FN_earlyV, 1)))
        tprofile_frameNo_DRA_longer.append(particle_frame_number)

        tprofile_frameNo_DRA_longer.append(list(range(end_FN_earlyV, end_extended_DRA, 1)))

        tprofile_frameNo_DRA_longer = [item for sublist in tprofile_frameNo_DRA_longer for item in sublist]

        if len(tprofile_frameNo_DRA_longer) != V_smooth_follow.shape[0]:
            pass

        return tprofile_frameNo_DRA_longer, start_FN_earlyV, end_FN_earlyV

    def plot_contrast_extraction(self, particles, batch_size, video_frame_num, MinPeakWidth,
                                 MinPeakProminence, pixel_size, particles_num='#0'):

        if type(particles) is list:
            particles = protein_trajectories_list2dic(particles)

        if type(particles) is dict:
            for key, item in particles.items():
                if key == particles_num:
                    intensity_horizontal = item['intensity_horizontal']
                    intensity_vertical = item['intensity_vertical']
                    center_int = item['center_int']
                    center_int_flow = item['center_int_flow']
                    frame_number = item['frame_number']
                    sigma = item['sigma']
                    x_center = item['x_center']
                    y_center = item['y_center']
                    particle_ID = item['particle_ID']

                    num_parameters = len(item)

                    if num_parameters == 21:
                        fit_intensity = item['fit_intensity']
                        fit_x = item['fit_x'] * pixel_size
                        fit_y = item['fit_y'] * pixel_size
                        fit_X_sigma = item['fit_X_sigma'] * pixel_size
                        fit_Y_sigma = item['fit_Y_sigma'] * pixel_size
                        fit_Bias = item['fit_Bias']
                        fit_intensity_error = item['fit_intensity_error']
                        fit_x_error = item['fit_x_error'] * pixel_size
                        fit_y_error = item['fit_y_error'] * pixel_size
                        fit_X_sigma_error = item['fit_X_sigma_error'] * pixel_size
                        fit_Y_sigma_error = item['fit_Y_sigma_error'] * pixel_size
                        fit_Bias_error = item['fit_Bias_error']

                    else:
                        fit_intensity = None
                        fit_X_sigma = None
                        fit_Y_sigma = None

                    properties, fit_properties = self.data_handling(center_int, center_int_flow, '', sigma,
                                                                    num_parameters, fit_intensity, fit_X_sigma,
                                                                    fit_Y_sigma, frame_number, batch_size,
                                                                    video_frame_num, MinPeakWidth=MinPeakWidth,
                                                                    MinPeakProminence=MinPeakProminence)

                    if properties is not None:

                        self.axs.plot(properties[5], properties[1], 'C0--', linewidth=5, zorder=0)
                        self.axs.plot(frame_number, properties[0], 'b', linewidth=5, zorder=0)
                        self.axs.plot(properties[2][1], properties[2][0], 'gv', label='Peak', markersize=7, zorder=4)
                        self.axs.axvline(properties[6], linestyle='--', zorder=5)
                        self.axs.axvline(properties[7], linestyle='--', zorder=6)
                        if math.isnan(properties[4][1]) and math.isnan(properties[4][0]):
                            pass
                        else:
                            if properties[2][0] >= 0:
                                self.axs.vlines(x=properties[4][1], ymin=properties[4][0] - properties[4][3],
                                                ymax=properties[4][0], color="C1", label='Prominence')
                            else:
                                self.axs.vlines(x=properties[4][1], ymin=properties[4][0] + properties[4][3],
                                                ymax=properties[4][0], color="C1", label='Prominence')
                        self.axs.set_ylim(np.min(properties[1]) + 0.5*np.min(properties[1]), np.max(properties[1]) + 0.5*np.max(properties[1]))
                        self.axs.grid()
                        self.axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        handles, labels = self.axs.get_legend_handles_labels()

                        handles = [handles[2], handles[0], handles[1], handles[3], handles[4]]
                        labels = [labels[2], labels[0], labels[1], labels[3], labels[4]]
                        self.axs.legend(handles, labels)
                        plt.show()
                    else:
                        raise Exception('---Data can not extract!---')

    def plot_histogram(self, bins=None, upper_limitation=1, lower_limitation=-1,
                       step_range=1e-7, face='g', edge='y', Flag_GMM_fit=True, max_n_components=3, imgSizex=20,
                       imgSizey=20, font_size=12):
        """
        This method plots histograms for different contrast extraction methods for black PSFs, white PSFs and all together.

        Parameters
        ----------
        bins: int
            Number of histogram bins.

        upper_limitation: float
            The upper limit for trimming histogram.

        lower_limitation: float
            The lower limit for trimming histogram.

        step_range: float
            The resolution that is used for GMM plotting.

        face: str
            Face color of the histogram.

        edge: str
            Edge color of the histogram.

        Flag_GMM_fit: bool
            Activate/Deactivate GMM.

        max_n_components: int
            The maximum number of components that GMM used for AIC and BIC tests. This helps to find an optimum number of the mixture.

        imgSizex: int
            The width of the histogram figure.

        imgSizey: int
            The height of the histogram figure.

        font_size: float
            The font size of the text in the table information.

        """

        df, list_data, title = self.extract_hist_information(con_intersections=self.t_contrast_intersection,
                                                             con_peaks=self.t_contrast_peaks,
                                                             con_proms=self.t_contrast_Prominence,
                                                             upper_limitation=upper_limitation,
                                                             lower_limitation=lower_limitation,
                                                             max_n_components=max_n_components,
                                                             Flag_GMM_fit=Flag_GMM_fit)

        fig = plt.figure(figsize=(imgSizex, imgSizey))
        outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.2)

        inner = gridspec.GridSpecFromSubplotSpec(3, 3,
                                                 subplot_spec=outer[0], wspace=0.3, hspace=0.7)

        for p_index in range(len(list_data)):
            if Flag_GMM_fit:
                try:
                    key = title[p_index]
                    index_ = df.index.to_list()
                    pdfs = []
                    means = []
                    stdevs = []
                    weights =[]
                    for idx_ in index_:
                        if 'GMM_mean' in idx_:
                            means.append(df[key][idx_])
                        elif 'GMM_std' in idx_:
                            stdevs.append(df[key][idx_])
                        elif 'GMM_weight' in idx_:
                            weights.append(df[key][idx_])

                    d_ = np.abs(list_data[p_index])
                    min_data = np.min(d_) - 0.5 * np.min(d_)
                    max_data = np.max(d_) + 0.5 * np.max(d_)

                    x = np.arange(min_data, max_data, step_range)

                    pdfs = [float(p) * ss.norm.pdf(x, float(mu), float(sd)) for mu, sd, p in zip(means, stdevs, weights)
                            if mu is not None and sd is not None and p is not None]
                    density = np.sum(np.array(pdfs), axis=0)
                    ax = plt.Subplot(fig, inner[p_index])
                    ax.hist(d_, bins=bins, fc=face, ec=edge, density=True)
                    ax.plot(x.ravel(), density.ravel())
                    ax.set_ylabel('Density')
                    ax.set_title(title[p_index])
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    fig.add_subplot(ax)

                except:
                    d_ = np.abs(list_data[p_index])

                    ax = plt.Subplot(fig, inner[p_index])
                    ax.hist(d_, bins=bins, fc=face, ec=edge, density=False)
                    ax.set_ylabel('#Counts')
                    ax.set_title(title[p_index])
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    fig.add_subplot(ax)

            else:
                d_ = np.abs(list_data[p_index])

                ax = plt.Subplot(fig, inner[p_index])
                ax.hist(d_, bins=bins, fc=face, ec=edge, density=False)
                ax.set_ylabel('#Counts')
                ax.set_title(title[p_index])
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                fig.add_subplot(ax)

        ax2 = plt.Subplot(fig, outer[1])
        font_size = font_size
        bbox = [0, 0, 1, 1]
        ax2.axis('off')
        mpl_table = ax2.table(cellText=df.values, rowLabels=df.index, bbox=bbox, colLabels=df.columns)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        fig.add_subplot(ax2)

        plt.show()

    def plot_fit_histogram(self, bins=None, upper_limitation=1, lower_limitation=-1, step_range=1e-7, face='g',
                           edge='y',
                           Flag_GMM_fit=True, max_n_components=3, imgSizex=20, imgSizey=20, font_size=12):
        """
        This method plots histograms for 2D Gaussian fitting contrast for black PSFs, white PSFs and all together.

       Parameters
       ----------
       bins: int
           Number of histogram bins.

       upper_limitation: float
           The upper limit for trimming histogram.

       lower_limitation: float
           The lower limit for trimming histogram.

       step_range: float
           The resolution that is used for GMM plotting.

       face: str
           Face color of the histogram.

       edge: str
           Edge color of the histogram.

       Flag_GMM_fit: bool
           Activate/Deactivate GMM.

       max_n_components: int
           The maximum number of components that GMM used for AIC and BIC tests. This helps to find an optimum number of the mixture.

       imgSizex: int
           The width of the histogram figure.

       imgSizey: int
           The height of the histogram figure.

       font_size: float
           The font size of the text in the table information.

        """

        df, list_data, title = self.extract_hist_information(con_intersections=self.t_contrast_fit_intersection,
                                                             con_peaks=self.t_contrast_fit_peaks,
                                                             con_proms=None,
                                                             upper_limitation=upper_limitation,
                                                             lower_limitation=lower_limitation,
                                                             max_n_components=max_n_components,
                                                             Flag_GMM_fit=Flag_GMM_fit)

        fig = plt.figure(figsize=(imgSizex, imgSizey))
        outer = gridspec.GridSpec(2, 1, wspace=0.3, hspace=0.2)

        inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], wspace=0.3, hspace=0.7)

        for p_index in range(len(list_data)):
            if Flag_GMM_fit:
                key = title[p_index]
                means = df[key]['GMM_Means']
                stdevs = df[key]['GMM_Stds']
                weights = df[key]['GMM_Weights']
                if means is None or stdevs is None or weights is None:
                    d_ = np.abs(list_data[p_index])
                    nan_array = np.isnan(d_)
                    not_nan_array = ~ nan_array
                    d_ = d_[not_nan_array]

                    min_data = np.nanmin(d_)
                    max_data = np.nanmax(d_)

                    x = np.arange(min_data, max_data, step_range)
                    pdfs = [p * ss.norm.pdf(x, mu, sd) for mu, sd, p in zip(means, stdevs, weights)]
                    density = np.sum(np.array(pdfs), axis=0)
                    ax = plt.Subplot(fig, inner[p_index])
                    ax.hist(d_, bins=bins, fc=face, ec=edge, density=True)
                    ax.plot(x.ravel(), density.ravel())
                    ax.xlim(lower_limitation, upper_limitation)
                    ax.set_ylabel('Density')
                    ax.set_title(title[p_index])
                    fig.add_subplot(ax)

                else:
                    d_ = np.abs(list_data[p_index])
                    nan_array = np.isnan(d_)
                    not_nan_array = ~ nan_array
                    d_ = d_[not_nan_array]

                    ax = plt.Subplot(fig, inner[p_index])
                    ax.hist(d_, bins=bins, fc=face, ec=edge, density=False)
                    ax.set_ylabel('#Counts')
                    ax.set_title(title[p_index])
                    fig.add_subplot(ax)

            else:
                d_ = list_data[p_index]
                nan_array = np.isnan(d_)
                not_nan_array = ~ nan_array
                d_ = d_[not_nan_array]

                ax = plt.Subplot(fig, inner[p_index])
                ax.hist(d_, bins=bins, fc=face, ec=edge, density=False)
                ax.set_ylabel('#Counts')
                ax.set_title(title[p_index])
                fig.add_subplot(ax)

        ax2 = plt.Subplot(fig, outer[1])
        font_size = font_size
        bbox = [0, 0, 1, 1]
        ax2.axis('off')
        mpl_table = ax2.table(cellText=df.values, rowLabels=df.index, bbox=bbox, colLabels=df.columns)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        fig.add_subplot(ax2)

        plt.show()

    def plot_fit_sigma(self):
        mean_sigmX = np.mean(self.t_iPSFCentroidSigmas_fit_x)
        mean_sigmY = np.mean(self.t_iPSFCentroidSigmas_fit_y)
        if mean_sigmY >= mean_sigmX:
            ratio_sigma = mean_sigmY / mean_sigmX
        else:
            ratio_sigma = mean_sigmX / mean_sigmY

        plt.figure()
        plt.hist(self.t_iPSFCentroidSigmas_fit_x)
        plt.hist(self.t_iPSFCentroidSigmas_fit_y)
        plt.xlabel('sigma(nm)')
        plt.ylabel('#Counts')
        plt.legend('Fitted sigma_x', 'Fitted sigma_y')
        plt.title('Ratio mean of Sigma_Max/Sigma_Min=' + str(ratio_sigma))
        plt.show()

    def save_hist_data(self, dirName, name, upper_limitation=1, lower_limitation=-1, Flag_GMM_fit=True, max_n_components=3):
        """
        This function save the histogram data with HDF5 format.

        Parameters
        ----------
        dirName: str
            Path for saving data.

        name: str
            Name that use for saving data.

        upper_limitation: float
            The upper limit for trimming histogram.

        lower_limitation: float
            The lower limit for trimming histogram.

        Flag_GMM_fit: bool
            Activate/Deactivate GMM.

        max_n_components: int
            The maximum number of components that GMM used for AIC and BIC tests. This helps to find an optimum number of the mixture.

        """

        df, list_data, title = self.extract_hist_information(con_intersections=self.t_contrast_intersection,
                                                             con_peaks=self.t_contrast_peaks,
                                                             con_proms=self.t_contrast_Prominence,
                                                             upper_limitation=upper_limitation,
                                                             lower_limitation=lower_limitation,
                                                             max_n_components=max_n_components,
                                                             Flag_GMM_fit=Flag_GMM_fit)
        dic_data = {}
        for d_, t_ in zip(list_data, title):
            dic_data[t_] = d_

        read_write_data.save_dic_to_hdf5(dic_data=dic_data, path=dirName, name=name)











