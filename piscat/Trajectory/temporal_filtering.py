import numpy as np

from tqdm.autonotebook import tqdm


class TemporalFilter:

    def __init__(self, video, batchSize):
        """
        Filters to be applied to temporal features are included in this class.

        Parameters
        ----------
        video: NDArray
            Input video.

        batchSize: int
            Batch size that is used for DRA.

        """
        self.video = video
        self.batchSize = batchSize

    def filter_tarj_base_length(self, df_PSFs, threshold):
        """
        This function removes the particle from data frames that have a temporal length smaller than threshold values.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations and ID (x, y, frame, sigma, particle, ...)

        threshold: int
            The minimum acceptable temporal length of one particle.

        Returns
        -------
        particles: pandas dataframe
            The filter data frame (x, y, frame, sigma, particle, ...)

        his_all_particles: list
            List that shows the statistics of particles length.
        """
        if df_PSFs.shape[0] == 0 or df_PSFs is None:
            raise ValueError('---data frames is empty!---')

        his_all_particles = df_PSFs['particle'].value_counts()
        temp = his_all_particles.where(his_all_particles >= threshold)
        select_particles = temp[~temp.isnull()]
        index_particles = select_particles.index
        particles = df_PSFs.loc[df_PSFs['particle'].isin(index_particles)]
        return particles, his_all_particles

    def v_trajectory(self, df_PSFs, threshold):
        """
        This function extract v-shape of the particle that have a temporal length bigger than threshold values.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations and ID (x, y, frame, sigma, particle, ...).

        threshold: int
            The minimum acceptable temporal length of one particle.

        Returns
        -------
        all_trajectories: List of list
            Returns list of extracted data (i.e [List of list])

            | [intensity_horizontal, intensity_vertical, particle_center_intensity,
                particle_center_intensity_follow, particle_frame, particle_sigma, particle_X, particle_Y, particle_ID,
                optional(fit_intensity, fit_x, fit_y, fit_X_sigma, fit_Y_sigma, fit_Bias, fit_intensity_error,
                fit_x_error, fit_y_error, fit_X_sigma_error, fit_Y_sigma_error, fit_Bias_error)]

        particles: pandas dataframe
             The dataframe after using temporal filter (x, y, frame, sigma, particle, ...)

        his_all_particles: list
            List that shows the statistics of particles length.
        """
        if df_PSFs.shape[0] == 0 or df_PSFs is None:
            raise ValueError('---data frames is empty!---')

        particles, his_all_particles = self.filter_tarj_base_length(df_PSFs=df_PSFs, threshold=threshold)
        all_trajectories = self.v_profile(df_PSFs=particles, window_size=self.batchSize)
        return all_trajectories, particles, his_all_particles

    def v_profile(self, df_PSFs, window_size=2000):
        """
        The V-Shape trajectories and extended version are calculated.

        Parameters
        ----------
        window_size: int
            The maximum number of the frames that follow the V-Shape contrast from both sides.

        Returns
        -------
        all_trajectories: List of list
            Returns array contains the following information for each particle.

            | [intensity_horizontal, intensity_vertical, particle_center_intensity,
                particle_center_intensity_follow, particle_frame, particle_sigma, particle_X, particle_Y, particle_ID,
                optional(fit_intensity, fit_x, fit_y, fit_X_sigma, fit_Y_sigma, fit_Bias, fit_intensity_error,
                fit_x_error, fit_y_error, fit_X_sigma_error, fit_Y_sigma_error, fit_Bias_error)]
        """

        all_trajectories = []
        intensity_trajectories_dic = dict()
        index_particles = df_PSFs['particle'].unique().tolist()

        print("\nstart V_trajectories without parallel loop--->", end=" ")
        window_size = 2 * window_size

        for j_ in tqdm(index_particles):

            particle = df_PSFs.loc[df_PSFs['particle'] == j_]
            particle_ID = particle['particle'].tolist()
            particle_X = particle['x'].tolist()
            particle_Y = particle['y'].tolist()
            particle_frame = particle['frame'].tolist()
            particle_center_intensity = particle['center_intensity'].tolist()
            particle_sigma = particle['sigma'].tolist()

            if 'Fit_Amplitude' in particle.keys():
                fit_intensity = particle['Fit_Amplitude'].tolist()
                fit_x = particle['Fit_X-Center'].tolist()
                fit_y = particle['Fit_Y-Center'].tolist()
                fit_X_sigma = particle['Fit_X-Sigma'].tolist()
                fit_Y_sigma = particle['Fit_Y-Sigma'].tolist()
                fit_Bias = particle['Fit_Bias'].tolist()

                fit_intensity_error = particle["Fit_errors_Amplitude"].tolist()
                fit_x_error = particle["Fit_errors_X-Center"].tolist()
                fit_y_error = particle["Fit_errors_Y-Center"].tolist()
                fit_X_sigma_error = particle["Fit_errors_X-Sigma"].tolist()
                fit_Y_sigma_error = particle["Fit_errors_Y-Sigma"].tolist()
                fit_Bias_error = particle["Fit_errors_Bias"].tolist()

            intensity_vertical = []
            intensity_horizontal = []
            for frame_num, y, x, sigma in zip(particle_frame, particle_Y, particle_X, particle_sigma):
                len_profile = int(5 * sigma)
                min_v = np.max([y - len_profile, 0])
                max_v = np.min([y + len_profile, self.video.shape[1]])

                intensity_vertical.append(self.video[int(frame_num), int(min_v):int(max_v), int(x)])

                min_h = np.max([x - len_profile, 0])
                max_h = np.min([x + len_profile, self.video.shape[1]])

                intensity_horizontal.append(self.video[int(frame_num), int(y), int(min_h):int(max_h)])

            first_frame = int(particle_frame[0])
            x_first_frame = int(particle_X[0])
            y_first_frame = int(particle_Y[0])

            last_frame = int(particle_frame[-1])
            x_last_frame = int(particle_X[-1])
            y_last_frame = int(particle_Y[-1])

            start_frame = np.max([0, first_frame - window_size])
            end_frame = np.min([self.video.shape[0], last_frame + window_size])

            particle_center_intensity_follow_backward = self.video[int(start_frame):int(first_frame),  y_first_frame, x_first_frame]
            particle_center_intensity_follow_forward = self.video[int(last_frame):int(end_frame),  y_last_frame, x_last_frame]

            particle_center_intensity_follow = np.concatenate((particle_center_intensity_follow_backward,
                                                               particle_center_intensity, particle_center_intensity_follow_forward), axis=0)
            trajectories = []
            trajectories.append(intensity_horizontal)
            trajectories.append(intensity_vertical)
            trajectories.append(particle_center_intensity)
            trajectories.append(particle_center_intensity_follow)
            trajectories.append(particle_frame)
            trajectories.append(particle_sigma)
            trajectories.append(particle_X)
            trajectories.append(particle_Y)
            trajectories.append(particle_ID)
            if 'Fit_Amplitude' in particle.keys():

                trajectories.append(fit_intensity)
                trajectories.append(fit_x)
                trajectories.append(fit_y)
                trajectories.append(fit_X_sigma)
                trajectories.append(fit_Y_sigma)
                trajectories.append(fit_Bias)
                trajectories.append(fit_intensity_error)
                trajectories.append(fit_x_error)
                trajectories.append(fit_y_error)
                trajectories.append(fit_X_sigma_error)
                trajectories.append(fit_Y_sigma_error)
                trajectories.append(fit_Bias_error)

            all_trajectories.append(trajectories)
        print('Done')

        return all_trajectories



