from piscat.InputOutput import read_status_line, reading_videos
from piscat.InputOutput import read_write_data
from piscat.BackgroundCorrection import DRA
from piscat.Preproccessing import normalization
from piscat.Localization import localization_filtering
from piscat.Localization import particle_localization
from piscat.Trajectory import particle_linking
from piscat.Trajectory import temporal_filtering

import os
import numpy as np


def protein_analysis(paths, video_names, hyperparameters, flags, name_mkdir):
    """
    This function analysis several videos based on the setting that the user defines in the ``hyperparameters`` and ``flags``.

    Parameters
    ----------
    paths: list
        List of all paths for videos.

    video_names: str
        list of all video names.

    hyperparameters: dic
        The dictionary is used to define different parameters for analysis. In the following you can see the example of this dictionary:

            | hyperparameters = {'function': 'dog', 'batch_size': 3000, 'min_V_shape_width': 1500,
                                    'search_range': 2, 'memory': 20, 'min_sigma': 1.3, 'max_sigma': 3, 'sigma_ratio': 1.1,
                                    'PSF_detection_thr': 4e-5, 'overlap': 0, 'outlier_frames_thr': 20, 'Mode_PSF_Segmentation': 'BOTH',
                                    'symmetric_PSFs_thr': 0.6, 'im_size_x': 72,
                                    'im_size_y': 72, 'image_format': '<u2'}

    flags: dic
         The dictionary is used to active/deactivate different parts in analyzing pipelines. In the following you can see the example of this dictionary:

            | flags = {'PN': True, 'outlier_frames_filter': True, 'Dense_Filter': True, 'symmetric_PSFs_Filter': True, 'FFT_flag': True}

    name_mkdir: str
        It defines the name of the folder that automatically creates next to each video to save the results of the analysis and setting history.

    Returns
    -------
    The following information will be saved
    
        * `hyperparameters`

        * `flags`

        * `Number of particles and PSFs after each steps`

        * `All extracted trajectories with 'HDF5' and 'Matlab' format.`

            * `MATLAB` saves array contains the following information for each particle:

            | [intensity_horizontal, intensity_vertical, particle_center_intensity,
                                particle_center_intensity_follow, particle_frame, particle_sigma, particle_X, particle_Y, particle_ID,
                                optional(fit_intensity, fit_x, fit_y, fit_X_sigma, fit_Y_sigma, fit_Bias, fit_intensity_error,
                                fit_x_error, fit_y_error, fit_X_sigma_error, fit_Y_sigma_error, fit_Bias_error)]

            * `HDF5` saves dictionary similar to the following structures:

            | {"#0": {'intensity_horizontal': ..., 'intensity_vertical': ..., ..., 'particle_ID': ...},
                        "#1": {}, ...}

        * `Table of PSFs information`

            * `particles`: pandas dataframe

            | Saving the clean data frame (x, y, frame, sigma, particle, ...)

    """
    PSFs_Particels_num = {'#Totall_PSFs': None,
                          '#PSFs_afterOutlierFramesFilter': None,
                          '#PSFs_afterDenseFilter': None,
                          '#PSFs_afterSymmetricPSFsFilter': None,
                          '#Totall_Particles': None,
                          '#Particles_after_V_shapeFilter': None,
                          '#Totall_frame_num_DRA': None}

    for p_, n_ in zip(paths, video_names):
        print("---" + p_ + "---")
        try:
            dr_mk = os.path.join(p_, name_mkdir)
            os.mkdir(dr_mk)
            print("Directory ", name_mkdir, " Created ")
        except FileExistsError:
            print("Directory ", name_mkdir, " already exists")

        s_dir_ = os.path.join(p_, name_mkdir)
        read_write_data.save_dic2json(data_dictionary=hyperparameters, path=s_dir_, name='hyperparameters')
        read_write_data.save_dic2json(data_dictionary=flags, path=s_dir_, name='flags')

        video = reading_videos.read_binary(file_name=p_ + '/' + n_, img_width=hyperparameters['im_size_x'], img_height=hyperparameters['im_size_y'],
                                           image_type=np.dtype(hyperparameters['image_format']))
        status_ = read_status_line.StatusLine(video)
        video, status_info = status_.find_status_line()

        if flags['PN']:
            video_pn, _ = normalization.Normalization(video=video).power_normalized()
        else:
            video_pn = video

        DRA_ = DRA.DifferentialRollingAverage(video=video_pn, batchSize=hyperparameters['batch_size'])

        RVideo_PN = DRA_.differential_rolling(FFT_flag=flags['FFT_flag'])

        PSFs_Particels_num['#Totall_frame_num_DRA'] = RVideo_PN.shape[0]

        PSF_l = particle_localization.PSFsExtraction(video=RVideo_PN)
        df_PSFs = PSF_l.psf_detection(function=hyperparameters['function'],  # function='log', 'doh', 'dog'
                                      min_sigma=hyperparameters['min_sigma'], max_sigma=hyperparameters['max_sigma'],
                                      sigma_ratio=hyperparameters['sigma_ratio'],
                                      threshold=hyperparameters['PSF_detection_thr'],
                                      overlap=hyperparameters['overlap'], mode=hyperparameters['Mode_PSF_Segmentation'])

        PSFs_Particels_num['#Totall_PSFs'] = df_PSFs.shape[0]

        if PSFs_Particels_num['#Totall_PSFs'] > 1:
            s_filters = localization_filtering.SpatialFilter()

            if flags['outlier_frames_filter']:
                df_PSFs_s_filter = s_filters.outlier_frames(df_PSFs,
                                                            threshold=hyperparameters['outlier_frames_thr'])
                PSFs_Particels_num['#PSFs_afterOutlierFramesFilter'] = df_PSFs_s_filter.shape[0]
            else:
                df_PSFs_s_filter = df_PSFs

            if flags['Dense_Filter']:
                df_PSFs_s_filter = s_filters.dense_PSFs(df_PSFs_s_filter, threshold=0)
                PSFs_Particels_num['#PSFs_afterDenseFilter'] = df_PSFs_s_filter.shape[0]

            if flags['symmetric_PSFs_Filter'] and df_PSFs_s_filter.shape[0] != 0:
                df_PSF_2Dfit = PSF_l.fit_Gaussian2D_wrapper(PSF_List=df_PSFs, scale=5, internal_parallel_flag=True)
                df_PSFs_s_filter = s_filters.symmetric_PSFs(df_PSFs=df_PSF_2Dfit, threshold=hyperparameters['symmetric_PSFs_thr'])
                if df_PSFs_s_filter is not None:
                    PSFs_Particels_num['#PSFs_afterSymmetricPSFsFilter'] = df_PSFs_s_filter.shape[0]

            if df_PSFs_s_filter.shape[0] > 1 and df_PSFs_s_filter is not None:
                linking_ = particle_linking.Linking()
                df_PSFs_link = linking_.create_link(psf_position=df_PSFs_s_filter,
                                                    search_range=hyperparameters['search_range'],
                                                    memory=hyperparameters['memory'])
                PSFs_Particels_num['#Totall_Particles'] = linking_.trajectory_counter(df_PSFs_link)

                t_filters = temporal_filtering.TemporalFilter(video=RVideo_PN,
                                                              batchSize=hyperparameters['batch_size'])
                all_trajectories, df_PSFs_t_filter = t_filters.v_trajectory(df_PSFs=df_PSFs_link,
                                                                            threshold=hyperparameters[
                                                                                'min_V_shape_width'])
                PSFs_Particels_num['#Particles_after_V_shapeFilter'] = linking_.trajectory_counter(df_PSFs_t_filter)

                read_write_data.save_mat(data=all_trajectories, path=s_dir_, name='all_trajectories')
                read_write_data.save_df2csv(df_PSFs_t_filter, path=s_dir_, name='position_PSFs')
                read_write_data.save_list_to_hdf5(list_data=all_trajectories, path=s_dir_, name='histData')

        read_write_data.save_dic2json(data_dictionary=PSFs_Particels_num, path=s_dir_, name='PSFs_Particels_num')
