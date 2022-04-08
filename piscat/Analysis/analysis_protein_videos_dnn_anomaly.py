from piscat.InputOutput import read_status_line, reading_videos
from piscat.InputOutput import read_write_data, write_video
from piscat.BackgroundCorrection import DRA
from piscat.Preproccessing import normalization, filtering
from piscat.Preproccessing.applying_mask import Mask2Video
from piscat.Localization import localization_filtering
from piscat.Localization import particle_localization
from piscat.Trajectory import particle_linking
from piscat.Trajectory import temporal_filtering
from piscat.Anomaly.hand_crafted_feature_genration import CreateFeatures
from piscat.Anomaly.anomaly_localization import BinaryToiSCATLocalization
from piscat.Preproccessing.normalization import Normalization
from piscat.DNNModel.FastDVDNet import FastDVDNet
from piscat.Anomaly.DNN_anomaly import DNNAnomaly

import os
import numpy as np


def dnn_anomaly_protein_analysis(paths, video_names, hyperparameters, flags, name_mkdir):
    """
    This function analyzes several videos based on the user's settings in the "hyperparameters" and "flags" for
    training/testing DNN and feeding to isolation forest to extract protein contrasts.

    Parameters
    ----------
    paths: list
        List of all paths for videos.

    video_names: str
        list of all video names.

    hyperparameters: dic
        The dictionary is used to define different parameters for analysis. In the following you can see the example of this dictionary:

            | hyperparameters = {   ##### Reading video + Background correction ######
                                    'im_size_x': 72, 'im_size_y': 72, 'start_fr': 29000, 'end_fr': 89000, 'image_format': '<u2',
                                    'batch_size': 2500,
                                    'mode_FPN': 'mFPN', 'select_correction_axis': 'Both',
                                    ##### Setting for extracting feature matrix (DNN) and anomaly ######
                                    'dog_min_sigmaX': 1.7, 'dog_min_sigmaY': 1.7, 'dog_max_sigmaX': 1.8,
                                    'dog_max_sigmaY': 1.8, 'train_frame_stride': 5, 'DNN_batch_size':20, 'epochs': 25,
                                    'shuffle': False, 'validation_split': 0.33, 'downsampling': 1,
                                    'anomaly_method': 'IsolationForest', 'contamination':0.003, 'IF_step': 10, 'IF_stride': 10,
                                    'morphological_threshold': 4, 'experimental_PSF_size_(pixel)': 1.7,
                                    ### Localization and Linking method ###
                                    'Mode_PSF_Segmentation': 'BOTH', 'center_int_method': 'dog',
                                    'localization_dog_min_sigma':1.7, 'localization_dog_max_sigma':1.8,
                                    'localization_dog_sigma_ratio': 1.1, 'localization_dog_PSF_detection_thr': 1e-5,
                                    'localization_dog_overlap': 0, 'localization_window_scale': 5,
                                    'search_range': 4, 'memory': 20,
                                    ### Spatio_themporal filters ###
                                    'outlier_frames_thr': 10, 'symmetric_PSFs_thr': 0.6, 'min_V_shape_width': 700,
                                    'circlur_mask_redius': 60}

    flags: dic
         The dictionary is used to active/deactivate different parts in analyzing pipelines. In the following you can see the example of this dictionary:

            | flags = { ### Background correction flags ###
                        'PN': True, 'FPNc': True, 'DRA': True, 'filter_hotPixels': True, 'FFT_flag': True,
                        ### iPSF filter flags ###
                        'outlier_frames_filter': True, 'Dense_Filter': False, 'symmetric_PSFs_Filter': False,
                        'Side_lobes_Filter': False, 'Edge_mask' : True,
                        ### feature mode flags ###
                        'DNN_train': False, 'DNN_test': True}




    name_mkdir: str
        It defines the name of the folder that automatically creates next to each video to save the results of the analysis and setting history.

    Returns
    -------
    The following informations will be saved
        * `hyperparameters`

        * `flags`

        * `Number of particles and PSFs after each steps`

        * `All extracted trajectoried with 'HDF5' and 'Matlab' format.`

            * `Matlab` saves array contains the following information for each particle:

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

        * `Binary video result of the anomaly detection`

    """

    PSFs_Particels_num = {'#Totall_PSFs': None,
                          '#PSFs_afterOutlierFramesFilter': None,
                          '#PSFs_afterDenseFilter': None,
                          '#PSFs_afterSymmetricPSFsFilter': None,
                          '#Totall_Particles': None,
                          '#Particles_after_V_shapeFilter': None,
                          '#Totall_frame_num_DRA': None}

    for idx_vid, (p_, n_) in enumerate(zip(paths, video_names)):
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

        if isinstance(hyperparameters['start_fr'], int) and isinstance(hyperparameters['end_fr'], int):
            video_ = reading_videos.read_binary(file_name=p_ + '/' + n_,
                                               img_width=hyperparameters['im_size_x'], img_height=hyperparameters['im_size_y'],
                                               image_type=np.dtype(hyperparameters['image_format']))

            video_ = video_[hyperparameters['start_fr']:hyperparameters['end_fr'], :, :]
        else:
            video_ = reading_videos.read_binary(file_name=p_ + '/' + n_,
                                               img_width=hyperparameters['im_size_x'],
                                               img_height=hyperparameters['im_size_y'],
                                               image_type=np.dtype(hyperparameters['image_format']))

        status_ = read_status_line.StatusLine(video_)
        video_status, status_info = status_.find_status_line()

        F, H, W = video_status.shape
        min_dim = np.min((H, W))
        center_image_x = int(round(H/2))
        center_image_y = int(round(W/2))
        w_x = w_y = int(0.5 * 8 *  int(min_dim/8))
        video = video_[:, center_image_x-w_x:center_image_x+w_x, center_image_y-w_y:center_image_y+w_y]


        if flags['PN']:
            video_pn, _ = normalization.Normalization(video=video).power_normalized()
        else:
            video_pn = video

        if flags['DRA']:
            DRA_ = DRA.DifferentialRollingAverage(video=video_pn, batchSize=hyperparameters['batch_size'],
                                                  mode_FPN=hyperparameters['mode_FPN'])

            RVideo_PN_FPN_, _ = DRA_.differential_rolling(FPN_flag=flags['FPNc'],
                                                         select_correction_axis=hyperparameters['select_correction_axis'],
                                                         FFT_flag=flags['FFT_flag'], inter_flag_parallel_active=True)
        else:
            RVideo_PN_FPN_ = video_pn

        if flags['filter_hotPixels']:
            RVideo_PN_FPN = filtering.Filters(RVideo_PN_FPN_).median(3)
            video_norm = Normalization(RVideo_PN_FPN).normalized_image_global()

        else:
            video_norm = RVideo_PN_FPN_

        PSFs_Particels_num['#Totall_frame_num_DRA'] = video_norm.shape[0]
        s_dir_weights = os.path.dirname(p_)

        if flags['DNN_train']:
            if idx_vid == 0:
                flag_warm_train = False
                print("---Train from random weights!---")
            else:
                flag_warm_train = True
                print("---Train from previous weights!---")

            dnn_ = FastDVDNet(video_original=video_norm)
            batch_original_array, video_original_crop = dnn_.data_handling(hyperparameters['train_frame_stride'])
            dnn_.train(video_input_array=batch_original_array,
                       DNN_param=hyperparameters,
                       video_label_array=None,
                       path_save=s_dir_weights, name_weights="anomaly_weights.h5", flag_warm_train=flag_warm_train)

        if flags['DNN_test']:
            dnn_ = FastDVDNet(video_original=video_norm)
            batch_original_array, video_original_crop = dnn_.data_handling()
            feature_DNN = dnn_.test(batch_original_array, path_save=s_dir_weights, name_weights="anomaly_weights.h5")

            diff_vid = np.abs(video_original_crop[2:-2, ...] - feature_DNN)

            if flags['Edge_mask']:
                M2Vid = Mask2Video(diff_vid, mask=None, inter_flag_parallel_active=True)
                circlur_mask = M2Vid.mask_generating_circle(center=(int(diff_vid.shape[1]/2), int(diff_vid.shape[1]/2)), redius=hyperparameters['circlur_mask_redius'])
                video_mask = M2Vid.apply_mask(flag_nan=False)
            else:
                video_mask = diff_vid

            write_video.write_binary(dir_path=s_dir_, file_name='feature_map_type_' + str(diff_vid.dtype) + '_shape' + str(diff_vid.shape) + '.raw', data=video_mask)

            feature_maps_spatio = CreateFeatures(video=video_norm[2:-2, ...])
            dog_features = feature_maps_spatio.dog2D_creater(
                low_sigma=[hyperparameters['dog_min_sigmaX'], hyperparameters['dog_min_sigmaY']],
                high_sigma=[hyperparameters['dog_max_sigmaX'], hyperparameters['dog_max_sigmaY']])

            if_dnn = DNNAnomaly([video_mask, dog_features], contamination=hyperparameters['contamination'])
            mask_video_ = if_dnn.apply_IF_spacial_temporal(step=hyperparameters['IF_step'],
                                                           stride=hyperparameters['IF_stride'])

            r_diff = int(0.5 * abs(mask_video_.shape[1] - RVideo_PN_FPN.shape[1]))
            c_diff = int(0.5 * abs(mask_video_.shape[2] - RVideo_PN_FPN.shape[2]))

            mask_video_pad = np.pad(mask_video_, ((0, 0), (r_diff, r_diff), (c_diff, c_diff)), 'constant', constant_values=((0, 0), (1, 1), (1, 1)))

            result_anomaly_ = mask_video_pad.copy()
            result_anomaly_[mask_video_pad == -1] = 0
            result_anomaly_ = Normalization(video=result_anomaly_.astype(int)).normalized_image_specific()

            write_video.write_binary(dir_path=s_dir_,
                                     file_name='mask_map_type_' + str(result_anomaly_.dtype) + '_shape' + str(result_anomaly_.shape) + '.raw',
                                     data=result_anomaly_)

            binery_localization = BinaryToiSCATLocalization(video_binary=result_anomaly_, video_iSCAT=RVideo_PN_FPN[2:-2, ...],
                                                            area_threshold=hyperparameters['morphological_threshold'], sigma=hyperparameters['experimental_PSF_size_(pixel)'])

            if hyperparameters['center_int_method'] == 'extremum':
                binery_localization.local_extremum_in_window(scale=hyperparameters['localization_window_scale'])
                df_PSFs = binery_localization.df_pos
            elif hyperparameters['center_int_method'] == 'fit2D':
                df_PSFs = binery_localization.gaussian2D_fit_iSCAT(scale=hyperparameters['localization_window_scale'])
            elif hyperparameters['center_int_method'] == 'dog':
                df_PSFs = binery_localization.local_dog_in_window(min_sigma=hyperparameters['localization_dog_min_sigma'],
                                                                  max_sigma=hyperparameters['localization_dog_max_sigma'],
                                                                  sigma_ratio=hyperparameters['localization_dog_sigma_ratio'],
                                                                  threshold=hyperparameters['localization_dog_PSF_detection_thr'],
                                                                  overlap=hyperparameters['localization_dog_overlap'],
                                                                  scale=hyperparameters['localization_window_scale'])

            PSF_l = particle_localization.PSFsExtraction(video=RVideo_PN_FPN)

            if df_PSFs is not None:
                PSFs_Particels_num['#Totall_PSFs'] = df_PSFs.shape[0]
            else:
                PSFs_Particels_num['#Totall_PSFs'] = 0

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

                if flags['Side_lobes_Filter']:
                    df_PSFs_s_filter = s_filters.remove_side_lobes_artifact(df_PSFs_s_filter, threshold=0)
                    PSFs_Particels_num['#PSFs_afterSideLobesFilter'] = df_PSFs_s_filter.shape[0]

                if flags['symmetric_PSFs_Filter'] and df_PSFs_s_filter.shape[0] != 0:
                    df_PSF_2Dfit = PSF_l.fit_Gaussian2D_wrapper(PSF_List=df_PSFs, scale=6, internal_parallel_flag=True)
                    df_PSFs_s_filter = s_filters.symmetric_PSFs(df_PSFs=df_PSF_2Dfit, threshold=hyperparameters['symmetric_PSFs_thr'])
                    if df_PSFs_s_filter is not None:
                        PSFs_Particels_num['#PSFs_afterSymmetricPSFsFilter'] = df_PSFs_s_filter.shape[0]

                if df_PSFs_s_filter.shape[0] > 1 and df_PSFs_s_filter is not None:

                    read_write_data.save_df2csv(df_PSFs_s_filter, path=s_dir_, name='position_PSFs_before_temporal')

                    linking_ = particle_linking.Linking()
                    df_PSFs_link = linking_.create_link(psf_position=df_PSFs_s_filter,
                                                        search_range=hyperparameters['search_range'],
                                                        memory=hyperparameters['memory'])
                    PSFs_Particels_num['#Totall_Particles'] = linking_.trajectory_counter(df_PSFs_link)

                    t_filters = temporal_filtering.TemporalFilter(video=RVideo_PN_FPN,
                                                                  batchSize=hyperparameters['batch_size'])

                    all_trajectories, df_PSFs_t_filter, his_all_particles = t_filters.v_trajectory(df_PSFs=df_PSFs_link,
                                                                                threshold=hyperparameters[
                                                                                    'min_V_shape_width'])
                    PSFs_Particels_num['#Particles_after_V_shapeFilter'] = linking_.trajectory_counter(df_PSFs_t_filter)

                    if len(all_trajectories) > 0:
                        read_write_data.save_mat(data=all_trajectories, path=s_dir_, name='all_trajectories')
                        read_write_data.save_list_to_hdf5(list_data=all_trajectories, path=s_dir_, name='histData')

                    if df_PSFs_t_filter.shape[0] > 0:
                        read_write_data.save_df2csv(df_PSFs_t_filter, path=s_dir_, name='position_PSFs')

            read_write_data.save_dic2json(data_dictionary=PSFs_Particels_num, path=s_dir_, name='PSFs_Particels_num')
