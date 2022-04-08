from piscat.InputOutput import read_status_line, reading_videos
from piscat.InputOutput import read_write_data, write_video
from piscat.BackgroundCorrection import DRA
from piscat.Preproccessing import normalization, filtering
from piscat.Localization import localization_filtering
from piscat.Localization import particle_localization
from piscat.Trajectory import particle_linking
from piscat.Trajectory import temporal_filtering
from piscat.Anomaly.hand_crafted_feature_genration import CreateFeatures
from piscat.Anomaly.spatio_temporal_anomaly import SpatioTemporalAnomalyDetection
from piscat.Anomaly.anomaly_localization import BinaryToiSCATLocalization
from piscat.Visualization.display import Display, DisplaySubplot
from piscat.Preproccessing.normalization import Normalization

import os
import numpy as np
import matplotlib.pylab as plt


def create_features_list(video_pn_, RVideo_PN_FPN_, hyperparameters, flags):
    features_list = []
    if flags['M1'] or flags['M2'] or flags['S1'] or flags['S2'] or flags['diff']:
        feature_maps_temporal = CreateFeatures(video=video_pn_)
        out_feature_t = feature_maps_temporal.temporal_features(batchSize=hyperparameters['batch_size'], flag_dc=False)
        if flags['M1']:
            features_list.append(out_feature_t[0])
        if flags['M2']:
            features_list.append(out_feature_t[1])
        if flags['S1']:
            features_list.append(out_feature_t[2])
        if flags['S2']:
            features_list.append(out_feature_t[3])
        if flags['diff']:
            features_list.append(out_feature_t[4])

    if flags['M1-M12'] or flags['M2-M12']:
        feature_maps_temporal = CreateFeatures(video=video_pn_)
        out_feature_t_dc = feature_maps_temporal.temporal_features(batchSize=hyperparameters['batch_size'],
                                                                   flag_dc=True)
        if flags['M1-M12']:
            features_list.append(out_feature_t_dc[0])
        if flags['M2-M12']:
            features_list.append(out_feature_t_dc[1])

    if flags['dog']:
        feature_maps_spatio = CreateFeatures(video=RVideo_PN_FPN_)
        dog_features = feature_maps_spatio.dog2D_creater(
            low_sigma=[hyperparameters['dog_min_sigmaX'], hyperparameters['dog_min_sigmaY']],
            high_sigma=[hyperparameters['dog_max_sigmaX'], hyperparameters['dog_max_sigmaY']])

        features_list.append(dog_features)
    if flags['rvt']:
        rvt_ = filtering.RadialVarianceTransform(inter_flag_parallel_active=True)
        filtered_video = rvt_.rvt_video(video=RVideo_PN_FPN_, rmin=hyperparameters['rmin'],
                                        rmax=hyperparameters['rmax'],
                                        kind=hyperparameters['rvt_kind'],
                                        highpass_size=hyperparameters['rvt_highpass_size'],
                                        upsample=hyperparameters['rvt_upsample'],
                                        rweights=hyperparameters['rvt_rweights'],
                                        coarse_factor=hyperparameters['rvt_factor'],
                                        coarse_mode=hyperparameters['rvt_coarse_mode'],
                                        pad_mode=hyperparameters['rvt_pad_mode'])

        features_list.append(filtered_video)
    if flags['dra']:
        features_list.append(RVideo_PN_FPN_)
    return features_list


def anomaly_protein_analysis(paths, video_names, hyperparameters, flags, name_mkdir):
    """
    This function analyzes several videos based on the user's settings in the "hyperparameters" and "flags" for
    extracting spatiotemporal handcraft features matrix and feeding to isolation forest to extract protein contrasts.

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
                                    ##### Setting for extracting feature matrix and anomaly ######
                                    'dog_min_sigmaX': 1.7, 'dog_min_sigmaY': 1.7, 'dog_max_sigmaX': 1.8,
                                    'dog_max_sigmaY': 1.8,'rvt_kind': 'basic', 'rmin': 2, 'rmax': 4,
                                    'rvt_highpass_size': None, 'rvt_upsample': 1, 'rvt_rweights': None,
                                    'rvt_factor': 1, 'rvt_coarse_mode': 'add', 'rvt_pad_mode': 'constant',
                                    'downsampling': 1, 'anomaly_method': 'IsolationForest', 'contamination':0.003,
                                    'morphological_threshold': 4, 'experimental_PSF_size_(pixel)': 1.7,
                                    ### Localization and Linking method ###
                                    'Mode_PSF_Segmentation': 'BOTH', 'center_int_method': 'dog',
                                    'localization_dog_min_sigma':1.7, 'localization_dog_max_sigma':1.8,
                                    'localization_dog_sigma_ratio': 1.1, 'localization_dog_PSF_detection_thr': 1e-5,
                                    'localization_dog_overlap': 0, 'localization_window_scale': 5,
                                    'search_range': 4, 'memory': 20,
                                    ### Spatio_themporal filters ###
                                    'outlier_frames_thr': 10, 'symmetric_PSFs_thr': 0.6, 'min_V_shape_width': 700}

    flags: dic
         The dictionary is used to active/deactivate different parts in analyzing pipelines. In the following you can see the example of this dictionary:

            | flags = { ### feature selection flags ###
                        'M1': True, 'M2': True, 'S1': False, 'S2': False, 'diff': True, 'M1-M12': False, 'M2-M12': False,
                        'dog': True, 'rvt': False, 'dra': True,
                        ### Background correction flags ###
                        'PN': True, 'FPNc': True, 'DRA': True, 'filter_hotPixels': True, 'FFT_flag': True,
                        ### iPSF filter flags ###
                        'outlier_frames_filter': True, 'Dense_Filter': False, 'symmetric_PSFs_Filter': False,
                        'Side_lobes_Filter': False}

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

        if isinstance(hyperparameters['start_fr'], int) and isinstance(hyperparameters['end_fr'], int):
            video = reading_videos.read_binary(file_name=p_ + '/' + n_,
                                               img_width=hyperparameters['im_size_x'], img_height=hyperparameters['im_size_y'],
                                               s_frame=hyperparameters['start_fr'], e_frame=hyperparameters['end_fr'],
                                               image_type=np.dtype(hyperparameters['image_format']))
        else:
            video = reading_videos.read_binary(file_name=p_ + '/' + n_,
                                               img_width=hyperparameters['im_size_x'],
                                               img_height=hyperparameters['im_size_y'],
                                               image_type=np.dtype(hyperparameters['image_format']))

        status_ = read_status_line.StatusLine(video)
        video, status_info = status_.find_status_line()

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
            video_pn_ = filtering.Filters(video_pn).median(3)
        else:
            RVideo_PN_FPN = RVideo_PN_FPN_

        PSFs_Particels_num['#Totall_frame_num_DRA'] = RVideo_PN_FPN.shape[0]

        features_list = create_features_list(video_pn_=video_pn_, RVideo_PN_FPN_=RVideo_PN_FPN,
                             hyperparameters=hyperparameters, flags=flags)

        anomaly_st = SpatioTemporalAnomalyDetection(features_list)
        binary_st, _ = anomaly_st.fun_anomaly(scale=hyperparameters['downsampling'],
                                                           method=hyperparameters['anomaly_method'],
                                                           contamination=hyperparameters['contamination'])

        PSFs_Particels_num['#Totall_frame_num_anomaly_mask'] = binary_st.shape[0]

        if binary_st.shape != RVideo_PN_FPN:
            dim_x = min(binary_st.shape[1], RVideo_PN_FPN.shape[1])
            dim_y = min(binary_st.shape[2], RVideo_PN_FPN.shape[2])
            result_anomaly = binary_st[:, 0:dim_x, 0:dim_y]
        else:
            result_anomaly = binary_st

        result_anomaly_ = result_anomaly.copy()
        result_anomaly_[result_anomaly == True] = 1
        result_anomaly_[result_anomaly == False] = 0
        result_anomaly_ = Normalization(video=result_anomaly_.astype(int)).normalized_image_specific()

        write_video.write_binary(dir_path=s_dir_, file_name='type_' + str(result_anomaly_.dtype) + '_shape' + str(result_anomaly_.shape) + '.raw', data=result_anomaly_)

        binery_localization = BinaryToiSCATLocalization(video_binary=result_anomaly, video_iSCAT=RVideo_PN_FPN,
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
                                                              scale=hyperparameters['localization_window_scale'],)

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
