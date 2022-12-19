from piscat.BackgroundCorrection.DRA import *
from piscat.InputOutput.reading_videos import video_reader
from piscat.InputOutput import read_status_line
from piscat.Preproccessing import normalization
from piscat.BackgroundCorrection import DifferentialRollingAverage
from particle_localization_test import psf_detection_preview, psf_detection
from piscat.Localization import particle_localization
from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering

import os
import pickle


current_path = os.path.abspath(os.path.join('..'))


def save_fixture(obj, method, frame_number, mode, filename, flag_preview=True):
    if flag_preview:
        output = psf_detection_preview(obj, method, frame_number=frame_number, mode=mode)
    else:
        output = psf_detection(obj, method, mode=mode)
    with open(filename, 'wb') as file:
        pickle.dump(output, file)


def save_video(video_array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(video_array, file)


def load_remove_status_video(file_path):
    loaded_video = video_reader(file_name=file_path, type='binary', img_width=128, img_height=128,
                                image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)
    loaded_video = loaded_video[0:20, :, :]
    status_ = read_status_line.StatusLine(loaded_video)
    loaded_video, status_information = status_.find_status_line()
    return loaded_video


if __name__ == "__main__":
    path = os.path.join(current_path, 'TestData/Video/')
    file_path = os.path.join(path, '5nm_GNPs_128x128_uint16_3333fps_10Acc.raw')
    video = load_remove_status_video(file_path)
    file_path = os.path.join(path, '00_darkframes_fullFPS.raw')
    loaded_dark_video = load_remove_status_video(file_path)
    mean_dark_frame = np.mean(loaded_dark_video, axis=0)
    video = np.subtract(video, mean_dark_frame)
    video, _ = normalization.Normalization(video=video).power_normalized()
    dra_obj = DifferentialRollingAverage(video=video, batchSize=3)
    video_dra, _ = dra_obj.differential_rolling(FFT_flag=False)
    file_name_save = os.path.join(path, 'test_localization_input_video.pck')
    save_video(video_dra, file_name_save)
    file_name_save = os.path.join(path, 'test_localization_input_video_doh.pck')
    save_video(video, file_name_save)
    test_obj = particle_localization.PSFsExtraction(video=video_dra)
    test_obj_doh = particle_localization.PSFsExtraction(video=video)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_dog_video_both.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='BOTH', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_dog_video_Bright.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='Bright', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_dog_video_Dark.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='Dark', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_doh_video.pck')
    save_fixture(test_obj_doh, 'doh', frame_number=[0], mode='BOTH', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_log_video_both.pck')
    save_fixture(test_obj, 'log', frame_number=[0], mode='BOTH', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_log_video_Bright.pck')
    save_fixture(test_obj, 'log', frame_number=[0], mode='Bright', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_log_video_Dark.pck')
    save_fixture(test_obj, 'log', frame_number=[0], mode='Dark', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_preview_frst_one_psf_video.pck')
    save_fixture(test_obj, 'frst_one_psf', frame_number=[0], mode='BOTH', filename=file_name_save)

    temporary = test_obj.video
    test_obj.video = np.repeat(test_obj.video, 2, axis=1)
    file_name_save = os.path.join(path, 'test_psf_detection_preview_frst_one_psf_repeat_video.pck')
    save_fixture(test_obj, 'frst_one_psf', frame_number=[0], mode='BOTH', filename=file_name_save)
    test_obj.video = temporary

    file_name_save = os.path.join(path, 'test_psf_detection_preview_RVT_video.pck')
    save_fixture(test_obj_doh, 'RVT', frame_number=[0], mode='BOTH', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_psf_detection_dog_video_both.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='BOTH', filename=file_name_save, flag_preview=False)

    file_name_save = os.path.join(path, 'test_psf_detection_dog_video_Bright.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='Bright', filename=file_name_save, flag_preview=False)

    file_name_save = os.path.join(path, 'test_psf_detection_dog_video_Dark.pck')
    save_fixture(test_obj, 'dog', frame_number=[0], mode='Dark', filename=file_name_save, flag_preview=False)

    file_name_save = os.path.join(path, 'test_fit_Gaussian2D_wrapper.pck')
    psf_dataframe_p = psf_detection(test_obj, 'dog', mode='Bright')
    psf_dataframe_gaussian_p = test_obj.fit_Gaussian2D_wrapper(PSF_List=psf_dataframe_p, scale=5,
                                                               internal_parallel_flag=True)
    with open(file_name_save, 'wb') as file:
        pickle.dump(psf_dataframe_gaussian_p, file)

    linking_ = Linking()
    linked_PSFs = linking_.create_link(psf_position=psf_dataframe_gaussian_p, search_range=2, memory=10)
    linked_PSFs = linked_PSFs.reset_index()
    test_obj = localization_filtering.SpatialFilter()
    filtered_psf = test_obj.outlier_frames(linked_PSFs, threshold=20)
    file_name_save = os.path.join(path, 'test_outlier_frames.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(filtered_psf, file)

    filtered_psf = test_obj.dense_PSFs(linked_PSFs, threshold=1)
    file_name_save = os.path.join(path, 'test_dense_PSFs.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(filtered_psf, file)

    filtered_psf = test_obj.symmetric_PSFs(linked_PSFs, threshold=0.7)
    file_name_save = os.path.join(path, 'symmetric_PSFs.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(filtered_psf, file)


