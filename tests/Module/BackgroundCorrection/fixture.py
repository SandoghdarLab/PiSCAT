from piscat.BackgroundCorrection.DRA import *
from piscat.InputOutput.reading_videos import video_reader
import unittest
import os
import numpy as np
import pickle


def save_fixture(obj, FFT_flag, FPNc_flag, select_correction_axis, filename):
    output_video, _ = obj.differential_rolling(FFT_flag=FFT_flag, FPN_flag=FPNc_flag, select_correction_axis=select_correction_axis)
    with open(filename, 'wb') as file:
        pickle.dump(output_video, file)


if __name__ == "__main__":
    current_path = os.path.abspath(os.path.join('..'))
    path = os.path.join(current_path, 'TestData/Video/')
    file_name = 'control_4999_128_128_uint16_2.33FPS.raw'
    file_path = os.path.join(path, file_name)
    video = video_reader(file_name=file_path, type='binary', img_width=128, img_height=128,
                         image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)
    video = video[0:50, :, :]
    test_obj = DifferentialRollingAverage(video, batchSize=10)
    file_name_save = os.path.join(path, 'test_differential_rolling.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=False, select_correction_axis=1, filename=file_name_save)

    test_obj.mode_FPN = 'mFPN'
    file_name_save = os.path.join(path, 'test_differential_rolling_mFPN_column.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=1, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_mFPN_row.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=0, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_mFPN_both.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis='Both', filename=file_name_save)

    test_obj.mode_FPN = 'cpFPN'
    file_name_save = os.path.join(path, 'test_differential_rolling_cpFPN_column.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=1, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_cpFPN_row.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=0, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_cpFPN_both.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis='Both', filename=file_name_save)

    test_obj.mode_FPN = 'fFPN'
    file_name_save = os.path.join(path, 'test_differential_rolling_fFPN_column.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=1, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_fFPN_row.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=0, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_fFPN_both.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis='Both', filename=file_name_save)

    test_obj.mode_FPN = 'wFPN'
    file_name_save = os.path.join(path, 'test_differential_rolling_wFPN_column.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=1, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_wFPN_row.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis=0, filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_wFPN_both.pck')
    save_fixture(test_obj, FFT_flag=False, FPNc_flag=True, select_correction_axis='Both', filename=file_name_save)

    file_name_save = os.path.join(path, 'test_differential_rolling_FFT_flag.pck')
    save_fixture(test_obj, FFT_flag=True, FPNc_flag=False, select_correction_axis=1, filename=file_name_save)