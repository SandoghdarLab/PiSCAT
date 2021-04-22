import pickle

from ..Fixtures.shared import *
from piscat.Preproccessing import Normalization
import numpy as np


def test_preprocessing_normalization(control_video):

    video_pn, power_fluctuation = Normalization(video=control_video).power_normalized()

    expected = _load_expected(DATA_PATH / 'preprocessing_normalization.pck')

    _compare(video_pn, power_fluctuation, expected)


def test_preprocessing_normalization_parallel_vs_serial(control_video):

    vid_pn_parallel, fluct_parallel = Normalization(video=control_video).power_normalized(inter_flag_parallel_active=True)
    vid_pn_serial, fluct_serial = Normalization(video=control_video).power_normalized(inter_flag_parallel_active=False)

    np.testing.assert_array_almost_equal(vid_pn_serial, vid_pn_parallel)
    np.testing.assert_array_almost_equal(fluct_serial, fluct_parallel)


def _load_expected(file_name):
    with open(file_name, 'rb') as f:
        video_pn_first, video_pn_last, flucts = pickle.load(f)
    return video_pn_first, video_pn_last, flucts

def _compare(video_pn, power_fluctuation, expected):
    np.testing.assert_array_almost_equal(video_pn[0, ...], expected[0])
    np.testing.assert_array_almost_equal(video_pn[-1, ...], expected[1])
    np.testing.assert_array_almost_equal(power_fluctuation, expected[2])