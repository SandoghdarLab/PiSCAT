import wget
import pathlib
import zipfile
import os

import numpy as np

import pytest
import pickle

from piscat.InputOutput import StatusLine, DirectoryType, reading_videos, video_reader
from piscat.Visualization import JupyterDisplay_StatusLine, JupyterDisplay
from piscat.Preproccessing import Normalization
from piscat.BackgroundCorrection import DifferentialRollingAverage, NoiseFloor

data_path = pathlib.Path(__file__).parent / 'data'


@pytest.fixture
def control_video_path():

    video_path =  data_path / 'Control' / 'control_4999_128_128_uint16_2.33FPS.raw'
    zip_file = data_path / 'Control.zip'
    if not os.path.exists(video_path):
        zip_file = wget.download('https://owncloud.gwdg.de/index.php/s/tzRZ7ytBd1weNDl/download')
        zip = zipfile.ZipFile(zip_file)
        zip.extractall(data_path)
        zip.close()
    if os.path.exists(zip_file):
        os.remove(zip_file)

    return video_path


@pytest.fixture()
def control_video(control_video_path):
    return reading_videos.video_reader(file_name=control_video_path, type='binary', img_width=128, img_height=128,
                                        image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)


def test_can_read_raw_video(control_video_path):

    df_video = DirectoryType(data_path / 'Control', type_file='raw').return_df()
    video = video_reader(file_name=control_video_path, type='binary', img_width=128, img_height=128,
                                        image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)

    assert len(df_video) == 1
    assert isinstance(video, np.ndarray)
    assert video.shape == (4999, 128, 128)


def test_statusline(control_video):
    assert JupyterDisplay_StatusLine(control_video, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5,
                              IntSlider_width='500px',
                              step=1)


def test_normalization(control_video):
    video_pn, power_fluctuation = Normalization(video=control_video).power_normalized()

    assert (video_pn >= 0).all()
    assert np.max(video_pn) == pytest.approx(3611, 1)
    assert power_fluctuation.shape == (4999, )
    assert video_pn.shape == (4999, 128, 128)


def test_background_correction_dra(control_video):

    DRA_PN = DifferentialRollingAverage(video=control_video, batchSize=200,)
    RVideo_PN_ = DRA_PN.differential_rolling(FFT_flag=False)

    with open(data_path / 'expected_dra_result_1.pck', 'rb') as f:
        expected = pickle.load(f)

    np.testing.assert_array_almost_equal(RVideo_PN_[0, ...], expected)
    assert len(control_video) > len(RVideo_PN_)


def _test_tutorial2(control_video):
    control_video = control_video[:500, ...]
    status_ = StatusLine(control_video)  # Reading the status line
    video_remove_status, status_information = status_.find_status_line()  # Examining the status line & removing it

    # Normalization of the power in the frames of a video
    video_pn, _ = Normalization(video=video_remove_status).power_normalized()

    DRA_PN = DifferentialRollingAverage(video=video_pn, batchSize=200)
    RVideo_PN_, _ = DRA_PN.differential_rolling(FFT_flag=False)
    disp = JupyterDisplay(RVideo_PN_, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=100)
    l_range = list(range(30, 200, 30))
    noise_floor_DRA_pn = NoiseFloor(video_pn, list_range=l_range, n_jobs=1)
