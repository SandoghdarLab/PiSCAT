import pytest
import pathlib
import os
import wget
import zipfile
import numpy as np

from piscat.InputOutput import video_reader

DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'

@pytest.fixture
def control_video_path():

    video_path = DATA_PATH / 'Control' / 'control_4999_128_128_uint16_2.33FPS.raw'
    zip_file = DATA_PATH / 'Control.zip'
    if not os.path.exists(video_path):
        zip_file = wget.download('https://owncloud.gwdg.de/index.php/s/tzRZ7ytBd1weNDl/download')
        zip = zipfile.ZipFile(zip_file)
        zip.extractall(DATA_PATH)
        zip.close()
    if os.path.exists(zip_file):
        os.remove(zip_file)

    return video_path


@pytest.fixture()
def control_video(control_video_path):
    return video_reader(file_name=control_video_path, type='binary', img_width=128, img_height=128,
                                        image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)
