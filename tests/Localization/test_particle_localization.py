import numpy as np
import time
from piscat.BackgroundCorrection import NoiseFloor, DifferentialRollingAverage
from piscat.Preproccessing import Normalization
from piscat.InputOutput import StatusLine
from piscat.Localization import PSFsExtraction

from ..Fixtures.shared import *


def test_PSFsExtraction_non_jupyter(control_video):
    status_ = StatusLine(control_video)  # Reading the status line
    video_remove_status, status_information = status_.find_status_line()  # Examining the status line & removing it

    video_pn, _ = Normalization(video=video_remove_status).power_normalized()
    DRA_PN_cpFPNc = DifferentialRollingAverage(video=video_pn, batchSize=120)
    RVideo_PN_cpFPNc = DRA_PN_cpFPNc.differential_rolling(FFT_flag=False)
    PSF_l = PSFsExtraction(video=RVideo_PN_cpFPNc)
    PSFs = PSF_l.psf_detection_preview(function='dog',
                                min_sigma=1.6, max_sigma=1.7, sigma_ratio=1.1, threshold=5e-5,
                                overlap=0, mode='BOTH', frame_number=[1000], IntSlider_width='400px',
                                       title='Localization threshold on cpFPNc')

    assert len(PSFs) == 15
