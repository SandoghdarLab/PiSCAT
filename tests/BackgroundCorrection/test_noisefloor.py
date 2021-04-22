#
# import numpy as np
# import time
# from piscat.BackgroundCorrection import NoiseFloor
# from piscat.Preproccessing import Normalization
# from piscat.InputOutput import StatusLine
#
# from ..Fixtures.shared import *
#
# def _test_noisefloor_serial_vs_parallel(control_video):
#     status_ = StatusLine(control_video)
#     video_remove_status, status_information = status_.find_status_line()
#
#     video_pn, _ = Normalization(video=video_remove_status).power_normalized()
#     l_range = list(range(30, 150, 30))
#     start = time.time()
#     noise_floor_DRA_pn_parallel = NoiseFloor(video_pn, list_range=l_range, inter_flag_parallel_active=True)
#     stop = time.time()
#     dt_parallel = stop - start
#     start = time.time()
#     noise_floor_DRA_pn_serial = NoiseFloor(video_pn, list_range=l_range, inter_flag_parallel_active=False)
#     stop = time.time()
#     dt_serial = stop - start
#
#     assert dt_serial > dt_parallel
#
#     assert (noise_floor_DRA_pn_serial.mean >= 0).all()
#     np.testing.assert_array_almost_equal(noise_floor_DRA_pn_parallel.mean, noise_floor_DRA_pn_serial.mean)
#
#
# def test_noisefloor_results(control_video):
#     status_ = StatusLine(control_video)
#     video_remove_status, status_information = status_.find_status_line()
#
#     video_pn, _ = Normalization(video=video_remove_status).power_normalized()
#     l_range = list(range(90, 150, 30))
#     noise_floor_DRA_pn = NoiseFloor(video_pn, list_range=l_range, inter_flag_parallel_active=False)
#     assert np.nanargmin(noise_floor_DRA_pn.mean) == 1
#     assert np.min(noise_floor_DRA_pn.mean) == pytest.approx(2.7E-4, rel=0.1)
