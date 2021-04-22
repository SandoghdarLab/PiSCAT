
import pickle

from piscat.BackgroundCorrection import DifferentialRollingAverage

from ..Fixtures.shared import *


def test_DifferentialRollingAverage_differential_rolling_defaults(control_video):

    # Arrange
    control_video = control_video[:2000, :, :]
    orig_video = control_video.copy()

    with open(DATA_PATH / 'differential_rolling.pck', 'rb') as f:
        expected_out_first = pickle.load(f)
        expected_out_last = pickle.load(f)
        expected_gain_first = pickle.load(f)
        expected_gain_last = pickle.load(f)

    DRA_PN = DifferentialRollingAverage(video=control_video)

    # Act
    output = DRA_PN.differential_rolling()

    # Assert

    # Test for side effects
    np.testing.assert_array_equal(orig_video, control_video)

    # Test for stable results
    np.testing.assert_array_almost_equal(output[0, ...], expected_out_first)
    np.testing.assert_array_almost_equal(output[-1, ...], expected_out_last)


def test_DifferentialRollingAverage_differential_rolling_FFT_flag(control_video):

    # Arrange
    control_video = control_video[:2000, :, :]

    expected = _load_expected(DATA_PATH / 'differential_rolling_FFT_flag.pck')

    DRA_PN = DifferentialRollingAverage(video=control_video)

    # Act
    output = DRA_PN.differential_rolling(FFT_flag=True)

    _compare(output, expected)

def _compare(output, expected):
    np.testing.assert_array_almost_equal(output[0, ...], expected[0])
    np.testing.assert_array_almost_equal(output[-1, ...], expected[1])


def _load_expected(file_name):
    with open(file_name, 'rb') as f:
        expected_out_first, expected_out_last, \
        expected_gain_first, expected_gain_last = pickle.load(f)
    return expected_out_first, expected_out_last, expected_gain_first, expected_gain_last
