from __future__ import print_function

from scipy import optimize
from matplotlib.patches import Ellipse

import numpy as np
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y, b):
    (x, y) = xy_mesh

    # make the 2D Gaussian matrix
    gauss = b + amp * np.exp(-(((x - xc) ** 2 / ((sigma_x ** 2))) + ((y - yc) ** 2 / ((sigma_y ** 2)))))

    # flatten the 2D Gaussian down to 1D
    return np.ravel(gauss)


def fit_2D_Gaussian_varAmp(image, sigma_x, sigma_y, display_flag=False):
    """
    This function uses non-linear squares to fit 2D Gaussian.

    Parameters
    ----------
    image: NDArray
        2D numpy array, image.

    sigma_x: float
        It is initial value for sigma X.

    sigma_y: float
        It is initial value for sigma y.

    display_flag: bool
        This flag is used to display the result of fitting for each PSF.

    Returns
    -------
    @return: (list)
            [sigma_ratio, fit_params, fit_errors]
    """
    x = np.linspace(0, image.shape[0] - 1, image.shape[0], dtype=np.int)
    y = np.linspace(0, image.shape[1] - 1, image.shape[1], dtype=np.int)
    xy_mesh = np.meshgrid(y, x)

    data = np.transpose(image)
    data = np.reshape(data, (data.shape[0] * data.shape[1], 1))
    try:
        if (np.median(data) - np.min(data)) > (np.max(data) - np.median(data)):
            i_amp = - (np.median(data) - np.min(data))
        else:
            i_amp = np.max(data) - np.median(data)
    except ValueError:
        i_amp = 1

    amp = i_amp
    b = np.median(data)
    xc = int(image.shape[1] / 2)
    yc = int(image.shape[0] / 2)
    sigma_x = sigma_x
    sigma_y = sigma_y
    guess_vals = [amp, xc, yc, sigma_x, sigma_y, b]
    try:
        fit_params, cov_mat = optimize.curve_fit(gaussian_2d, xy_mesh, np.ravel(image), p0=guess_vals, maxfev=5000)
        fit_errors = np.sqrt(np.diag(cov_mat))

        fit_residual = image - gaussian_2d(xy_mesh, *fit_params).reshape(np.outer(x, y).shape)
        fit_Rsquared = 1 - np.var(fit_residual) / np.var(image)

        if display_flag is True:

            print('Fit R-squared:', fit_Rsquared, '\n')
            print('Fit Amplitude:', fit_params[0], '\u00b1', fit_errors[0])
            print('Fit X-Center: ', fit_params[1], '\u00b1', fit_errors[1])
            print('Fit Y-Center: ', fit_params[2], '\u00b1', fit_errors[2])
            print('Fit X-Sigma:  ', fit_params[3], '\u00b1', fit_errors[3])
            print('Fit Y-Sigma:  ', fit_params[4], '\u00b1', fit_errors[4])
            print('Fit Bias:  ', fit_params[5], '\u00b1', fit_errors[5])

            plt.figure()
            plt.imshow(image)
            # draw the ellipse
            ax = plt.gca()
            ax.add_patch(Ellipse((fit_params[2], fit_params[1]), width=math.sqrt(2) * fit_params[3],
                                 height=math.sqrt(2) * fit_params[4],
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=5))

            if abs(fit_params[3]) < abs(fit_params[4]):
                sigma_ratio = abs(fit_params[3] / fit_params[4])
                print('X-Sigma/Y-Sigma:  ', fit_params[3] / fit_params[4])
            else:
                sigma_ratio = abs(fit_params[4] / fit_params[3])
                print('Y-Sigma/X-Sigma:  ', fit_params[4] / fit_params[3])

        else:
            if abs(fit_params[3]) < abs(fit_params[4]):
                sigma_ratio = abs(fit_params[3] / fit_params[4])
            else:
                sigma_ratio = abs(fit_params[4] / fit_params[3])
    except:
        sigma_ratio = 1
        fit_params = None
        fit_errors = None
    return [sigma_ratio, fit_params, fit_errors]
