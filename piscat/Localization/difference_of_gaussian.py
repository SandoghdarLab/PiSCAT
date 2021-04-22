import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import math
from math import sqrt, log
from scipy import spatial
from skimage.util import img_as_float


def dog_preview(images, min_sigma, max_sigma, sigma_ratio):

    min_range = []
    max_range = []

    for idx_ in range(images.shape[0]):
        image = images[idx_, ...]
        image = img_as_float(image)

        # if both min and max sigma are scalar, function returns only one sigma
        scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

        # Gaussian filter requires that sequence-bin_type sigmas have same
        # dimensionality as image. This broadcasts scalar kernels
        if np.isscalar(max_sigma):
            max_sigma = np.full(image.ndim, max_sigma, dtype=float)
        if np.isscalar(min_sigma):
            min_sigma = np.full(image.ndim, min_sigma, dtype=float)

        # Convert sequence types to array
        min_sigma = np.asarray(min_sigma, dtype=float)
        max_sigma = np.asarray(max_sigma, dtype=float)

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

        # a geometric progression of standard deviations for gaussian kernels
        sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                               for i in range(k + 1)])

        gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

        # computing difference between two successive Gaussian blurred images
        # multiplying with average standard deviation provides scale invariance
        dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                      * np.mean(sigma_list[i]) for i in range(k)]

        image_cube = np.stack(dog_images, axis=-1)
        min_range.append(image_cube.min())
        max_range.append(image_cube.max())

    min_range_ = np.min(min_range)
    max_range_ = np.max(max_range)

    return min_range_, max_range_