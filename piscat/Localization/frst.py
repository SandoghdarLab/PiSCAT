from __future__ import print_function
import numpy as np
import math

from math import sqrt
from scipy import spatial
from skimage.feature import peak_local_max
from skimage import img_as_float
from piscat.Preproccessing.filtering import FastRadialSymmetryTransform


def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
      Distance between centers.

    r1 : float
      Radius of the first disk.

    r2 : float
      Radius of the second disk.

    Returns
    -------
    fraction: float
      Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
          0.5 * sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.

    r1 : float
        Radius of the first sphere.

    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.

    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d) ** 2 *
           (d ** 2 + 2 * d * (r1 + r2) - 3 * (r1 ** 2 + r2 ** 2) + 6 * r1 * r2))
    return vol / (4. / 3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area.

    Parameters
    ----------
    blob1 : sequence of arrays
      A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
      where ``row, col`` (or ``(pln, row, col)``) are coordinates
      of blob and ``sigma`` is the standard deviation of the Gaussian kernel
      which detected the blob.

    blob2 : sequence of arrays
      A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
      where ``row, col`` (or ``(pln, row, col)``) are coordinates
      of blob and ``sigma`` is the standard deviation of the Gaussian kernel
      which detected the blob.

    Returns
    -------
    f : float
      Fraction of overlapped area (or volume in 3D).
    """
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)

    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))
    if d > r1 + r2:
        return 0

    if d <= abs(r1 - r2):
        return 1

    if n_dim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:
        return _compute_sphere_overlap(d, r1, r2)


def _prune_blobs(blobs_array, overlap):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : NDArray
      A 2d array with each row representing 3 (or 4) values,
      ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
      where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
      and ``sigma`` is the standard deviation of the Gaussian kernel which
      detected the blob.
      This array must not have a dimension of size 0.

    overlap : float
      A value between 0 and 1. If the fraction of area overlapping for 2
      blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A :(NDArray
      `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
          blob1, blob2 = blobs_array[i], blobs_array[j]
          if _blob_overlap(blob1, blob2) > overlap:
            if blob1[-1] > blob2[-1]:
              blob2[-1] = 0
            else:
              blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])


def blob_frst(image, min_radial=1, max_radial=50, radial_step=1.6, threshold=2.0, alpha=2, beta=1e-3, stdFactor=4,
              mode='BOTH', overlap=.5, *, exclude_border=False):
    """
    This function uses scikit-image's local maximum function to localize the PSFs using a ``frst``.

    Parameters
    ----------
    img: NDArray
       Input_video image, grayscale.

    alpha: float
       Strictness of symmetry transform (higher=more strict; 2 is good place to start)

    beta: float
        Gradient threshold parameter, float in [0,1]

    stdFactor: float
       Standard deviation factor for gaussian kernel

    mode: str
       BRIGHT, DARK, or BOTH

    min_radial: int
       integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

    max_radial: int
       integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

    threshold: float
        The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this
        to detect blobs with less intensities.

    overlap: float
        A value between 0 and 1. If the area of two blobs are overlapping by a fraction greater than threshold, smaller blobs are eliminated.

    exclude_border: int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.

    Returns
    -------
    local_maxima_: list
        [y, x, sigma]

    """

    image = img_as_float(image)

    # Gaussian filter requires that sequence-bin_type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_radial):
      max_radial = np.full(image.ndim, max_radial, dtype=float)
    if np.isscalar(min_radial):
      min_radial = np.full(image.ndim, min_radial, dtype=float)

    # Convert sequence types to array
    min_radial = np.asarray(min_radial, dtype=float)
    max_radial = np.asarray(max_radial, dtype=float)

    # a geometric progression of standard deviations for gaussian kernels
    radial_list = np.array([radial for radial in range(int(min_radial[0]), int(max_radial[0] + 1), int(radial_step))])

    frst = FastRadialSymmetryTransform()
    frst_images = [frst.frst(image, radii=r, alpha=alpha, beta=beta, stdFactor=stdFactor, mode=mode) for r in radial_list]

    image_cube = np.stack(frst_images, axis=-1)
    image_cube = np.power(image_cube, 2)
    # image_cube = np.std(image_cube, axis=2)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)
    # Catch no peaks
    if local_maxima.size == 0:
      return np.empty((0, 3))

    local_maxima_ = []

    for idx, l_m in enumerate(local_maxima):
        local_maxima_.append([l_m[0], l_m[1], radial_list[l_m[2]]])

    return local_maxima_



