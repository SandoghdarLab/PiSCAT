import cv2 as cv
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from piscat.InputOutput.cpu_configurations import CPUConfigurations


class Mask2Video:
    def __init__(self, video, mask, inter_flag_parallel_active=True):
        """
        This class produces a masked version of the input video.

        Parameters
        ----------
        video: NDArray
            The video is 3D-numpy (number of frames, width, height).

        mask: NDArray
            The mask is 2D-numpy with binary values and the same dimension as the input video.

         internal_parallel_flag: bool
            Internal flag for activating parallel computation. Default is True!
        """

        self.video = video
        self.cpu = CPUConfigurations()
        self.inter_flag_parallel_active = inter_flag_parallel_active
        if mask is not None:
            if mask.ndim == 2:
                print("--- Same mask is used for all frame! ---")
                self.mask_repeater(mask)

            elif video.shape == mask.shape:
                print("--- Mask has same shape with video! ---")
                self.mask = mask
            else:
                print("--- Mask does not have same shape with video! ---")
        else:
            print("--- Mask is not define! ---")

    def apply_mask(self, flag_nan=True):
        """
        This method is used to apply a mask on a video. The masked values might be defined as nan, or they can be chosen from the median values of each frame.

        Parameters
        ----------
        flag_nan: bool
           A flag that defined how masked values are chosen.

        Returns
        -------
        mask_video_: NDArray
            The video after applying the mask.

        """
        mask_video = []
        if self.inter_flag_parallel_active and self.cpu.parallel_active:
            print("\n---apply mask with Parallel---")
            mask_video = Parallel(
                n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose
            )(delayed(self.apply_mask_kernel)(f_) for f_ in tqdm(range(self.video.shape[0])))
            mask_video_ = np.asarray(mask_video)

        else:
            print("\n---apply mask without Parallel---")
            for f_ in tqdm(range(self.video.shape[0])):
                img_bg = cv.bitwise_and(self.video[f_], self.video[f_], mask=self.mask[f_])
                mask_video.append(img_bg)
            mask_video_ = np.asarray(mask_video)
        if flag_nan:
            mask_video_[mask_video_ == 0] = np.nan
        else:
            mask_video_ = mask_video_.astype("float64")
            mask_video_[mask_video_ == 0.0] = np.nan

            for f_ in tqdm(range(mask_video_.shape[0])):
                img_mask = mask_video_[f_, ...]
                tmp_median = np.nanmedian(img_mask)
                inds = np.where(np.isnan(img_mask))
                img_mask[inds] = tmp_median
                mask_video_[f_, ...] = img_mask
        return mask_video_

    def apply_mask_kernel(self, f_):
        return cv.bitwise_and(self.video[f_], self.video[f_], mask=self.mask[f_])

    def mask_repeater(self, mask):
        """
        Making a 3D numpy array from a 2D mask.

        Parameters
        ----------
        mask: NDArray
            The mask is 2D-numpy with binary values and the same dimension as the input video.
        """
        mask = np.expand_dims(mask, axis=0)
        self.mask = np.repeat(mask, self.video.shape[0], axis=0)

    def mask_generating_circle(self, center=(10, 10), redius=1):
        """
        This method generates a circular mask depending on the rediuse and origin parameters.

        Parameters
        ----------
        center: list
            A list of two integer numbers indicating the mask's center.

        redius: int
            Circular mask redius.

        Returns
        -------
            The mask that has the same shape as the input video.
        """
        mask = np.zeros(self.video[0].shape, dtype="uint8")
        cv.circle(mask, center, redius, 1, -1)
        self.mask_repeater(mask)
        return self.mask
