import cv2 as cv
import numpy as np

from skimage import exposure


class ContrastAdjustment():

    def __init__(self, video):
        """
        This class is used in GUI to adjust the brightness and contrast of visualization

        Parameters
        ----------
        video: NDArray
         Input video
        """
        self.video = video

    def pixel_transforms(self, image, alpha, beta, min_intensity, max_intensity):
        """
        Adjust the brightness and contrast of the current image based on the value of hyperparameters.

        Parameters
        ----------
        image: NDArray

        alpha: float

        beta: float

        min_intensity: float

        max_intensity: float

        Returns
        -------
        image_: NDArray
        """
        video_rescale = exposure.rescale_intensity(image, out_range=(min_intensity, max_intensity))
        image_ = cv.convertScaleAbs(video_rescale, alpha=alpha, beta=beta)
        return image_

    def auto_pixelTransforms(self, image):
        """
        Adjusting the brightness and contrast of the current image automatically.

        Parameters
        ----------
        image: NDArray

        alpha: float

        beta: float

        min_intensity: float

        max_intensity: float

        Returns
        -------
        @return NDArray
        """
        min_intensity = image.min()
        max_intensity = image.max()
        alpha = 255 / (max_intensity - min_intensity)
        beta = -min_intensity * alpha
        return self.pixel_transforms(image, alpha, beta, min_intensity, max_intensity)


