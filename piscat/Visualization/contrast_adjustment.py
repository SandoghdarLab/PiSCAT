import cv2 as cv
import numpy as np

from skimage import exposure


class ContrastAdjustment():

    def __init__(self, video):
        """
        This class is used in the GUI to change the visualization's brightness and contrast.

        Parameters
        ----------
        video: NDArray
            Input video.
        """
        self.video = video

    def pixel_transforms(self, image, alpha, beta, min_intensity, max_intensity):
        """
        Using the value of the hyperparameters, adjust the brightness and contrast of the current image.

        Parameters
        ----------
        image: NDArray
            Input image (2D-Numpy).

        alpha: float
            Scale factor.

        beta: float
            Delta added to the scaled values.

        min_intensity: float
            Min intensity values of output image.

        max_intensity: float
            Max intensity values of output image.

        Returns
        -------
        image_: NDArray
            Output image (2D-Numpy)
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
            Input image (2D-Numpy).

        alpha: float
            Scale factor.

        beta: float
            Delta added to the scaled values.

        min_intensity: float
            Min intensity values of output image.

        max_intensity: float
            Max intensity values of output image.

        Returns
        -------
        image_: NDArray
            Output image (2D-Numpy)
        """
        min_intensity = image.min()
        max_intensity = image.max()
        alpha = 255 / (max_intensity - min_intensity)
        beta = -min_intensity * alpha
        return self.pixel_transforms(image, alpha, beta, min_intensity, max_intensity)


