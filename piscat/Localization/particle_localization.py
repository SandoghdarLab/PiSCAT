import warnings
import numpy as np
import pandas as pd
import os
from PySide2.QtCore import Slot
from joblib import Parallel, delayed
from skimage import feature
from tqdm.autonotebook import tqdm
from ipywidgets import widgets
from ipywidgets import Layout, interact
from skimage.feature import peak_local_max

from piscat.Preproccessing import normalization
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Localization import data_handeling, frst
from piscat.Localization import gaussian_2D_fit
from piscat.Preproccessing.filtering import RadialVarianceTransform
from piscat.Localization import radial_symmetry_centering
from piscat.Visualization.display_jupyter import JupyterPSFs_localizationPreviewDisplay
from piscat.Localization.difference_of_gaussian import dog_preview


class PSFsExtraction():

    def __init__(self, video, flag_transform=False, flag_GUI=False, **kwargs):

        """
        This class employs a variety of PSF localization methods, including DoG/LoG/DoH/RS/RVT.

        It returns a list containing the following details:

        |[[frame number, center y, center x, sigma], [frame number, center y, center x, sigma], ...]

        Parameters
        ----------
        video: NDArray
            Numpy 3D video array.

        flag_transform: bool
            In case it is defined as true, the input video is already transformed.
            So it does not need to run this task during localization.

        flag_GUI: bool
            While the GUI is calling this class, it is true.

        """
        super(PSFsExtraction, self).__init__()
        self.cpu = CPUConfigurations()
        self.kwargs = kwargs
        self.counter = 0

        self.flag_GUI = flag_GUI

        self.video = video
        self.flag_transform = flag_transform
        self.norm_vid = None

        self.psf_dog = None
        self.psf_doh = None
        self.psf_log = None
        self.psf_hog = None
        self.psf_frst = None

        self.psf_hog_1D_feature = None

        self.min_sigma = None
        self.max_sigma = None
        self.sigma_ratio = None
        self.threshold = None
        self.overlap = None
        self.radii = None
        self.alpha = None
        self.beta = None
        self.stdFactor = None
        self.mode = None
        self.function = None

        self.df_PSF = None

    @Slot()
    def run(self):
        result = self.psf_detection(**self.kwargs)
        self.signals.result.emit(result)

    def dog(self, image):
        """
        PSF localization using DoG.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        tmp: list
            [y, x, sigma]
        """

        return feature.blob_dog(image, min_sigma=self.min_sigma, max_sigma=self.max_sigma, sigma_ratio=self.sigma_ratio,
                                threshold=self.threshold, overlap=self.overlap, exclude_border=True)

    def doh(self, image):
        """
        PSF localization using DoH.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        tmp: list
            [y, x, sigma]
        """

        tmp = feature.blob_doh(image, min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                               num_sigma=int(self.sigma_ratio),
                               threshold=self.threshold, overlap=self.overlap)
        return tmp

    def log(self, image):
        """
        PSF localization using LoG.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        @returns: list
            [y, x, sigma]
        """

        return feature.blob_log(image, min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                                num_sigma=int(self.sigma_ratio),
                                threshold=self.threshold, overlap=self.overlap, exclude_border=True)

    def frst(self, image):
        """
        PSF localization using frst.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        tmp: list
            [y, x, sigma]

        References
        ----------
            [1] Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for detecting points of interest. Computer Vision, ECCV 2002.
        """

        normaliz_image = normalization.Normalization(image).normalized_image()

        tmp = frst.blob_frst(normaliz_image, min_radial=self.min_radial, max_radial=self.max_radial,
                             radial_step=self.radial_step, threshold=self.threshold, alpha=self.alpha, beta=self.beta,
                             stdFactor=self.stdFactor, mode=self.mode)
        return tmp

    def rvt(self, image):
        """
        PSF localization using RVT.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        tmp: list
            [y, x, sigma]
        """
        if self.flag_transform:
            tr_img = image
        else:
            rvt_ = RadialVarianceTransform()
            tr_img = rvt_.rvt(img=image, rmin=self.min_radial, rmax=self.max_radial, kind=self.rvt_kind, highpass_size=self.highpass_size,
                        upsample=self.upsample, rweights=self.rweights, coarse_factor=self.coarse_factor, coarse_mode=self.coarse_mode,
                        pad_mode=self.pad_mode)

        local_maxima = peak_local_max(
            tr_img,
            threshold_abs=self.threshold,
            footprint=np.ones((3,) * (image.ndim)),
            threshold_rel=0.0,
            exclude_border=True,
        )

        # Catch no peaks
        if local_maxima.size == 0:
            return np.empty((0, 3))

        sigmas = (self.min_radial/np.sqrt(2)) * np.ones((local_maxima.shape[0], 1))
        tmp = np.concatenate((local_maxima, sigmas), axis=1)

        return tmp

    def fit_Gaussian2D_wrapper(self, PSF_List, scale=5, internal_parallel_flag=False):
        """
        PSF localization using fit_Gaussian2D.

        Parameters
        ----------
        PSF_List: pandas dataframe
            The data frame contains PSFs locations( x, y, frame, sigma)

        scale: int
            The ROI around PSFs is defined using this scale, which is based on their sigmas.

        internal_parallel_flag: bool
            Internal flag for activating parallel computation. Default is True!

        Returns
        -------
        df: pandas dataframe
            The data frame contains PSFs locations ( 'y', 'x', 'frame', 'center_intensity', 'sigma', 'Sigma_ratio') and fitting information.
            fit_params is a list include ('Fit_Amplitude', 'Fit_X-Center', 'Fit_Y-Center', 'Fit_X-Sigma', 'Fit_Y-Sigma',
            'Fit_Bias', 'Fit_errors_Amplitude', 'Fit_errors_X-Center', 'Fit_errors_Y-Center', 'Fit_errors_X-Sigma', 'Fit_errors_Y-Sigma', 'Fit_errors_Bias'].
        """

        if type(PSF_List) is list:
            df_PSF = data_handeling.list2dataframe(feature_position=PSF_List, video=self.video)
        elif type(PSF_List) is pd.core.frame.DataFrame:
            df_PSF = PSF_List
        else:
            raise ValueError('PSF_List does not have correct bin_type')

        self.df2numpy = df_PSF.to_numpy()

        if self.cpu.parallel_active and internal_parallel_flag:
            print('\n---Fitting 2D gaussian with parallel loop---')
            list_df = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(
                self.fit_2D_gussian_kernel)(i_, scale) for i_ in tqdm(range(self.df2numpy.shape[0])))

        else:
            print('\n---Fitting 2D gaussian without parallel loop---')
            list_df = []
            for i_ in tqdm(range(self.df2numpy.shape[0])):
                tmp = self.fit_2D_gussian_kernel(i_, scale)
                list_df.append(tmp)

        df2numpy = np.asarray(list_df)

        if df2numpy.shape[0] != 0:
            df = pd.DataFrame(data=df2numpy, columns=['y', 'x', 'frame', 'center_intensity', 'sigma', 'Sigma_ratio',
                                                       'Fit_Amplitude', 'Fit_X-Center', 'Fit_Y-Center', 'Fit_X-Sigma',
                                                       'Fit_Y-Sigma', 'Fit_Bias', 'Fit_errors_Amplitude',
                                                       'Fit_errors_X-Center', 'Fit_errors_Y-Center', 'Fit_errors_X-Sigma',
                                                       'Fit_errors_Y-Sigma', 'Fit_errors_Bias'])
        else:
            df = None
        return df

    def fit_2D_gussian_kernel(self, i_, scale=5, flag_init=False, image=None, start_sigma=[None, None], display_flag=False):

        if flag_init:
            fit_params_ = gaussian_2D_fit.fit_2D_Gaussian_varAmp(image, sigma_x=start_sigma[0],
                                                                 sigma_y=start_sigma[1],
                                                                 display_flag=display_flag)
            return fit_params_
        else:

            sigma_0 = self.df2numpy[i_, 4]
            cen_int = self.df2numpy[i_, 3]
            frame_num = self.df2numpy[i_, 2]
            p_x = self.df2numpy[i_, 1]
            p_y = self.df2numpy[i_, 0]

            window_size = scale * np.sqrt(2) * sigma_0
            start_sigma = sigma_0

            if p_x > window_size and p_y > window_size:

                window_frame = self.video[int(frame_num), int(p_y - window_size) + 1:int(p_y + window_size),
                               int(p_x - window_size) + 1:int(p_x + window_size)]
                w_s = window_size
            else:
                window_size_temp = window_size

                while p_x < window_size_temp or p_y < window_size_temp:
                    window_size_temp = window_size_temp - 2

                window_frame = self.video[int(frame_num),
                               int(p_y - window_size_temp) + 1:int(p_y + window_size_temp),
                               int(p_x - window_size_temp) + 1:int(p_x + window_size_temp)]

                w_s = window_size_temp

            fit_params = gaussian_2D_fit.fit_2D_Gaussian_varAmp(window_frame, sigma_x=start_sigma,
                                                                sigma_y=start_sigma,
                                                                display_flag=display_flag)

            params = [p_y, p_x, frame_num, cen_int, sigma_0]
            if fit_params[0] is not None:
                params.append(fit_params[0])
            else:
                params.append(np.nan)

            if fit_params[1] is not None:
                params.append(fit_params[1][0])
                params.append(fit_params[1][1] + p_x - w_s)
                params.append(fit_params[1][2] + p_y - w_s)
                params.append(fit_params[1][3])
                params.append(fit_params[1][4])
                params.append(fit_params[1][5])
            else:
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)

            if fit_params[2] is not None:
                params.append(fit_params[2][0])
                params.append(fit_params[2][1])
                params.append(fit_params[2][2])
                params.append(fit_params[2][3])
                params.append(fit_params[2][4])
                params.append(fit_params[2][5])
            else:
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
                params.append(np.nan)
            return params

    def frst_one_PSF(self, image):
        """
        The lateral position of PSFs with subpixel resolution is returned by this function.
        Because this function only works when there is only one PSF in the field of view, it
        is typically used after coarse localization to extract fine localization for each PSF.

        Parameters
        ----------
        image: NDArray
            image is an input numpy array.

        Returns
        -------
        @returns: list
            [y, x, sigma]

        References
        ----------
            [1] Parthasarathy, R. Rapid, accurate particle tracking by calculation of radial symmetry centers.
            Nat Methods 9, 724–726 (2012). https://doi.org/10.1038/nmeth.2071
        """
        if image.shape[0] == image.shape[1]:
            xc, yc, sigma = radial_symmetry_centering.RadialCenter().radialcenter(Image=image)

        elif image.shape[0] > image.shape[1]:
            tmp = np.median(image) + np.zeros((image.shape[0], image.shape[0]))
            tmp[:image.shape[0], :image.shape[1]] = image
            offset = image.shape[0] - image.shape[1]
            xc, yc, sigma = radial_symmetry_centering.RadialCenter().radialcenter(Image=tmp)
            xc = xc - offset

        elif image.shape[0] < image.shape[1]:
            tmp = np.median(image) + np.zeros((image.shape[1], image.shape[1]))
            tmp[:image.shape[0], :image.shape[1]] = image
            offset = image.shape[1] - image.shape[0]
            xc, yc, sigma = radial_symmetry_centering.RadialCenter().radialcenter(Image=tmp)
            yc = yc - offset

        return [yc, xc, sigma]

    def improve_localization_with_frst(self, df_PSFs, scale, flag_preview=False):
        """
        It extracts subpixels localization based on initial pixel localization using ``frst_one_PSF`` methods for all detected PSFs.

        Parameters
        ----------
        df_PSFs: pandas dataframe
            The data frame contains PSFs locations( x, y, frame, sigma)

        scale: int
            The ROI around PSFs is defined using this scale, which is based on their sigmas.

        flag_preview: bool
            When the GUI calls these functions, this flag is set as True.

        Returns
        -------
        sub_pixel_localization: pandas dataframe
            The data frame contains subpixels PSFs locations( x, y, frame, sigma)
        """
        sigma = df_PSFs['sigma'].tolist()
        psf_position_x = df_PSFs['x'].tolist()
        psf_position_y = df_PSFs['y'].tolist()
        psf_position_frame = df_PSFs['frame'].tolist()

        if self.cpu.parallel_active and flag_preview is not True:
            print("\n---start improve_localization with parallel loop---")
            try:
                result0 = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                    delayed(self.frst_wrapper)(p_, scale) for p_ in tqdm(zip(psf_position_frame, psf_position_x, psf_position_y, sigma)))

            except:
                warnings.warn("switch backend to threading", DeprecationWarning)
                result0 = Parallel(n_jobs=self.cpu.n_jobs, backend='threading', verbose=self.cpu.verbose)(
                    delayed(self.frst_wrapper)(p_, scale) for p_ in tqdm(zip(psf_position_frame, psf_position_x, psf_position_y, sigma)))

        else:
            print("\n---start improve_localization without parallel loop---")

            result0 = [self.frst_wrapper(p_, scale) for p_ in tqdm(zip(psf_position_frame, psf_position_x, psf_position_y, sigma))]

        tmp2 = [tmp for tmp in result0 if isinstance(tmp, np.ndarray)]  # remove non values
        sub_pixel_localization = data_handeling.list2dataframe(feature_position=tmp2, video=self.video)

        return sub_pixel_localization

    def frst_wrapper(self, p_, scale=4):
        subPixel = None

        f_ = int(p_[0])
        x_ = int(p_[1])
        y_ = int(p_[2])
        sigma_ = p_[3]
        win_size = int(scale * sigma_)

        tmp_x = [0, (x_ - win_size)]
        x_start = np.max(tmp_x)

        tmp_x = [(x_ + win_size), self.video.shape[1]]
        x_end = np.min(tmp_x)

        tmp_y = [0, (y_ - win_size)]
        y_start = np.max(tmp_y)

        tmp_y = [(y_ + win_size), self.video.shape[2]]
        y_end = np.min(tmp_y)

        particle_windows = self.video[f_, y_start:y_end, x_start:x_end]
        if particle_windows.shape[0] == particle_windows.shape[1]:
            new_position = self.frst_one_PSF(particle_windows)

            if (new_position[0] + y_ - win_size) > self.video.shape[1]:
                l_y = None
            else:
                l_y = new_position[0] + y_ - win_size

            if (new_position[1] + x_ - win_size) > self.video.shape[2]:
                l_x = None
            else:
                l_x = new_position[1] + x_ - win_size

            if l_x is not None and l_y is not None:
                tmp = np.expand_dims(np.asarray([f_, l_y, l_x, new_position[2]]), axis=0)

                if subPixel is None:
                    subPixel = tmp
                else:
                    subPixel = np.concatenate((subPixel, tmp), axis=0)

            return subPixel

    def psf_detection(self, function, min_sigma=1, max_sigma=2, sigma_ratio=1.1, threshold=0, overlap=0,
                      min_radial=1, max_radial=2, radial_step=0.1, alpha=2, beta=1, stdFactor=1,
                      rvt_kind="basic",  highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add",
                      pad_mode="constant", mode='BOTH', flag_GUI_=False):
        """
        This function is a wrapper for calling various PSF localization methods.

        Parameters
        ----------
        function: str
            PSF localization algorithm which should be selected  from : (``'dog'``, ``'log'``, ``'doh'``, ``'frst'``,
            ``'frst_one_psf'``, ``'RVT'``)

        mode: str
            Defines which PSFs will be detected (``'BRIGHT'``, ``'DARK'``, or ``'BOTH'``).

        flag_GUI_: bool
            Only is used when GUI calls this function.

        optional_1:
            These parameters are used when ``'dog'``, ``'log'``, ``'doh'`` are defined as function.

            * `min_sigma`: float, list of floats
                The is the minimum standard deviation for the kernel. The lower the value, the smaller blobs can be detected.
                The standard deviations of the filter are given for each axis in sequence or with a single number which is considered for both axis.

            * `max_sigma`: float, list of floats
                The is the maximum standard deviation for the kernel. The higher the value, the bigger blobs can be detected.
                The standard deviations of the filter are given for each axis in sequence or with a single number
                which is considered for both axis.

            * `sigma_ratio`: float
                * The ratio between the standard deviation of Kernels which is used for computing the DoG and LoG.
                * The number of intermediate values of standard deviations between min_sigma and max_sigma for computing the DoH.

            * `threshold`: float
                The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this
                to detect blobs with less intensities.

            * `overlap`: float
                A value between 0 and 1. If the area of two blobs are overlapping by a fraction greater than threshold, smaller blobs are eliminated.

        optional_2:
            These parameters are used when ``'frst'`` is defined as function.

            * `min_radial`: int
                integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

            * `max_radial`: int
                integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

            * `alpha`: str
                Strictness of symmetry transform (higher=more strict; 2 is good place to start)

            * `beta`: float
                gradient threshold parameter, float in range [0,1]

            * `stdFactor`:
                Standard deviation factor for gaussian kernel

        optional_3:
            These parameters are used when ``"RVT"`` is defined as function.

            * `min_radial`:
                minimal radius (inclusive)

            * `max_radial`:
                maximal radius (inclusive)

            * `rvt_kind`:
                either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
                normalized version increases subpixel bias, but it works better at lower SNR

            * `highpass_size`:
                size of the high-pass filter; ``None`` means no filter (effectively, infinite size)

            * `upsample`: int
                integer image upsampling factor;
                `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
                if ``upsample>1``, the resulting image size is multiplied by ``upsample``

            * `rweights`:
                relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
                coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
                coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
                or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

            * `coarse_factor`:
                the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

            * `coarse_mode`:
                the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
                or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

            * `pad_mode`:
                edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
                or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
                ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
                note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

        Returns
        -------
        df_PSF: pandas dataframe
            The dataframe for PSFs that contains the ['x', 'y', 'frame number', 'sigma'] for each PSF.
        """

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold
        self.overlap = overlap
        self.min_radial = min_radial
        self.max_radial = max_radial
        self.radial_step = radial_step
        self.alpha = alpha
        self.beta = beta
        self.stdFactor = stdFactor
        self.mode = mode

        self.rvt_kind = rvt_kind
        self.highpass_size = highpass_size
        self.upsample = upsample
        self.rweights = rweights
        self.coarse_factor = coarse_factor
        self.coarse_mode = coarse_mode
        self.pad_mode = pad_mode

        if self.cpu.parallel_active:
            self.function = function

            print("\n---start PSF detection with parallel loop---")
            if flag_GUI_ is False:
                try:
                    result0 = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                        delayed(self.psf_detection_kernel)(x) for x in tqdm(range(self.video.shape[0])))
                    result = [x for x in result0 if x is not None]
                except:
                    warnings.warn("switch backend to threading", DeprecationWarning)

                    result0 = Parallel(n_jobs=self.cpu.n_jobs, backend='threading', verbose=self.cpu.verbose)(
                        delayed(self.psf_detection_kernel)(x) for x in range(self.video.shape[0]))
                    result = [x for x in result0 if x is not None]
            else:
                result0 = Parallel(n_jobs=self.cpu.n_jobs, backend='threading', verbose=self.cpu.verbose)(
                    delayed(self.psf_detection_kernel)(x) for x in tqdm(range(self.video.shape[0])))
                result = [x for x in result0 if x is not None]

        else:
            print("\n---start PSF detection without parallel loop---")
            self.function = function
            result0 = [self.psf_detection_kernel(x) for x in tqdm(range(self.video.shape[0]))]
            result = [x for x in result0 if x is not None]

        df_PSF = data_handeling.list2dataframe(feature_position=result, video=self.video)

        return df_PSF

    def psf_detection_preview(self,  function, min_sigma=1, max_sigma=2, sigma_ratio=1.1, threshold=0, overlap=0,
                                min_radial=1, max_radial=2, radial_step=0.1, alpha=2, beta=1, stdFactor=1,
                                rvt_kind="basic",  highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add",
                                pad_mode="constant", mode='BOTH', frame_number=0, median_filter_flag=False, color='gray',
                                imgSizex=5, imgSizey=5, IntSlider_width='500px', title=''):

        """
        This function is a preview wrapper for calling various PSF localization methods.

        Parameters
        ----------
        function: str
            PSF localization algorithm which should be selected  from : (``'dog'``, ``'log'``, ``'doh'``, ``'frst'``, ``'frst_one_psf``')

        mode: str
            Defines which PSFs will be detected (``'BRIGHT'``, ``'DARK'``, or ``'BOTH'``).

        frame_number: int
            Selecting frame number that PSFs detection should apply on it.

        optional_1:
            These parameters are used when ``'dog'``, ``'log'``, ``'doh'`` are defined as function.

            * `min_sigma`: float, list of floats
                The is the minimum standard deviation for the kernel. The lower the value, the smaller blobs can be detected.
                The standard deviations of the filter are given for each axis in sequence or with a single number which is considered for both axis.

            * `max_sigma`: float, list of floats
                The is the maximum standard deviation for the kernel. The higher the value, the bigger blobs can be detected.
                The standard deviations of the filter are given for each axis in sequence or with a single number
                which is considered for both axis.

            * `sigma_ratio`: float
                * The ratio between the standard deviation of Kernels which is used for computing the DoG and LoG.
                * The number of intermediate values of standard deviations between min_sigma and max_sigma for computing the DoH.

            * `threshold`: float
                The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this
                to detect blobs with less intensities.

            * `overlap`: float
                A value between 0 and 1. If the area of two blobs are overlapping by a fraction greater than threshold, smaller blobs are eliminated.

        optional_2:
            These parameters are used when ``'frst'`` is defined as function.

            * `min_radial`: int
                integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

            * `max_radial`: int
                integer value for radius size in pixels (n in the original paper); also is used as gaussian kernel size

            * `alpha`: str
                Strictness of symmetry transform (higher=more strict; 2 is good place to start)

            * `beta`: float
                gradient threshold parameter, float in range [0,1]

            * `stdFactor`:
                Standard deviation factor for gaussian kernel

        optional_3:
            These parameters are used when ``'RVT'`` is defined as function.

            * `min_radial`:
                minimal radius (inclusive)

            * `max_radial`:
                maximal radius (inclusive)

            * `rvt_kind`:
                either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
                normalized version increases subpixel bias, but it works better at lower SNR

            * `highpass_size`:
                size of the high-pass filter; ``None`` means no filter (effectively, infinite size)

            * `upsample`: int
                integer image upsampling factor;
                `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
                if ``upsample>1``, the resulting image size is multiplied by ``upsample``

            * `rweights`:
                relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
                coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
                coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
                or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

            * `coarse_factor`:
                the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision

            * `coarse_mode`:
                the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
                or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)

            * `pad_mode`:
                edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
                or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
                ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
                note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

        Returns
        -------
        df_PSF: pandas dataframe
            The dataframe for PSFs that contains the ['x', 'y', 'frame number', 'sigma'] for each PSF
        """

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_ratio = sigma_ratio
        self.overlap = overlap
        self.min_radial = min_radial
        self.max_radial = max_radial
        self.radial_step = radial_step
        self.alpha = alpha
        self.beta = beta
        self.stdFactor = stdFactor
        self.mode = mode
        self.function = function
        self.frame_number = frame_number
        self.title = title

        self.rvt_kind = rvt_kind
        self.highpass_size = highpass_size
        self.upsample = upsample
        self.rweights = rweights
        self.coarse_factor = coarse_factor
        self.coarse_mode = coarse_mode
        self.pad_mode = pad_mode

        self.median_filter_flag = median_filter_flag
        self.color = color
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.IntSlider_width = IntSlider_width

        if "JPY_PARENT_PID" in os.environ:
            display_ = JupyterPSFs_localizationPreviewDisplay(video=self.video, df_PSFs=None,
                                                              frame_num=self.frame_number, title=self.title,
                                                               median_filter_flag=self.median_filter_flag,
                                                               color=self.color, imgSizex=self.imgSizex,
                                                               imgSizey=self.imgSizey,
                                                               IntSlider_width=self.IntSlider_width)
            def _preview(threshold):
                self.threshold = threshold
                df_PSF = self.psf_preview_kernel()

                display_.set_df(df_PSF)
                display_.display_run()

            selected_frame = self.video[self.frame_number, ...]
            min_range_, max_range_ = dog_preview(images=selected_frame, min_sigma=self.min_sigma,
                                                 max_sigma=self.max_sigma, sigma_ratio=self.sigma_ratio)
            max_range_0 = 1.5 * max_range_
            sci_num = lambda x: "{:.2e}".format(x)
            tmp = sci_num(max_range_).split('e')
            power = int(tmp[-1])
            if power >= 0:
                step = 0.1
            elif power <= -3:
                step = 1e-5
            else:
                p_ = power - 2
                step = 10**p_

            interact(_preview,
                     threshold=widgets.FloatSlider(value=threshold, min=0, max=max_range_0, step=step,
                                                   continuous_update=False, readout=True, readout_format='.7f',
                                                    description='Threshold', layout=Layout(width=IntSlider_width)))
        else:
            self.threshold = threshold
            df_PSF = self.psf_preview_kernel()
            return df_PSF

    def psf_preview_kernel(self):

        if type(self.frame_number) == list:
            result = [self.psf_detection_kernel(f_) for f_ in self.frame_number]
        else:
            result = self.psf_detection_kernel(self.frame_number)

        df_PSF = data_handeling.list2dataframe(feature_position=result, video=self.video)

        return df_PSF

    def psf_detection_kernel(self, i_):
        i_ = int(i_)

        if len(self.video.shape) == 3 and self.video.shape[0] > 0:

            if self.function == 'dog':

                if self.mode == 'BOTH':
                    positive_psf = self.dog(self.video[i_, :, :])
                    negative_psf = self.dog(-1 * self.video[i_, :, :])
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Bright':
                    positive_psf = self.dog(self.video[i_, :, :])
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Dark':
                    positive_psf = []
                    negative_psf = self.dog(-1 * self.video[i_, :, :])
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'doh':

                if self.mode == 'BOTH':
                    positive_psf = self.doh(self.video[i_, :, :])
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'log':

                if self.mode == 'BOTH':
                    positive_psf = self.log(self.video[i_, :, :])
                    negative_psf = self.log(-1 * self.video[i_, :, :])
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Bright':
                    positive_psf = self.log(self.video[i_, :, :])
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Dark':
                    positive_psf = []
                    negative_psf = self.log(-1 * self.video[i_, :, :])
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'frst':
                b_psf = self.frst(self.video[i_, :, :])
                temp2 = self.concatenateBrightDark(b_psf, [], i_)

            elif self.function == 'frst_one_psf':
                b_psf = self.frst_one_PSF(self.video[i_, :, :])
                temp2 = self.concatenateBrightDark(b_psf, [], i_)
                temp2 = np.expand_dims(temp2, axis=0)

            elif self.function == 'RVT':
                b_psf = self.rvt(self.video[i_, :, :])
                temp2 = self.concatenateBrightDark(b_psf, [], i_)

        else:

            if self.function == 'dog':

                if self.mode == 'BOTH':
                    positive_psf = self.dog(self.video)
                    negative_psf = self.dog(-1 * self.video)
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Bright':
                    positive_psf = self.dog(self.video)
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Dark':
                    positive_psf = []
                    negative_psf = self.dog(-1 * self.video)
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'doh':

                if self.mode == 'BOTH':
                    positive_psf = self.doh(self.video)
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'log':

                if self.mode == 'BOTH':
                    positive_psf = self.log(self.video)
                    negative_psf = self.log(-1 * self.video)
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Bright':
                    positive_psf = self.log(self.video)
                    negative_psf = []
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)
                elif self.mode == 'Dark':
                    positive_psf = []
                    negative_psf = self.log(-1 * self.video)
                    temp2 = self.concatenateBrightDark(positive_psf, negative_psf, i_)

            elif self.function == 'frst':
                b_psf = self.frst(self.video)
                temp2 = self.concatenateBrightDark(b_psf, [], i_)

            elif self.function == 'frst_one_psf':
                b_psf = self.frst_one_PSF(self.video)
                temp2 = self.concatenateBrightDark(b_psf, [], i_)
                temp2 = np.expand_dims(temp2, axis=0)

            elif self.function == 'RVT':
                b_psf = self.rvt(self.video)
                temp2 = self.concatenateBrightDark(b_psf, [], i_)
                temp2 = np.expand_dims(temp2, axis=0)

        return temp2

    def concatenateBrightDark(self, bright_psf, dark_psf, i_):
        if len(bright_psf) != 0 and len(dark_psf) != 0:
            psf = np.unique(np.concatenate((bright_psf, dark_psf)), axis=0)

            frame_num = i_ * np.ones((psf.shape[0], 1), dtype=int)
            temp2 = np.concatenate((frame_num, psf), axis=1)
        elif len(bright_psf) != 0 and len(dark_psf) == 0:
            psf = np.asarray(bright_psf)
            frame_num = i_ * np.ones((psf.shape[0], 1), dtype=int)

            if len(psf.shape) != len(frame_num.shape) and len(psf.shape) == 1:
                temp2 = np.concatenate(([i_], psf), axis=0)
            else:
                temp2 = np.concatenate((frame_num, psf), axis=1)

        elif len(bright_psf) == 0 and len(dark_psf) != 0:
            psf = np.asarray(dark_psf)

            frame_num = i_ * np.ones((psf.shape[0], 1), dtype=int)
            temp2 = np.concatenate((frame_num, psf), axis=1)
        else:
            temp2 = None
        return temp2




