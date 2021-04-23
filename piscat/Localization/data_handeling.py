import numpy as np
import pandas as pd


def feature2df(feature_position, videos):

    if feature_position.shape[1] == 4:

        sigma = feature_position[:, 3]
        psf_position_x = feature_position[:, 2]
        psf_position_y = feature_position[:, 1]
        psf_position_frame = np.asarray(feature_position[:, 0], dtype=int)

        center_intensity = np.asarray([videos[int(psf_position_frame[i_]), int(psf_position_y[i_]),
                                              int(psf_position_x[i_])] for i_ in range(psf_position_frame.shape[0])])

        dict = {'y': psf_position_y, 'x': psf_position_x, 'frame': psf_position_frame, 'center_intensity': center_intensity, 'sigma': sigma}

        df_features = pd.DataFrame(dict)

        return df_features

    elif feature_position.shape[1] == 5:
        sigma = (feature_position[:, 3], feature_position[:, 4])
        sigma = np.asarray(sigma)
        psf_position_x = feature_position[:, 2]
        psf_position_y = feature_position[:, 1]
        psf_position_frame = np.asarray(feature_position[:, 0], dtype=np.int)

        center_intensity = np.asarray([videos[int(psf_position_frame[i_]), int(
            psf_position_y[i_]), int(psf_position_x[i_])] for i_ in
                                       range(psf_position_frame.shape[0])])

        dict = {'y': psf_position_y, 'x': psf_position_x, 'frame': psf_position_frame, 'center_intensity': center_intensity,
                'sigma_x': feature_position[:, 3], 'sigma_y': feature_position[:, 4]}

        df_features = pd.DataFrame(dict)
        df_features['sigma'] = 0.5 * (df_features['sigma_x'] + df_features['sigma_y'])
        return df_features


def list2dataframe(feature_position, video):
    """
    This function converts the output of ``particle_localization.PSFsExtraction`` method from list to data frame.

    Parameters
    ----------
    feature_position: list
        List of position of PSFs (x, y, frame, sigma)

    video: NDArray
        The video is 3D-numpy (number of frames, width, height).

    Returns
    -------
    df_features: pandas dataframe
        PSF positions are stored in the data frame. ( 'y', 'x', 'frame', 'center_intensity', 'sigma', 'Sigma_ratio', ...).
    """
    if feature_position is not None:
        if type(feature_position) is list:
            try:
                feature_position = [l_ for l_ in feature_position if l_ is not None]
                if len(feature_position) >= 1:
                    feature_position = np.concatenate(feature_position, axis=0)
                    df_features = feature2df(feature_position, video)
                else:
                    df_features = None
            except:
                raise Exception('---List is empty!---')

        elif type(feature_position) is np.ndarray:
            try:
                df_features = feature2df(feature_position, video)
            except:
                raise Exception('---List is empty!---')
        else:
            df_features = None

        return df_features
    else:
        df_features = None

        return df_features


