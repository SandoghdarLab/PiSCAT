from sklearn import svm
from sklearn.ensemble import IsolationForest
from skimage.transform import rescale
from skimage.filters import threshold_isodata
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Visualization.display import Display, DisplaySubplot

import matplotlib.pylab as plt
import numpy as np


class SpatioTemporalAnomalyDetection:

    def __init__(self, feature_list, inter_flag_parallel_active=True):
        """
        This class generates the feature matrix based on the received feature list and using anomaly detection algorithms
        to identify each pixel in the video as normal or abnormal.

        Parameters
        ----------
        feature_list: list
            This is a list of various 3D arrays that define features.

        inter_flag_parallel_active: bool
            If the user wants to enable general parallel tasks in the CPU configuration, he or she can only use this flag to enable or disable this process.
        """

        self.cpu = CPUConfigurations()

        self.feature_list = feature_list

        self.inter_flag_parallel_active = inter_flag_parallel_active

        self.feature_list_rescale = None

        self.video_shape = feature_list[0].shape

    def fun_anomaly(self, scale=1, method='IsolationForest', contamination='auto'):
        """
        Using the 'IsolationForest' or 'OneClassSVM' methods.

        Parameters
        ----------
        scale: float
           It specifies the scale of image downsampling.

        method: str
            Defines the methods we utilized to detect anomalies (for example, 'IsolationForest' or 'OneClassSVM').

        contamination : 'auto' or float, default='auto'
            This value is only used when `IsolationForest` is used.
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the scores of the samples.

                - If 'auto', the threshold is determined as in the
                  original paper.
                - If float, the contamination should be in the range (0, 0.5].
        """
        self.rng = np.random.RandomState(42)

        if method == 'IsolationForest':
            self.clf = IsolationForest(max_samples=100, random_state=self.rng, bootstrap=False, warm_start=True,
                                       n_jobs=None, contamination=contamination, verbose=0)
        elif method == 'OneClassSVM':
            self.clf = svm.OneClassSVM(nu=0.06, kernel='rbf', gamma='scale', verbose=False, tol=1e-3)

        feature_list_rescale = None

        print('\nstart feature matrix genration ' + '--->', end=" ")
        for feature_ in self.feature_list:
                tmp = rescale(feature_, (1, scale, scale), anti_aliasing=False)

                feature_tmp_ = np.expand_dims(tmp, axis=0)

                if feature_list_rescale is None:
                    feature_list_rescale = feature_tmp_
                else:
                    feature_list_rescale = np.concatenate((feature_list_rescale, feature_tmp_), axis=0)

        self.feature_list_rescale = feature_list_rescale
        print('Done')

        if self.cpu.parallel_active and self.inter_flag_parallel_active:
            print("\n---start anomaly with Parallel---")
            results = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                delayed(self.anomaly_kernel)(f_) for f_ in tqdm(range(self.feature_list_rescale.shape[1])))
        else:
            print("\n---start anomaly without Parallel---")
            results = [self.anomaly_kernel(f_) for f_ in tqdm(range(self.feature_list_rescale.shape[1]))]

        anomaly_mask = np.asarray(results)
        thresh = threshold_isodata(anomaly_mask)
        binary = anomaly_mask > thresh

        return binary, anomaly_mask

    def anomaly_kernel(self, i_):
        features_matrix_2D = None
        for s_ in range(self.feature_list_rescale.shape[0]):
            f_2D_tmp = self.feature_list_rescale[s_, i_, :, :]
            f_1D_tmp = np.reshape(f_2D_tmp, (-1, 1))

            if features_matrix_2D is None:
                features_matrix_2D = f_1D_tmp
            else:
                features_matrix_2D = np.concatenate((features_matrix_2D, f_1D_tmp), axis=1)

        self.clf.fit(features_matrix_2D)
        seq_vid = self.clf.predict(features_matrix_2D)
        label2D = np.reshape(seq_vid, (f_2D_tmp.shape[0], f_2D_tmp.shape[1]))
        return label2D



