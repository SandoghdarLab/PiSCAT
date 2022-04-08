from piscat.InputOutput.cpu_configurations import CPUConfigurations
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from tqdm.autonotebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os


class DNNAnomaly:
    def __init__(self, feature_list, contamination=0.002):
        self.feature_list = feature_list
        self.video_shape = feature_list[0].shape
        self.cpu = CPUConfigurations()
        self.clf = IsolationForest(max_samples=100, random_state=0, bootstrap=False, warm_start=False,
                              n_jobs=None, contamination=contamination, verbose=0)

    def apply_IF_spacial_temporal(self, step=1, stride=1):
        self.feature_matrix_new = []
        for feature_ in self.feature_list:
            tmp_reshape = np.reshape(feature_, (feature_.shape[0], -1), order='C')
            feature_matrix_ = tmp_reshape.transpose()
            self.feature_matrix_new.append(feature_matrix_)
        print("\n---start anomaly with Parallel---")
        mask_video = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(
            self._feature_genration_kernel)(f_, step, stride) for f_ in tqdm(range(0, self.feature_matrix_new[0].shape[1])))
        mask_video_ = np.asarray(mask_video)
        return mask_video_

    def _feature_genration_kernel(self, f_, step, stride=1):
        # if (f_ + step) <= self.feature_matrix_new[0].shape[1] and (f_-step) >= 0:
        #     X = self.feature_matrix_new[0][:, f_-step:f_+step:stride]
        #     Y = self.feature_matrix_new[1][:, f_-step:f_+step:stride]
        #
        # elif (f_ + step) <= self.feature_matrix_new[0].shape[1] and (f_ - step) < 0:
        #     X = self.feature_matrix_new[0][:, f_:f_ + step:stride]
        #     Y = self.feature_matrix_new[1][:, f_:f_ + step:stride]
        #
        # elif (f_ + step) > self.feature_matrix_new[0].shape[1] and (f_ - step) >= 0:
        #     X = self.feature_matrix_new[0][:, f_-step:f_:stride]
        #     Y = self.feature_matrix_new[1][:, f_-step:f_:stride]
        #
        # elif (f_ + step) > self.feature_matrix_new[0].shape[1] and (f_ - step) < 0:
        #     X = self.feature_matrix_new[0][:, f_]
        #     Y = self.feature_matrix_new[1][:, f_]

        if (f_ + step) <= self.feature_matrix_new[0].shape[1] and (f_-step) >= 0:
            selected_features = []
            for feature_ in self.feature_matrix_new[1:]:
                X = feature_[:, f_ - step:f_ + step:stride]
                selected_features.append(X)

        elif (f_ + step) <= self.feature_matrix_new[0].shape[1] and (f_ - step) < 0:
            selected_features = []
            for feature_ in self.feature_matrix_new[1:]:
                X = feature_[:, 0:f_ + step:stride]
                selected_features.append(X)

        elif (f_ + step) > self.feature_matrix_new[0].shape[1] and (f_ - step) >= 0:
            selected_features = []
            for feature_ in self.feature_matrix_new[1:]:
                X = feature_[:, f_-step::stride]
                selected_features.append(X)

        elif (f_ + step) > self.feature_matrix_new[0].shape[1] and (f_ - step) < 0:
            print('warning!!!')
            selected_features = []
            for feature_ in self.feature_matrix_new[1:]:
                X = feature_[:, f_]
                selected_features.append(X)

        # f_xy = np.concatenate((X, Y), axis=1)
        f_xy = selected_features[0]
        for feature_ in selected_features[1:]:
            f_xy = np.concatenate((f_xy, feature_), axis=1)
        self.clf.fit(f_xy)
        p = self.clf.predict(f_xy)
        img_ = np.reshape(p, (self.video_shape[1], self.video_shape[2]))
        return img_
