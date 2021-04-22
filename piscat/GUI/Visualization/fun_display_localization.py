from piscat.GUI.Visualization import image_viewer
from piscat.Preproccessing import Normalization
from skimage.draw import circle_perimeter

from PySide2 import QtGui
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2.QtCore import *

from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


class Visulization_localization(QtWidgets.QMainWindow):

    def __init__(self, filename=None):
        super(Visulization_localization, self).__init__()
        self.different_views = {}
        self.current_frame_number = 0
        self.threadpool = QThreadPool()
        self.filename = filename

    @QtCore.Slot()
    def get_sliceNumber(self, frame_num):
        self.current_frame_number = frame_num

    def new_display(self, display_input, original_video, object=None, title=None, mask_status=False, position_list=None):

            if mask_status is False:

                self.different_views[title] = image_viewer.ImageViewer(display_input, original_video, title)
                if title != "PNG":
                    self.slice_num = self.different_views[title].viewer.slice_num
                    current_pixmap = self.different_views[title].viewer.create_pixmap(display_input[self.slice_num, :, :])
                    self.different_views[title].viewer.update_slice(current_pixmap)
                elif title == "PNG":
                    current_pixmap = self.different_views[title].viewer.create_pixmap(self.filename)
                    self.different_views[title].viewer.update_slice(current_pixmap)

            elif mask_status is True:

                self.different_views[title] = image_viewer.ImageViewer(display_input, original_video, title, mask=mask_status, list_psf=position_list)

                if title != "PNG":
                    self.slice_num = self.different_views[title].viewer.slice_num
                    current_pixmap = self.different_views[title].viewer.create_pixmap(display_input[self.slice_num, :, :])
                    self.different_views[title].viewer.update_slice(current_pixmap)
                elif title == "PNG":
                    current_pixmap = self.different_views[title].viewer.create_pixmap(self.filename)
                    self.different_views[title].viewer.update_slice(current_pixmap)

            if object is not None:
                self.different_views[title].currentFrame.connect(object.get_sliceNumber)

    def create_circle(self, x, y, r, shape):
        rr, cc = circle_perimeter(x, y, r, shape=shape)
        return rr, cc

    def create_circle_apply_oneFrame_pd(self, x_, y_, s_, f_):
        y = int(x_)
        x = int(y_)
        radius = int(np.sqrt(2) * s_)
        rr, cc = self.create_circle(x, y, radius, self.input_mask[0, :, :].shape)
        self.input_mask[f_, rr, cc] = True

    def create_circle_apply_toAll_pd(self, input_mask, position_df, parallel_flag):
        if position_df is not None:
            frames = position_df['frame'].tolist()
            x_positions = position_df['x'].tolist()
            y_positions = position_df['y'].tolist()
            sigmas = position_df['sigma'].tolist()
            # particle_labels = particle['particle'].tolist()

            if parallel_flag:
                self.input_mask = input_mask
                Parallel(n_jobs=-1, backend='threading', verbose=0)(
                    delayed(self.create_circle_apply_oneFrame_pd)(x_, y_, s_, f_) for x_, y_, s_, f_ in tqdm(zip(x_positions, y_positions, sigmas, frames)))
                input_mask = self.input_mask
            else:

                for x_, y_, s_, f_ in tqdm(zip(x_positions, y_positions, sigmas, frames)):
                    y = int(x_)
                    x = int(y_)
                    radius = int(np.sqrt(2) * s_)
                    rr, cc = self.create_circle(x, y, radius, input_mask[0, :, :].shape)
                    input_mask[f_, rr, cc] = True
            print("-------------------MaskArray is now updated.-------------------")
        else:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Do not find any particle!")
            self.msg_box.exec_()
        return input_mask

    def create_circle_apply_toAll_list(self, input_mask, position_list, flag_preview):
        if position_list is not None:
            if flag_preview is False:
                for i_ in range(len(position_list)):
                    frame_number = int(position_list[i_][0, 0])
                    for j_ in range(position_list[i_].shape[0]):
                        y = int(position_list[i_][j_, 1])
                        x = int(position_list[i_][j_, 2])
                        sigma = position_list[i_][j_, 3]
                        radius = int(np.sqrt(2)*sigma)
                        rr, cc = self.create_circle(y, x, radius, input_mask[0, :, :].shape)
                        input_mask[frame_number, rr, cc] = True
            else:
                frame_number = int(position_list[0, 0])
                for j_ in range(position_list.shape[0]):
                    y = int(position_list[j_, 1])
                    x = int(position_list[j_, 2])
                    sigma = position_list[j_, 3]
                    radius = int(np.sqrt(2) * sigma)
                    rr, cc = self.create_circle(y, x, radius, input_mask[0, :, :].shape)
                    input_mask[frame_number, rr, cc] = True
            print("-------------------MaskArray is now updated.-------------------")
        else:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Do not find any particle!")
            self.msg_box.exec_()
        return input_mask

    def call_norm_2_display(self, in_video):
        self.flag_update_display_video = True
        worker = Normalization.Normalization(in_video, flag_GUI=True, flag_image_specific=True)
        worker.signals.result.connect(self.update_display_video)
        self.threadpool.start(worker)
        while self.flag_update_display_video:
            QtGui.qApp.processEvents()

        self.flag_update_display_video = True

        return self.norm_v

    @QtCore.Slot(np.ndarray)
    def bg_correction_update(self, in_video, label, object):
            # self.call_norm_2_display(in_video)
            self.new_display(in_video, in_video, object=object, title=label, mask_status=True)

    def update_display_video(self, r_):
        self.norm_v = r_
        self.flag_update_display_video = False

    @QtCore.Slot()
    def update_localization_onMask(self, video_in, df_PSFs, title, flag_preview=False):
        try:
            self.maskArray = np.zeros_like(video_in, dtype="bool")

            if type(df_PSFs) is np.ndarray:
                circled_mask = self.create_circle_apply_toAll_list(self.maskArray, df_PSFs, flag_preview=True)
                print('list_done')

            elif type(df_PSFs) is pd.core.frame.DataFrame:
                if flag_preview:
                    circled_mask = self.create_circle_apply_toAll_pd(self.maskArray, df_PSFs, parallel_flag=False)

                else:
                    circled_mask = self.create_circle_apply_toAll_pd(self.maskArray, df_PSFs, parallel_flag=True)

            self.different_views[title].viewer.maskArray = circled_mask
            self.different_views[title].viewer.mask_is_set = True
            mask_pixmap = self.different_views[title].viewer.create_mask_pixmap(circled_mask[self.current_frame_number, :, :])
            self.different_views[title].viewer.update_overlay(mask_pixmap)
            self.different_views[title].get_in(df_PSFs)

        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Error in load data-frame!")
            self.msg_box.exec_()






