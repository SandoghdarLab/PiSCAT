from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2 import QtGui
from PySide2.QtCore import *

from piscat.InputOutput import write_video
from piscat.Preproccessing.normalization import Normalization
from piscat.Visualization.display import Display, DisplayPSFs_subplotLocalizationDisplay
from piscat.GUI.Visualization.contrast_adjustment_GUI import Contrast_adjustment_GUI
from piscat.GUI.Visualization import slice_view
from piscat.GUI.InputOutput import save_GUI
from piscat.GUI.Visualization.play_setting import PlaySetting

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

play_flag = True


class WorkerSignals(QObject):

    updateProgress = Signal(int)
    result = Signal(object)
    finished = Signal()
    stop_signal = Signal()


class ImageViewer(QtWidgets.QMainWindow, QtCore.QObject):

    progressChanged = QtCore.Signal(int)
    sliceNumber = QtCore.Signal(int)
    currentFrame = QtCore.Signal(int)

    def __init__(self, input_image_series, original_video, name, mask=False, list_psf=None):
        super(ImageViewer, self).__init__()

        self.threadpool = QThreadPool()

        self.image_series = input_image_series
        self.original_video = original_video
        self.df_PSFs = list_psf
        self.sizeX = self.image_series.shape[1]
        self.sizeY = self.image_series.shape[2]
        self.frame_num = None
        self.current_frame = 0
        self.title = name
        self.mask = mask
        self.starter()
        self.save_video = None
        self.step_size = None
        self.time_delay = None
        self.fps = None
        self.info_image_save = None
        self.alpha = None
        self.beta = None
        self.max_intensity = None
        self.min_intensity = None
        self.frame_strides = 1
        self.signals = WorkerSignals()
        self.sliceNumber.connect(self.Update_SliceNumber)

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def starter(self):
        self.setWindowIcon(QtGui.QIcon(QtCore.QDir.currentPath() + "/icons/mpl.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(490, 100,  512+124,  512+93)
        self.main_Viewer_frame = QtWidgets.QFrame()
        self.setCentralWidget(self.main_Viewer_frame)
        self.main_Viewer_frame_layout = QtWidgets.QGridLayout()
        self.main_Viewer_frame.setLayout(self.main_Viewer_frame_layout)
        self.resize(500, 500)

        # Create an instant of GraphicView Class to show the image PixMap.
        self.viewer = slice_view.SliceView(self.image_series, self.original_video, mask=self.mask)
        self.viewer.frameClicked.connect(self.frame_read)
        self.viewer.photoClicked.connect(self.photoClicked)

        self.main_Viewer_frame_layout.addWidget(self.viewer, 0, 0)

        # checkbox median Filter
        self.checkbox_medianFilter = QtWidgets.QRadioButton("Median Filter")
        self.checkbox_medianFilter.toggled.connect(lambda: self.activeMedianFilter(self.checkbox_medianFilter))

        # Contrast
        self.con_adj = QtWidgets.QPushButton("Contrast adjustment")
        self.con_adj.setAutoDefault(False)
        self.con_adj.clicked.connect(self.open_adjustment)
        self.con_adj.setFixedWidth(130)
        self.con_adj.setStyleSheet("background-color : lightgrey")

        # Display
        self.Display = QtWidgets.QPushButton("Display")
        self.Display.setAutoDefault(False)
        self.Display.clicked.connect(self.matplotlib_display)
        self.Display.setFixedWidth(130)
        self.Display.setStyleSheet("background-color : lightgrey")

        # annotation
        self.annotation = QtWidgets.QPushButton("Live annotation")
        # self.annotation.setAutoDefault(False)
        self.annotation.setCheckable(True)
        self.annotation.clicked.connect(self.annotationFun)
        self.annotation.setFixedWidth(130)
        self.annotation.setStyleSheet("background-color : lightgrey")

        # Play
        self.playBtn = QtWidgets.QPushButton()
        self.playBtn.setAutoDefault(False)
        self.playBtn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playBtn.setFixedWidth(20)
        self.playBtn.clicked.connect(self.run_play)

        # create context menu
        self.playBtn.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.playBtn.customContextMenuRequested.connect(self.on_context_menu)

        # stop
        self.stopBtn = QtWidgets.QPushButton()
        self.stopBtn.setAutoDefault(False)
        self.stopBtn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stopBtn.setFixedWidth(20)
        self.stopBtn.clicked.connect(self.stop_video)

        # Pause
        self.pauseBtn = QtWidgets.QPushButton()
        self.pauseBtn.setAutoDefault(False)
        self.pauseBtn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self.pauseBtn.setFixedWidth(20)
        self.pauseBtn.clicked.connect(self.pause_video)
        self._running = False

        # save
        self.save = QtWidgets.QPushButton("save")
        self.save.setAutoDefault(False)
        self.save.clicked.connect(self.do_update_save)
        self.save.setFixedWidth(130)
        self.save.setStyleSheet("background-color : lightgrey")

        if self.mask is True:
            self.Display_mask = QtWidgets.QPushButton("Display_with_mask")
            self.Display_mask.setAutoDefault(False)
            self.Display_mask.clicked.connect(self.matplotlib_display_mask)
            self.Display_mask.setFixedWidth(130)
            self.Display_mask.setStyleSheet("background-color : lightgrey")

            # mp4
            self.mp4 = QtWidgets.QPushButton("MP4+localization")
            self.mp4.setAutoDefault(False)
            self.mp4.clicked.connect(self.make_mp4)
            self.mp4.setFixedWidth(130)
            self.mp4.setStyleSheet("background-color : lightgrey")

            # progressBar
            self.statusbar = self.statusBar()
            self.progressBar = QtWidgets.QProgressBar(self)
            self.statusBar().addPermanentWidget(self.progressBar)
            self.progressBar.setGeometry(30, 40, 200, 25)
            self.progressBar.setValue(0)
            self.progressChanged.connect(self.setProgress)

        # Create
        self.connect(QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H), self), QtCore.SIGNAL('activated()'), self.histogram)

        if self.title != "PNG":
            self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.image_series.shape[0]-1)
            self.slice_slider.setValue(0)
            self.slice_slider.valueChanged.connect(self.set_new_slice_num)

            self.main_Viewer_frame_layout.addWidget(self.createThirdExclusiveGroup(), 1, 0)

        self.main_Viewer_frame_layout.addWidget(self.createSecondExclusiveGroup(), 2, 0)

        self.statusbar = self.statusBar()
        self.show()

    def createSecondExclusiveGroup(self):

        self.groupBox_displayBtn = QtWidgets.QGroupBox("Video Display")

        self.grid_diplay = QtWidgets.QGridLayout()
        self.grid_diplay.addWidget(self.Display, 0, 0)
        self.grid_diplay.addWidget(self.annotation, 0, 1)
        self.grid_diplay.addWidget(self.con_adj,  0, 2)

        if self.mask is True:
            self.grid_diplay.addWidget(self.Display_mask, 1, 0)
            # self.grid_diplay.addWidget(self.mp4, 1, 1)

        self.grid_diplay.addWidget(self.save, 1, 2)
        self.groupBox_displayBtn.setLayout(self.grid_diplay)

        return self.groupBox_displayBtn

    def createThirdExclusiveGroup(self):

        play = QtWidgets.QGroupBox()

        self.grid_play = QtWidgets.QGridLayout()
        self.grid_play.addWidget(self.playBtn, 0, 0)
        self.grid_play.addWidget(self.pauseBtn, 0, 1)
        self.grid_play.addWidget(self.stopBtn, 0, 2)
        self.grid_play.addWidget(self.slice_slider, 0, 3)
        self.grid_play.addWidget(self.checkbox_medianFilter, 0, 4)

        play.setLayout(self.grid_play)
        return play

    def activeMedianFilter(self, box_status):
        if box_status.isChecked():
            self.viewer.medianFilterFlag = True
        else:
            self.viewer.medianFilterFlag = False

    def annotationFun(self):
        if self.annotation.isChecked():
            self.annotation.setStyleSheet("background-color : lightblue")

        else:
            self.annotation.setStyleSheet("background-color : lightgrey")

        self.viewer.annotation.emit()

    def closeEvent(self, event):
        self.stop_video()
        print("closed")

    @QtCore.Slot()
    def breakAll(self):
        self.make_mp4.kill()

    @QtCore.Slot()
    def Update_SliceNumber(self, frame_number):
        self.current_frame = frame_number
        self.currentFrame.emit(frame_number)

    def get_in(self, df_PSFs):
        self.df_PSFs = df_PSFs

    @QtCore.Slot()
    def call_save(self):
        if self.fps == '' or self.fps is None:
            self.fps = 10
        else:
            self.fps = int(self.fps)

        if self.info_image_save.video_type == "RAW":
            if self.info_image_save.bin_type is not None:
                type = self.original_video.astype(self.info_image_save.bin_type)
            else:
                type = 'original'

            file_name = str(self.original_video.shape[0]) + '_' + str(self.original_video.shape[1]) +'_' + \
                        str(self.original_video.shape[2]) + '_' + str(self.original_video.dtype) + '.raw'
            write_video.write_binary(dir_path=self.file_path, file_name=file_name, data=self.original_video, type=type)

        elif self.info_image_save.video_type == "MP4":

            self.flag_update_normalization = True
            worker = Normalization(video=self.original_video, flag_image_specific=True)
            worker.signals.result.connect(self.update_normalization)
            self.threadpool.start(worker)
            while self.flag_update_normalization:
                QtCore.QCoreApplication.processEvents()
            file_name = str(self.original_video.shape[0]) + '_' + str(self.original_video.shape[1]) + '_' + \
                        str(self.original_video.shape[2]) + '_' + str(self.original_video.dtype) + '.mp4'
            write_video.write_MP4(dir_path=self.file_path, file_name=file_name, data=self.save_video, fps=self.fps)

        elif self.info_image_save.video_type == "GIF":
            self.flag_update_normalization = True
            worker = Normalization(video=self.original_video, flag_image_specific=True)
            worker.signals.result.connect(self.update_normalization)
            self.threadpool.start(worker)
            while self.flag_update_normalization:
                QtCore.QCoreApplication.processEvents()
            file_name = str(self.original_video.shape[0]) + '_' + str(self.original_video.shape[1]) + '_' + \
                        str(self.original_video.shape[2]) + '_' + str(self.original_video.dtype) + '.mp4'

            write_video.write_GIF(dir_path=self.file_path, file_name=file_name, data=self.save_video, jump=self.frame_strides, fps=self.fps)

    @Slot()
    def update_normalization(self, video):
        self.save_video = video
        self.flag_update_normalization = False

    def do_update_save(self):
        if self.save.clicked:
            try:
                self.info_image_save = save_GUI.SaveVideo()
                self.info_image_save.signal_save_Done.connect(self.call_save)
                self.file_path = self.askdir()

            except:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Folder does not defined!")
                self.msg_box.exec_()

    def askdir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', '',
                                                                 QtWidgets.QFileDialog.ShowDirsOnly)
        return folder

    def matplotlib_display(self):

        if self.time_delay == '' or self.time_delay is None:
            self.time_delay = 0.1
        else:
            self.time_delay = float(self.time_delay)

        if self.Display.clicked:
            if self.step_size != '' and self.step_size is not None:
                Display(video=np.fliplr(self.original_video), time_delay=float(self.time_delay))
            else:
                Display(video=np.fliplr(self.original_video), time_delay=float(self.time_delay))

    def matplotlib_display_mask(self):

        if self.step_size == '' or self.step_size is None:
            self.step_size = 1
        else:
            self.step_size = int(self.step_size)

        if self.time_delay == '' or self.time_delay is None:
            self.time_delay = 0.1
        else:
            self.time_delay = float(self.time_delay)

        if self.Display_mask.clicked:
            if self.df_PSFs is not None:
                if type(self.df_PSFs) is pd.core.frame.DataFrame:
                    if self.step_size != '':

                        ani_ = DisplayPSFs_subplotLocalizationDisplay(list_videos=[self.original_video],
                                                                                list_df_PSFs=[self.df_PSFs], list_titles=None,
                                                                                numRows=1, numColumns=1, color='gray',
                                                                                median_filter_flag=self.viewer.medianFilterFlag,
                                                                                imgSizex=10, imgSizey=5, time_delay=self.time_delay)

                    else:
                        ani_ = DisplayPSFs_subplotLocalizationDisplay(list_videos=[self.original_video],
                                                                                list_df_PSFs=[self.df_PSFs], list_titles=None,
                                                                                numRows=1, numColumns=1, color='gray',
                                                                                median_filter_flag=self.viewer.medianFilterFlag,
                                                                                imgSizex=10, imgSizey=5, time_delay=self.time_delay)

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("The PSF location is not defined")
                self.msg_box.exec_()

    def make_mp4(self):

        if self.mp4.clicked:
            self.file_path = self.askdir()
            try:
                if self.step_size == '' or self.step_size is None:
                    self.step_size = 1
                else:
                    self.step_size = int(self.step_size)

                if self.time_delay == '' or self.time_delay is None:
                    self.time_delay = 0.1
                else:
                    self.time_delay = float(self.time_delay)

                if self.fps == '' or self.fps is None:
                    self.fps = 10
                else:
                    self.fps = int(self.fps)

                p_max = self.original_video.shape[0] - self.step_size
                self.progressBar.setValue(0)
                self.progressBar.setMaximum(p_max)

                if self.df_PSFs is not None:
                    if type(self.df_PSFs) is pd.core.frame.DataFrame:
                        self.flag_mp4 = True
                        save_path = os.path.join(self.file_path, 'PiSCAT_GUI_Localization.mp4')

                        ani_ = HTML_PSFs_subplotLocalizationDisplay(list_videos=[self.original_video],
                                                                          list_df_PSFs=[self.df_PSFs], list_titles=None,
                                                                          numRows=1, numColumns=1, cmap='gray',
                                                                          median_filter_flag=self.viewer.medianFilterFlag,
                                                                          imgSizex=10, imgSizey=5, time_delay=self.time_delay,
                                                                          save_path=save_path, fps=self.fps)
                        ani_.signals.finished.connect(self.finish_mp4)
                        self.threadpool.start(ani_)
                        while self.flag_mp4:
                            QtCore.QCoreApplication.processEvents()

                        self.msg_box = QtWidgets.QMessageBox()
                        self.msg_box.setWindowTitle("Warning!")
                        self.msg_box.setText("MP4 vide saved!")
                        self.msg_box.exec_()

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("The PSF location is not defined")
                    self.msg_box.exec_()
            except:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Error occurred in MP4!")
                self.msg_box.exec_()

    @QtCore.Slot()
    def finish_mp4(self):
        self.flag_mp4 = False

    def startProgressBar(self, fun, **kwargs):
        self.thread = fun(**kwargs)
        self.thread.obj_connection.updateProgress_.connect(self.setProgress)
        self.thread.obj_connection.finished.connect(self.thread_complete)
        self.threadpool.start(self.thread)

    @QtCore.Slot()
    def update_animate(self, setting):
        self.step_size = setting[0]
        self.time_delay = setting[1]
        self.fps = setting[2]
        self.raise_()

    def set_new_slice_num(self):
        self.viewer.slice_num = self.slice_slider.value()
        self.updata_viewer()
        self.sliceNumber.emit(self.viewer.slice_num)

    def enterEvent(self, event):
        self.setMouseTracking(True)

    def leaveEvent(self, event):
        self.setMouseTracking(False)

    def update_potion(self, result_):
        self.statusbar.showMessage(result_)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def wheelEvent(self, *args, **kwargs):
        pass

    def histogram(self):
        tmp = self.original_video[self.viewer.slice_num]
        plt.figure(figsize=(6, 6), dpi=100)
        plt.hist(tmp.ravel(), fc='g', ec='k')
        plt.title("Histogram of pixel intensity")
        plt.ylabel("#Counts")
        plt.show()

    @QtCore.Slot()
    def run_play(self):

        if self.step_size == '' or self.step_size is None:
            self.frame_strides = 1
        else:
            self.frame_strides = int(self.step_size)

        if self.time_delay == '' or self.time_delay is None:
            self.time_delay = 0.1
        else:
            self.time_delay = float(self.time_delay)

        self.playBtn.setDisabled(True)
        self.play_video()

    @QtCore.Slot()
    def play_video(self):
        global play_flag
        play_flag = True
        slice_num = self.slice_slider.value()
        self.playBtn.setDisabled(True)

        while slice_num < self.image_series.shape[0]-1 and play_flag:
            slice_num += self.frame_strides
            self.slice_slider.setValue(slice_num)
            self.viewer.slice_num = self.slice_slider.value()
            self.updata_viewer()
            QtCore.QCoreApplication.processEvents()

    @QtCore.Slot()
    def pause_video(self):
        global play_flag
        play_flag = False
        self.playBtn.setEnabled(True)

    @QtCore.Slot()
    def stop_video(self):

        global play_flag
        play_flag = False

        self.playBtn.setEnabled(True)

        self.slice_slider.setValue(0)
        self.viewer.slice_num = self.slice_slider.value()
        self.updata_viewer()

    def on_context_menu(self):
        self.animatePanel_ = PlaySetting()
        self.animatePanel_.procDone_Animation_panel.connect(self.update_animate)
        self.animatePanel_.show()

    def frame_read(self, frame_num):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.frame_num = frame_num

    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            value = self.original_video[self.frame_num, pos[0], pos[1]]
            self.statusbar.showMessage('X:%d, Y:%d, Frame:%d, Value:%f' % (pos[0], pos[1], self.frame_num, value))

    @QtCore.Slot(int)
    def setProgress(self, val):
        self.progressBar.setValue(val)

    @QtCore.Slot()
    def set_alpha(self, alpha):
        self.alpha = alpha

    @QtCore.Slot()
    def set_beta(self, beta):
        self.beta = beta

    @QtCore.Slot()
    def set_max_intensity(self, max_intensity):
        self.max_intensity = max_intensity
        self.updata_viewer()

    @QtCore.Slot()
    def set_min_intensity(self, min_intensity):
        self.min_intensity = min_intensity

    def open_adjustment(self):
        img_ = self.viewer.input_file
        min_intensity = img_.min()
        max_intensity = img_.max()
        alpha = 255 / (max_intensity - min_intensity)
        beta = -min_intensity * alpha
        self.con_adj_setting = Contrast_adjustment_GUI(alpha, beta, min_intensity, max_intensity)

        self.con_adj_setting.alpha_signal.connect(self.set_alpha)
        self.con_adj_setting.beta_signal.connect(self.set_beta)
        self.con_adj_setting.min_intensity_signal.connect(self.set_min_intensity)
        self.con_adj_setting.max_intensity_signal.connect(self.set_max_intensity)

    def updata_viewer(self):
        self.viewer.current_pixmap = self.viewer.create_pixmap(self.image_series[self.viewer.slice_num])
        self.viewer.update_slice(self.viewer.current_pixmap)
        if self.viewer.mask_is_set is True:
            self.viewer.current_mask_pixmap = self.viewer.create_mask_pixmap(
                self.viewer.maskArray[self.viewer.slice_num, :, :])
            self.viewer.update_overlay(self.viewer.current_mask_pixmap)






