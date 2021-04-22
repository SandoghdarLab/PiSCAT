from piscat.InputOutput import read_write_data
from piscat.GUI.Projects.tab_bg_correction import BgCorrection_GUI
from piscat.GUI.Projects.tab_localization import Localization_GUI
from piscat.GUI import Tracking_GUI
from piscat.GUI.Projects.tab_plot_histogram import Histogram_GUI
from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat.InputOutput.read_write_data import load_dict_from_hdf5

from PySide2 import QtCore, QtWidgets
from functools import partial
import pandas as pd
import numpy as np
import os


class ProteinTabs(QtWidgets.QMainWindow):

    new_update_df_PSFs = QtCore.Signal(object)
    new_update_bg_video = QtCore.Signal(object)
    new_update_trajectories = QtCore.Signal(object)
    new_update_DRA_video = QtCore.Signal(object)

    def __init__(self, video_in, batch_size, object_update_progressBar, parent=None):
        super(ProteinTabs, self).__init__(parent)
        self.window = QtWidgets.QWidget()

        self.all_tabs = {}
        self.input_video = video_in
        self.batch_size = batch_size
        self.bgCorrectedVideo = None
        self.df_PSFs = None
        self.type_bgCorrection = None
        self.trajectories = None
        self.current_frame = 0
        self.object_update_progressBar = object_update_progressBar

        self.history = {'Type_bg_correction': None, 'bg_setting': None,
                        'Localization_setting': None, 'Tracking_setting': None}

        self.PSFs_Particels_num = {'#PSFs': None, '#Particles': None}

        self.initUI()

        self.setWindowTitle("Protein")
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.window.show()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('PSF Tracking')
        self.setGeometry(1140, 100, 400, 200)

        # Tabs
        self.all_tabs = {"Bg_correction": BgCorrection_GUI(input_video=self.input_video, object_update_progressBar=self.object_update_progressBar),
                         "PSFs_Localization": Localization_GUI(),
                         "PSFs_Tracking": Tracking_GUI(self.batch_size),
                         "Histogram": Histogram_GUI()}

        self.all_tabs["Bg_correction"].output_Tab_bgCorrection.connect(self.Update_tab_bgCorrection)
        self.all_tabs["Bg_correction"].output_batchSize_Tab_bgCorrection.connect(self.Update_batchSize)
        self.all_tabs["Bg_correction"].output_setting_Tab_bgCorrection.connect(partial(self.history_update, key='bg_setting'))
        self.all_tabs["Bg_correction"].update_tab_index.connect(self.update_tab)

        self.all_tabs["PSFs_Localization"].preview_localization.connect(partial(self.Update_tab_localization, flag_preview=True))
        self.all_tabs["PSFs_Localization"].update_localization.connect(partial(self.Update_tab_localization, flag_preview=False))
        self.all_tabs["PSFs_Localization"].output_setting_Tab_Localization.connect(partial(self.history_update, key='Localization_setting'))
        self.all_tabs["PSFs_Localization"].output_number_PSFs_tracking.connect(partial(self.number_PSFs, key='#PSFs'))
        self.all_tabs["PSFs_Localization"].update_tab_index.connect(self.update_tab)

        self.new_update_bg_video.connect(self.all_tabs["PSFs_Localization"].update_in_data)

        self.all_tabs["PSFs_Tracking"].update_tracking.connect(self.Update_tab_localization)
        self.all_tabs["PSFs_Tracking"].update_trajectories.connect(self.Update_tab_trajectories)
        self.all_tabs["PSFs_Tracking"].output_setting_Tab_tracking.connect(partial(self.history_update, key='Tracking_setting'))
        self.all_tabs["PSFs_Tracking"].output_number_Particels_tracking.connect(partial(self.number_PSFs, key='#Particles'))
        self.new_update_df_PSFs.connect(self.all_tabs["PSFs_Tracking"].update_in_data)

        self.new_update_trajectories.connect(self.all_tabs["Histogram"].update_in_data)

        self.main_tabWidget = QtWidgets.QTabWidget()
        self.main_tabWidget.addTab(self.all_tabs["Bg_correction"], "Background correction")
        self.main_tabWidget.addTab(self.all_tabs["PSFs_Localization"], "PSFs localization")
        self.main_tabWidget.addTab(self.all_tabs["PSFs_Tracking"], "PSFs tracking")
        self.main_tabWidget.addTab(self.all_tabs["Histogram"], "Histogram")

        self.btn_save = QtWidgets.QPushButton("Save Data_History")
        self.btn_save.setAutoDefault(False)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_save.setFixedWidth(120)
        self.btn_save.setFixedHeight(20)

        self.btn_load_csv = QtWidgets.QPushButton("Load CSV")
        self.btn_load_csv.setAutoDefault(False)
        self.btn_load_csv.clicked.connect(self.load_csv_data)
        self.btn_load_csv.setFixedWidth(100)
        self.btn_load_csv.setFixedHeight(20)

        self.btn_load_hdf5 = QtWidgets.QPushButton("Load hdf5")
        self.btn_load_hdf5.setAutoDefault(False)
        self.btn_load_hdf5.clicked.connect(self.load_hdf5_data)
        self.btn_load_hdf5.setFixedWidth(120)
        self.btn_load_hdf5.setFixedHeight(20)

        self.statusbar = self.statusBar()
        self.statusBar().addPermanentWidget(self.btn_save)
        self.statusBar().addPermanentWidget(self.btn_load_csv)
        self.statusBar().addPermanentWidget(self.btn_load_hdf5)

        self.main_tabWidget.setCurrentIndex(0)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.main_tabWidget)
        self.layout.setContentsMargins(0, 0, 0, 0)

    @QtCore.Slot()
    def update_input(self, video_in):
        self.input_video = video_in

    def __del__(self):
        print('Destructor called, Employee deleted.')

    @QtCore.Slot(int)
    def update_tab(self, idx_):
        self.main_tabWidget.setCurrentIndex(idx_)

    @QtCore.Slot(int)
    def setProgress(self, val):
        self.progressBar.setValue(val)

    @QtCore.Slot()
    def Update_batchSize(self, batchSize):
        self.batch_size = batchSize
        self.all_tabs["PSFs_Tracking"].update_batchSize(batchSize=batchSize)

    @QtCore.Slot()
    def Update_tab_bgCorrection(self, data):
        if data[0] is not None:
            self.bgCorrectedVideo = data[0]
            self.history['Type_bg_correction'] = data[1]
            self.new_update_bg_video.emit(self.bgCorrectedVideo)

            if data[1] == 'DRA':
                self.new_update_DRA_video.emit([self.bgCorrectedVideo, 'DRA', None, self.batch_size])

            self.visualization_ = Visulization_localization()
            self.visualization_.bg_correction_update(in_video=self.bgCorrectedVideo, label=self.type_bgCorrection, object=self.all_tabs["PSFs_Localization"])
        else:
            self.msg_box2 = QtWidgets.QMessageBox()
            self.msg_box2.setWindowTitle("Warning!")
            self.msg_box2.setText("Input video does not find!")
            self.msg_box2.exec_()

    @QtCore.Slot()
    def Update_tab_localization(self, df_psfs, flag_preview=False):
        self.df_PSFs = df_psfs
        if self.bgCorrectedVideo is not None and self.df_PSFs is not None:
            if type(self.df_PSFs) is np.ndarray:
                self.visualization_.get_sliceNumber(frame_num=self.all_tabs["PSFs_Localization"].frame_num)
                self.visualization_.update_localization_onMask(video_in=self.bgCorrectedVideo,
                                                               title=self.type_bgCorrection, df_PSFs=self.df_PSFs)

            elif type(self.df_PSFs) is pd.core.frame.DataFrame:
                self.visualization_.get_sliceNumber(frame_num=self.all_tabs["PSFs_Localization"].frame_num)
                self.visualization_.update_localization_onMask(video_in=self.bgCorrectedVideo, title=self.type_bgCorrection,
                                                               df_PSFs=self.df_PSFs, flag_preview=flag_preview)

                self.new_update_df_PSFs.emit([self.bgCorrectedVideo, self.df_PSFs])

        elif self.bgCorrectedVideo is None:
            self.msg_box1 = QtWidgets.QMessageBox()
            self.msg_box1.setWindowTitle("Warning!")
            self.msg_box1.setText("Input Video is None!")
            self.msg_box1.exec_()

        elif self.df_PSFs is None:
            self.msg_box1 = QtWidgets.QMessageBox()
            self.msg_box1.setWindowTitle("Warning!")
            self.msg_box1.setText("Zero particle detected!")
            self.msg_box1.exec_()

    @QtCore.Slot()
    def Update_tab_trajectories(self, trajectories):
        self.trajectories = trajectories
        self.new_update_trajectories.emit([trajectories, self.input_video.shape[0], self.batch_size])

    @QtCore.Slot()
    def history_update(self, history, key):
        self.history[key] = history

    @QtCore.Slot()
    def number_PSFs(self, data_in, key):
        self.PSFs_Particels_num[key] = data_in

    def save_data(self):
        self.file_path = False
        self.file_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        if self.file_path:
            if self.df_PSFs is not None:
                read_write_data.save_df2csv(self.df_PSFs, path=self.file_path, name='df_PSFs')

            fout = os.path.join(self.file_path, 'history.txt')
            fo = open(fout, "w")
            for k, v in self.history.items():
                fo.write(str(k) + ' >>> ' + str(v) + '\n\n')

            for k, v in self.PSFs_Particels_num.items():
                fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
            fo.close()

            if self.trajectories is not None:
                read_write_data.save_mat(data=self.trajectories, path=self.file_path, name='all_trajectories')
                read_write_data.save_list_to_hdf5(list_data=self.trajectories, path=self.file_path, name='histData')

    def load_hdf5_data(self):
        if self.bgCorrectedVideo is not None:
            try:
                fileName = QtWidgets.QFileDialog().getOpenFileName()
                filePath = str(fileName[0])
                data_dic = load_dict_from_hdf5(filePath)
                self.Update_tab_trajectories(data_dic)
            except:
                print('HDF5 can not load!')
        elif self.bgCorrectedVideo is None:
            self.msg_box1 = QtWidgets.QMessageBox()
            self.msg_box1.setWindowTitle("Warning!")
            self.msg_box1.setText("Update background tab!")
            self.msg_box1.exec_()

    def load_csv_data(self):
        if self.bgCorrectedVideo is not None:
            try:
                fileName = QtWidgets.QFileDialog().getOpenFileName()
                filePath = str(fileName[0])
                df_PSFs = pd.read_csv(filePath)
                self.Update_tab_localization(df_PSFs)
            except:
                print('CSV can not load!')
        elif self.bgCorrectedVideo is None:
            self.msg_box1 = QtWidgets.QMessageBox()
            self.msg_box1.setWindowTitle("Warning!")
            self.msg_box1.setText("Update background tab!")
            self.msg_box1.exec_()