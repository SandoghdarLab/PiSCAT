from piscat.InputOutput import read_write_data
from piscat.GUI.BatchAnalysis.tab_bg_correction import BgCorrection_GUI
from piscat.GUI.BatchAnalysis.tab_localization import Localization_GUI
from piscat.GUI.BatchAnalysis.tab_reading_data import LoadData_GUI
from piscat.GUI.BatchAnalysis.tab_plot_histogram import Histogram_GUI
from piscat.GUI.BatchAnalysis.tab_trajectory import Tracking_GUI
from piscat.GUI.BatchAnalysis.tab_analysis_protein import ProteinAnalysis_GUI


from piscat.InputOutput.read_write_data import load_dict_from_hdf5

from PySide6 import QtCore, QtWidgets
from functools import partial
import pandas as pd
import numpy as np
import time
import os


class BatchAnalysis(QtWidgets.QMainWindow):

    def __init__(self, object_update_progressBar, parent=None):
        super(BatchAnalysis, self).__init__(parent)
        self.window = QtWidgets.QWidget()

        self.all_tabs = {}
        self.type_bgCorrection = None
        self.current_frame = 0
        self.object_update_progressBar = object_update_progressBar

        self.type_file = None
        self.dirName = None
        self.name_mkdir = None

        self.hyperparameters = {'im_size_x': None, 'im_size_y': None, 'start_fr': None, 'end_fr': None, 'image_format': None,
                                'batch_size': None, 'mode_FPN': 'mFPN', 'select_correction_axis': 'Both',
                                'function': None, 'Mode_PSF_Segmentation': 'BOTH', 'min_sigma': None, 'max_sigma': None, 'sigma_ratio': None, 'PSF_detection_thr': None, 'overlap': None,
                                'search_range': None, 'memory': None,
                                'outlier_frames_thr': None, 'symmetric_PSFs_thr': None, 'min_V_shape_width': None}

        self.flags = {'PN': None, 'DRA': None, 'FPNc': None, 'outlier_frames_filter': None, 'Dense_Filter': None, 'symmetric_PSFs_Filter': None, 'FFT_flag': None}

        self.history = {'Type_bg_correction': None, 'bg_setting': None,
                        'PSFs_Localization_setting': None, 'PSFs_Tracking_setting': None}

        self.PSFs_Particels_num = {'#PSFs': None, '#Particles': None}

        self.initUI()

        self.setWindowTitle("Batch analysis")
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.window.show()

    def initUI(self):
        self.resize(400, 300)
        self.setWindowTitle('Batch analysis')
        self.setGeometry(1140, 100, 400, 200)

        # Tabs
        self.input_video = None
        self.batch_size = None
        self.all_tabs = {"Load_data": LoadData_GUI(),
                         "Bg_correction": BgCorrection_GUI(),
                         "PSFs_Localization": Localization_GUI(),
                         "PSFs_Tracking": Tracking_GUI(self.batch_size),
                         "Protein_Analysis": ProteinAnalysis_GUI(),
                         "Histogram": Histogram_GUI()}

        self.all_tabs["Load_data"].update_output_external.connect(self.update_read_data)
        self.all_tabs["Load_data"].update_tab_index.connect(self.update_tab)

        self.all_tabs["Bg_correction"].output_setting_Tab_bgCorrection.connect(self.update_BGc)
        self.all_tabs["Bg_correction"].update_tab_index.connect(self.update_tab)

        self.all_tabs["PSFs_Localization"].output_setting_Tab_Localization_external.connect(self.Update_tab_localization)
        self.all_tabs["PSFs_Localization"].update_tab_index.connect(self.update_tab)

        self.all_tabs["PSFs_Tracking"].output_setting_Tab_tracking.connect(self.Update_tab_trajectories)
        self.all_tabs["PSFs_Tracking"].update_tab_index.connect(self.update_tab)

        self.main_tabWidget = QtWidgets.QTabWidget()
        self.main_tabWidget.addTab(self.all_tabs["Load_data"], "Load data")
        self.main_tabWidget.addTab(self.all_tabs["Bg_correction"], "Background correction")
        self.main_tabWidget.addTab(self.all_tabs["PSFs_Localization"], "PSFs localization")
        self.main_tabWidget.addTab(self.all_tabs["PSFs_Tracking"], "PSFs tracking")
        self.main_tabWidget.addTab(self.all_tabs["Protein_Analysis"], "Protein analysis")
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

    def __del__(self):
        print('Destructor called, Employee deleted.')

    @QtCore.Slot(int)
    def update_tab(self, idx_):
        if idx_ == 4:
            self.all_tabs["Protein_Analysis"].updata_input_setting([self.hyperparameters, self.flags])
            self.main_tabWidget.setCurrentIndex(idx_)
        else:
            self.main_tabWidget.setCurrentIndex(idx_)

    @QtCore.Slot(int)
    def update_read_data(self, data_in):
        self.hyperparameters['im_size_x'] = data_in['img_width']
        self.hyperparameters['im_size_y'] = data_in['img_height']
        self.hyperparameters['width_size_s'] = data_in['width_size_s']
        self.hyperparameters['width_size_e'] = data_in['width_size_e']
        self.hyperparameters['height_size_s'] = data_in['height_size_s']
        self.hyperparameters['height_size_e'] = data_in['height_size_e']
        self.hyperparameters['start_fr'] = data_in['s_frame']
        self.hyperparameters['end_fr'] = data_in['e_frame']
        self.hyperparameters['frame_stride'] = data_in['frame_stride']
        self.hyperparameters['image_format'] = data_in['image_type']
        self.hyperparameters['type_file'] = data_in['type_file']
        self.hyperparameters['path'] = data_in['path']
        self.hyperparameters['name_mkdir'] = data_in['name_mkdir']

    @QtCore.Slot()
    def update_BGc(self, data_in):
        for key in data_in:
            if key == 'Batch_size (frames)':
                self.hyperparameters['batch_size'] = data_in[key]
            elif key == 'FPNc_axis':
                self.hyperparameters['select_correction_axis'] = data_in[key]
            elif key == 'Power_Normalization':
                self.flags['PN'] = data_in['Power_Normalization']
            elif key == 'FPNc':
                self.flags[key] = data_in[key]
            elif key == 'filter_hotPixels':
                self.flags[key] = data_in[key]
            elif key == 'type_BGc':
                if data_in['type_BGc'] == 'DRA':
                    self.flags['DRA'] = True
                else:
                    self.flags['DRA'] = False
            else:
                self.hyperparameters[key] = data_in[key]

        self.flags['FFT_flag'] = True

    @QtCore.Slot()
    def Update_tab_localization(self, data_in):

        for key in data_in:
            if key == 'mode':
                self.hyperparameters['Mode_PSF_Segmentation'] = data_in[key]
            elif key == 'threshold_min':
                self.hyperparameters['PSF_detection_thr'] = data_in[key]
            elif key == 'Asymmetry_PSFs_filtering_threshold':
                self.hyperparameters['symmetric_PSFs_thr'] = data_in[key]
            elif key == 'Outlier_frames_filter_max_number_PSFs':
                self.hyperparameters['outlier_frames_thr'] = data_in[key]
            elif key == 'Flag_fine_localization':
                self.flags['Fine_localization'] = data_in[key]
            elif key == 'Flag_fit_2DGaussian':
                self.flags['Flag_fit_2DGaussian'] = data_in[key]
            elif key == 'Side_lobes_Filter' or key == 'Dense_Filter' or key == 'outlier_frames_filter' or key == 'symmetric_PSFs_Filter':
                self.flags[key] = data_in[key]
            else:
                self.hyperparameters[key] = data_in[key]

    @QtCore.Slot()
    def Update_tab_trajectories(self, data_in):
        self.hyperparameters['search_range'] = data_in['Neighborhood_size (px)']
        self.hyperparameters['memory'] = data_in['Memory (frame)']
        self.hyperparameters['min_V_shape_width'] = data_in['Minimum_temporal_length (frame)']

    def save_data(self):
        self.file_path = False
        self.file_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name_mkdir = timestr
        try:
            dr_mk = os.path.join(self.file_path, name_mkdir)
            os.mkdir(dr_mk)
            print("Directory ", name_mkdir, " Created ")
        except FileExistsError:
            dr_mk = os.path.join(self.file_path, name_mkdir)
            print("Directory ", name_mkdir, " already exists")

        self.file_path = dr_mk

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

