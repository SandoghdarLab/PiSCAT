import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from piscat.BackgroundCorrection.noise_floor import NoiseFloor
from piscat.Preproccessing import Normalization


class Noise_Floor(QtWidgets.QMainWindow):
    def __init__(self, video, parent=None):
        super(Noise_Floor, self).__init__(parent)
        self.window = QtWidgets.QWidget()

        self.original_video = video
        self.original_video_pn = None

        self.min_radius = None
        self.max_radius = None
        self.step_radius = None
        self.radius_list = None
        self.mode = None
        self.file_path = None
        self.find_radius_update_tab_flag = True

        # self.checkbox_power_normalization = QtWidgets.QCheckBox("Laser power normalization", self)

        self.groupBox_PN = QtWidgets.QGroupBox("Laser Power normalization:")
        self.groupBox_PN.setCheckable(True)
        self.groupBox_PN.setChecked(False)

        self.start_win_size_x = QtWidgets.QLineEdit()
        self.start_win_size_x.setPlaceholderText("ROI X start")
        self.start_win_size_x_label = QtWidgets.QLabel("start width pixel:")

        self.end_win_size_x = QtWidgets.QLineEdit()
        self.end_win_size_x.setPlaceholderText("ROI X end")
        self.end_win_size_x_label = QtWidgets.QLabel("end width pixel:")

        self.start_win_size_y = QtWidgets.QLineEdit()
        self.start_win_size_y.setPlaceholderText("ROI Y start")
        self.start_win_size_y_label = QtWidgets.QLabel("start hight pixel:")

        self.end_win_size_y = QtWidgets.QLineEdit()
        self.end_win_size_y.setPlaceholderText("ROI X end")
        self.end_win_size_y_label = QtWidgets.QLabel("start hight pixel:")

        self.selected_frame = QtWidgets.QLineEdit()
        self.selected_frame.setPlaceholderText("selected frame")
        self.selected_frame_label = QtWidgets.QLabel("selected preview frame:")
        self.selected_frame.setText("0")

        self.preview_roi = QtWidgets.QPushButton("ROI preview")
        self.preview_roi.setAutoDefault(False)
        self.preview_roi.clicked.connect(self.preview_roi_plot)
        self.preview_roi.setFixedHeight(20)
        self.preview_roi.setFixedWidth(100)

        grid_pn = QtWidgets.QGridLayout()
        grid_pn.addWidget(self.start_win_size_x_label, 0, 0)
        grid_pn.addWidget(self.start_win_size_x, 0, 1)

        grid_pn.addWidget(self.end_win_size_x_label, 0, 2)
        grid_pn.addWidget(self.end_win_size_x, 0, 3)

        grid_pn.addWidget(self.start_win_size_y_label, 1, 0)
        grid_pn.addWidget(self.start_win_size_y, 1, 1)

        grid_pn.addWidget(self.end_win_size_y_label, 1, 2)
        grid_pn.addWidget(self.end_win_size_y, 1, 3)

        grid_pn.addWidget(self.selected_frame_label, 2, 0)
        grid_pn.addWidget(self.selected_frame, 2, 1)
        grid_pn.addWidget(self.preview_roi, 2, 2)

        self.groupBox_PN.setLayout(grid_pn)

        self.checkbox_save = QtWidgets.QCheckBox("Saving as CSV", self)
        self.checkbox_save.toggled.connect(lambda: self.save_active())

        self.checkbox_loglog_scale = QtWidgets.QRadioButton("log scale")
        self.checkbox_loglog_scale.setChecked(True)
        self.checkbox_normal_scale = QtWidgets.QRadioButton("normal scale")

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)
        self.ok.setFixedWidth(100)

        self.plot = QtWidgets.QPushButton("Plot")
        self.plot.setAutoDefault(False)
        self.plot.clicked.connect(self.do_plot)
        self.plot.setFixedWidth(100)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.createThirdExclusiveGroup(), 2, 0)
        self.grid.addWidget(self.groupBox_PN, 5, 0)
        self.grid.addWidget(self.checkbox_save, 6, 0)
        self.grid.addWidget(self.checkbox_loglog_scale, 7, 0)
        self.grid.addWidget(self.checkbox_normal_scale, 8, 0)
        self.grid.addWidget(self.ok, 9, 0)
        self.grid.addWidget(self.plot, 10, 0)

        self.setWindowTitle("Noise Floor")
        self.window.setLayout(self.grid)
        self.window.show()

    def __del__(self):
        del self
        print("Destructor called, Employee deleted.")

    def createFirstExclusiveGroup(self):
        self.groupBox_range = QtWidgets.QGroupBox("Give range for batch sizes:")
        self.groupBox_range.setCheckable(True)
        self.groupBox_range.setChecked(False)
        self.grid1 = QtWidgets.QGridLayout()
        self.add_line_edit1()
        self.groupBox_range.setLayout(self.grid1)

        return self.groupBox_range

    def createSecondExclusiveGroup(self):
        self.groupBox_list = QtWidgets.QGroupBox("Give list for batch sizes")
        self.groupBox_list.setCheckable(True)
        self.groupBox_list.setChecked(False)
        self.grid2 = QtWidgets.QGridLayout()
        self.add_line_edit2()
        self.groupBox_list.setLayout(self.grid2)

        return self.groupBox_list

    def createThirdExclusiveGroup(self):
        self.groupBox_FPNc = QtWidgets.QGroupBox("FPNc:")
        self.groupBox_FPNc.setCheckable(True)
        self.groupBox_FPNc.setChecked(False)

        self.FPN_mode_group = QtWidgets.QButtonGroup()
        self.radio_wFPN_mode = QtWidgets.QRadioButton("Wavelet FPNc")
        self.radio_mFPN_mode = QtWidgets.QRadioButton("Median FPNc")
        self.radio_cpFPN_mode = QtWidgets.QRadioButton("Column projection FPNc")
        self.radio_fft_FPN_mode = QtWidgets.QRadioButton("FFT2D FPNc")

        self.FPN_mode_group.addButton(self.radio_wFPN_mode)
        self.FPN_mode_group.addButton(self.radio_mFPN_mode)
        self.FPN_mode_group.addButton(self.radio_cpFPN_mode)
        self.FPN_mode_group.addButton(self.radio_fft_FPN_mode)

        self.radio_wFPN_mode.toggled.connect(self.axis_selection)
        self.radio_mFPN_mode.toggled.connect(self.axis_selection)
        self.radio_cpFPN_mode.toggled.connect(self.axis_selection)
        self.radio_fft_FPN_mode.toggled.connect(self.axis_selection)

        self.axis_group = QtWidgets.QButtonGroup()
        self.radio_axis_1 = QtWidgets.QRadioButton("FPNc in axis 0")
        self.radio_axis_2 = QtWidgets.QRadioButton("FPNc in axis 1")
        self.radio_axis_3 = QtWidgets.QRadioButton("FPNc in Both axis")
        self.radio_axis_2.setChecked(True)
        self.radio_mFPN_mode.setChecked(True)

        self.axis_group.addButton(self.radio_axis_1)
        self.axis_group.addButton(self.radio_axis_2)
        self.axis_group.addButton(self.radio_axis_3)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.radio_mFPN_mode, 1, 0)
        grid.addWidget(self.radio_cpFPN_mode, 2, 0)
        grid.addWidget(self.radio_wFPN_mode, 3, 0)
        grid.addWidget(self.radio_fft_FPN_mode, 4, 0)

        grid.addWidget(self.radio_axis_1, 1, 1)
        grid.addWidget(self.radio_axis_2, 2, 1)
        grid.addWidget(self.radio_axis_3, 3, 1)

        self.groupBox_FPNc.setLayout(grid)
        return self.groupBox_FPNc

    def axis_selection(self):
        if self.radio_mFPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_wFPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_fft_FPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_cpFPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

    @QtCore.Slot()
    def do_update(self):
        if self.ok.clicked:
            self.get_values()
            if self.groupBox_range.isChecked() and not (self.groupBox_list.isChecked()):
                if self.min_radius != "" and self.max_radius != "" and self.step_radius != "":
                    self.min_radius = int(self.min_radius)
                    self.max_radius = int(self.max_radius)
                    self.step_radius = int(self.step_radius)
                    self.radius_list = list(
                        range(self.min_radius, self.max_radius, self.step_radius)
                    )
                    self.mode = "Range"
                    self.find_radius_update_tab_flag = False
                    self.run_noiseFloor()

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Please insert radius size!")
                    self.msg_box.exec_()

            if self.groupBox_list.isChecked() and not (self.groupBox_range.isChecked()):
                if self.radius_list != "":
                    self.radius_list = eval(self.radius_list)
                    self.mode = "List"
                    self.find_radius_update_tab_flag = False
                    self.run_noiseFloor()
                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Please insert radius size!")
                    self.msg_box.exec_()

            elif self.groupBox_list.isChecked() and self.groupBox_range.isChecked():
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Only one method should active!")
                self.msg_box.exec_()

            elif not (self.groupBox_list.isChecked()) and not (self.groupBox_range.isChecked()):
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Please select one of the methods!")
                self.msg_box.exec_()

    def run_noiseFloor(self):
        if self.groupBox_PN.isChecked():
            try:
                self.s_x_win_pn = int(self.start_win_size_x.text())
                self.e_x_win_pn = int(self.end_win_size_x.text())
                self.s_y_win_pn = int(self.start_win_size_y.text())
                self.e_y_win_pn = int(self.end_win_size_y.text())
            except:
                self.s_x_win_pn = None
                self.e_x_win_pn = None
                self.s_y_win_pn = None
                self.e_y_win_pn = None

            self.original_video_pn, _ = Normalization(video=self.original_video).power_normalized(
                roi_x=(self.s_x_win_pn, self.e_x_win_pn),
                roi_y=(self.s_y_win_pn, self.e_y_win_pn),
            )
        else:
            self.original_video_pn = self.original_video

        if self.groupBox_FPNc.isChecked():
            FPN_flag = True
            if self.radio_axis_1.isChecked():
                self.axis = 0
            elif self.radio_axis_2.isChecked():
                self.axis = 1
            elif self.radio_axis_3.isChecked():
                self.axis = "Both"

            if self.radio_mFPN_mode.isChecked():
                self.mode_FPN = "mFPN"
            elif self.radio_cpFPN_mode.isChecked():
                self.mode_FPN = "cpFPN"
            elif self.radio_wFPN_mode.isChecked():
                self.mode_FPN = "wFPN"
            elif self.radio_fft_FPN_mode.isChecked():
                self.mode_FPN = "fFPN"

        else:
            FPN_flag = False
            self.axis = None
            self.mode_FPN = "mFPN"

        result_flag = True
        n_jobs = os.cpu_count()
        inter_flag_parallel_active = True
        flag_first_except = False
        while result_flag:
            try:
                self.noise_floor_ = NoiseFloor(
                    self.original_video_pn,
                    list_range=self.radius_list,
                    select_correction_axis=self.axis,
                    FPN_flag=FPN_flag,
                    mode_FPN=self.mode_FPN,
                    n_jobs=None,
                    inter_flag_parallel_active=inter_flag_parallel_active,
                )

                if self.checkbox_loglog_scale.isChecked():
                    self.noise_floor_.plot_result(flag_log=True)
                else:
                    self.noise_floor_.plot_result(flag_log=False)

                if self.checkbox_save.isChecked() and self.file_path is not None:
                    noise_floor = {
                        "batch size": self.noise_floor_.list_range,
                        "SNR": self.noise_floor_.mean,
                    }
                    noise_floor_df = pd.DataFrame(data=noise_floor)
                    noise_floor_df.to_csv(self.file_path)
                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Nothing Save!")
                    self.msg_box.exec_()

                result_flag = False
            except:
                inter_flag_parallel_active = False
                if flag_first_except:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Memory Error! Please off parallel CPU.")
                    self.msg_box.exec_()
                    result_flag = False
                flag_first_except = True

    def do_plot(self):
        try:
            if self.checkbox_loglog_scale.isChecked():
                self.noise_floor_.plot_result(flag_log=True)
            else:
                self.noise_floor_.plot_result(flag_log=False)
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please update first!5")
            self.msg_box.exec_()

    def preview_roi_plot(self, selected_frame):
        if self.groupBox_PN.isChecked():
            self.flag_power_normalization = True
            try:
                self.s_x_win_pn = int(self.start_win_size_x.text())
                self.e_x_win_pn = int(self.end_win_size_x.text())
                self.s_y_win_pn = int(self.start_win_size_y.text())
                self.e_y_win_pn = int(self.end_win_size_y.text())
                selected_frame = int(self.selected_frame.text())

                img_ = self.original_video[selected_frame, :, :]
                roi_img_ = img_[
                    self.s_x_win_pn : self.e_x_win_pn, self.s_y_win_pn : self.e_y_win_pn
                ]

                fig, ax = plt.subplots()
                ax.imshow(img_, cmap="gray")
                rect = patches.Rectangle(
                    (self.s_x_win_pn, self.s_y_win_pn),
                    roi_img_.shape[0],
                    roi_img_.shape[1],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                plt.show()

            except:
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("The ROI is not defined!")
                self.msg_box1.exec_()

    def get_values(self):
        if self.groupBox_range.isChecked():
            self.min_radius = self.le1.text()
            self.max_radius = self.le2.text()
            self.step_radius = self.le3.text()

        elif self.groupBox_list.isChecked():
            self.radius_list = self.le4.text()

    def add_line_edit1(self):
        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText("Min. batch size")
        self.le_1_label = QtWidgets.QLabel("Min batch size:")

        self.le2 = QtWidgets.QLineEdit()
        self.le2.setPlaceholderText("Max. batch size")
        self.le_2_label = QtWidgets.QLabel("Max batch size:")

        self.le3 = QtWidgets.QLineEdit()
        self.le3.setPlaceholderText("step")
        self.le_3_label = QtWidgets.QLabel("Stride between batch size:")

        self.grid1.addWidget(self.le_1_label, 2, 0)
        self.grid1.addWidget(self.le_2_label, 3, 0)
        self.grid1.addWidget(self.le_3_label, 4, 0)

        self.grid1.addWidget(self.le1, 2, 1)
        self.grid1.addWidget(self.le2, 3, 1)
        self.grid1.addWidget(self.le3, 4, 1)

    def add_line_edit2(self):
        self.le4 = QtWidgets.QLineEdit()
        self.le4.setPlaceholderText("list of batches size")
        self.le_4_label = QtWidgets.QLabel("List of all batches")

        self.grid2.addWidget(self.le_4_label, 0, 0)
        self.grid2.addWidget(self.le4, 1, 1)

    def save_active(self):
        if self.checkbox_save.isChecked():
            self.file_path = False

            self.file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Noise floor", QtCore.QDir.currentPath()
            )

            self.file_path = self.file_path + "_noise_floor.csv"

    def closeEvent(self, event):
        event.accept()  # let the window close
