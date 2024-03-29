import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore, QtWidgets

from piscat.GUI.BackgroundCorrection import FUN_DRA
from piscat.GUI.InputOutput import Reading
from piscat.Preproccessing.filtering import Filters


class BgCorrection_GUI(QtWidgets.QWidget):
    output_Tab_bgCorrection = QtCore.Signal(object)
    output_batchSize_Tab_bgCorrection = QtCore.Signal(int)
    output_setting_Tab_bgCorrection = QtCore.Signal(object)
    update_tab_index = QtCore.Signal(int)

    def __init__(self, input_video, object_update_progressBar, parent=None):
        super(BgCorrection_GUI, self).__init__(parent)
        self.different_views = {}

        self.input_video = input_video
        self.object_update_progressBar = object_update_progressBar
        self.setting_bg_correction = {}

        self.flag_remove_box = True
        self.flag_update_class = True

        self.output = None
        self.diff_video = None
        self.original_video_bg = None
        self.reconstruction_background = None
        self.empty_value_box_flag = False
        self.flag_no_warning = True

        self.transform_algorithm = None
        self.n_comp = None
        self.n_iter = None
        self.alpha = None
        self.random_select = None
        self.non_zero_coeff = None
        self.patch_size = None
        self.strides = None

        self.resize(600, 600)
        self.setWindowTitle("Background Correction")

        self.combo_bg_filter = QtWidgets.QComboBox(self)
        self.combo_bg_filter.addItem("-Select the background correction methods-")
        self.combo_bg_filter.addItem("Differential rolling average")
        self.combo_bg_filter.addItem("Spatial median Filter")
        self.combo_bg_filter.addItem("Spatial gaussian Filter")
        self.combo_bg_filter.addItem("Background correction temporal median")
        self.combo_bg_filter.addItem("Flat field (Gaussian filter)")
        self.combo_bg_filter.addItem("Flat field (mean background subtraction)")
        self.combo_bg_filter.currentIndexChanged.connect(self.layout_based_on_select_bg)

        self.btn = QtWidgets.QPushButton("Update")
        self.btn.setAutoDefault(False)
        self.btn.clicked.connect(self.do_update)
        self.btn.setFixedWidth(70)
        self.btn.setFixedHeight(20)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.combo_bg_filter, 0, 0)
        self.grid.addWidget(self.createFirstExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.btn, 2, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print("Destructor called, Employee deleted.")

    def createFirstExclusiveGroup(self):
        self.groupBox_update = QtWidgets.QGroupBox("Background correction Setting:")
        self.grid1 = QtWidgets.QGridLayout()

        self.groupBox_update.setLayout(self.grid1)
        return self.groupBox_update

    def massage_filled(self):
        self.flag_no_warning = False
        self.msg_box1 = QtWidgets.QMessageBox()
        self.msg_box1.setWindowTitle("Warning!")
        self.msg_box1.setText("Please filled all parameters!")
        self.msg_box1.exec_()

    @QtCore.Slot()
    def do_update(self):
        if self.btn.clicked:
            if self.combo_bg_filter.currentText() == "Differential rolling average":
                self.batch_size = self.le_batchSize.text()

                if self.groupBox_FPNc.isChecked():
                    self.flag_FPN = True
                    if self.radio_axis_1.isChecked():
                        self.axis = 0
                    elif self.radio_axis_2.isChecked():
                        self.axis = 1
                    elif self.radio_axis_3.isChecked():
                        self.axis = "Both"

                    if self.radio_cpFPN_mode.isChecked():
                        self.mode_FPN = "cpFPN"
                    elif self.radio_wFPN_mode.isChecked():
                        self.mode_FPN = "wFPN"
                    elif self.radio_FFT2D_FPN_mode.isChecked():
                        self.mode_FPN = "fFPN"
                    elif self.radio_median_FPN_mode.isChecked():
                        self.mode_FPN = "mFPN"
                else:
                    self.flag_FPN = False
                    self.axis = 0
                    self.mode_FPN = "fFPN"

                if self.groupBox_PN.isChecked():
                    self.flag_power_normalization = True
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
                else:
                    self.flag_power_normalization = False
                    self.s_x_win_pn = None
                    self.e_x_win_pn = None
                    self.s_y_win_pn = None
                    self.e_y_win_pn = None

                if self.batch_size != "":
                    self.batch_size = int(self.batch_size)
                    self.output_batchSize_Tab_bgCorrection.emit(self.batch_size)

                    dra_ = FUN_DRA(
                        video=self.input_video,
                        object_update_progressBar=self.object_update_progressBar,
                    )
                    dra_video = dra_.run_DRA_from_bgtabs(
                        mode_FPN=self.mode_FPN,
                        batch_size=self.batch_size,
                        flag_power_normalization=self.flag_power_normalization,
                        roi_x_pn=(self.s_x_win_pn, self.e_x_win_pn),
                        roi_y_pn=(self.s_y_win_pn, self.e_y_win_pn),
                        flag_FPN=self.flag_FPN,
                        axis=self.axis,
                    )

                    self.output = [dra_video, "DRA"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction["mode_FPN"] = self.mode_FPN
                    self.setting_bg_correction["Batch_size (frames)"] = self.batch_size
                    self.setting_bg_correction[
                        "Power_Normalization"
                    ] = self.flag_power_normalization
                    self.setting_bg_correction["FPNc"] = self.flag_FPN
                    self.setting_bg_correction["FPNc_axis"] = self.axis

                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Please insert batch size!")
                    self.msg_box.exec_()

            elif self.combo_bg_filter.currentText() == "Spatial median Filter":
                median_size = self.line_edit1.text()
                if median_size != "":
                    print("\nMedian is Applying--->", end="")
                    median_size = int(self.line_edit1.text())
                    blur = Filters(self.input_video)
                    blur_video = blur.median(median_size)
                    self.output = [blur_video, "Spatial median Filter"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction["Spatial_median_kernel_size"] = median_size
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    self.output_batchSize_Tab_bgCorrection.emit(0)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Spatial gaussian Filter":
                gaussian_sigma = self.line_edit2.text()
                if gaussian_sigma != "":
                    print("\nGaussian is applying--->", end="")
                    gaussian_sigma = float(self.line_edit2.text())
                    blur = Filters(self.input_video)
                    blur_video = blur.gaussian(gaussian_sigma)
                    self.output = [blur_video, "Spatial gaussian Filter"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction["Spatial_gaussian_sigma"] = gaussian_sigma
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    self.output_batchSize_Tab_bgCorrection.emit(0)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Background correction temporal median":
                print("\nBackground correction temporal median--->", end="")
                blur = Filters(self.input_video)
                self.output = [blur.temporal_median(), "Background correction temporal median"]
                self.output_Tab_bgCorrection.emit(self.output)
                self.output_batchSize_Tab_bgCorrection.emit(0)
                print("Done!")

            elif self.combo_bg_filter.currentText() == "Flat field (Gaussian filter)":
                sigma = self.line_edit5.text()
                if sigma != "":
                    print("\nBackground correction Flat Field--->", end="")
                    sigma = int(self.line_edit5.text())
                    blur = Filters(self.input_video)
                    self.output = [blur.flat_field(sigma=sigma), "Flat field (Gaussian filter)"]
                    self.output_Tab_bgCorrection.emit(self.output)
                    self.setting_bg_correction["Flat_field_sigma"] = sigma
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    self.output_batchSize_Tab_bgCorrection.emit(0)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Flat field (mean background subtraction)":
                if self.original_video_bg is not None:
                    print("\nBackground correction Flat Field back ground--->", end="")

                    if (
                        self.original_video_bg.shape[1] == self.input_video.shape[1]
                        and self.original_video_bg.shape[2] == self.input_video.shape[2]
                    ):
                        self.output = np.divide(
                            self.input_video, np.mean(self.original_video_bg, axis=0)
                        )
                        self.output_Tab_bgCorrection.emit(
                            [self.output, "Flat field (mean background subtraction)"]
                        )
                        self.output_batchSize_Tab_bgCorrection.emit(0)
                        print("Done!")

                    else:
                        self.flag_no_warning = False
                        self.msg_box1 = QtWidgets.QMessageBox()
                        self.msg_box1.setWindowTitle("Warning!")
                        self.msg_box1.setText("The size of these two videos is not the same!")
                        self.msg_box1.exec_()

                else:
                    self.massage_filled()

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Continuing without using any filter!")
                self.msg_box.exec_()
                self.output = self.input_video
                self.output_Tab_bgCorrection.emit([self.output, "RAW"])
                self.output_batchSize_Tab_bgCorrection.emit(0)

            self.update_tab_index.emit(1)

    def on_select(self):
        self.transform_algorithm = self.combo.currentText()
        self.create_como_values()

    def axis_selection(self):
        if self.radio_cpFPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_wFPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_median_FPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

        elif self.radio_FFT2D_FPN_mode.isChecked():
            self.radio_axis_1.setEnabled(True)
            self.radio_axis_2.setEnabled(True)
            self.radio_axis_3.setEnabled(True)

    def layout_based_on_select_bg(self):
        if self.combo_bg_filter.currentText() == "Differential rolling average":
            while self.flag_remove_box:
                self.remove_widgets()
            self.flag_remove_box = True
            self.groupBatchSize = QtWidgets.QGroupBox("Batch size:")

            self.le_batchSize = QtWidgets.QLineEdit()
            self.le_batchSize.setPlaceholderText("radius")
            self.le_batchSize_label = QtWidgets.QLabel("Batch size (frames):")
            self.le_batchSize.setFixedWidth(50)

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

            grid_batch = QtWidgets.QGridLayout()
            grid_batch.addWidget(self.le_batchSize_label, 0, 0)
            grid_batch.addWidget(self.le_batchSize, 0, 1)
            self.groupBatchSize.setLayout(grid_batch)

            self.groupBox_FPNc = QtWidgets.QGroupBox("FPNc:")
            self.groupBox_FPNc.setCheckable(True)
            self.groupBox_FPNc.setChecked(False)

            self.FPN_mode_group = QtWidgets.QButtonGroup()
            self.radio_wFPN_mode = QtWidgets.QRadioButton("Wavelet FPNc")
            self.radio_cpFPN_mode = QtWidgets.QRadioButton("Column_projection FPNc")
            self.radio_median_FPN_mode = QtWidgets.QRadioButton("Median FPNc")
            self.radio_FFT2D_FPN_mode = QtWidgets.QRadioButton("FFT2D FPNc")

            self.FPN_mode_group.addButton(self.radio_wFPN_mode)
            self.FPN_mode_group.addButton(self.radio_cpFPN_mode)
            self.FPN_mode_group.addButton(self.radio_FFT2D_FPN_mode)
            self.FPN_mode_group.addButton(self.radio_median_FPN_mode)

            self.radio_wFPN_mode.toggled.connect(self.axis_selection)
            self.radio_cpFPN_mode.toggled.connect(self.axis_selection)
            self.radio_FFT2D_FPN_mode.toggled.connect(self.axis_selection)
            self.radio_median_FPN_mode.toggled.connect(self.axis_selection)

            self.axis_group = QtWidgets.QButtonGroup()
            self.radio_axis_1 = QtWidgets.QRadioButton("FPNc in axis 0")
            self.radio_axis_2 = QtWidgets.QRadioButton("FPNc in axis 1")
            self.radio_axis_3 = QtWidgets.QRadioButton("FPNc in Both axis")
            self.radio_axis_2.setChecked(True)
            self.radio_cpFPN_mode.setChecked(True)

            self.axis_group.addButton(self.radio_axis_1)
            self.axis_group.addButton(self.radio_axis_2)
            self.axis_group.addButton(self.radio_axis_3)

            grid = QtWidgets.QGridLayout()
            grid.addWidget(self.radio_cpFPN_mode, 0, 0)
            grid.addWidget(self.radio_median_FPN_mode, 1, 0)
            grid.addWidget(self.radio_wFPN_mode, 2, 0)
            grid.addWidget(self.radio_FFT2D_FPN_mode, 3, 0)

            grid.addWidget(self.radio_axis_1, 0, 1)
            grid.addWidget(self.radio_axis_2, 1, 1)
            # grid.addWidget(self.radio_axis_3, 2, 1)

            self.groupBox_FPNc.setLayout(grid)

            self.grid1.addWidget(self.groupBatchSize, 1, 0)
            self.grid1.addWidget(self.groupBox_PN, 2, 0)
            self.grid1.addWidget(self.groupBox_FPNc, 3, 0)

        elif self.combo_bg_filter.currentText() == "Spatial median Filter":
            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.line_edit1 = QtWidgets.QLineEdit(self)
            self.line_edit1_label = QtWidgets.QLabel("Neighborhood size (px):")
            self.line_edit1.setFixedWidth(50)

            self.grid1.addWidget(self.line_edit1_label, 2, 0)
            self.grid1.addWidget(self.line_edit1, 2, 1)

        elif self.combo_bg_filter.currentText() == "Spatial gaussian Filter":
            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.line_edit2 = QtWidgets.QLineEdit(self)
            self.line_edit2_label = QtWidgets.QLabel("Sigma (px):")
            self.line_edit2.setFixedWidth(50)

            self.grid1.addWidget(self.line_edit2_label, 2, 0)
            self.grid1.addWidget(self.line_edit2, 2, 1)

        elif self.combo_bg_filter.currentText() == "Background correction temporal median":
            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

        elif self.combo_bg_filter.currentText() == "Flat field (Gaussian filter)":
            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.line_edit5 = QtWidgets.QLineEdit(self)
            self.line_edit5_label = QtWidgets.QLabel("Sigma (px):")
            self.line_edit5.setFixedWidth(50)

            self.grid1.addWidget(self.line_edit5_label, 2, 0)
            self.grid1.addWidget(self.line_edit5, 2, 1)

        elif self.combo_bg_filter.currentText() == "Flat field (mean background subtraction)":
            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.btn_read_file = QtWidgets.QPushButton("load")
            self.btn_read_file.setAutoDefault(False)
            self.btn_read_file.clicked.connect(self.load_video)
            self.btn_read_file.setFixedHeight(20)
            self.btn_read_file.setFixedWidth(50)

            self.grid1.addWidget(self.btn_read_file, 1, 0)

    def remove_widgets(self):
        for i_ in range(0, 7, 1):
            for j_ in range(0, 7, 1):
                if i_ == 0 and j_ == 0:
                    pass
                else:
                    layout = self.grid1.itemAtPosition(i_, j_)
                    if layout is not None:
                        layout.widget().deleteLater()
                        self.grid1.removeItem(layout)

        self.flag_remove_box = False

    def load_video(self):
        self.reading = Reading()
        self.reading.update_output.connect(self.updata_bg_video)
        self.reading.read_video()

    def updata_bg_video(self, video_in):
        self.original_video_bg = video_in[0]

    def preview_roi_plot(self, selected_frame):
        if self.groupBox_PN.isChecked():
            self.flag_power_normalization = True
            try:
                self.s_x_win_pn = int(self.start_win_size_x.text())
                self.e_x_win_pn = int(self.end_win_size_x.text())
                self.s_y_win_pn = int(self.start_win_size_y.text())
                self.e_y_win_pn = int(self.end_win_size_y.text())
                selected_frame = int(self.selected_frame.text())

                img_ = self.input_video[selected_frame, :, :]
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

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
