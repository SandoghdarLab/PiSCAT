from PySide6 import QtCore, QtWidgets

from piscat.GUI.InputOutput import Reading


class BgCorrection_GUI(QtWidgets.QWidget):
    output_setting_Tab_bgCorrection = QtCore.Signal(object)
    update_tab_index = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(BgCorrection_GUI, self).__init__(parent)
        self.different_views = {}

        self.setting_bg_correction = {}

        self.flag_remove_box = True
        self.flag_update_class = True
        self.flag_hotPixels = False

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

        self.btn = QtWidgets.QPushButton("Next")
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

                if self.checkbox_power_normalization.isChecked():
                    self.flag_power_normalization = True
                else:
                    self.flag_power_normalization = False

                if self.checkbox_hotPixels.isChecked():
                    self.flag_hotPixels = True
                else:
                    self.flag_hotPixels = False

                if self.batch_size != "":
                    self.batch_size = int(self.batch_size)

                    self.setting_bg_correction["type_BGc"] = "DRA"
                    self.setting_bg_correction["mode_FPN"] = self.mode_FPN
                    self.setting_bg_correction["Batch_size (frames)"] = self.batch_size
                    self.setting_bg_correction[
                        "Power_Normalization"
                    ] = self.flag_power_normalization
                    self.setting_bg_correction["FPNc"] = self.flag_FPN
                    self.setting_bg_correction["FPNc_axis"] = self.axis
                    self.setting_bg_correction["filter_hotPixels"] = self.flag_hotPixels

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

                    self.setting_bg_correction["type_BGc"] = "Spatial median Filter"
                    self.setting_bg_correction["Spatial_median_kernel_size"] = median_size
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Spatial gaussian Filter":
                gaussian_sigma = self.line_edit2.text()
                if gaussian_sigma != "":
                    print("\nGaussian is applying--->", end="")
                    gaussian_sigma = float(self.line_edit2.text())
                    self.setting_bg_correction["type_BGc"] = "Spatial gaussian Filter"
                    self.setting_bg_correction["Spatial_gaussian_sigma"] = gaussian_sigma
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Background correction temporal median":
                print("\nBackground correction temporal median--->", end="")
                self.setting_bg_correction["type_BGc"] = "Background correction temporal median"
                self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                print("Done!")

            elif self.combo_bg_filter.currentText() == "Flat field (Gaussian filter)":
                sigma = self.line_edit5.text()
                if sigma != "":
                    print("\nBackground correction Flat Field--->", end="")
                    sigma = int(self.line_edit5.text())
                    self.setting_bg_correction["type_BGc"] = "Flat field (Gaussian filter)"
                    self.setting_bg_correction["Flat_field_sigma"] = sigma
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Flat field (mean background subtraction)":
                if self.original_video_bg is not None:
                    print("\nBackground correction Flat Field back ground--->", end="")
                    self.setting_bg_correction[
                        "type_BGc"
                    ] = 'Flat field (mean background subtraction)"'
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")

                else:
                    self.massage_filled()
            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Continuing without using any filter!")
                self.msg_box.exec_()
                self.setting_bg_correction["type_BGc"] = "RAW"
                self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)

            self.update_tab_index.emit(2)

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

            self.checkbox_power_normalization = QtWidgets.QCheckBox(
                "Laser power normalization", self
            )
            self.checkbox_hotPixels = QtWidgets.QCheckBox("Hot pixels correction", self)

            grid_batch = QtWidgets.QGridLayout()
            grid_batch.addWidget(self.le_batchSize_label, 0, 0)
            grid_batch.addWidget(self.le_batchSize, 0, 1)
            grid_batch.addWidget(self.checkbox_power_normalization, 1, 0)
            grid_batch.addWidget(self.checkbox_hotPixels, 2, 0)

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
            grid.addWidget(self.radio_axis_3, 2, 1)

            self.groupBox_FPNc.setLayout(grid)

            self.grid1.addWidget(self.groupBatchSize, 1, 0)
            self.grid1.addWidget(self.groupBox_FPNc, 2, 0)

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

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
