from piscat.Localization import particle_localization, localization_filtering
from PySide2 import QtGui, QtCore, QtWidgets

import pandas as pd


class Localization_GUI(QtWidgets.QWidget):
    preview_localization = QtCore.Signal(object)
    update_localization = QtCore.Signal(object)
    output_setting_Tab_Localization = QtCore.Signal(object)
    output_number_PSFs_tracking = QtCore.Signal(object)
    update_tab_index = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(Localization_GUI, self).__init__(parent)

        self.input_video = None
        self.min_radial = None
        self.max_radial = None
        self.rvt_kind = None
        self.highpass_size = None
        self.upsample = None
        self.rweights = None
        self.coarse_factor = None
        self.coarse_mode = None
        self.pad_mode = None
        self.min_sigma = None
        self.max_sigma = None
        self.sigma_ratio = None
        self.threshold = None
        self.overlap = None
        self.frame_num = 0
        self.setting_localization = {}
        self.PSFs_Particels_num = {}

        self.crappy_thr = None
        self.scale = None
        self.thr_sigma = None
        self.win_size_Fitting = None

        self.empty_value_box_flag = False
        self.flag_remove_box = True
        self.mode = None
        self.df_PSFs = None
        self.resize(300, 300)
        self.setWindowTitle('2D Localization')

        self.btn_FineLocalizationUpdate = QtWidgets.QPushButton("Update Fine Localization")
        self.btn_FineLocalizationUpdate.clicked.connect(self.do_fineLocalization)
        self.btn_FineLocalizationUpdate.setFixedWidth(150)

        self.btn_localizationUpdate = QtWidgets.QPushButton("Update Localization")
        self.btn_localizationUpdate.clicked.connect(self.do_localization)
        self.btn_localizationUpdate.setFixedWidth(150)

        self.btn_previewLocalization = QtWidgets.QPushButton('Preview', self)
        self.btn_previewLocalization.clicked.connect(self.previewLocalization)
        self.btn_previewLocalization.setIconSize(QtCore.QSize(24, 24))
        self.btn_previewLocalization.setFixedWidth(100)

        self.btn_filtering = QtWidgets.QPushButton('Filtering', self)
        self.btn_filtering.clicked.connect(self.do_Filtering)
        self.btn_filtering.setIconSize(QtCore.QSize(24, 24))
        self.btn_filtering.setMaximumSize(100, 70)

        self.mode_list = {"Bright PSF": "Bright", "Dark PSF": "Dark", "Bright & Dark PSF": "BOTH"}
        self.method_list = {"Difference of Gaussian": "dog", "Laplacian of Gaussian": "log", "Determinant of Hessian": "doh",
                            "Radial Symmetry": "frst_one_psf", 'RVT': 'RVT'}

        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("-Select the method-")
        self.combo.addItem("Difference of Gaussian")
        self.combo.addItem("Laplacian of Gaussian")
        self.combo.addItem("Determinant of Hessian")
        self.combo.addItem("Radial Symmetry")
        self.combo.addItem("RVT")
        self.combo.currentIndexChanged.connect(self.on_select)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.createThirdExclusiveGroup(), 2, 0)
        self.grid.addWidget(self.createFourthExclusiveGroup(), 3, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print('Destructor called, Employee deleted.')

    @QtCore.Slot(int)
    def get_sliceNumber(self, frame_number):
        self.frame_num = frame_number

    @QtCore.Slot()
    def update_in_data(self, input_video):
        self.input_video = input_video

    def createFirstExclusiveGroup(self):
        groupBox = QtWidgets.QGroupBox("PSF Localization")

        self.grid_Localization = QtWidgets.QGridLayout()
        self.grid_Localization.addWidget(self.combo, 0, 0)
        groupBox.setLayout(self.grid_Localization)
        return groupBox

    def createSecondExclusiveGroup(self):

        self.le_win_size = QtWidgets.QLineEdit()
        self.le_win_size.setPlaceholderText("win size")
        self.le_win_size_label = QtWidgets.QLabel("Neighborhood size(px)")

        self.groupBox_fine_localization = QtWidgets.QGroupBox("Fine localization:")
        self.groupBox_fine_localization.setCheckable(True)
        self.groupBox_fine_localization.setChecked(False)

        self.gridFine_Localization = QtWidgets.QGridLayout()
        self.gridFine_Localization.addWidget(self.le_win_size_label, 0, 0)
        self.gridFine_Localization.addWidget(self.le_win_size, 0, 1)

        self.groupBox_fine_localization.setLayout(self.gridFine_Localization)
        return self.groupBox_fine_localization

    def createThirdExclusiveGroup(self):

        self.groupBox_update = QtWidgets.QGroupBox("Update:")

        self.grid_applyLocalization = QtWidgets.QGridLayout()
        self.grid_applyLocalization.addWidget(self.btn_previewLocalization, 0, 0)
        self.grid_applyLocalization.addWidget(self.btn_localizationUpdate, 0, 1)
        self.grid_applyLocalization.addWidget(self.btn_FineLocalizationUpdate, 0, 2)

        self.groupBox_update.setLayout(self.grid_applyLocalization)
        return self.groupBox_update

    def createFourthExclusiveGroup(self):

        self.checkbox_filter_double_PSF = QtWidgets.QCheckBox("Filter dense PSFs", self)
        self.checkbox_filter_asymmetry_PSF = QtWidgets.QCheckBox("Filter asymmetry PSFs", self)
        self.checkbox_2DFitting = QtWidgets.QCheckBox("2D Gaussian Fitting", self)
        self.checkbox_remove_overlay = QtWidgets.QCheckBox("Remove PSFs with overlay", self)
        self.checkbox_crappy_frames = QtWidgets.QCheckBox("Filter outlier frames", self)

        self.checkbox_filter_asymmetry_PSF.toggled.connect(lambda: self.add_line_asymmetry_PSF())
        self.checkbox_2DFitting.toggled.connect(lambda: self.add_line_2DFitting())
        self.checkbox_crappy_frames.toggled.connect(lambda: self.add_line_crappy_frames())

        self.groupBox_filters = QtWidgets.QGroupBox("Spatial filters:")
        self.groupBox_filters.setCheckable(True)
        self.groupBox_filters.setChecked(False)

        self.grid_filters = QtWidgets.QGridLayout()
        self.grid_filters.addWidget(self.checkbox_crappy_frames, 0, 0)
        self.grid_filters.addWidget(self.checkbox_filter_double_PSF, 1, 0)
        self.grid_filters.addWidget(self.checkbox_remove_overlay, 2, 0)
        self.grid_filters.addWidget(self.checkbox_filter_asymmetry_PSF, 3, 0)
        self.grid_filters.addWidget(self.checkbox_2DFitting, 4, 0)
        self.grid_filters.addWidget(self.btn_filtering, 5, 0)

        self.groupBox_filters.setLayout(self.grid_filters)

        return self.groupBox_filters

    def add_line_crappy_frames(self):
        if self.checkbox_crappy_frames.isChecked():
            self.line_edit_crappy = QtWidgets.QLineEdit(self)
            self.line_edit_crappy.setPlaceholderText('Max. number PSFs')
            self.line_edit_crappy_label = QtWidgets.QLabel("Max #PSFs:")

            self.grid_filters.addWidget(self.line_edit_crappy_label, 0, 1)
            self.grid_filters.addWidget(self.line_edit_crappy, 0, 2)

        if not self.checkbox_crappy_frames.isChecked():

            for i_ in range(1, 5, 1):
                layout = self.grid_filters.itemAtPosition(0, i_)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.grid_filters.removeItem(layout)

    def add_line_asymmetry_PSF(self):
        if self.checkbox_filter_asymmetry_PSF.isChecked():
            self.line_asymmetry_PSF = QtWidgets.QLineEdit(self)
            self.line_asymmetry_PSF.setPlaceholderText('Scale based on sigma size')
            self.line_asymmetry_PSF_label = QtWidgets.QLabel("Scale win_size:")

            self.line_asymmetry_PSF_Thr = QtWidgets.QLineEdit(self)
            tmp_str = "\u03C3"
            self.line_asymmetry_PSF_Thr.setPlaceholderText(tmp_str + "_x" + "/" + tmp_str + "_y")
            self.line_asymmetry_PSF_Thr_label = QtWidgets.QLabel("Symmetry threshold:")

            self.grid_filters.addWidget(self.line_asymmetry_PSF_label, 3, 1)
            self.grid_filters.addWidget(self.line_asymmetry_PSF, 3, 2)

            self.grid_filters.addWidget(self.line_asymmetry_PSF_Thr_label, 3, 3)
            self.grid_filters.addWidget(self.line_asymmetry_PSF_Thr, 3, 4)

        if not self.checkbox_filter_asymmetry_PSF.isChecked():

            for i_ in range(1, 5, 1):
                layout = self.grid_filters.itemAtPosition(3, i_)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.grid_filters.removeItem(layout)

    def add_line_2DFitting(self):
        if self.checkbox_2DFitting.isChecked():
            self.line_2DFitting = QtWidgets.QLineEdit(self)
            self.line_2DFitting.setPlaceholderText('Scale based on sigma size')
            self.line_2DFitting_label = QtWidgets.QLabel("Scale win_size:")

            self.grid_filters.addWidget(self.line_2DFitting_label, 4, 1)
            self.grid_filters.addWidget(self.line_2DFitting, 4, 2)

        if not self.checkbox_2DFitting.isChecked():

            for i_ in range(1, 5, 1):
                layout = self.grid_filters.itemAtPosition(4, i_)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.grid_filters.removeItem(layout)

    def on_select(self):

        if self.combo.currentText() == "Difference of Gaussian":

            while self.flag_remove_box:
                self.remove_extra_box()

            self.flag_remove_box = True
            self.combo_item = self.method_list[self.combo.currentText()]
            self.PSF_mode()
            self.create_como_values()

        elif self.combo.currentText() == "Laplacian of Gaussian":

            while self.flag_remove_box:
                self.remove_extra_box()

            self.flag_remove_box = True
            self.combo_item = self.method_list[self.combo.currentText()]
            self.PSF_mode()
            self.create_como_values()

        elif self.combo.currentText() == "Determinant of Hessian":

            while self.flag_remove_box:
                self.remove_extra_box()

            self.flag_remove_box = True
            self.combo_item = self.method_list[self.combo.currentText()]
            self.create_como_values()

        elif self.combo.currentText() == "Radial Symmetry":

            while self.flag_remove_box:
                self.remove_extra_box()

            self.flag_remove_box = True
            self.combo_item = self.method_list[self.combo.currentText()]
            self.msg_box2 = QtWidgets.QMessageBox()
            self.msg_box2.setWindowTitle("Warning!")
            self.msg_box2.setText("This function only works when you have only one PSF in each frame!")
            self.msg_box2.exec_()

        elif self.combo.currentText() == "RVT":

            while self.flag_remove_box:
                self.remove_extra_box()

            self.flag_remove_box = True
            self.combo_item = self.method_list[self.combo.currentText()]
            self.create_rvt_como_values()

    def PSF_mode(self):
        self.combo_mode = QtWidgets.QComboBox(self)
        self.combo_mode.addItem("-Select the Mode-")
        self.combo_mode.addItem("Bright PSF")
        self.combo_mode.addItem("Dark PSF")
        self.combo_mode.addItem("Bright & Dark PSF")
        self.combo_mode.currentIndexChanged.connect(self.on_select_mode)
        self.grid_Localization.addWidget(self.combo_mode, 0, 1)

    def on_select_mode(self):
        self.mode = self.mode_list[self.combo_mode.currentText()]

    def create_como_values(self):
        self.le_1 = QtWidgets.QLineEdit()
        self.le_1.setPlaceholderText("Min Sigma")
        self.le_1_label = QtWidgets.QLabel("Min Sigma (px): ")

        self.le_2 = QtWidgets.QLineEdit()
        self.le_2.setPlaceholderText("Max Sigma")
        self.le_2_label = QtWidgets.QLabel("Max Sigma (px): ")

        self.le_3 = QtWidgets.QLineEdit()
        self.le_3.setPlaceholderText("Sigma Ration/Num Sigma")
        self.le_3_label = QtWidgets.QLabel("Sigma Ration/Num Sigma: ")

        self.le_4 = QtWidgets.QLineEdit()
        self.le_4.setPlaceholderText("Threshold")
        self.le_4_label = QtWidgets.QLabel("Threshold: ")

        self.le_5 = QtWidgets.QLineEdit()
        self.le_5.setPlaceholderText("Overlap")
        self.le_5_label = QtWidgets.QLabel("Overlap (px): ")

        self.grid_Localization.addWidget(self.le_1_label, 1, 0)
        self.grid_Localization.addWidget(self.le_2_label, 2, 0)
        self.grid_Localization.addWidget(self.le_3_label, 3, 0)
        self.grid_Localization.addWidget(self.le_4_label, 1, 2)
        self.grid_Localization.addWidget(self.le_5_label, 2, 2)

        self.grid_Localization.addWidget(self.le_1, 1, 1)
        self.grid_Localization.addWidget(self.le_2, 2, 1)
        self.grid_Localization.addWidget(self.le_3, 3, 1)
        self.grid_Localization.addWidget(self.le_4, 1, 3)
        self.grid_Localization.addWidget(self.le_5, 2, 3)

    def create_rvt_como_values(self):
        self.le_1 = QtWidgets.QLineEdit()
        self.le_1.setPlaceholderText("Min radius")
        self.le_1_label = QtWidgets.QLabel("Min radius (px): ")

        self.le_2 = QtWidgets.QLineEdit()
        self.le_2.setPlaceholderText("Max radius")
        self.le_2_label = QtWidgets.QLabel("Max radius (px): ")

        self.le_3 = QtWidgets.QLineEdit()
        self.le_3.setPlaceholderText("highpass_size")
        self.le_3_label = QtWidgets.QLabel("Highpass_size: ")

        self.le_4 = QtWidgets.QLineEdit()
        self.le_4.setPlaceholderText("upsample")
        self.le_4_label = QtWidgets.QLabel("Upsample: ")

        self.le_5 = QtWidgets.QLineEdit()
        self.le_5.setPlaceholderText("rweights")
        self.le_5_label = QtWidgets.QLabel("rweights: ")

        self.le_6 = QtWidgets.QLineEdit()
        self.le_6.setPlaceholderText("coarse_factor")
        self.le_6_label = QtWidgets.QLabel("Coarse_factor: ")

        self.le_7 = QtWidgets.QLineEdit()
        self.le_7.setPlaceholderText("Threshold")
        self.le_7_label = QtWidgets.QLabel("Threshold: ")

        self.pad_mode_group = QtWidgets.QButtonGroup()
        self.radio_pad_mode_constant = QtWidgets.QRadioButton("Pad mode (constant)")
        self.radio_pad_mode_reflect = QtWidgets.QRadioButton("Pad mode (reflect)")
        self.radio_pad_mode_edge = QtWidgets.QRadioButton("Pad mode (edge)")
        self.radio_pad_mode_fast = QtWidgets.QRadioButton("Pad mode (fast)")

        self.pad_mode_group.addButton(self.radio_pad_mode_constant)
        self.pad_mode_group.addButton(self.radio_pad_mode_reflect)
        self.pad_mode_group.addButton(self.radio_pad_mode_edge)
        self.pad_mode_group.addButton(self.radio_pad_mode_fast)
        self.radio_pad_mode_constant.setChecked(True)

        self.coarse_mode_group = QtWidgets.QButtonGroup()
        self.radio_coarse_mode_add = QtWidgets.QRadioButton("Coarse mode (add)")
        self.radio_coarse_mode_skip = QtWidgets.QRadioButton("Coarse mode (skip)")

        self.coarse_mode_group.addButton(self.radio_coarse_mode_add)
        self.coarse_mode_group.addButton(self.radio_coarse_mode_skip)
        self.radio_coarse_mode_add.setChecked(True)

        self.kind_group = QtWidgets.QButtonGroup()
        self.radio_kind_basic = QtWidgets.QRadioButton("Kind (basic)")
        self.radio_kind_normalized = QtWidgets.QRadioButton("Kind (normalized)")

        self.kind_group.addButton(self.radio_kind_basic)
        self.kind_group.addButton(self.radio_kind_normalized)
        self.radio_kind_basic.setChecked(True)

        self.grid_Localization.addWidget(self.le_1_label, 1, 0)
        self.grid_Localization.addWidget(self.le_2_label, 2, 0)
        self.grid_Localization.addWidget(self.le_3_label, 3, 0)
        self.grid_Localization.addWidget(self.le_4_label, 1, 2)
        self.grid_Localization.addWidget(self.le_5_label, 2, 2)
        self.grid_Localization.addWidget(self.le_6_label, 3, 2)
        self.grid_Localization.addWidget(self.le_7_label, 5, 2)

        self.grid_Localization.addWidget(self.le_1, 1, 1)
        self.grid_Localization.addWidget(self.le_2, 2, 1)
        self.grid_Localization.addWidget(self.le_3, 3, 1)
        self.grid_Localization.addWidget(self.le_4, 1, 3)
        self.grid_Localization.addWidget(self.le_5, 2, 3)
        self.grid_Localization.addWidget(self.le_6, 3, 3)
        self.grid_Localization.addWidget(self.le_7, 5, 3)

        self.grid_Localization.addWidget(self.radio_pad_mode_constant, 4, 0)
        self.grid_Localization.addWidget(self.radio_pad_mode_reflect, 4, 1)
        self.grid_Localization.addWidget(self.radio_pad_mode_edge, 4, 2)
        self.grid_Localization.addWidget(self.radio_pad_mode_fast, 4, 3)

        self.grid_Localization.addWidget(self.radio_coarse_mode_add, 5, 0)
        self.grid_Localization.addWidget(self.radio_coarse_mode_skip, 5, 1)

        self.grid_Localization.addWidget(self.radio_kind_basic, 6, 0)
        self.grid_Localization.addWidget(self.radio_kind_normalized, 6, 1)

    def remove_extra_box(self):
        layout_combo = self.grid_Localization.itemAtPosition(0, 1)
        if layout_combo is not None:
            layout_combo.widget().deleteLater()
            self.grid_Localization.removeItem(layout_combo)

        layout_pre_label = self.grid_Localization.itemAtPosition(1, 0)
        layout_pre = self.grid_Localization.itemAtPosition(1, 1)

        if layout_pre_label is not None:
            layout_pre_label.widget().deleteLater()
            self.grid_Localization.removeItem(layout_pre_label)

        if layout_pre is not None:
            layout_pre.widget().deleteLater()
            self.grid_Localization.removeItem(layout_pre)

        for i_ in range(1, 7, 1):
            for j_ in range(0, 6, 1):
                layout = self.grid_Localization.itemAtPosition(i_, j_)
                if layout is not None:
                    layout.widget().deleteLater()
                    self.grid_Localization.removeItem(layout)
        self.flag_remove_box = False

    def get_localization_method_parameters(self):
        try:
            if self.combo.currentText() != "Radial Symmetry" and self.combo.currentText() != "RVT":

                self.min_sigma = eval(self.le_1.text())
                self.max_sigma = eval(self.le_2.text())
                self.sigma_ratio = float(self.le_3.text())
                self.threshold = float(self.le_4.text())
                self.overlap = float(self.le_5.text())

            elif self.combo.currentText() == "RVT":
                self.min_radial = eval(self.le_1.text())
                self.max_radial = eval(self.le_2.text())

                if self.radio_kind_basic.isChecked():
                    self.rvt_kind = 'basic'
                elif self.radio_kind_normalized.isChecked():
                    self.rvt_kind = 'normalized'

                self.highpass_size = self.le_3.text()
                if self.highpass_size == '':
                    self.highpass_size = None
                else:
                    self.highpass_size = float(self.highpass_size)

                self.upsample = self.le_4.text()
                if self.upsample == '':
                    self.upsample = 1
                else:
                    self.upsample = int(self.upsample)

                self.rweights = self.le_5.text()
                if self.rweights == '':
                    self.rweights = None
                else:
                    self.rweights = eval(self.rweights)

                self.coarse_factor = self.le_6.text()
                if self.coarse_factor == '':
                    self.coarse_factor = 1
                else:
                    self.coarse_factor = float(self.coarse_factor)

                if self.radio_coarse_mode_add.isChecked():
                    self.coarse_mode = 'add'
                elif self.radio_coarse_mode_skip.isChecked():
                    self.coarse_mode = 'skip'

                if self.radio_pad_mode_constant.isChecked():
                    self.pad_mode = "constant"
                elif self.radio_pad_mode_reflect.isChecked():
                    self.pad_mode = "reflect"
                elif self.radio_pad_mode_edge.isChecked():
                    self.pad_mode = "edge"
                elif self.radio_pad_mode_fast.isChecked():
                    self.pad_mode = "fast"

                if self.le_7.text() == '':
                    self.threshold = 0
                else:
                    self.threshold = float(self.le_7.text())

            if self.groupBox_fine_localization.isChecked() and self.combo.currentText() != "Radial Symmetry":

                self.win_size = int(self.le_win_size.text())

            elif self.combo.currentText() == "Radial Symmetry" and self.groupBox_fine_localization.isChecked():

                self.win_size = int(self.le_win_size.text())

            elif self.combo.currentText() == "Radial Symmetry" and not self.groupBox_fine_localization.isChecked():

                pass

            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def get_values_crappy_frames(self):
        try:
            self.crappy_thr = int(self.line_edit_crappy.text())
            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def get_values_asymmetry_filtering(self):
        try:
            self.scale = int(self.line_asymmetry_PSF.text())
            self.thr_sigma = float(self.line_asymmetry_PSF_Thr.text())
            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def get_values_2DFitting(self):
        try:
            self.win_size_Fitting = int(self.line_2DFitting.text())
            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def removekey(self, d, key):
        r = dict(d)
        try:
            del r[key]
        except:
            print('there is no key with the name ' + key)
        return r

    @QtCore.Slot()
    def previewLocalization(self):

        layout_combo = self.grid_Localization.itemAtPosition(0, 1)
        if layout_combo is not None:
            if self.combo.currentText() == "-Select the method-" or self.combo_mode.currentText() == "-Select the Mode-":
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("Please select one of the methods and modes.")
                self.msg_box1.exec_()
                flag_run = False
            else:
                flag_run = True

        elif layout_combo is None:
            if self.combo.currentText() == "-Select the method-":
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("Please select one of the methods.")
                self.msg_box1.exec_()
                flag_run = False
            else:
                flag_run = True

        if flag_run:

            if self.input_video is None:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("Please update the previous tab!")
                self.msg_box3.exec_()
            else:
                self.get_localization_method_parameters()
                if self.empty_value_box_flag:
                    psf = particle_localization.PSFsExtraction(self.input_video)
                    self.df_PSFs_preview = psf.psf_detection_preview(function=self.combo_item, min_sigma=self.min_sigma,
                                                                     max_sigma=self.max_sigma, sigma_ratio=self.sigma_ratio,
                                                                     threshold=self.threshold, overlap=self.overlap,
                                                                     frame_number=self.frame_num, mode=self.mode,
                                                                     min_radial=self.min_radial, max_radial=self.max_radial,
                                                                     highpass_size=self.highpass_size, upsample=self.upsample,
                                                                     rweights=self.rweights, coarse_factor=self.coarse_factor,
                                                                     coarse_mode=self.coarse_mode, pad_mode=self.pad_mode,
                                                                     rvt_kind=self.rvt_kind)
                    if self.groupBox_fine_localization.isChecked():
                        self.df_PSFs_preview = psf.improve_localization_with_frst(df_PSFs=self.df_PSFs_preview,
                                                                                  scale=self.win_size, flag_preview=True)

                    self.setting_localization['function'] = self.combo_item
                    self.setting_localization['min_sigma'] = self.min_sigma
                    self.setting_localization['max_sigma'] = self.max_sigma
                    self.setting_localization['sigma_ratio'] = self.sigma_ratio
                    self.setting_localization['threshold'] = self.threshold
                    self.setting_localization['overlap'] = self.overlap
                    self.setting_localization['mode'] = self.mode
                    self.setting_localization['min_radial'] = self.min_radial
                    self.setting_localization['max_radial'] = self.max_radial
                    self.setting_localization['highpass_size'] = self.highpass_size
                    self.setting_localization['upsample'] = self.upsample
                    self.setting_localization['rweights'] = self.rweights
                    self.setting_localization['coarse_factor'] = self.coarse_factor
                    self.setting_localization['coarse_mode'] = self.coarse_mode
                    self.setting_localization['pad_mode'] = self.pad_mode
                    self.setting_localization['rvt_kind'] = self.rvt_kind

                    self.preview_localization.emit(self.df_PSFs_preview)
                    self.output_setting_Tab_Localization.emit(self.setting_localization)
                    self.empty_value_box_flag = False

    @QtCore.Slot()
    def do_localization(self):
        layout_combo = self.grid_Localization.itemAtPosition(0, 1)
        if layout_combo is not None:
            if self.combo.currentText() == "-Select the method-" or self.combo_mode.currentText() == "-Select the Mode-":
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("Please select one of the methods and modes.")
                self.msg_box1.exec_()
                flag_run = False
            else:
                flag_run = True

        elif layout_combo is None:
            if self.combo.currentText() == "-Select the method-":
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("Please select one of the methods.")
                self.msg_box1.exec_()
                flag_run = False
            else:
                flag_run = True

        if flag_run:
                self.get_localization_method_parameters()
                if self.empty_value_box_flag:
                    psf = particle_localization.PSFsExtraction(video=self.input_video, flag_GUI=True)
                    self.df_PSFs = psf.psf_detection(function=self.combo_item, min_sigma=self.min_sigma,
                                                                     max_sigma=self.max_sigma,
                                                                     sigma_ratio=self.sigma_ratio,
                                                                     threshold=self.threshold, overlap=self.overlap,
                                                                     mode=self.mode,
                                                                     min_radial=self.min_radial,
                                                                     max_radial=self.max_radial,
                                                                     highpass_size=self.highpass_size,
                                                                     upsample=self.upsample,
                                                                     rweights=self.rweights,
                                                                     coarse_factor=self.coarse_factor,
                                                                     coarse_mode=self.coarse_mode,
                                                                     pad_mode=self.pad_mode,
                                                                     rvt_kind=self.rvt_kind)

                    self.PSFs_Particels_num['Total_number_PSFs'] = self.df_PSFs.shape[0]

                    self.setting_localization['function'] = self.combo_item
                    self.setting_localization['min_sigma'] = self.min_sigma
                    self.setting_localization['max_sigma'] = self.max_sigma
                    self.setting_localization['sigma_ratio'] = self.sigma_ratio
                    self.setting_localization['threshold'] = self.threshold
                    self.setting_localization['overlap'] = self.overlap
                    self.setting_localization['mode'] = self.mode
                    self.setting_localization['min_radial'] = self.min_radial
                    self.setting_localization['max_radial'] = self.max_radial
                    self.setting_localization['highpass_size'] = self.highpass_size
                    self.setting_localization['upsample'] = self.upsample
                    self.setting_localization['rweights'] = self.rweights
                    self.setting_localization['coarse_factor'] = self.coarse_factor
                    self.setting_localization['coarse_mode'] = self.coarse_mode
                    self.setting_localization['pad_mode'] = self.pad_mode
                    self.setting_localization['rvt_kind'] = self.rvt_kind

                    self.update_localization.emit(self.df_PSFs)
                    self.output_setting_Tab_Localization.emit(self.setting_localization)
                    self.output_number_PSFs_tracking.emit(self.PSFs_Particels_num)
                    self.empty_value_box_flag = False

    @QtCore.Slot()
    def do_fineLocalization(self):
        if self.groupBox_fine_localization.isChecked():
            if self.df_PSFs is not None:
                self.get_localization_method_parameters()
                if self.empty_value_box_flag:
                    psf = particle_localization.PSFsExtraction(video=self.input_video, flag_GUI=True)
                    self.df_PSFs = psf.improve_localization_with_frst(df_PSFs=self.df_PSFs, scale=self.win_size,
                                                                      flag_preview=False)
                    self.setting_localization['fine_localization'] = True
                    self.setting_localization['fine_localization_winSize'] = self.win_size

                    self.update_localization.emit(self.df_PSFs)
                    self.output_setting_Tab_Localization.emit(self.setting_localization)
                    self.output_number_PSFs_tracking.emit(self.PSFs_Particels_num)
                    self.empty_value_box_flag = False
            else:
                self.msg_box1 = QtWidgets.QMessageBox()
                self.msg_box1.setWindowTitle("Warning!")
                self.msg_box1.setText("Please update localization!")
                self.msg_box1.exec_()
        else:
            self.msg_box1 = QtWidgets.QMessageBox()
            self.msg_box1.setWindowTitle("Warning!")
            self.msg_box1.setText("Please active fine localization box!")
            self.msg_box1.exec_()

    @QtCore.Slot()
    def do_Filtering(self):

        if self.input_video is None:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please update the previous tab!")
            self.msg_box3.exec_()

        elif self.df_PSFs is None or self.df_PSFs.shape[0] <= 0:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Press the 'update' or 'loading' button to localize PSFs!")
            self.msg_box3.exec_()

        else:
            s_filters = localization_filtering.SpatialFilter()

            if self.checkbox_crappy_frames.isChecked():
                self.get_values_crappy_frames()
                if self.empty_value_box_flag:
                    self.df_PSFs_s_filter = s_filters.outlier_frames(self.df_PSFs, threshold=self.crappy_thr)
                    self.setting_localization['outlier_frames_filtering'] = True
                    self.setting_localization['outlier_frames_filtering_threshold'] = self.crappy_thr
                    self.PSFs_Particels_num['#PSFs_after_outlier_frames_filtering'] = self.df_PSFs_s_filter.shape[0]
                    self.empty_value_box_flag = False

            else:
                self.df_PSFs_s_filter = self.df_PSFs

            if self.checkbox_filter_double_PSF.isChecked():
                self.df_PSFs_s_filter = s_filters.dense_PSFs(self.df_PSFs_s_filter, threshold=0)
                self.setting_localization['dense_PSFs_filtering'] = True
                self.PSFs_Particels_num['#PSFs_after_dense_PSFs_filtering'] = self.df_PSFs_s_filter.shape[0]

            if self.checkbox_2DFitting.isChecked() and self.checkbox_filter_asymmetry_PSF.isChecked():
                 pass

            if self.checkbox_2DFitting.isChecked():
                self.get_values_2DFitting()
                if self.empty_value_box_flag:
                    psf_localization = particle_localization.PSFsExtraction(video=self.input_video)
                    self.df_PSFs_s_filter = psf_localization.fit_Gaussian2D_wrapper(PSF_List=self.df_PSFs_s_filter,
                                                                                    scale=self.win_size_Fitting,
                                                                                    internal_parallel_flag=True)

                    self.setting_localization['fit_2DGaussian'] = True
                    self.setting_localization['fit_2DGaussian_winSize_scale'] = self.scale
                    self.empty_value_box_flag = False

            if self.checkbox_filter_asymmetry_PSF.isChecked() and self.checkbox_2DFitting.isChecked():
                self.get_values_asymmetry_filtering()
                if self.empty_value_box_flag:
                    self.df_PSFs_s_filter = s_filters.symmetric_PSFs(self.df_PSFs_s_filter, threshold=self.thr_sigma)

                    self.setting_localization['asymmetry_PSFs_filtering'] = True
                    self.setting_localization['asymmetry_PSFs_filtering_threshold'] = self.thr_sigma
                    self.setting_localization['asymmetry_PSFs_filtering_scale'] = self.scale
                    self.PSFs_Particels_num['#PSFs_after_asymmetry_PSFs_filtering'] = self.df_PSFs_s_filter.shape[0]
                    self.empty_value_box_flag = False

            if self.checkbox_remove_overlay.isChecked():
                self.df_PSFs_s_filter = s_filters.remove_overlay_particles(df_PSFs=self.df_PSFs_s_filter, filter_thr=0)
                self.setting_localization['remove_overlay_PSFs'] = True

            self.update_localization.emit(self.df_PSFs_s_filter)
            self.output_setting_Tab_Localization.emit(self.setting_localization)
            self.output_number_PSFs_tracking.emit(self.PSFs_Particels_num)

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
