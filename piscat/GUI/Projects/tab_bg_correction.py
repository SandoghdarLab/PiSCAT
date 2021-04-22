from piscat.Preproccessing.filtering import Filters
from piscat.GUI.InputOutput import Reading
from piscat.GUI.BackgroundCorrection import FUN_DRA

from PySide2 import QtCore, QtWidgets

import numpy as np


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
        self.setWindowTitle('Background Correction')

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
        print('Destructor called, Employee deleted.')

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

                if self.checkbox_power_normalization.isChecked():
                    self.flag_power_normalization = True
                else:
                    self.flag_power_normalization = False

                if self.batch_size != '':
                    self.batch_size = int(self.batch_size)
                    self.output_batchSize_Tab_bgCorrection.emit(self.batch_size)

                    dra_ = FUN_DRA(video=self.input_video, object_update_progressBar=self.object_update_progressBar)
                    dra_video = dra_.run_DRA_from_bgtabs(batch_size=self.batch_size,
                                                         flag_power_normalization=self.flag_power_normalization)

                    self.output = [dra_video, "DRA"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction['batch_size'] = self.batch_size
                    self.setting_bg_correction['Power_Normalization'] = self.flag_power_normalization

                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Please insert batch size!")
                    self.msg_box.exec_()

            elif self.combo_bg_filter.currentText() == "Spatial median Filter":
                median_size = self.line_edit1.text()
                if median_size != '':
                    print("\nMedian is Applying--->", end='')
                    median_size = int(self.line_edit1.text())
                    blur = Filters(self.input_video)
                    blur_video = blur.median(median_size)
                    self.output = [blur_video, 'Spatial median Filter']
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction['median_kernel_size'] = median_size
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Spatial gaussian Filter":
                gaussian_sigma = self.line_edit2.text()
                if gaussian_sigma != '':
                    print("\nGaussian is applying--->", end='')
                    gaussian_sigma = float(self.line_edit2.text())
                    blur = Filters(self.input_video)
                    blur_video = blur.gaussian(gaussian_sigma)
                    self.output = [blur_video, "Spatial gaussian Filter"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction['gaussian_sigma'] = gaussian_sigma
                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Background correction temporal median":
                print("\nBackground correction temporal median--->", end='')
                blur = Filters(self.input_video)
                self.output = [blur.temporal_median(), "Background correction temporal median"]
                self.output_Tab_bgCorrection.emit(self.output)
                print("Done!")

            elif self.combo_bg_filter.currentText() == "Flat field (Gaussian filter)":
                sigma = self.line_edit5.text()
                if sigma != '':
                    print("\nBackground correction Flat Field--->", end='')
                    sigma = int(self.line_edit5.text())
                    blur = Filters(self.input_video)
                    self.output = [blur.flat_field(sigma=sigma), "Flat field (Gaussian filter)"]
                    self.output_Tab_bgCorrection.emit(self.output)

                    self.setting_bg_correction['Flat_field_sigma'] = sigma

                    self.output_setting_Tab_bgCorrection.emit(self.setting_bg_correction)
                    print("Done!")
                else:
                    self.massage_filled()

            elif self.combo_bg_filter.currentText() == "Flat field (mean background subtraction)":
                if self.original_video_bg is not None:
                    print("\nBackground correction Flat Field back ground--->", end='')

                    if self.original_video_bg.shape[1] == self.input_video.shape[1] and self.original_video_bg.shape[2] == \
                            self.input_video.shape[2]:

                        self.output = np.divide(self.input_video, np.mean(self.original_video_bg, axis=0))
                        self.output_Tab_bgCorrection.emit([self.output, "Flat field (mean background subtraction)"])
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
                self.output_Tab_bgCorrection.emit([self.output, 'RAW'])

            self.update_tab_index.emit(1)

    def on_select(self):
        self.transform_algorithm = self.combo.currentText()
        self.create_como_values()

    def layout_based_on_select_bg(self):
        if self.combo_bg_filter.currentText() == "Differential rolling average":
            while self.flag_remove_box:
                self.remove_widgets()
            self.flag_remove_box = True
            self.groupBatchSize = QtWidgets.QGroupBox("Batch size:")

            self.le_batchSize = QtWidgets.QLineEdit()
            self.le_batchSize.setPlaceholderText('radius')
            self.le_batchSize_label = QtWidgets.QLabel("Batch size (#frames):")
            self.le_batchSize.setFixedWidth(50)

            self.checkbox_power_normalization = QtWidgets.QCheckBox("Laser power normalization", self)

            grid_batch = QtWidgets.QGridLayout()
            grid_batch.addWidget(self.le_batchSize_label, 0, 0)
            grid_batch.addWidget(self.le_batchSize, 0, 1)
            grid_batch.addWidget(self.checkbox_power_normalization, 0, 2)
            self.groupBatchSize.setLayout(grid_batch)

            self.grid1.addWidget(self.groupBatchSize, 1, 0)

        elif self.combo_bg_filter.currentText() == "Spatial median Filter":

            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.line_edit1 = QtWidgets.QLineEdit(self)
            self.line_edit1_label = QtWidgets.QLabel("Neighborhood size(px):")
            self.line_edit1.setFixedWidth(50)

            self.grid1.addWidget(self.line_edit1_label, 2, 0)
            self.grid1.addWidget(self.line_edit1, 2, 1)

        elif self.combo_bg_filter.currentText() == "Spatial gaussian Filter":

            while self.flag_remove_box:
                self.remove_widgets()

            self.flag_remove_box = True

            self.line_edit2 = QtWidgets.QLineEdit(self)
            self.line_edit2_label = QtWidgets.QLabel("Sigma(px):")
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
            self.line_edit5_label = QtWidgets.QLabel("Sigma(px):")
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

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
