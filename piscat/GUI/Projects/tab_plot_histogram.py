from PySide6 import QtCore, QtWidgets

from piscat.Analysis.plot_protein_histogram import PlotProteinHistogram
import os
import time


class Histogram_GUI(QtWidgets.QWidget):
    update_tracking = QtCore.Signal(object)
    output_setting_Tab_tracking = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(Histogram_GUI, self).__init__(parent)

        self.input_data = None
        self.lower_limit = None
        self.upper_limit = None
        self.max_n_components = None
        self.minPeak_width = None
        self.n_bins = None

        self.empty_value_box_flag = False

        self.step_limit = 1e-6
        self.setting_tracking = {}

        self.resize(300, 300)
        self.setWindowTitle("Histogram")

        self.plot_hist = QtWidgets.QPushButton("plot_histogram")
        self.plot_hist.clicked.connect(self.do_hist)
        self.plot_hist.setFixedWidth(150)

        self.save_hist = QtWidgets.QPushButton("save_histogram")
        self.save_hist.clicked.connect(self.save_histogram)
        self.save_hist.setFixedWidth(150)

        self.le_lower_contrast_trim = QtWidgets.QLineEdit()
        self.le_lower_contrast_trim.setPlaceholderText("lower_limitation")
        self.le_lower_contrast_trim_label = QtWidgets.QLabel("lower limitation (Contrast):")
        self.le_lower_contrast_trim.setFixedWidth(150)

        self.le_upper_contrast_trim = QtWidgets.QLineEdit()
        self.le_upper_contrast_trim.setPlaceholderText("upper_limitation")
        self.le_upper_contrast_trim_label = QtWidgets.QLabel("upper limitation (Contrast):")
        self.le_upper_contrast_trim.setFixedWidth(150)

        self.le_MaxGMM_mode = QtWidgets.QLineEdit()
        self.le_MaxGMM_mode.setPlaceholderText("max_n_components for GMM")
        self.le_MaxGMM_mode_label = QtWidgets.QLabel("max n_components:")
        self.le_MaxGMM_mode.setFixedWidth(150)

        self.le_hist_bin = QtWidgets.QLineEdit()
        self.le_hist_bin.setPlaceholderText("#bins")
        self.le_hist_bin_label = QtWidgets.QLabel("Histogram bins:")
        self.le_hist_bin.setFixedWidth(150)

        self.min_peak_width = QtWidgets.QLineEdit()
        self.min_peak_width.setPlaceholderText("Min Peak Width")
        self.min_peak_width_label = QtWidgets.QLabel("Min Peak Width:")
        self.min_peak_width.setFixedWidth(150)

        self.axis_limit = QtWidgets.QLineEdit()
        self.axis_limit.setPlaceholderText("scale * upper_limitation")
        self.le_axis_limit = QtWidgets.QLabel("Scale x-axis:")
        self.axis_limit.setFixedWidth(150)

        self.GMM_visualization_mode_group = QtWidgets.QButtonGroup()
        self.radio_mode_1 = QtWidgets.QRadioButton("Separate mixture")
        self.radio_mode_2 = QtWidgets.QRadioButton("Superposition mixture")
        self.GMM_visualization_mode_group.addButton(self.radio_mode_1)
        self.GMM_visualization_mode_group.addButton(self.radio_mode_2)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.plot_hist, 2, 0)
        self.grid.addWidget(self.save_hist, 3, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print("Destructor called, Employee deleted.")

    def createFirstExclusiveGroup(self):
        self.groupBox_hist = QtWidgets.QGroupBox("Histogram setting")

        self.grid3 = QtWidgets.QGridLayout()

        self.grid3.addWidget(self.le_lower_contrast_trim_label, 0, 0)
        self.grid3.addWidget(self.le_lower_contrast_trim, 1, 0)
        self.grid3.addWidget(self.le_upper_contrast_trim_label, 0, 1)
        self.grid3.addWidget(self.le_upper_contrast_trim, 1, 1)
        self.grid3.addWidget(self.min_peak_width_label, 2, 0)
        self.grid3.addWidget(self.min_peak_width, 3, 0)
        self.grid3.addWidget(self.le_hist_bin_label, 2, 1)
        self.grid3.addWidget(self.le_hist_bin, 3, 1)
        self.grid3.addWidget(self.le_axis_limit, 4, 0)
        self.grid3.addWidget(self.axis_limit, 5, 0)

        self.groupBox_hist.setLayout(self.grid3)
        return self.groupBox_hist

    def createSecondExclusiveGroup(self):
        self.groupBox_gmm = QtWidgets.QGroupBox("Gaussian mixture models (GMM):")
        self.groupBox_gmm.setCheckable(True)
        self.groupBox_gmm.setChecked(False)

        self.grid4 = QtWidgets.QGridLayout()
        self.grid4.addWidget(self.le_MaxGMM_mode_label, 0, 0)
        self.grid4.addWidget(self.le_MaxGMM_mode, 0, 1)
        self.grid4.addWidget(self.radio_mode_1, 0, 2)
        self.grid4.addWidget(self.radio_mode_2, 0, 3)

        self.groupBox_gmm.setLayout(self.grid4)
        return self.groupBox_gmm

    def do_hist(self):
        self.step_limit = 1e-6

        self.get_values_histogram_plot()
        if self.empty_value_box_flag:
            if self.groupBox_gmm.isChecked():
                self.his_ = PlotProteinHistogram(intersection_display_flag=False)
                self.his_(
                    folder_name="",
                    particles=self.input_data,
                    batch_size=self.batch_size,
                    video_frame_num=self.number_frames,
                    MinPeakWidth=self.minPeak_width,
                    MinPeakProminence=0,
                )

                self.his_.plot_histogram(
                    bins=self.n_bins,
                    upper_limitation=self.upper_limit,
                    lower_limitation=self.lower_limit,
                    step_range=self.step_limit,
                    face="g",
                    edge="k",
                    Flag_GMM_fit=True,
                    max_n_components=self.max_n_components,
                )

                #
                # self.his_. plot_fit_histogram(bins=self.n_bins,
                #                     upper_limitation=self.upper_limit,
                #                     lower_limitation=self.lower_limit,
                #                     step_range=self.step_limit,
                #                     face="g",
                #                     edge="y",
                #                     Flag_GMM_fit=True,
                #                     max_n_components=self.max_n_components,
                #                 )
            else:
                self.his_ = PlotProteinHistogram(intersection_display_flag=False)
                self.his_(
                    folder_name="",
                    particles=self.input_data,
                    batch_size=self.batch_size,
                    video_frame_num=self.number_frames,
                    MinPeakWidth=self.minPeak_width,
                    MinPeakProminence=0,
                )
                self.his_.plot_histogram(
                    bins=self.n_bins,
                    upper_limitation=self.upper_limit,
                    lower_limitation=self.lower_limit,
                    step_range=self.step_limit,
                    face="g",
                    edge="k",
                    Flag_GMM_fit=False,
                    max_n_components=self.max_n_components,
                )

                # self.his_.plot_fit_histogram(bins=self.n_bins,
                #                         upper_limitation=self.upper_limit,
                #                         lower_limitation=self.lower_limit,
                #                         step_range=self.step_limit,
                #                         face="g",
                #                         edge="y",
                #                         Flag_GMM_fit=False,
                #                         max_n_components=self.max_n_components,
                #                         )

            self.empty_value_box_flag = False

    def save_histogram(self):

        self.file_path = False
        self.file_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder", os.path.expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly
        )
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
            if self.groupBox_gmm.isChecked():
                self.his_.save_hist_data(self.file_path,
                name='plot_hist_',
                upper_limitation=self.upper_limit,
                lower_limitation=self.lower_limit,
                Flag_GMM_fit=True,
                max_n_components=self.max_n_components)
            else:
                self.his_.save_hist_data(self.file_path,
                                         name='plot_hist.h5',
                                         upper_limitation=self.upper_limit,
                                         lower_limitation=self.lower_limit,
                                         Flag_GMM_fit=False,
                                         max_n_components=self.max_n_components)

    def get_values_histogram_plot(self):
        try:
            self.minPeak_width = float(self.min_peak_width.text())
            self.lower_limit = float(self.le_lower_contrast_trim.text())
            self.upper_limit = float(self.le_upper_contrast_trim.text())
            self.n_bins = int(self.le_hist_bin.text())

            try:
                self.max_n_components = int(self.le_MaxGMM_mode.text())
            except:
                pass

            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    @QtCore.Slot()
    def update_in_data(self, data_in):
        self.input_data = data_in[0]
        self.number_frames = data_in[1]
        self.batch_size = data_in[2]

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
