from piscat.Analysis.plot_protein_histogram import PlotProteinHistogram

from PySide2 import QtGui, QtCore, QtWidgets


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
        self.setWindowTitle('Histogram')

        self.plot_hist = QtWidgets.QPushButton("plot_histogram")
        self.plot_hist.clicked.connect(self.do_hist)

        self.le_lower_contrast_trim = QtWidgets.QLineEdit()
        self.le_lower_contrast_trim.setPlaceholderText("lower_limitation")
        self.le_lower_contrast_trim_label = QtWidgets.QLabel("lower limitation (Contrast):")

        self.le_upper_contrast_trim = QtWidgets.QLineEdit()
        self.le_upper_contrast_trim.setPlaceholderText("upper_limitation")
        self.le_upper_contrast_trim_label = QtWidgets.QLabel("upper limitation (Contrast):")

        self.le_MaxGMM_mode = QtWidgets.QLineEdit()
        self.le_MaxGMM_mode.setPlaceholderText("max_n_components for GMM")
        self.le_MaxGMM_mode_label = QtWidgets.QLabel("max n_components:")

        self.le_hist_bin = QtWidgets.QLineEdit()
        self.le_hist_bin.setPlaceholderText("#bins")
        self.le_hist_bin_label = QtWidgets.QLabel("bins:")

        self.min_peak_width = QtWidgets.QLineEdit()
        self.min_peak_width.setPlaceholderText("Min Peak Width")
        self.min_peak_width_label = QtWidgets.QLabel("Min Peak Width:")

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def createFirstExclusiveGroup(self):
        self.checkbox_GMM_FIT = QtWidgets.QCheckBox("Histogram and GMM", self)

        self.groupBox_hist = QtWidgets.QGroupBox("Histogram and GMM")

        self.grid3 = QtWidgets.QGridLayout()
        self.grid3.addWidget(self.min_peak_width_label, 0, 0)
        self.grid3.addWidget(self.min_peak_width, 0, 1)
        self.grid3.addWidget(self.le_lower_contrast_trim_label, 1, 0)
        self.grid3.addWidget(self.le_lower_contrast_trim, 1, 1)
        self.grid3.addWidget(self.le_upper_contrast_trim_label, 2, 0)
        self.grid3.addWidget(self.le_upper_contrast_trim, 2, 1)
        self.grid3.addWidget(self.le_MaxGMM_mode_label, 3, 0)
        self.grid3.addWidget(self.le_MaxGMM_mode, 3, 1)
        self.grid3.addWidget(self.checkbox_GMM_FIT, 3, 2)
        self.grid3.addWidget(self.le_hist_bin_label, 4, 0)
        self.grid3.addWidget(self.le_hist_bin, 4, 1)
        self.grid3.addWidget(self.plot_hist, 5, 0)

        self.groupBox_hist.setLayout(self.grid3)
        return self.groupBox_hist

    def do_hist(self):
        self.step_limit = 1e-6

        self.get_values_histogram_plot()
        if self.empty_value_box_flag:
            if self.checkbox_GMM_FIT.isChecked():
                his_ = PlotProteinHistogram(intersection_display_flag=False)
                his_(folder_name='', particles=self.input_data, batch_size=self.batch_size,
                          video_frame_num=self.number_frames, MinPeakWidth=self.minPeak_width,
                          MinPeakProminence=0)

                his_.plot_histogram(bins=self.n_bins, upper_limitation=self.upper_limit, lower_limitation=self.lower_limit, step_range=self.step_limit,
                                    face='g',
                                    edge='k', Flag_GMM_fit=True, max_n_components=self.max_n_components)
            else:
                his_ = PlotProteinHistogram(intersection_display_flag=False)
                his_(folder_name='', particles=self.input_data, batch_size=self.batch_size,
                     video_frame_num=self.number_frames, MinPeakWidth=self.minPeak_width,
                     MinPeakProminence=0)
                his_.plot_histogram(bins=self.n_bins, upper_limitation=self.upper_limit,
                                    lower_limitation=self.lower_limit, step_range=self.step_limit,
                                    face='g',
                                    edge='k', Flag_GMM_fit=False, max_n_components=self.max_n_components)
            self.empty_value_box_flag = False

    def get_values_histogram_plot(self):

        try:
            self.minPeak_width = float(self.min_peak_width.text())
            self.lower_limit = float(self.le_lower_contrast_trim.text())
            self.upper_limit = float(self.le_upper_contrast_trim.text())
            self.max_n_components = int(self.le_MaxGMM_mode.text())
            self.n_bins = int(self.le_hist_bin.text())

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
