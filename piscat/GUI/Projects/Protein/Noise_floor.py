from piscat.BackgroundCorrection.noise_floor import NoiseFloor
from piscat.Preproccessing import Normalization

from PySide2 import QtGui, QtCore, QtWidgets
import pandas as pd
import os


class Noise_Floor(QtWidgets.QMainWindow):

    def __init__(self, video, parent=None):
        super(Noise_Floor, self).__init__(parent)
        self.window = QtWidgets.QWidget()

        self.original_video = video

        self.min_radius = None
        self.max_radius = None
        self.step_radius = None
        self.radius_list = None
        self.mode = None
        self.file_path = None
        self.find_radius_update_tab_flag = True

        self.checkbox_power_normalization = QtWidgets.QCheckBox("Laser power normalization", self)

        self.checkbox_save = QtWidgets.QCheckBox("Saving as CSV", self)
        self.checkbox_save.toggled.connect(lambda: self.save_active())

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
        self.grid.addWidget(self.checkbox_power_normalization, 5, 0)
        self.grid.addWidget(self.checkbox_save, 6, 0)
        self.grid.addWidget(self.ok, 7, 0)
        self.grid.addWidget(self.plot, 8, 0)

        self.setWindowTitle("Noise Floor")
        self.window.setLayout(self.grid)
        self.window.show()

    def __del__(self):
        del self
        print('Destructor called, Employee deleted.')

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
        self.groupBox_list .setLayout(self.grid2)

        return self.groupBox_list

    @QtCore.Slot()
    def do_update(self):

        if self.ok.clicked:
            self.get_values()
            if self.groupBox_range.isChecked() and not(self.groupBox_list.isChecked()):

                if self.min_radius != '' and self.max_radius != '' and self.step_radius != '':
                    self.min_radius = int(self.min_radius)
                    self.max_radius = int(self.max_radius)
                    self.step_radius = int(self.step_radius)
                    self.radius_list = list(range(self.min_radius, self.max_radius, self.step_radius))
                    self.mode = 'Range'
                    self.find_radius_update_tab_flag = False
                    self.run_noiseFloor()

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Please insert radius size!")
                    self.msg_box.exec_()

            if self.groupBox_list.isChecked() and not(self.groupBox_range.isChecked()):

                if self.radius_list != '':
                    self.radius_list = eval(self.radius_list)
                    self.mode = 'List'
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

            elif not(self.groupBox_list.isChecked()) and not(self.groupBox_range.isChecked()):
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Please select one of the methods!")
                self.msg_box.exec_()

    def run_noiseFloor(self):

        if self.checkbox_power_normalization.isChecked():
            self.original_video, _ = Normalization(video=self.original_video).power_normalized()

        result_flag = True
        n_jobs = os.cpu_count()
        inter_flag_parallel_active = True
        flag_first_except = False
        while result_flag:
            try:
                self.noise_floor_ = NoiseFloor(self.original_video, list_range=self.radius_list, n_jobs=None,
                                               inter_flag_parallel_active=inter_flag_parallel_active)
                self.noise_floor_.plot_result()

                if self.checkbox_save.isChecked() and self.file_path is not None:

                    noise_floor = {'batch size': self.noise_floor_.list_range, 'SNR': self.noise_floor_.mean}
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
            self.noise_floor_.plot_result()
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please update first!5")
            self.msg_box.exec_()

    def get_values(self):

        if self.groupBox_range.isChecked():
            self.min_radius = self.le1.text()
            self.max_radius = self.le2.text()
            self.step_radius = self.le3.text()

        elif self.groupBox_list.isChecked():
            self.radius_list = self.le4.text()

    def add_line_edit1(self):
        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText('Min. batch size')
        self.le_1_label = QtWidgets.QLabel("Min batch size:")

        self.le2 = QtWidgets.QLineEdit()
        self.le2.setPlaceholderText('Max. batch size')
        self.le_2_label = QtWidgets.QLabel("Max batch size:")

        self.le3 = QtWidgets.QLineEdit()
        self.le3.setPlaceholderText('step')
        self.le_3_label = QtWidgets.QLabel("Stride between batch size:")

        self.grid1.addWidget(self.le_1_label, 2, 0)
        self.grid1.addWidget(self.le_2_label, 3, 0)
        self.grid1.addWidget(self.le_3_label, 4, 0)

        self.grid1.addWidget(self.le1, 2, 1)
        self.grid1.addWidget(self.le2, 3, 1)
        self.grid1.addWidget(self.le3, 4, 1)

    def add_line_edit2(self):
        self.le4 = QtWidgets.QLineEdit()
        self.le4.setPlaceholderText('list of batches size')
        self.le_4_label = QtWidgets.QLabel("List of all batches")

        self.grid2.addWidget(self.le_4_label, 0, 0)
        self.grid2.addWidget(self.le4, 1, 1)

    def save_active(self):
        if self.checkbox_save.isChecked():
            self.file_path = False

            self.file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Noise floor",
                                                                      QtCore.QDir.currentPath())

            self.file_path = self.file_path + '_noise_floor.csv'

    def closeEvent(self, event):
        event.accept()  # let the window close


