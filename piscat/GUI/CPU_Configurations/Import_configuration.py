from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtWidgets import *
from PySide2 import QtCore, QtGui
from PySide2.QtGui import *
from PySide2.QtCore import *

import psutil


class CPU_Setting(QtWidgets.QMainWindow):
    update_CPU_Setting = QtCore.Signal()

    def __init__(self):
        super(CPU_Setting,  self).__init__()
        self.n_jobs = None
        self.backend = None
        self.verbose = None
        self.parallel_active = None
        self.threshold_for_parallel_run = None
        self.cpu_configuration_update_flag = True
        self.empty_value_box_flag = False

        self.cpu_backend = {"loky": "loky", "multiprocessing": "multiprocessing", "threading": "threading"}

        self.window = QtWidgets.QWidget()

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)

        self._ProgressBar = QtWidgets.QProgressBar(self)
        self._ProgressBar.setValue(0)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)

        self.grid.addWidget(self.ok, 3, 0)
        # self.grid.addWidget(self._ProgressBar, 3, 1)

        self.setWindowTitle("CPU setting")
        # self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def createFirstExclusiveGroup(self):
        self.groupBox_CPU = QtWidgets.QGroupBox("CPU parallel setting:")
        self.groupBox_CPU.setCheckable(True)
        self.groupBox_CPU.setChecked(False)
        # groupBox.isChecked(False)
        self.grid2 = QtWidgets.QGridLayout()
        self.add_line_edit1()
        self.groupBox_CPU .setLayout(self.grid2)

        return self.groupBox_CPU

    def setProgress(self, step=1000):
        value = psutil.cpu_percent()

        self._ProgressBar.setMaximum(step * 100)
        self._ProgressBar.setValue(step * value)
        self._ProgressBar.setFormat("%.02f %%" % value)
        self._ProgressBar.setAlignment(Qt.AlignCenter)

    @QtCore.Slot()
    def do_update(self):
        if self.ok.clicked:
            if self.groupBox_CPU.isChecked():
                self.parallel_active = True
                self.get_values()

                if self.n_jobs != '':
                    self.n_jobs = int(self.n_jobs)
                else:
                    self.n_jobs = -1
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("all CPU cores are selected!")
                    self.msg_box.exec_()

                if self.verbose != '':
                    self.verbose = int(self.verbose)
                else:
                    self.verbose = 10
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Default Verbose is selected!")
                    self.msg_box.exec_()

                if self.backend is None:
                    self.backend = 'multiprocessing'
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Default backend is selected!")
                    self.msg_box.exec_()

            else:
                self.parallel_active = False

            self.cpu_configuration_update_flag = False
            self.update_CPU_Setting.emit()
            self.window.close()

    def on_select(self):
        if self.combo.currentText() == "-Select the backend-":
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please select the backend!")
            self.msg_box.exec_()
        else:
            self.backend = self.cpu_backend[self.combo.currentText()]

    def add_line_edit1(self):
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("-Select the backend-")
        self.combo.addItem("loky")
        self.combo.addItem("multiprocessing")
        self.combo.addItem("threading")

        self.combo.currentIndexChanged.connect(self.on_select)

        self.le1_cpu_cores = QtWidgets.QLineEdit()
        self.le1_cpu_cores.setPlaceholderText('Number of use CPU cores')
        self.le1_cpu_cores_label = QtWidgets.QLabel("Number cores:")

        self.le2_verbose = QtWidgets.QLineEdit()
        self.le2_verbose.setPlaceholderText('Verbose')
        self.le2_verbose_label = QtWidgets.QLabel("Verbose:")

        self.grid2.addWidget(self.combo, 1, 0)
        self.grid2.addWidget(self.le1_cpu_cores_label, 2, 0)
        self.grid2.addWidget(self.le2_verbose_label, 3, 0)

        self.grid2.addWidget(self.combo, 1, 1)
        self.grid2.addWidget(self.le1_cpu_cores, 2, 1)
        self.grid2.addWidget(self.le2_verbose, 3, 1)

    def get_values(self):
        self.on_select()
        try:
            self.n_jobs = self.le1_cpu_cores.text()
            self.verbose = self.le2_verbose.text()
            self.threshold_for_parallel_run = self.le3_threshold_for_parallel_run.text()
            self.empty_value_box_flag = True
        except:
            self.empty_value_box_flag = False

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()
