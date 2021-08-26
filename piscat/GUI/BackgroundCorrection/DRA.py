from PySide2 import QtGui, QtCore, QtWidgets


class DRA(QtWidgets.QMainWindow):

    signal_finish = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_size = None
        self.axis = 1
        self.mode_FPN = 'cpFPN'

        self.flag_display = False
        self.flag_FPN = False
        self.flag_power_normalization = False

        self.window = QtWidgets.QWidget()

        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText('radius')

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)

        self.checkbox_display = QtWidgets.QCheckBox("Display", self)
        self.checkbox_FPN = QtWidgets.QCheckBox("Fix pattern noise", self)
        self.checkbox_power_normalization = QtWidgets.QCheckBox("Laser power normalization", self)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.le1, 0, 0)
        self.grid.addWidget(self.checkbox_display, 0, 1)
        self.grid.addWidget(self.checkbox_power_normalization, 1, 1)
        self.grid.addWidget(self.createFirstExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.ok, 2, 0)

        self.window.setWindowTitle("Rolling Average")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def createFirstExclusiveGroup(self):

        self.groupBox_FPNc = QtWidgets.QGroupBox("FPNc:")
        self.groupBox_FPNc.setCheckable(True)
        self.groupBox_FPNc.setChecked(False)

        self.FPN_mode_group = QtWidgets.QButtonGroup()
        self.radio_wFPN_mode = QtWidgets.QRadioButton("Wavelet FPNc")
        self.radio_cpFPN_mode = QtWidgets.QRadioButton("Column_projection FPNc")
        self.radio_wf_FPN_mode = QtWidgets.QRadioButton("Wavelet_FFT2D FPNc")
        self.radio_cpFPN_mode.setChecked(True)

        self.FPN_mode_group.addButton(self.radio_wFPN_mode)
        self.FPN_mode_group.addButton(self.radio_cpFPN_mode)
        self.FPN_mode_group.addButton(self.radio_wf_FPN_mode)

        self.axis_group = QtWidgets.QButtonGroup()
        self.radio_axis_1 = QtWidgets.QRadioButton("FPNc in axis 0")
        self.radio_axis_2 = QtWidgets.QRadioButton("FPNc in axis 1")
        self.radio_axis_3 = QtWidgets.QRadioButton("FPNc in Both axis")
        self.radio_axis_2.setChecked(True)

        self.axis_group.addButton(self.radio_axis_1)
        self.axis_group.addButton(self.radio_axis_2)
        self.axis_group.addButton(self.radio_axis_3)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.radio_cpFPN_mode, 0, 0)
        grid.addWidget(self.radio_wFPN_mode, 0, 1)
        grid.addWidget(self.radio_wf_FPN_mode, 0, 2)

        grid.addWidget(self.radio_axis_1, 1, 0)
        grid.addWidget(self.radio_axis_2, 1, 1)
        grid.addWidget(self.radio_axis_3, 1, 2)

        self.groupBox_FPNc.setLayout(grid)
        return self.groupBox_FPNc

    def do_update(self):
        if self.ok.clicked:
            self.get_values()
            if self.checkbox_display.isChecked():
                self.flag_display = True
            else:
                self.flag_display = False

            if self.groupBox_FPNc.isChecked():
                self.flag_FPN = True
                if self.radio_axis_1.isChecked():
                    self.axis = 0
                elif self.radio_axis_2.isChecked():
                    self.axis = 1
                elif self.radio_axis_3.isChecked():
                    self.axis = 'Both'

                if self.radio_cpFPN_mode.isChecked():
                    self.mode_FPN = 'cpFPN'
                elif self.radio_wFPN_mode.isChecked():
                    self.mode_FPN = 'wFPN'
                elif self.radio_wf_FPN_mode.isChecked():
                    self.mode_FPN = 'fFPN'

            if self.checkbox_power_normalization.isChecked():
                self.flag_power_normalization = True

            if self.batch_size != '':
                self.batch_size = int(self.batch_size)
                self.signal_finish.emit(True)

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Please insert batch size!")
                self.msg_box.exec_()

    def get_values(self):
        self.batch_size = self.le1.text()

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()

