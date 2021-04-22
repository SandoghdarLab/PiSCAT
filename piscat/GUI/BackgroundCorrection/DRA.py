from PySide2 import QtGui, QtCore, QtWidgets


class DRA(QtWidgets.QMainWindow):

    signal_finish = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_size = None

        self.flag_display = False
        self.flag_power_normalization = False

        self.window = QtWidgets.QWidget()

        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText('radius')

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)

        self.checkbox_display = QtWidgets.QCheckBox("Display", self)
        self.checkbox_power_normalization = QtWidgets.QCheckBox("Laser power normalization", self)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.le1, 0, 0)
        self.grid.addWidget(self.checkbox_display, 0, 1)
        self.grid.addWidget(self.checkbox_power_normalization, 1, 1)
        self.grid.addWidget(self.ok, 2, 0)

        self.window.setWindowTitle("Rolling Average")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def do_update(self):
        if self.ok.clicked:
            self.get_values()
            if self.checkbox_display.isChecked():
                self.flag_display = True
            else:
                self.flag_display = False

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

