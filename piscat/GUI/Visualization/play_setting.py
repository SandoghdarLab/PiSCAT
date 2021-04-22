from PySide2 import QtWidgets, QtCore, QtGui


class PlaySetting(QtWidgets.QWidget):
    procDone_Animation_panel = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(PlaySetting, self).__init__(parent)

        # ok
        self.ok = QtWidgets.QPushButton("ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)
        self.ok.setFixedWidth(100)

        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText('step_size')
        self.le_1_label = QtWidgets.QLabel("Stride between frame:")
        self.le1.setFixedWidth(100)

        self.le2 = QtWidgets.QLineEdit()
        self.le2.setPlaceholderText('Time delay')
        self.le_2_label = QtWidgets.QLabel("Time delay (ms):")
        self.le2.setFixedWidth(100)

        self.le3 = QtWidgets.QLineEdit()
        self.le3.setPlaceholderText('fps')
        self.le_3_label = QtWidgets.QLabel("Speed:")
        self.le3.setFixedWidth(100)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.le_1_label, 0, 0)
        self.grid.addWidget(self.le1, 0, 1)
        self.grid.addWidget(self.le_2_label, 1, 0)
        self.grid.addWidget(self.le2, 1, 1)
        self.grid.addWidget(self.le_3_label, 2, 0)
        self.grid.addWidget(self.le3, 2, 1)
        self.grid.addWidget(self.ok, 3, 1)
        self.setLayout(self.grid)

        self.setWindowTitle("Animation panel")
        self.setLayout(self.grid)

    @QtCore.Slot()
    def do_update(self):

        try:
            self.step_size = int(self.le1.text())
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Incorrect input_video for step size! \nThe default value will be selected")
            self.msg_box.exec_()
            self.step_size = 1

        try:
            self.time_delay = float(self.le2.text())
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Incorrect input_video for time delay! \nThe default value will be selected")
            self.msg_box.exec_()
            self.time_delay = 0.1

        try:
            self.fps = float(self.le3.text())
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Incorrect input_video for Speed! \nThe default value will be selected")
            self.msg_box.exec_()
            self.fps = 10.0

        self.procDone_Animation_panel.emit([self.step_size, self.time_delay, self.fps])
        self.close()