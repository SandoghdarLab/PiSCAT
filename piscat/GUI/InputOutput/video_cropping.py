from PySide2 import QtGui, QtCore, QtWidgets


class Cropping(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.width_size_s = None
        self.width_size_e = None

        self.height_size_s = None
        self.height_size_e = None

        self.height_size = -1
        self.width_size = -1

        self.frame_s = None
        self.frame_e = None
        self.frame_jump = None

        self.type = None

        self.flag_display = False

        self.raw_data_update_flag = True

        self.window = QtWidgets.QWidget()

        self.checkbox_display = QtWidgets.QCheckBox("Display", self)

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.setFixedWidth(100)
        self.ok.clicked.connect(self.do_update)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createSecondExclusiveGroup(), 0, 0)

        self.grid.addWidget(self.checkbox_display, 5, 0)
        self.grid.addWidget(self.ok, 6, 0)

        self.setWindowTitle("Reading video")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()

    def createSecondExclusiveGroup(self):

        self.groupBox_cropping = QtWidgets.QGroupBox("Video Cropping")
        self.groupBox_cropping.setCheckable(True)
        self.groupBox_cropping.setChecked(False)
        self.grid2 = QtWidgets.QGridLayout()
        self.info_cut_edith()
        self.groupBox_cropping.setLayout(self.grid2)

        return self.groupBox_cropping

    def do_update(self):
        if self.ok.clicked:
            self.get_values()
            if self.checkbox_display.isChecked():
                self.flag_display = True

            if self.groupBox_cropping.isChecked():
                if self.width_size_s != '' and self.width_size_e != '':
                    self.width_size_s = int(self.width_size_s)
                    self.width_size_e = int(self.width_size_e)
                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("All rows are selected")
                    self.msg_box.exec_()
                    self.width_size_s = 0
                    self.width_size_e = self.width_size

                if self.height_size_s != '' and self.height_size_e != '':
                    self.height_size_s = int(self.height_size_s)
                    self.height_size_e = int(self.height_size_e)
                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("All columns are selected")
                    self.msg_box.exec_()
                    self.height_size_s = 0
                    self.height_size_e = self.height_size

                if self.frame_s != '' and self.frame_e != '' and self.frame_jump != '':
                    self.frame_s = int(self.frame_s)
                    self.frame_e = int(self.frame_e)
                    self.frame_jump = int(self.frame_jump)

                else:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("All frames are selected")
                    self.msg_box.exec_()
                    self.frame_s = 0
                    self.frame_e = -1
                    self.frame_jump = 1


            self.raw_data_update_flag = False
            self.window.close()

    def check_cut_size_odd(self):

        if (self.width_size_s % 2) != 0:
            self.width_size_s = self.width_size_s + 1

        if (self.width_size_e % 2) != 0:
            self.width_size_e = self.width_size_e - 1

        if (self.height_size_s % 2) != 0:
            self.height_size_s = self.height_size_s + 1

        if (self.height_size_e % 2) != 0:
            self.height_size_e = self.height_size_e - 1

    def info_cut_edith(self):
        self.le3 = QtWidgets.QLineEdit()
        self.le3.setPlaceholderText('width_start')
        self.le_3_label = QtWidgets.QLabel("start width pixel:")

        self.le4 = QtWidgets.QLineEdit()
        self.le4.setPlaceholderText('width_end')
        self.le_4_label = QtWidgets.QLabel("end width pixel:")

        self.le5 = QtWidgets.QLineEdit()
        self.le5.setPlaceholderText('height_start')
        self.le_5_label = QtWidgets.QLabel("start height pixel:")

        self.le6 = QtWidgets.QLineEdit()
        self.le6.setPlaceholderText('height_end')
        self.le_6_label = QtWidgets.QLabel("end height pixel:")

        self.le7 = QtWidgets.QLineEdit()
        self.le7.setPlaceholderText('frame_start')
        self.le_7_label = QtWidgets.QLabel("start frame:")

        self.le8 = QtWidgets.QLineEdit()
        self.le8.setPlaceholderText('frame_end')
        self.le_8_label = QtWidgets.QLabel("end frame:")

        self.le9 = QtWidgets.QLineEdit()
        self.le9.setPlaceholderText('frame_jump')
        self.le_9_label = QtWidgets.QLabel("frame stride:")

        self.grid2.addWidget(self.le_3_label, 1, 0)
        self.grid2.addWidget(self.le_4_label, 1, 2)
        self.grid2.addWidget(self.le_5_label, 2, 0)
        self.grid2.addWidget(self.le_6_label, 2, 2)
        self.grid2.addWidget(self.le_7_label, 3, 0)
        self.grid2.addWidget(self.le_8_label, 3, 2)
        self.grid2.addWidget(self.le_9_label, 3, 4)

        self.grid2.addWidget(self.le3, 1, 1)
        self.grid2.addWidget(self.le4, 1, 3)
        self.grid2.addWidget(self.le5, 2, 1)
        self.grid2.addWidget(self.le6, 2, 3)
        self.grid2.addWidget(self.le7, 3, 1)
        self.grid2.addWidget(self.le8, 3, 3)
        self.grid2.addWidget(self.le9, 3, 5)

    def get_values(self):
        if self.groupBox_cropping.isChecked():
            self.width_size_s = self.le3.text()
            self.width_size_e = self.le4.text()

            self.height_size_s = self.le5.text()
            self.height_size_e = self.le6.text()

            self.frame_s = self.le7.text()
            self.frame_e = self.le8.text()
            self.frame_jump = self.le9.text()

    def handleClose(self):
        print("closing Tab B")
