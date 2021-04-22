from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat.GUI.InputOutput import Reading

from PySide2 import QtCore, QtWidgets
from functools import partial
import numpy as np


class AnalysisConstant(QtWidgets.QMainWindow):
    update_output = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(600, 1000)
        self.flag_analysis_finish = True
        self.flag_remove_box = True

        self.original_video_1 = None
        self.out_video = None
        self.constant_number = None
        self.different_views = {}

        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("-Select the analysis-")
        self.combo.addItem("Subtraction")
        self.combo.addItem("ADD")
        self.combo.addItem("Divide")
        self.combo.addItem("Multiplication")
        self.combo.currentIndexChanged.connect(self.on_select)

        self.load1 = QtWidgets.QPushButton("load video")
        self.load1.setAutoDefault(False)
        self.load1.clicked.connect(self.read_data1)
        self.load1.setEnabled(False)
        self.load1.setFixedWidth(100)

        self.inputFileLineEdit1 = QtWidgets.QTextEdit(self)
        self.inputFileLineEdit1.setFixedHeight(20)
        self.inputFileLineEdit1.setFixedWidth(500)

        self.inputNumber = QtWidgets.QLineEdit(self)
        self.inputNumber.setFixedHeight(20)
        self.inputNumber.setFixedWidth(100)
        self.inputNumber_label = QtWidgets.QLabel("Number:")

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.combo, 0, 0)
        self.grid.addWidget(self.load1, 1, 0)
        self.grid.addWidget(self.inputFileLineEdit1, 1, 1)
        self.grid.addWidget(self.inputNumber_label, 2, 0)
        self.grid.addWidget(self.inputNumber, 2, 1)
        self.grid.addWidget(self.ok, 3, 0)

        self.window = QtWidgets.QWidget()
        self.setWindowTitle("Video calculator")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setGeometry(450, 90, 600, 50)
        self.window.setLayout(self.grid)
        self.window.show()

    def on_select(self):

        if self.combo.currentText() == "Subtraction":

            self.load1.setEnabled(True)

        elif self.combo.currentText() == "ADD":

            self.load1.setEnabled(True)

        elif self.combo.currentText() == "Divide":

            self.load1.setEnabled(True)

        elif self.combo.currentText() == "Multiplication":

            self.load1.setEnabled(True)

    def updata_input_video(self, data_in, label):
        if label == 0:
            self.original_video_1 = data_in[0]
            self.title_0 = data_in[1]
            self.file_name_0 = data_in[2]
        elif label == 1:
            self.original_video_2 = data_in[0]
            self.title_1 = data_in[1]
            self.file_name_1 = data_in[2]

    def read_data1(self):
        reading = Reading()
        reading.update_output.connect(partial(self.updata_input_video, label=0))
        reading.read_video()

        self.inputFileLineEdit1.setText(self.file_name_0)

    def display_result(self, input_video, headr_name):
        visualization_ = Visulization_localization()
        visualization_.new_display(input_video, input_video, object=None, title=headr_name)

    def do_update(self):

        if self.ok.clicked:
            flag_read_number = self.get_values()
            if flag_read_number:
                if self.combo.currentText() == "Subtraction":
                    if self.original_video_1 is not None:
                        self.out_video = np.subtract(self.original_video_1, self.constant_number)
                        self.display_result(self.out_video, "Subtraction")
                        self.update_output.emit([self.out_video, self.combo.currentText(), None])

                    else:
                        self.msg_box = QtWidgets.QMessageBox()
                        self.msg_box.setWindowTitle("Warning!")
                        self.msg_box.setText("Please import videos!")
                        self.msg_box.exec_()

                elif self.combo.currentText() == "ADD":
                    if self.original_video_1 is not None:
                        self.out_video = np.add(self.original_video_1, self.constant_number)
                        self.display_result(self.out_video, "ADD")
                        self.update_output.emit([self.out_video, self.combo.currentText(), None])

                    else:
                        self.msg_box = QtWidgets.QMessageBox()
                        self.msg_box.setWindowTitle("Warning!")
                        self.msg_box.setText("Please import videos!")
                        self.msg_box.exec_()

                elif self.combo.currentText() == "Divide":
                    if self.original_video_1 is not None:
                        self.out_video = np.divide(self.original_video_1, self.constant_number)
                        self.display_result(self.out_video, "Divide")
                        self.update_output.emit([self.out_video, self.combo.currentText(), None])

                    else:
                        self.msg_box = QtWidgets.QMessageBox()
                        self.msg_box.setWindowTitle("Warning!")
                        self.msg_box.setText("Please import videos!")
                        self.msg_box.exec_()

                elif self.combo.currentText() == "Multiplication":

                    if self.original_video_1 is not None:
                        self.out_video = np.multiply(self.original_video_1, self.constant_number)
                        self.display_result(self.out_video, "Multiplication")
                        self.update_output.emit([self.out_video, self.combo.currentText(), None])
                    else:
                        self.msg_box = QtWidgets.QMessageBox()
                        self.msg_box.setWindowTitle("Warning!")
                        self.msg_box.setText("Please import video!")
                        self.msg_box.exec_()

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Please fill the number text box!")
                self.msg_box.exec_()

    def get_values(self):
        try:
            self.constant_number = float(self.inputNumber.text())
            flag_read_number = True
        except:
            flag_read_number = False
        return flag_read_number

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
        print("closing PlaySetting")

