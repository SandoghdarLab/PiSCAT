import matplotlib.pyplot as plt
from piscat.InputOutput import reading_videos
from piscat.InputOutput.image_to_video import Image2Video

from PySide2 import QtCore
from PySide2 import QtWidgets


class Image2Video(QtWidgets.QMainWindow):

    def __init__(self, fileName):
        super(Image2Video, self).__init__()
        self.fileName = fileName

        self.im_type = None

        self.width_size = None
        self.height_size = None

        self.width_size_s = None
        self.width_size_e = None

        self.height_size_s = None
        self.height_size_e = None

        self.frame_s = None
        self.frame_e = None
        self.frame_jump = None

        self.set_bit_order = None
        self.type = None

        self.flag_display = False

        self.little_endian_flag = False
        self.big_endian_flag = False

        self.image_format = {"int_8": "i1", "int_16": "i2", "int_32": "i4", "int_64": "i8", "uint_8": "u1", "uint_16": "u2", "uint_32": "u4",
                             "uint_64": "u8", "float_16": "f2", "float_32": "f4", "float_64": "f8"}

        self.reader_format = {"binary": "binary", "tif": "tif", "avi": "avi", "png": "png"}
        self.raw_data_update_flag = True

        self.window = QtWidgets.QWidget()

        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItem("-Select the image bin_type-")
        self.combo.addItem("int_8")
        self.combo.addItem("int_16")
        self.combo.addItem("int_32")
        self.combo.addItem("int_64")

        self.combo.addItem("uint_8")
        self.combo.addItem("uint_16")
        self.combo.addItem("uint_32")
        self.combo.addItem("uint_64")

        self.combo.addItem("float_16")
        self.combo.addItem("float_32")
        self.combo.addItem("float_64")
        self.combo.currentIndexChanged.connect(self.on_select)

        self.combo_reader_type = QtWidgets.QComboBox(self)
        self.combo_reader_type.addItem("-Select the video reader bin_type-")
        self.combo_reader_type.addItem("binary")
        self.combo_reader_type.addItem("tif")
        self.combo_reader_type.addItem("avi")
        self.combo_reader_type.addItem("png")
        self.combo_reader_type.currentIndexChanged.connect(self.on_select_vid_reader)

        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText('width')
        self.le_1_label = QtWidgets.QLabel("Image size width(px):")
        self.le1.setFixedWidth(50)

        self.le2 = QtWidgets.QLineEdit()
        self.le2.setPlaceholderText('height')
        self.le_2_label = QtWidgets.QLabel("Image size height(px):")
        self.le2.setFixedWidth(50)

        self.le_imType = QtWidgets.QLineEdit()
        self.le_imType.setPlaceholderText('image_type')
        self.le_imType_label = QtWidgets.QLabel("image_type(\*.---):")
        self.le_imType.setFixedWidth(50)

        self.checkbox_0 = QtWidgets.QRadioButton("little-endian byte order")
        self.checkbox_0.toggled.connect(lambda: self.btnstate(self.checkbox_0))
        self.checkbox_0.setChecked(True)

        self.checkbox_1 = QtWidgets.QRadioButton("big-endian byte order")
        self.checkbox_1.toggled.connect(lambda: self.btnstate(self.checkbox_1))

        self.checkbox_display = QtWidgets.QCheckBox("Display", self)

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.setFixedWidth(100)
        self.ok.clicked.connect(self.do_update)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.ok, 6, 0)

        self.setWindowTitle("Reading video")
        self.window.setLayout(self.grid)
        self.window.show()

    def createFirstExclusiveGroup(self):
        groupBox = QtWidgets.QGroupBox("Raw information:")
        self.grid1 = QtWidgets.QGridLayout()
        self.grid1.addWidget(self.combo, 0, 0)
        self.grid1.addWidget(self.combo_reader_type, 0, 1)
        self.grid1.addWidget(self.checkbox_0, 0, 2)
        self.grid1.addWidget(self.checkbox_1, 0, 3)
        self.grid1.addWidget(self.checkbox_display, 0, 4)
        self.grid1.addWidget(self.le_1_label, 1, 0)
        self.grid1.addWidget(self.le1, 1, 1)
        self.grid1.addWidget(self.le_2_label, 2, 0)
        self.grid1.addWidget(self.le2, 2, 1)
        self.grid1.addWidget(self.le_imType_label, 3, 0)
        self.grid1.addWidget(self.le_imType, 3, 1)
        groupBox.setLayout(self.grid1)

        return groupBox

    def createSecondExclusiveGroup(self):
        self.groupBox_cropping = QtWidgets.QGroupBox("Video Cropping")
        self.groupBox_cropping.setCheckable(True)
        self.groupBox_cropping.setChecked(False)
        self.grid2 = QtWidgets.QGridLayout()
        self.info_cut_edith()
        self.groupBox_cropping .setLayout(self.grid2)

        return self.groupBox_cropping

    def do_update(self):
        if self.ok.clicked:
            self.on_select()
            self.on_select_vid_reader()
            self.get_values()
            self.check_button()
            if self.checkbox_display.isChecked():
                self.flag_display = True

            if self.height_size != '' and self.width_size != '':
                self.width_size = int(self.width_size)
                self.height_size = int(self.height_size)
                self.frame_s = int(0)
                self.frame_e = int(-1)
                self.frame_jump = 1

                if self.groupBox_cropping .isChecked():

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

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Please insert image size!")
                self.msg_box.exec_()

    def check_cut_size_odd(self):

        if (self.width_size_s % 2) != 0:
            self.width_size_s = self.width_size_s + 1

        if (self.width_size_e % 2) != 0:
            self.width_size_e = self.width_size_e - 1

        if (self.height_size_s % 2) != 0:
            self.height_size_s = self.height_size_s + 1

        if (self.height_size_e % 2) != 0:
            self.height_size_e = self.height_size_e - 1

    def on_select(self):
        if self.combo.currentText() == "-Select the image bin_type-":
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please select bin_type of image!")
            self.msg_box.exec_()
        else:
            self.type = self.image_format[self.combo.currentText()]

    def on_select_vid_reader(self):
        if self.combo_reader_type.currentText() == "-Select the video reader bin_type-":
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please select bin_type of reader!")
            self.msg_box.exec_()
        else:
            self.video_reader_type = self.reader_format[self.combo_reader_type.currentText()]

    def check_button(self):
        bit_order = {'native': '=', 'little_endian': '<', 'big_endian': '>'}

        if self.little_endian_flag:
            self.set_bit_order = bit_order["little_endian"]
        elif self.big_endian_flag:
            self.set_bit_order = bit_order["big_endian"]
        else:
            self.set_bit_order = bit_order["native"]

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
        self.width_size = self.le1.text()
        self.height_size = self.le2.text()
        self.im_type = self.le_imType.text()

        if self.groupBox_cropping.isChecked():

            self.width_size_s = self.le3.text()
            self.width_size_e = self.le4.text()

            self.height_size_s = self.le5.text()
            self.height_size_e = self.le6.text()

            self.frame_s = self.le7.text()
            self.frame_e = self.le8.text()
            self.frame_jump = self.le9.text()

    def btnstate(self, b):

        if b.text() == "little-endian byte order":
            if b.isChecked() == True:
                self.little_endian_flag = True
            else:
                self.little_endian_flag = False

        if b.text() == "big-endian byte order":
            if b.isChecked() == True:
                self.big_endian_flag = True
            else:
                self.big_endian_flag = False

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
        print("closing PlaySetting")
