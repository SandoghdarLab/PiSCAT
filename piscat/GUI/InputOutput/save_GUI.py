from PySide2 import QtGui, QtCore, QtWidgets


#############Define MyWindow Class Here ############
class SaveVideo(QtWidgets.QMainWindow):
    signal_save_Done = QtCore.Signal()

    ##-----------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.bin_type = None
        self.video_type = None

        self.flag_display = False

        self.image_format = {"int_8": "i1", "int_16": "i2", "int_32": "i4", "int_64": "i8", "uint_8": "u1", "uint_16": "u2", "uint_32": "u4",
                             "uint_64": "u8", "float_16": "f2", "float_32": "f4", "float_64": "f8"}

        self.raw_data_update_flag = True

        self.window = QtWidgets.QWidget()

        self.combo_video = QtWidgets.QComboBox(self)
        self.combo_video.addItem("-Select the video type-")
        self.combo_video.addItem("RAW")
        self.combo_video.addItem("MP4")
        self.combo_video.addItem("GIF")

        self.combo_bin = QtWidgets.QComboBox(self)
        self.combo_bin.addItem("-Select the binary type-")
        self.combo_bin.addItem("int_8")
        self.combo_bin.addItem("int_16")
        self.combo_bin.addItem("int_32")
        self.combo_bin.addItem("int_64")

        self.combo_bin.addItem("uint_8")
        self.combo_bin.addItem("uint_16")
        self.combo_bin.addItem("uint_32")
        self.combo_bin.addItem("uint_64")

        self.combo_bin.addItem("float_16")
        self.combo_bin.addItem("float_32")
        self.combo_bin.addItem("float_64")
        self.combo_bin.currentIndexChanged.connect(self.on_select)

        self.ok = QtWidgets.QPushButton("Ok")
        self.ok.setAutoDefault(False)
        self.ok.setFixedWidth(100)
        self.ok.clicked.connect(self.do_update)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.ok, 6, 0)

        self.setWindowTitle("Saving video")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def createFirstExclusiveGroup(self):
        self.groupBox_cropping = QtWidgets.QGroupBox("Cast to a specified bin_type")
        self.groupBox_cropping.setCheckable(True)
        self.groupBox_cropping.setChecked(True)

        # groupBox.isChecked(False)
        self.grid1 = QtWidgets.QGridLayout()
        self.grid1.addWidget(self.combo_video, 0, 0)
        self.grid1.addWidget(self.combo_bin, 0, 1)
        self.groupBox_cropping .setLayout(self.grid1)

        return self.groupBox_cropping

    def do_update(self):
        if self.ok.clicked:
            self.on_select()
            if self.video_type is not None:
                if self.video_type == 'RAW':
                    if self.bin_type is None:
                        self.signal_save_Done.emit()
                    else:
                        self.signal_save_Done.emit()
                    self.raw_data_update_flag = False
                elif self.video_type == 'MP4':
                    self.signal_save_Done.emit()
                elif self.video_type == 'GIF':
                    self.signal_save_Done.emit()

    def on_select(self):
        if self.combo_bin.currentText() == "-Select the binary type-":
            self.bin_type = None
        else:
            self.bin_type = self.image_format[self.combo_bin.currentText()]
            
        if self.combo_video.currentText() == "-Select the video type-":
            self.video_type = None
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please select type of video writer!")
            self.msg_box.exec_()
        else:
            self.video_type = self.combo_video.currentText()

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()
