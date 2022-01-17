from piscat.GUI.InputOutput import Reading
from functools import partial


from PySide2 import QtCore
from PySide2 import QtWidgets

import os
import numpy as np


class LoadData_GUI(QtWidgets.QWidget):

    update_output_internal = QtCore.Signal(object)
    update_output_external = QtCore.Signal(object)

    update_tab_index = QtCore.Signal(int)

    def __init__(self):
        super(LoadData_GUI, self).__init__()
        self.original_video = None
        self.filename = None
        self.folder = None
        self.data_in = None

        self.update_output_internal.connect(self.updata_setting)

        self.info_video_setting = None
        self.empty_value_box_flag = False

        self.load = QtWidgets.QPushButton("load video")
        self.load.setAutoDefault(False)
        self.load.clicked.connect(self.read_data)
        self.load.setEnabled(True)
        self.load.setFixedWidth(100)

        self.Next = QtWidgets.QPushButton("Next")
        self.Next.setAutoDefault(False)
        self.Next.clicked.connect(self.update_tab)
        self.Next.setEnabled(True)
        self.Next.setFixedWidth(100)

        self.LineEdit_path = QtWidgets.QTextEdit(self)
        self.LineEdit_path_label = QtWidgets.QLabel("Path:")
        self.LineEdit_path.setFixedHeight(20)
        self.LineEdit_path.setFixedWidth(200)

        self.LineEdit_type = QtWidgets.QLineEdit()
        self.LineEdit_type.setPlaceholderText(".bin")
        self.LineEdit_type_label = QtWidgets.QLabel("Data type:")
        self.LineEdit_type.setFixedHeight(20)
        self.LineEdit_type.setFixedWidth(200)

        self.name_mkdir = QtWidgets.QLineEdit()
        self.name_mkdir.setPlaceholderText("folder_name")
        self.name_mkdir_label = QtWidgets.QLabel("Save folder name:")
        self.name_mkdir.setFixedHeight(20)
        self.name_mkdir.setFixedWidth(200)

        self.LineEdit_img_width = QtWidgets.QTextEdit(self)
        self.LineEdit_img_width_label = QtWidgets.QLabel("img_width:")
        self.LineEdit_img_width.setFixedHeight(20)
        self.LineEdit_img_width.setFixedWidth(200)

        self.LineEdit_img_height = QtWidgets.QTextEdit(self)
        self.LineEdit_img_height_label = QtWidgets.QLabel("img_height:")
        self.LineEdit_img_height.setFixedHeight(20)
        self.LineEdit_img_height.setFixedWidth(200)

        self.LineEdit_img_type = QtWidgets.QTextEdit(self)
        self.LineEdit_img_type_label = QtWidgets.QLabel("image_type:")
        self.LineEdit_img_type.setFixedHeight(20)
        self.LineEdit_img_type.setFixedWidth(200)

        self.LineEdit_frame = QtWidgets.QTextEdit(self)
        self.LineEdit_frame_label = QtWidgets.QLabel("frame:")
        self.LineEdit_frame.setFixedHeight(20)
        self.LineEdit_frame.setFixedWidth(200)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)


        self.setLayout(self.grid)

    def createFirstExclusiveGroup(self):
        groupBox = QtWidgets.QGroupBox("File browser:")

        self.grid_file_browser = QtWidgets.QGridLayout()
        self.grid_file_browser.addWidget(self.load, 0, 0)
        self.grid_file_browser.addWidget(self.LineEdit_path_label, 1, 0)
        self.grid_file_browser.addWidget(self.LineEdit_path, 1, 1)
        self.grid_file_browser.addWidget(self.LineEdit_type_label, 2, 0)
        self.grid_file_browser.addWidget(self.LineEdit_type, 2, 1)

        self.grid_file_browser.addWidget(self.LineEdit_img_width_label, 3, 0)
        self.grid_file_browser.addWidget(self.LineEdit_img_width, 3, 1)
        self.grid_file_browser.addWidget(self.LineEdit_img_height_label, 4, 0)
        self.grid_file_browser.addWidget(self.LineEdit_img_height, 4, 1)
        self.grid_file_browser.addWidget(self.LineEdit_frame_label, 5, 0)
        self.grid_file_browser.addWidget(self.LineEdit_frame, 5, 1)
        self.grid_file_browser.addWidget(self.LineEdit_img_type_label, 6, 0)
        self.grid_file_browser.addWidget(self.LineEdit_img_type, 6, 1)
        self.grid_file_browser.addWidget(self.name_mkdir_label, 7, 0)
        self.grid_file_browser.addWidget(self.name_mkdir, 7, 1)

        self.grid_file_browser.addWidget(self.Next, 8, 0)

        groupBox.setLayout(self.grid_file_browser)
        return groupBox


    def read_data(self):
        reading = Reading()
        reading.update_output.connect(self.updata_input_setting)
        reading.load_batch_data()

    @QtCore.Slot()
    def updata_input_setting(self, data_in):
        self.info_video_setting = data_in[0]

        self.filename = self.info_video_setting['path']
        self.LineEdit_path.setText(self.filename)
        # self.LineEdit_type.setText(self.info_video_setting['title'])

        img_width = self.info_video_setting['img_width']
        img_height = self.info_video_setting['img_height']
        s_frame = self.info_video_setting['s_frame']
        e_frame = self.info_video_setting['e_frame']
        frame_stride = self.info_video_setting['frame_stride']
        width_size_s = self.info_video_setting['width_size_s']
        width_size_e = self.info_video_setting['width_size_e']
        height_size_s = self.info_video_setting['height_size_s']
        height_size_e = self.info_video_setting['height_size_e']
        image_type = str(self.info_video_setting['image_type'])

        imageSize_w = 'img_width:' + str(img_width) + ',' + str(width_size_s) + ':' + str(width_size_e)
        imageSize_h = 'img_height:' + str(img_height) + ',' + str(height_size_s) + ':' + str(height_size_e)
        imageSize_f = str(s_frame) + ':' + str(e_frame) + ':' + str(frame_stride)

        self.LineEdit_img_type.setText(image_type)
        self.LineEdit_frame.setText(imageSize_f)
        self.LineEdit_img_width.setText(imageSize_w)
        self.LineEdit_img_height.setText(imageSize_h)

        self.update_output_internal.emit(data_in[0])

    def get_values(self):
        try:
            self.file_type = self.LineEdit_type.text()
            self.name_mkdir_ = self.name_mkdir.text()
            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()
            self.empty_value_box_flag = False

    @QtCore.Slot()
    def updata_setting(self, data_in):
        self.data_in = data_in

    @QtCore.Slot()
    def update_tab(self):
        self.get_values()
        if self.empty_value_box_flag:
            self.data_in['type_file'] = self.file_type
            self.data_in['name_mkdir'] = self.name_mkdir_

            self.update_output_external.emit(self.data_in)
            self.update_tab_index.emit(1)




