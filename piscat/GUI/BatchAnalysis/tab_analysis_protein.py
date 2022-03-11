from piscat.GUI.InputOutput import Reading
from functools import partial

from piscat.InputOutput import reading_videos
from piscat.Analysis.analysis_protein_videos import protein_analysis


from PySide6 import QtCore
from PySide6 import QtWidgets

import os
import numpy as np


class ProteinAnalysis_GUI(QtWidgets.QWidget):

    update_output_internal = QtCore.Signal(object)
    update_output_external = QtCore.Signal(object)

    update_tab_index = QtCore.Signal(int)

    def __init__(self):
        super(ProteinAnalysis_GUI, self).__init__()
        self.hyperparameters = None
        self.flags = None

        self.info_video_setting = None
        self.empty_value_box_flag = False

        self.Run = QtWidgets.QPushButton("Run")
        self.Run.setAutoDefault(False)
        self.Run.clicked.connect(self.run_analysis_scr)
        self.Run.setEnabled(True)
        self.Run.setFixedWidth(100)

        self.LineEdit_hyperparameters = QtWidgets.QTextEdit(self)
        self.LineEdit_hyperparameters_label = QtWidgets.QLabel("Hyperparameters:")
        # self.LineEdit_hyperparameters.setFixedHeight(20)
        # self.LineEdit_hyperparameters.setFixedWidth(200)

        self.LineEdit_flags = QtWidgets.QTextEdit(self)
        self.LineEdit_flags_label = QtWidgets.QLabel("flags:")
        # self.LineEdit_flags.setFixedHeight(20)
        # self.LineEdit_flags.setFixedWidth(200)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)

        self.setLayout(self.grid)

    def createFirstExclusiveGroup(self):
        groupBox = QtWidgets.QGroupBox("File browser:")

        self.grid_file_browser = QtWidgets.QGridLayout()
        self.grid_file_browser.addWidget(self.LineEdit_hyperparameters_label, 1, 0)
        self.grid_file_browser.addWidget(self.LineEdit_hyperparameters, 1, 1)
        self.grid_file_browser.addWidget(self.LineEdit_flags_label, 2, 0)
        self.grid_file_browser.addWidget(self.LineEdit_flags, 2, 1)
        self.grid_file_browser.addWidget(self.Run, 8, 0)

        groupBox.setLayout(self.grid_file_browser)
        return groupBox

    @QtCore.Slot()
    def updata_input_setting(self, data_in):
        self.hyperparameters = data_in[0]
        self.flags = data_in[1]

        self.LineEdit_hyperparameters.setText(str(self.hyperparameters))
        self.LineEdit_flags.setText(str(self.flags))

    def run_analysis_scr(self):
        if self.hyperparameters is not None and self.flags is not None:
            type_file = self.hyperparameters['type_file']
            dirName = self.hyperparameters['path']
            name_mkdir = self.hyperparameters['name_mkdir']

            df_video = reading_videos.DirectoryType(dirName, type_file).return_df()

            paths = df_video['Directory'].tolist()
            video_names = df_video['File'].tolist()
            protein_analysis(paths=paths, video_names=video_names, hyperparameters=self.hyperparameters, flags=self.flags,
                             name_mkdir=name_mkdir)
        else:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Hyperparameters or Flags is not defined!")
            self.msg_box.exec_()











