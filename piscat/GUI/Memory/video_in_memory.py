import psutil
import gc

from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtWidgets import *
from PySide2 import QtCore, QtGui
from PySide2.QtGui import *
from PySide2.QtCore import *
from functools import partial


class VideoInMemory(QtWidgets.QMainWindow):
    display_trigger = QtCore.Signal(str)

    def __init__(self, list_available_video, parent=None):
        super().__init__(parent)
        self.list_available_video = list_available_video
        self.display_flag_original = False
        self.display_flag_DRA = False
        self.display_flag_memory = True

        self.clear_flag_original = False
        self.clear_flag_DRA = False
        self.clear_flag_memory = True

        self.window = QtWidgets.QWidget()

        self.le_1_label = QtWidgets.QLabel("Original video:")
        self.le_2_label = QtWidgets.QLabel("DRA video:")
        self.le_3_label = QtWidgets.QLabel("Memory usage:")

        # Display
        self.Display_1 = QtWidgets.QPushButton("Display")
        self.Display_1.setAutoDefault(False)
        self.Display_1.setDisabled(True)
        self.Display_1.clicked.connect(partial(self.do_update_display, mode="original_video"))
        if 'original_video' in self.list_available_video.keys():
            if self.list_available_video['original_video']:
                self.Display_1.setDisabled(False)

        self.Display_2 = QtWidgets.QPushButton("Display")
        self.Display_2.setAutoDefault(False)
        self.Display_2.setDisabled(True)
        self.Display_2.clicked.connect(partial(self.do_update_display, mode="DRA_video"))
        if 'DRA_video' in self.list_available_video.keys():
            if self.list_available_video['DRA_video']:
                self.Display_2.setDisabled(False)

        # clear
        self.Clear_1 = QtWidgets.QPushButton("Clear")
        self.Clear_1.setAutoDefault(False)
        self.Clear_1.setDisabled(True)
        self.Clear_1.clicked.connect(partial(self.do_update_clear, mode="original_video"))
        if 'original_video' in self.list_available_video.keys():
            if self.list_available_video['original_video']:
                self.Clear_1.setDisabled(False)

        self.Clear_2 = QtWidgets.QPushButton("Clear")
        self.Clear_2.setAutoDefault(False)
        self.Clear_2.setDisabled(True)
        self.Clear_2.clicked.connect(partial(self.do_update_clear, mode="DRA_video"))
        if 'DRA_video' in self.list_available_video.keys():
            if self.list_available_video['DRA_video']:
                self.Clear_2.setDisabled(False)

        self._ProgressBar = QtWidgets.QProgressBar(self)
        self._ProgressBar.setValue(0)
        self.setProgress()

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.le_1_label, 0, 0)
        # self.grid.addWidget(self.Clear_1, 0, 1)
        self.grid.addWidget(self.Display_1, 0, 2)

        self.grid.addWidget(self.le_2_label, 1, 0)
        # self.grid.addWidget(self.Clear_2, 1, 1)
        self.grid.addWidget(self.Display_2, 1, 2)

        self.grid.addWidget(self.le_3_label, 3, 0)
        self.grid.addWidget(self._ProgressBar, 3, 2)

        self.window.setWindowTitle("Display video in Memory")
        # self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.window.setLayout(self.grid)
        self.window.show()

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def get_in(self, list_available_video):
        self.list_available_video = list_available_video

        self.display_flag_original = False
        self.display_flag_DRA = False
        self.display_flag_memory = True

        self.clear_flag_original = False
        self.clear_flag_DRA = False
        self.clear_flag_memory = True

        # Display
        self.Display_1.setDisabled(True)
        if self.list_available_video['original_video']:
            self.Display_1.setDisabled(False)

        self.Display_2.setDisabled(True)
        if self.list_available_video['DRA_video']:
            self.Display_2.setDisabled(False)

        # clear
        self.Clear_1.setDisabled(True)
        if self.list_available_video['original_video']:
            self.Clear_1.setDisabled(False)

        self.Clear_2.setDisabled(True)
        if self.list_available_video['DRA_video']:
            self.Clear_2.setDisabled(False)

    # @QtCore.Slot(int)
    def setProgress(self, step=1000):
        gc.collect()
        memory = dict(psutil.virtual_memory()._asdict())
        value = memory['percent']
        # print(Memory)
        self._ProgressBar.setMaximum(step * 100)
        self._ProgressBar.setValue(step * value)
        self._ProgressBar.setFormat("%.02f %%" % value)
        self._ProgressBar.setAlignment(Qt.AlignCenter)

    def do_update_display(self, mode):
        if mode == 'original_video':
            self.display_flag_original = True
            self.display_flag_memory = False
            self.display_trigger.emit('original_video')

        elif mode == 'DRA_video':
            self.display_flag_DRA = True
            self.display_flag_memory = False
            self.display_trigger.emit('DRA_video')

    def do_update_clear(self, mode):
        if mode == 'original_video':
            self.clear_flag_original = True
            self.clear_flag_memory = False

        elif mode == 'DRA_video':
            self.clear_flag_DRA = True
            self.clear_flag_memory = False

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()

