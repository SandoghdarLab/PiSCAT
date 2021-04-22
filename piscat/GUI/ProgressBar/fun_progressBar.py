# from GUI_RUN.Main import PiSCAT_GUI
from PySide2 import QtCore

class ProgressBar():

    def __init__(self, object):
        super(ProgressBar, self).__init__()
        self.progressBar = object

    @QtCore.Slot(int)
    def setProgress(self, val):
        self.progressBar.setValue(val)

    def setRange(self, max):
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(max)

    def setLabel(self, str):
        self.progressBar.setFormat(str)
