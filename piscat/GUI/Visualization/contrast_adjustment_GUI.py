from piscat.Visualization.contrast_adjustment import ContrastAdjustment

from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Contrast_adjustment_GUI(QtWidgets.QMainWindow):
    min_intensity_signal = QtCore.Signal(object)
    max_intensity_signal = QtCore.Signal(object)
    alpha_signal = QtCore.Signal(object)
    beta_signal = QtCore.Signal(object)

    def __init__(self, alpha, beta, min_intensity, max_intensity, parent=None):
        super().__init__(parent)

        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.alpha = alpha
        self.beta = beta

        self.window = QtWidgets.QWidget()

        # Min hist
        self.slider_min_label = QLabel('Minimum:')
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setRange(min_intensity, max_intensity)
        self.slider_min.setValue(min_intensity)
        self.slider_min.setTracking(True)
        # self.slider_min.setTickPosition(QSlider.TicksBothSides)
        self.slider_min.valueChanged.connect(self.apply)

        # Max hist
        self.slider_max_label = QLabel('Maximum:')
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setRange(min_intensity, max_intensity)
        self.slider_max.setValue(max_intensity)
        self.slider_max.setTracking(True)
        # self.slider_max.setTickPosition(QSlider.TicksBothSides)
        self.slider_max.valueChanged.connect(self.apply)

        # Brightness
        self.slider_alpha_label = QLabel('Brightness:')
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(1.0, 10.0)
        self.slider_alpha.setValue(alpha)
        self.slider_alpha.setTracking(True)
        # self.slider_alpha.setTickPosition(QSlider.TicksBothSides)
        self.slider_alpha.valueChanged.connect(self.apply)

        # Contrast
        self.slider_beta_label = QLabel('Contrast:')
        self.slider_beta = QSlider(Qt.Horizontal)
        self.slider_beta.setRange(0, 100)
        self.slider_beta.setValue(beta)
        self.slider_beta.setTracking(True)
        # self.slider_beta.setTickPosition(QSlider.TicksBothSides)
        self.slider_beta.valueChanged.connect(self.apply)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setAutoDefault(False)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setFixedWidth(100)

        self.auto_btn = QtWidgets.QPushButton("Auto")
        self.auto_btn.setAutoDefault(False)
        self.auto_btn.clicked.connect(self.auto)
        self.auto_btn.setFixedWidth(100)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.auto_btn, 1, 0)
        self.grid.addWidget(self.reset_btn, 2, 0)

        self.setWindowTitle("Contrast_adjustment")
        self.window.setLayout(self.grid)
        self.window.show()

    def createFirstExclusiveGroup(self):
        self.groupBox_slideBar = QtWidgets.QGroupBox("Adjustment:")

        self.gridSlideBar = QtWidgets.QGridLayout()
        self.gridSlideBar.addWidget(self.slider_min_label, 0, 0)
        self.gridSlideBar.addWidget(self.slider_min, 0, 1)

        self.gridSlideBar.addWidget(self.slider_max_label, 1, 0)
        self.gridSlideBar.addWidget(self.slider_max, 1, 1)

        self.gridSlideBar.addWidget(self.slider_alpha_label, 2, 0)
        self.gridSlideBar.addWidget(self.slider_alpha, 2, 1)

        self.gridSlideBar.addWidget(self.slider_beta_label, 3, 0)
        self.gridSlideBar.addWidget(self.slider_beta, 3, 1)

        self.groupBox_slideBar.setLayout(self.gridSlideBar)

        return self.groupBox_slideBar

    def valueHandler(self, value):
        scaledValue = float(value) / 100  # bin_type of "value" is int so you need to convert it to float in order to get float bin_type for "scaledValue"

    def frame_getter(self, image):
        self.image = image

    def apply(self):
        self.alpha_signal.emit(self.slider_alpha.value())
        self.beta_signal.emit(self.slider_beta.value())
        self.min_intensity_signal.emit(self.slider_min.value())
        self.max_intensity_signal.emit(self.slider_max.value())

    def auto(self):
        self.alpha_signal.emit(None)
        self.beta_signal.emit(None)
        self.min_intensity_signal.emit(None)
        self.max_intensity_signal.emit(None)

        self.slider_min.setValue(self.min_intensity)
        self.slider_max.setValue(self.max_intensity)
        self.slider_alpha.setValue(self.alpha)
        self.slider_beta.setValue(self.beta)

    def reset(self):
        self.slider_min.setValue(0)
        self.slider_max.setValue(self.max_intensity)
        self.slider_alpha.setValue(5)
        self.slider_beta.setValue(50)

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()

