from PySide6 import QtCore, QtWidgets


class TemporalAnnotationSetting(QtWidgets.QWidget):
    temporalAnnotation_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(TemporalAnnotationSetting, self).__init__(parent)
        self.flag_smooth_filter = False
        self.window_size = 1

        # ok
        self.ok = QtWidgets.QPushButton("ok")
        self.ok.setAutoDefault(False)
        self.ok.clicked.connect(self.do_update)
        self.ok.setFixedWidth(100)

        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText("Temporal window (#frames)")
        self.le_1_label = QtWidgets.QLabel("Temporal window:")
        self.le1.setFixedWidth(100)

        self.marker_size = QtWidgets.QLineEdit()
        self.marker_size.setPlaceholderText("Marker size (default=5))")
        self.marker_size_label = QtWidgets.QLabel("Marker size:")
        self.marker_size.setFixedWidth(100)

        self.checkbox_smooth_filter = QtWidgets.QCheckBox("Smoothness filter", self)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.le_1_label, 0, 0)
        self.grid.addWidget(self.le1, 0, 1)
        self.grid.addWidget(self.marker_size_label, 1, 0)
        self.grid.addWidget(self.marker_size, 1, 1)
        self.grid.addWidget(self.checkbox_smooth_filter, 2, 0)
        self.grid.addWidget(self.ok, 3, 1)
        self.setLayout(self.grid)

        self.setWindowTitle("Temporal Annotation panel")
        self.setLayout(self.grid)

    @QtCore.Slot()
    def do_update(self):
        try:
            self.window_size = int(self.le1.text())
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText(
                "Incorrect input_video for step size! \nThe default value will be selected"
            )
            self.msg_box.exec_()
            self.window_size = 1

        try:
            self.marker_size = int(self.marker_size.text())
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText(
                "Incorrect input_video for step size! \nThe default value will be selected"
            )
            self.msg_box.exec_()
            self.marker_size = 5

        if self.checkbox_smooth_filter.isChecked():
            self.flag_smooth_filter = True
        else:
            self.flag_smooth_filter = False

        self.temporalAnnotation_signal.emit(
            [self.window_size, self.flag_smooth_filter, self.marker_size]
        )
        self.close()
