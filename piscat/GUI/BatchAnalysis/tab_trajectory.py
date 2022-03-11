from piscat.Trajectory import particle_linking, temporal_filtering
from piscat.Visualization import plot, plot_histogram

from PySide6 import QtGui, QtCore, QtWidgets


class Tracking_GUI(QtWidgets.QWidget):
    output_setting_Tab_tracking = QtCore.Signal(object)
    update_tab_index = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(Tracking_GUI, self).__init__(parent)

        self.his_all_particles = None
        self.memory = None
        self.search_range = None
        self.temporal_length = None

        self.setting_tracking = {}

        self.empty_value_box_flag = False
        self.empty_value_optional_box_flag_1 = False
        self.empty_value_optional_box_flag_2 = False

        self.resize(300, 300)
        self.setWindowTitle('PSFs Tracking')

        self.btn_filterLinking = QtWidgets.QPushButton("Next")
        self.btn_filterLinking.setFixedWidth(120)
        self.btn_filterLinking.setFixedHeight(30)
        self.btn_filterLinking.clicked.connect(self.do_update_tfilter)

        self.line_memory = QtWidgets.QLineEdit()
        self.line_memory.setPlaceholderText("Memory")
        self.line_memory_label = QtWidgets.QLabel("Memory (frame):")

        self.line_search_range = QtWidgets.QLineEdit()
        self.line_search_range.setPlaceholderText("Move")
        self.line_searchRange_label = QtWidgets.QLabel("Neighborhood size(px):")

        self.line_min_V_shape = QtWidgets.QLineEdit()
        self.line_min_V_shape.setPlaceholderText("Minimum_temporal_length (frame)")
        self.line_min_V_shape_label = QtWidgets.QLabel("Minimum_temporal_length (frame):")

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def createFirstExclusiveGroup(self):

        groupBox = QtWidgets.QGroupBox("Linking")
        self.checkbox_sorting_based_lenght = QtWidgets.QCheckBox("Sorting", self)

        self.grid_linking = QtWidgets.QGridLayout()

        self.grid_linking.addWidget(self.line_memory_label, 0, 0)
        self.grid_linking.addWidget(self.line_memory, 0, 1)

        self.grid_linking.addWidget(self.line_searchRange_label, 1, 0)
        self.grid_linking.addWidget(self.line_search_range, 1, 1)

        self.grid_linking.addWidget(self.line_min_V_shape_label, 2, 0)
        self.grid_linking.addWidget(self.line_min_V_shape, 2, 1)
        self.grid_linking.addWidget(self.btn_filterLinking, 3, 0)

        self.grid_linking.addWidget(self.checkbox_sorting_based_lenght, 3, 3)

        groupBox.setLayout(self.grid_linking)
        return groupBox

    def do_update_tfilter(self):
        self.get_values_1()
        self.get_values_2()
        if self.empty_value_box_flag:
            self.setting_tracking['Memory (frame)'] = self.memory
            self.setting_tracking['Neighborhood_size (px)'] = self.search_range
            self.setting_tracking['Minimum_temporal_length (frame)'] = self.temporal_length
            self.output_setting_Tab_tracking.emit(self.setting_tracking)
            self.update_tab_index.emit(4)


    def get_values_1(self):
        try:
            self.memory = int(self.line_memory.text())
            self.search_range = int(self.line_search_range.text())

            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def get_values_2(self):
        try:
            self.temporal_length = int(self.line_min_V_shape.text())

            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
