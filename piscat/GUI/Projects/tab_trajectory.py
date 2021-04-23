from piscat.Trajectory import particle_linking, temporal_filtering
from piscat.Visualization import plot, plot_histogram

from PySide2 import QtGui, QtCore, QtWidgets


class Tracking_GUI(QtWidgets.QWidget):
    update_tracking = QtCore.Signal(object)
    update_trajectories = QtCore.Signal(object)
    output_setting_Tab_tracking = QtCore.Signal(object)
    output_number_Particels_tracking = QtCore.Signal(object)

    def __init__(self, batch_size, parent=None):
        super(Tracking_GUI, self).__init__(parent)

        self.batch_size = batch_size

        self.input_video = None
        self.link_df_PSFS = None
        self.df_psfs = None
        self.his_all_particles = None
        self.memory = None
        self.search_range = None
        self.temporal_length = None
        self.df_PSFs_link = None
        self.pixel_size = 1
        self.axisScale = 'nm'

        self.setting_tracking = {}
        self.PSFs_Particels_num = {}

        self.empty_value_box_flag = False
        self.empty_value_optional_box_flag_1 = False
        self.empty_value_optional_box_flag_2 = False

        self.resize(300, 300)
        self.setWindowTitle('PSFs Tracking')

        self.btn_hist_plot = QtWidgets.QPushButton("Plot Len histogram")
        self.btn_hist_plot.setFixedWidth(150)
        self.btn_hist_plot.setFixedHeight(30)
        self.btn_hist_plot.clicked.connect(self.plot_len_hist)

        self.btn_linking = QtWidgets.QPushButton("Update linking")
        self.btn_linking.setFixedWidth(120)
        self.btn_linking.setFixedHeight(30)
        self.btn_linking.clicked.connect(self.do_update)

        self.btn_filterLinking = QtWidgets.QPushButton("Temporal filter")
        self.btn_filterLinking.setFixedWidth(120)
        self.btn_filterLinking.setFixedHeight(30)
        self.btn_filterLinking.clicked.connect(self.do_update_tfilter)

        self.btn_plot2Dlocalization = QtWidgets.QPushButton("Plot 2D localization")
        self.btn_plot2Dlocalization.setFixedWidth(120)
        self.btn_plot2Dlocalization.setFixedHeight(20)
        self.btn_plot2Dlocalization.clicked.connect(self.do_plot2D)

        self.line_memory = QtWidgets.QLineEdit()
        self.line_memory.setPlaceholderText("Memory")
        self.line_memory_label = QtWidgets.QLabel("Memory (#frames):")

        self.line_search_range = QtWidgets.QLineEdit()
        self.line_search_range.setPlaceholderText("Move")
        self.line_searchRange_label = QtWidgets.QLabel("Neighborhood size(px):")

        self.line_min_V_shape = QtWidgets.QLineEdit()
        self.line_min_V_shape.setPlaceholderText("V_shape_width")
        self.line_min_V_shape_label = QtWidgets.QLabel("V_shape_width (#frames):")

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)

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
        self.grid_linking.addWidget(self.btn_linking, 3, 0)
        self.grid_linking.addWidget(self.btn_hist_plot, 3, 1)
        self.grid_linking.addWidget(self.btn_filterLinking, 3, 2)

        self.grid_linking.addWidget(self.checkbox_sorting_based_lenght, 3, 3)

        groupBox.setLayout(self.grid_linking)
        return groupBox

    def createSecondExclusiveGroup(self):

        self.checkbox_display_label = QtWidgets.QCheckBox("Display label of particles", self)

        self.line_pixel_size = QtWidgets.QLineEdit()
        self.line_pixel_size.setPlaceholderText("Pixel Size")
        self.line_pixel_size_label = QtWidgets.QLabel("Pixel size:")

        self.line_axis_scale = QtWidgets.QLineEdit()
        self.line_axis_scale.setPlaceholderText("Scale 'nm' ")
        self.line_axis_scale_label = QtWidgets.QLabel("Axis scale")

        groupBox = QtWidgets.QGroupBox("Optional")

        self.grid1 = QtWidgets.QGridLayout()
        self.grid1.addWidget(self.checkbox_display_label, 0, 0)
        self.grid1.addWidget(self.line_pixel_size_label, 1, 0)
        self.grid1.addWidget(self.line_pixel_size, 1, 1)
        self.grid1.addWidget(self.line_axis_scale_label, 2, 0)
        self.grid1.addWidget(self.line_axis_scale, 2, 1)
        self.grid1.addWidget(self.btn_plot2Dlocalization, 3, 1)

        groupBox.setLayout(self.grid1)
        return groupBox

    @QtCore.Slot()
    def update_in_data(self, data_in):
        self.input_video = data_in[0]
        self.df_psfs = data_in[1]

    @QtCore.Slot()
    def update_batchSize(self, batchSize):
        self.batch_size = batchSize

    @QtCore.Slot()
    def do_update(self):

        if self.df_psfs is None:
            self.msg_box2 = QtWidgets.QMessageBox()
            self.msg_box2.setWindowTitle("Warning!")
            self.msg_box2.setText("Please localized PSFs!")
            self.msg_box2.exec_()

        else:

            self.get_values_1()
            if self.empty_value_box_flag:
                if self.df_psfs.shape[0] > 1:
                    linking_ = particle_linking.Linking()
                    df_PSFs_link = linking_.create_link(psf_position=self.df_psfs,
                                                        search_range=self.search_range,
                                                        memory=self.memory)
                    self.df_PSFs_link = df_PSFs_link

                    self.his_all_particles = df_PSFs_link['particle'].value_counts()
                    plot_histogram(self.his_all_particles, 'Lengths of linking')
                    self.PSFs_Particels_num['Total_number_Particles'] = linking_.trajectory_counter(df_PSFs_link)
                else:
                    self.msg_box3 = QtWidgets.QMessageBox()
                    self.msg_box3.setWindowTitle("Warning!")
                    self.msg_box3.setText("No PSFs found for linking!")
                    self.msg_box3.exec_()

    def plot_len_hist(self):
        if self.his_all_particles is not None:
            plot_histogram(self.his_all_particles, 'Lengths of linking')
        else:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please update linking!")
            self.msg_box3.exec_()

    def do_update_tfilter(self):
        if self.df_PSFs_link is None:
            self.msg_box2 = QtWidgets.QMessageBox()
            self.msg_box2.setWindowTitle("Warning!")
            self.msg_box2.setText("Please linking PSFs!")
            self.msg_box2.exec_()

        else:
            self.get_values_2()
            if self.empty_value_box_flag:
                if self.df_PSFs_link.shape[0] > 1:
                    t_filters = temporal_filtering.TemporalFilter(video=self.input_video,
                                                                  batchSize=self.batch_size)

                    all_trajectories, df_PSFs_t_filter, his_all_particles = t_filters.v_trajectory(df_PSFs=self.df_PSFs_link,
                                                                                                   threshold=self.temporal_length)

                    linking_ = particle_linking.Linking()
                    self.PSFs_Particels_num['#Particles_after_V_shapeFilter'] = linking_.trajectory_counter(df_PSFs_t_filter)

                    if self.checkbox_sorting_based_lenght.isChecked():
                        df_PSFs_t_filter = linking_.sorting_linking(df_PSFs=df_PSFs_t_filter)

                    self.link_df_PSFS = df_PSFs_t_filter

                    self.setting_tracking['search_range'] = self.search_range
                    self.setting_tracking['Memory'] = self.memory
                    self.setting_tracking['minimum_temporal_length'] = self.temporal_length

                    self.update_tracking.emit(self.link_df_PSFS)
                    self.update_trajectories.emit(all_trajectories)
                    self.output_setting_Tab_tracking.emit(self.setting_tracking)
                    self.output_number_Particels_tracking.emit(self.PSFs_Particels_num)
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("No PSFs found for temporal filtering!")
                self.msg_box3.exec_()

    def do_plot2D(self):
        try:
            self.pixel_size = int(self.line_pixel_size.text())
        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Pixel size set as default (1 nm)!")
            self.msg_box3.exec_()

            self.pixel_size = 1

        try:
            self.axisScale = str(self.line_axis_scale.text())
        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("axis scale set as default (nm)!")
            self.msg_box3.exec_()

            self.axisScale = '(nm)'

        if self.link_df_PSFS is None:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("No Particles found!")
            self.msg_box3.exec_()
        else:
            if self.checkbox_display_label.isChecked():
                label = True
            else:
                label = False

            plot.plot2df(self.link_df_PSFS, pixel_size=self.pixel_size, scale=self.axisScale, title='', flag_label=label)

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
