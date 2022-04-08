from piscat.Anomaly.hand_crafted_feature_genration import CreateFeatures
from piscat.Preproccessing import normalization, filtering
from piscat.Anomaly.spatio_temporal_anomaly import SpatioTemporalAnomalyDetection

from PySide6 import QtGui, QtCore, QtWidgets


class HandCraftedFeatureMatrix_GUI(QtWidgets.QWidget):
    output_Tab_anomaly = QtCore.Signal(object)
    output_setting_Tab_anomaly = QtCore.Signal(object)
    update_tab_index = QtCore.Signal(int)

    def __init__(self, input_video_raw=None, input_video_BGc=None, batch_size=None, parent=None):
        super(HandCraftedFeatureMatrix_GUI, self).__init__(parent)

        self.video_in_raw = input_video_raw
        self.video_in_BGc = input_video_BGc

        self.batch_size = batch_size
        self.features_list = None
        self.setting_anomaly = {}

        self.resize(300, 300)
        self.setWindowTitle('Hand crafted feature matrix')

        self.le_downsampling = QtWidgets.QLineEdit()
        self.le_downsampling.setPlaceholderText("downsampling (default values is 1)")
        self.le_downsampling_label = QtWidgets.QLabel("Downsampling:")

        self.le_contamination = QtWidgets.QLineEdit()
        self.le_contamination.setPlaceholderText("contamination (default values is 0.003)")
        self.le_contamination_label = QtWidgets.QLabel("Contamination:")

        self._hotpixel = QtWidgets.QCheckBox("Hot pixel correction", self)

        self.temporal_filter_m1 = QtWidgets.QCheckBox("Mean batch 1 (M1)", self)
        self.temporal_filter_m2 = QtWidgets.QCheckBox("Mean batch 2 (M2)", self)
        self.temporal_filter_m1_2 = QtWidgets.QCheckBox("abs(M1-M2)", self)
        self.temporal_filter_m1_m12 = QtWidgets.QCheckBox("M1-offset", self)
        self.temporal_filter_m2_m12 = QtWidgets.QCheckBox("M2-offset", self)
        self.temporal_filter_s1 = QtWidgets.QCheckBox("Standard deviation batch 1 (S1)", self)
        self.temporal_filter_s2 = QtWidgets.QCheckBox("Standard deviation batch 2 (S2)", self)

        self.spatial_filter_input = QtWidgets.QCheckBox("input", self)

        self.btn_apply_anomaly = QtWidgets.QPushButton('Apply iForest', self)
        self.btn_apply_anomaly.clicked.connect(self.apply_anomaly)
        self.btn_apply_anomaly.setIconSize(QtCore.QSize(24, 24))
        self.btn_apply_anomaly.setFixedWidth(100)

        self.btn_features = QtWidgets.QPushButton('Create features', self)
        self.btn_features.clicked.connect(self.create_feature_matrix)
        self.btn_features.setIconSize(QtCore.QSize(24, 24))
        self.btn_features.setFixedWidth(150)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.btn_features, 2, 0)
        self.grid.addWidget(self.createThirdExclusiveGroup(), 3, 0)

        self.setLayout(self.grid)

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def createFirstExclusiveGroup(self):

        self.groupBox_tm_features = QtWidgets.QGroupBox("Temporal Features")

        self.grid_temporal = QtWidgets.QGridLayout()

        self.grid_temporal.addWidget(self.temporal_filter_m1, 0, 0)
        self.grid_temporal.addWidget(self.temporal_filter_m2, 0, 1)
        self.grid_temporal.addWidget(self.temporal_filter_s1, 0, 2)
        self.grid_temporal.addWidget(self.temporal_filter_s2, 1, 0)
        self.grid_temporal.addWidget(self.temporal_filter_m1_2, 1, 1)
        self.grid_temporal.addWidget(self.temporal_filter_m1_m12, 1, 2)
        self.grid_temporal.addWidget(self.temporal_filter_m2_m12, 2, 0)
        self.grid_temporal.addWidget(self._hotpixel, 2, 1)

        self.groupBox_tm_features.setLayout(self.grid_temporal)
        return self.groupBox_tm_features

    def createSecondExclusiveGroup(self):

        self.groupBox_sp_features = QtWidgets.QGroupBox("Temporal Features")

        self.grid_dog = QtWidgets.QGridLayout()
        self.groupBox_sp_dog = QtWidgets.QGroupBox("DoG transform")
        self.groupBox_sp_dog.setCheckable(True)
        self.groupBox_sp_dog.setChecked(False)

        self.create_dog_como_values()
        self.groupBox_sp_dog.setLayout(self.grid_dog)

        self.grid_rvt = QtWidgets.QGridLayout()
        self.groupBox_sp_rvt = QtWidgets.QGroupBox("RVT transform")
        self.groupBox_sp_rvt.setCheckable(True)
        self.groupBox_sp_rvt.setChecked(False)

        self.create_rvt_como_values()
        self.groupBox_sp_rvt.setLayout(self.grid_rvt)

        self.grid_spatial = QtWidgets.QGridLayout()
        self.grid_spatial.addWidget(self.groupBox_sp_dog, 0, 0)
        self.grid_spatial.addWidget(self.groupBox_sp_rvt, 1, 0)
        self.grid_spatial.addWidget(self.spatial_filter_input, 2, 0)

        self.groupBox_sp_features.setLayout(self.grid_spatial)
        return self.groupBox_sp_features

    def createThirdExclusiveGroup(self):

        self.groupBox_anomaly = QtWidgets.QGroupBox("Apply anomaly")

        self.grid_anomaly = QtWidgets.QGridLayout()

        self.grid_anomaly.addWidget(self.le_downsampling_label, 0, 0)
        self.grid_anomaly.addWidget(self.le_downsampling, 0, 1)
        self.grid_anomaly.addWidget(self.le_contamination_label, 1, 0)
        self.grid_anomaly.addWidget(self.le_contamination, 1, 1)
        self.grid_anomaly.addWidget(self.btn_apply_anomaly, 2, 0)

        self.groupBox_anomaly.setLayout(self.grid_anomaly)
        return self.groupBox_anomaly

    def create_dog_como_values(self):
        self.le_1_dog = QtWidgets.QLineEdit()
        self.le_1_dog.setPlaceholderText("Min Sigma")
        self.le_1_label_dog = QtWidgets.QLabel("Min Sigma (px): ")

        self.le_2_dog = QtWidgets.QLineEdit()
        self.le_2_dog.setPlaceholderText("Max Sigma")
        self.le_2_label_dog = QtWidgets.QLabel("Max Sigma (px): ")

        self.grid_dog.addWidget(self.le_1_label_dog, 1, 0)
        self.grid_dog.addWidget(self.le_2_label_dog, 2, 0)

        self.grid_dog.addWidget(self.le_1_dog, 1, 1)
        self.grid_dog.addWidget(self.le_2_dog, 2, 1)

    def create_rvt_como_values(self):
        self.le_1_rvt = QtWidgets.QLineEdit()
        self.le_1_rvt.setPlaceholderText("Min radius")
        self.le_1_label_rvt = QtWidgets.QLabel("Min radius (px): ")

        self.le_2_rvt = QtWidgets.QLineEdit()
        self.le_2_rvt.setPlaceholderText("Max radius")
        self.le_2_label_rvt = QtWidgets.QLabel("Max radius (px): ")

        self.le_3_rvt = QtWidgets.QLineEdit()
        self.le_3_rvt.setPlaceholderText("highpass_size")
        self.le_3_label_rvt = QtWidgets.QLabel("Highpass_size: ")

        self.le_4_rvt = QtWidgets.QLineEdit()
        self.le_4_rvt.setPlaceholderText("upsample")
        self.le_4_label_rvt = QtWidgets.QLabel("Upsample: ")

        self.le_5_rvt = QtWidgets.QLineEdit()
        self.le_5_rvt.setPlaceholderText("rweights")
        self.le_5_label_rvt = QtWidgets.QLabel("rweights: ")

        self.le_6_rvt = QtWidgets.QLineEdit()
        self.le_6_rvt.setPlaceholderText("coarse_factor")
        self.le_6_label_rvt = QtWidgets.QLabel("Coarse_factor: ")

        self.le_7_rvt = QtWidgets.QLineEdit()
        self.le_7_rvt.setPlaceholderText("Threshold")
        self.le_7_label_rvt = QtWidgets.QLabel("Threshold: ")

        self.pad_mode_group = QtWidgets.QButtonGroup()
        self.radio_pad_mode_constant = QtWidgets.QRadioButton("Pad mode (constant)")
        self.radio_pad_mode_reflect = QtWidgets.QRadioButton("Pad mode (reflect)")
        self.radio_pad_mode_edge = QtWidgets.QRadioButton("Pad mode (edge)")
        self.radio_pad_mode_fast = QtWidgets.QRadioButton("Pad mode (fast)")

        self.pad_mode_group.addButton(self.radio_pad_mode_constant)
        self.pad_mode_group.addButton(self.radio_pad_mode_reflect)
        self.pad_mode_group.addButton(self.radio_pad_mode_edge)
        self.pad_mode_group.addButton(self.radio_pad_mode_fast)
        self.radio_pad_mode_constant.setChecked(True)

        self.coarse_mode_group = QtWidgets.QButtonGroup()
        self.radio_coarse_mode_add = QtWidgets.QRadioButton("Coarse mode (add)")
        self.radio_coarse_mode_skip = QtWidgets.QRadioButton("Coarse mode (skip)")

        self.coarse_mode_group.addButton(self.radio_coarse_mode_add)
        self.coarse_mode_group.addButton(self.radio_coarse_mode_skip)
        self.radio_coarse_mode_add.setChecked(True)

        self.kind_group = QtWidgets.QButtonGroup()
        self.radio_kind_basic = QtWidgets.QRadioButton("Kind (basic)")
        self.radio_kind_normalized = QtWidgets.QRadioButton("Kind (normalized)")

        self.kind_group.addButton(self.radio_kind_basic)
        self.kind_group.addButton(self.radio_kind_normalized)
        self.radio_kind_basic.setChecked(True)

        self.grid_rvt.addWidget(self.le_1_label_rvt, 1, 0)
        self.grid_rvt.addWidget(self.le_2_label_rvt, 2, 0)
        self.grid_rvt.addWidget(self.le_3_label_rvt, 3, 0)
        self.grid_rvt.addWidget(self.le_4_label_rvt, 1, 2)
        self.grid_rvt.addWidget(self.le_5_label_rvt, 2, 2)
        self.grid_rvt.addWidget(self.le_6_label_rvt, 3, 2)
        self.grid_rvt.addWidget(self.le_7_label_rvt, 5, 2)

        self.grid_rvt.addWidget(self.le_1_rvt, 1, 1)
        self.grid_rvt.addWidget(self.le_2_rvt, 2, 1)
        self.grid_rvt.addWidget(self.le_3_rvt, 3, 1)
        self.grid_rvt.addWidget(self.le_4_rvt, 1, 3)
        self.grid_rvt.addWidget(self.le_5_rvt, 2, 3)
        self.grid_rvt.addWidget(self.le_6_rvt, 3, 3)
        self.grid_rvt.addWidget(self.le_7_rvt, 5, 3)

        self.grid_rvt.addWidget(self.radio_pad_mode_constant, 4, 0)
        self.grid_rvt.addWidget(self.radio_pad_mode_reflect, 4, 1)
        self.grid_rvt.addWidget(self.radio_pad_mode_edge, 4, 2)
        self.grid_rvt.addWidget(self.radio_pad_mode_fast, 4, 3)

        self.grid_rvt.addWidget(self.radio_coarse_mode_add, 5, 0)
        self.grid_rvt.addWidget(self.radio_coarse_mode_skip, 5, 1)

        self.grid_rvt.addWidget(self.radio_kind_basic, 6, 0)
        self.grid_rvt.addWidget(self.radio_kind_normalized, 6, 1)

    def get_localization_method_parameters(self):
        try:
            if self.groupBox_sp_dog.isChecked():
                self.min_sigma = eval(self.le_1_dog.text())
                self.max_sigma = eval(self.le_2_dog.text())

            if self.groupBox_sp_rvt.isChecked():
                self.min_radial = eval(self.le_1_rvt.text())
                self.max_radial = eval(self.le_2_rvt.text())

                if self.radio_kind_basic.isChecked():
                    self.rvt_kind = 'basic'
                elif self.radio_kind_normalized.isChecked():
                    self.rvt_kind = 'normalized'

                self.highpass_size = self.le_3_rvt.text()
                if self.highpass_size == '':
                    self.highpass_size = None
                else:
                    self.highpass_size = float(self.highpass_size)

                self.upsample = self.le_4_rvt.text()
                if self.upsample == '':
                    self.upsample = 1
                else:
                    self.upsample = int(self.upsample)

                self.rweights = self.le_5_rvt.text()
                if self.rweights == '':
                    self.rweights = None
                else:
                    self.rweights = eval(self.rweights)

                self.coarse_factor = self.le_6_rvt.text()
                if self.coarse_factor == '':
                    self.coarse_factor = 1
                else:
                    self.coarse_factor = float(self.coarse_factor)

                if self.radio_coarse_mode_add.isChecked():
                    self.coarse_mode = 'add'
                elif self.radio_coarse_mode_skip.isChecked():
                    self.coarse_mode = 'skip'

                if self.radio_pad_mode_constant.isChecked():
                    self.pad_mode = "constant"
                elif self.radio_pad_mode_reflect.isChecked():
                    self.pad_mode = "reflect"
                elif self.radio_pad_mode_edge.isChecked():
                    self.pad_mode = "edge"
                elif self.radio_pad_mode_fast.isChecked():
                    self.pad_mode = "fast"

                if self.le_7_rvt.text() == '':
                    self.threshold = 0
                else:
                    self.threshold = float(self.le_7_rvt.text())

            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def get_anomaly_parameters(self):
        try:
            self.scale = int(self.le_downsampling.text())
            self.contamination = float(self.le_contamination.text())
            self.empty_value_box_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all anomaly parameters!")
            self.msg_box3.exec_()

            self.empty_value_box_flag = False

    def create_feature_matrix(self):
        features_list = []
        if self._hotpixel.isChecked():
            if self.video_in_raw is not None and self.video_in_BGc is not None:
                self.video_in_raw = filtering.Filters(self.video_in_raw).median(3)
                self.video_in_BGc = filtering.Filters(self.video_in_BGc).median(3)
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("videos can not find!")
                self.msg_box3.exec_()

        if self.temporal_filter_m1.isChecked() or self.temporal_filter_m2.isChecked() or \
            self.temporal_filter_s1.isChecked() or self.temporal_filter_s2.isChecked() or \
            self.temporal_filter_m1_2:

            if self.video_in_raw is not None and self.batch_size is not None:
                feature_maps_temporal = CreateFeatures(video=self.video_in_raw)
                out_feature_t1 = feature_maps_temporal.temporal_features(batchSize=self.batch_size, flag_dc=False)
                self.setting_anomaly['temporal_bach_size'] = self.batch_size

                if self.temporal_filter_m1.isChecked():
                    features_list.append(out_feature_t1[0])
                    self.setting_anomaly['flag_m1'] = True

                if self.temporal_filter_m2.isChecked():
                    features_list.append(out_feature_t1[1])
                    self.setting_anomaly['flag_m2'] = True

                if self.temporal_filter_s1.isChecked():
                    features_list.append(out_feature_t1[2])
                    self.setting_anomaly['flag_s1'] = True

                if self.temporal_filter_s2.isChecked():
                    features_list.append(out_feature_t1[3])
                    self.setting_anomaly['flag_s2'] = True

                if self.temporal_filter_m1_2.isChecked():
                    features_list.append(out_feature_t1[4])
                    self.setting_anomaly['flag_diff'] = True
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("videos/(batch size) can not find!")
                self.msg_box3.exec_()

        if self.temporal_filter_m1_m12.isChecked() or self.temporal_filter_m2_m12.isChecked():
            if self.video_in_raw is not None and self.batch_size is not None:
                feature_maps_temporal = CreateFeatures(video=self.video_in_raw)
                out_feature_t2 = feature_maps_temporal.temporal_features(batchSize=self.batch_size, flag_dc=True)

                if self.temporal_filter_m1_m12.isChecked():
                    features_list.append(out_feature_t2[0])
                    self.setting_anomaly['flag_M1_m12'] = True

                if self.temporal_filter_m2_m12.isChecked():
                    features_list.append(out_feature_t2[1])
                    self.setting_anomaly['flag_M2_m12'] = True
                    self.msg_box3 = QtWidgets.QMessageBox()
                    self.msg_box3.setWindowTitle("Warning!")
                    self.msg_box3.setText("videos/(batch size) can not find!")
                    self.msg_box3.exec_()

        if self.groupBox_sp_rvt.isChecked():
            self.empty_value_box_flag = False
            self.get_localization_method_parameters()

            if self.empty_value_box_flag and self.video_in_BGc is not None:
                rvt_ = filtering.RadialVarianceTransform(inter_flag_parallel_active=True)
                filtered_video = rvt_.rvt_video(video=self.video_in_BGc, rmin=self.min_radial,
                                                rmax=self.max_radial,
                                                kind=self.rvt_kind,
                                                highpass_size=self.highpass_size,
                                                upsample=self.upsample,
                                                rweights=self.rweights,
                                                coarse_factor=self.coarse_factor,
                                                coarse_mode=self.coarse_mode,
                                                pad_mode=self.pad_mode)

                features_list.append(filtered_video)

                self.setting_anomaly['rvt_rmin_anomaly'] = self.min_radial
                self.setting_anomaly['rvt_rmax_anomaly'] = self.max_radial
                self.setting_anomaly['rvt_kind_anomaly'] = self.rvt_kind
                self.setting_anomaly['rvt_upsample_anomaly'] = self.upsample
                self.setting_anomaly['rvt_rweights_anomaly'] = self.rweights
                self.setting_anomaly['rvt_coarse_factor_anomaly'] = self.coarse_factor
                self.setting_anomaly['rvt_coarse_mode_anomaly'] = self.coarse_mode
                self.setting_anomaly['rvt_pad_mode_anomaly'] = self.pad_mode
                self.setting_anomaly['flag_rvt'] = True
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("video BGc can not find!")
                self.msg_box3.exec_()

        if self.groupBox_sp_dog.isChecked():
            self.empty_value_box_flag = False
            self.get_localization_method_parameters()
            if self.empty_value_box_flag and self.video_in_BGc is not None:
                feature_maps_spatio = CreateFeatures(video=self.video_in_BGc)
                dog_features = feature_maps_spatio.dog2D_creater(low_sigma=[self.min_sigma, self.min_sigma],
                    high_sigma=[self.max_sigma, self.max_sigma])

                features_list.append(dog_features)

                self.setting_anomaly['dog_min_sigma_anomaly'] = self.min_sigma
                self.setting_anomaly['dog_max_sigma_anomaly'] = self.max_sigma
                self.setting_anomaly['flag_dog'] = True
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("video BGc can not find!")
                self.msg_box3.exec_()

        if self.spatial_filter_input.isChecked():
            if self.video_in_BGc is not None:
                features_list.append(self.video_in_BGc)
                self.setting_anomaly['flag_input'] = True
            else:
                self.msg_box3 = QtWidgets.QMessageBox()
                self.msg_box3.setWindowTitle("Warning!")
                self.msg_box3.setText("video BGc can not find!")
                self.msg_box3.exec_()

        self.features_list = features_list

    def apply_anomaly(self):
        if self.features_list is not None:
            self.empty_value_box_flag = False
            self.get_anomaly_parameters()
            if self.empty_value_box_flag:
                anomaly_st = SpatioTemporalAnomalyDetection(self.features_list)
                binary_st, _ = anomaly_st.fun_anomaly(scale=self.scale,
                                                      method='IsolationForest',
                                                      contamination=self.contamination)
                if binary_st.shape != self.video_in_BGc:
                    dim_x = min(binary_st.shape[1], self.video_in_BGc.shape[1])
                    dim_y = min(binary_st.shape[2], self.video_in_BGc.shape[2])
                    result_anomaly = binary_st[:, 0:dim_x, 0:dim_y]
                else:
                    result_anomaly = binary_st

                result_anomaly_ = result_anomaly.copy()
                result_anomaly_[result_anomaly == True] = 1
                result_anomaly_[result_anomaly == False] = 0
                result_anomaly_ = normalization.Normalization(video=result_anomaly_.astype(int)).normalized_image_specific()

                self.setting_anomaly['anomaly_scale'] = self.scale
                self.setting_anomaly['anomaly_method'] = 'IsolationForest'
                self.setting_anomaly['anomaly_contamination'] = self.contamination

                self.output_Tab_anomaly.emit(result_anomaly_)
                self.output_setting_Tab_anomaly.emit(self.setting_anomaly)

        else:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("The feature matrix does not exist!")
            self.msg_box3.exec_()

    @QtCore.Slot()
    def update_batchSize(self, batchSize):
        self.batch_size = batchSize

    @QtCore.Slot()
    def update_in_data(self, data_in):
        self.video_in_BGc = data_in

    @QtCore.Slot(int)
    def get_sliceNumber(self, frame_number):
        self.frame_num = frame_number

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
