import math

import matplotlib.pylab as plt
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from piscat.GUI.Visualization.updating_plots import UpdatingPlots_Image
from piscat.iPSF_model import ImagingSetupParameters
from piscat.iPSF_model.ScatteredFieldDifferentialPhase import ScatteredFieldDifferentialPhase


class GUI_iPSF(QtWidgets.QWidget):
    display_trigger = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(GUI_iPSF, self).__init__(parent)

        self.iPSFs_focalStack = None
        self.iPSFs_AxialStack = None

        self.empty_value_ImagingBox_flag = True
        self.empty_value_ModelBox_flag = True

        self.wavelength = QtWidgets.QLineEdit()
        self.wavelength.setToolTip("Wavelength of the light source in meters")
        self.wavelength.setPlaceholderText("Wavelength of the light source in meters")
        self.wavelength_label = QtWidgets.QLabel("wavelength:")
        self.wavelength.setFixedWidth(150)

        self.NA = QtWidgets.QLineEdit()
        self.NA.setToolTip("Numerical Aperture (NA) of the objective lens")
        self.NA.setPlaceholderText("Numerical Aperture (NA) of the objective lens")
        self.NA_label = QtWidgets.QLabel("Numerical Aperture (NA):")
        self.NA.setFixedWidth(150)

        self.ti0 = QtWidgets.QLineEdit()
        self.ti0.setToolTip("Thickness of the immersion oil, nominal value in meters")
        self.ti0.setPlaceholderText("Thickness of the immersion oil, nominal value in meters")
        self.ti0_label = QtWidgets.QLabel("Thickness:")
        self.ti0.setFixedWidth(150)

        self.ni0 = QtWidgets.QLineEdit()
        self.ni0.setToolTip("Refractive index of the immersion oil, nominal value")
        self.ni0.setPlaceholderText("Refractive index of the immersion oil, nominal value")
        self.ni0_label = QtWidgets.QLabel("Refractive index:")
        self.ni0.setFixedWidth(150)

        self.ni = QtWidgets.QLineEdit()
        self.ni.setToolTip("Refractive index of the immersion oil, experimental value")
        self.ni.setPlaceholderText("Refractive index of the immersion oil, experimental value")
        self.ni_label = QtWidgets.QLabel("Refractive index:")
        self.ni.setFixedWidth(150)

        self.tg0 = QtWidgets.QLineEdit()
        self.tg0.setToolTip("Thickness of the coverglass, nominal value in meters")
        self.tg0.setPlaceholderText("Thickness of the coverglass, nominal value in meters")
        self.tg0_label = QtWidgets.QLabel("Thickness:")
        self.tg0.setFixedWidth(150)

        self.tg = QtWidgets.QLineEdit()
        self.tg.setToolTip("Thickness of the coverglass, experimental value in meters")
        self.tg.setPlaceholderText("Thickness of the coverglass, experimental value in meters")
        self.tg_label = QtWidgets.QLabel("Thickness:")
        self.tg.setFixedWidth(150)

        self.ng0 = QtWidgets.QLineEdit()
        self.ng0.setToolTip("Refractive index of the coverglass, nominal value")
        self.ng0.setPlaceholderText("Refractive index of the coverglass, nominal value")
        self.ng0_label = QtWidgets.QLabel("Refractive index:")
        self.ng0.setFixedWidth(150)

        self.ng = QtWidgets.QLineEdit()
        self.ng.setToolTip("Refractive index of the coverglass, experimental value")
        self.ng.setPlaceholderText("Refractive index of the coverglass, experimental value")
        self.ng_label = QtWidgets.QLabel("Refractive index:")
        self.ng.setFixedWidth(150)

        self.ns = QtWidgets.QLineEdit()
        self.ns.setToolTip("Refractive index of the sample/medium")
        self.ns.setPlaceholderText("Refractive index of the sample/medium")
        self.ns_label = QtWidgets.QLabel("Refractive index:")
        self.ns.setFixedWidth(150)

        self.pixel_size_physical = QtWidgets.QLineEdit()
        self.pixel_size_physical.setToolTip("Physical size of the camera pixel in meters")
        self.pixel_size_physical.setPlaceholderText("Physical size of the camera pixel in meters")
        self.pixel_size_physical_label = QtWidgets.QLabel("Pixel size:")
        self.pixel_size_physical.setFixedWidth(150)

        self.pixel_size = QtWidgets.QLineEdit()
        self.pixel_size.setToolTip(
            "Imaging pixel size in meters, related to the physical pixel size through the magnification of the setup"
        )
        self.pixel_size.setPlaceholderText(
            "Imaging pixel size in meters, related to the physical pixel size through the magnification of the setup"
        )
        self.pixel_size_label = QtWidgets.QLabel("Pixel size:")
        self.pixel_size.setFixedWidth(150)

        self.s_range = QtWidgets.QLineEdit()
        self.s_range.setToolTip("The starting range across which the focus is swept")
        self.s_range.setPlaceholderText("focus start point in meter")
        self.s_range_label = QtWidgets.QLabel("focus is swept start point:")
        self.s_range.setFixedWidth(150)

        self.e_range = QtWidgets.QLineEdit()
        self.e_range.setToolTip("The ending range across which the focus is swept")
        self.e_range.setPlaceholderText("focus end point in meter")
        self.e_range_label = QtWidgets.QLabel("focus is swept end point:")
        self.e_range.setFixedWidth(150)

        self.step_range = QtWidgets.QLineEdit()
        self.step_range.setToolTip("The stride across which the focus is swept")
        self.step_range.setPlaceholderText("step in meter")
        self.step_range_label = QtWidgets.QLabel("focus is swept step:")
        self.step_range.setFixedWidth(150)

        self.s_range_particle = QtWidgets.QLineEdit()
        self.s_range_particle.setToolTip(
            "The starting axial range across which the particle is travelling"
        )
        self.s_range_particle.setPlaceholderText("particle start point in meter")
        self.s_range_label_particle = QtWidgets.QLabel("particle is swept start point:")
        self.s_range_particle.setFixedWidth(150)

        self.e_range_particle = QtWidgets.QLineEdit()
        self.e_range_particle.setToolTip(
            "The ending axial range across which the particle is travelling"
        )
        self.e_range_particle.setPlaceholderText("particle end point in meter")
        self.e_range_label_particle = QtWidgets.QLabel("particle is swept end point:")
        self.e_range_particle.setFixedWidth(150)

        self.step_range_particle = QtWidgets.QLineEdit()
        self.step_range_particle.setToolTip(
            "The stride of axial across which the particle is travelling"
        )
        self.step_range_particle.setPlaceholderText("step in meter")
        self.step_range_label_particle = QtWidgets.QLabel("particle is swept step:")
        self.step_range_particle.setFixedWidth(150)

        self.z_focus_particle = QtWidgets.QLineEdit()
        self.z_focus_particle.setToolTip("The position of the focal plane")
        self.z_focus_particle.setPlaceholderText("The position of the focal plane")
        self.z_focus_label_particle = QtWidgets.QLabel("The position of the focal plane:")
        self.z_focus_particle.setFixedWidth(150)

        self.position_particle3D = QtWidgets.QLineEdit()
        self.position_particle3D.setToolTip("The 3D position of the nanoparticle")
        self.position_particle3D.setPlaceholderText("[x, y, z]")
        self.position_particle3D_label = QtWidgets.QLabel("The 3D position of the nanoparticle:")
        self.position_particle3D.setFixedWidth(150)

        self.nx = QtWidgets.QLineEdit()
        self.nx.setToolTip("Number of lateral pixels over which the image is calculated")
        self.nx.setPlaceholderText("pixels")
        self.nx_label = QtWidgets.QLabel(
            "Number of lateral pixels over which the image is calculated :"
        )
        self.nx.setFixedWidth(150)

        self.nx_particle = QtWidgets.QLineEdit()
        self.nx_particle.setToolTip("Number of lateral pixels over which the image is calculated")
        self.nx_particle.setPlaceholderText("pixels")
        self.nx_label_particle = QtWidgets.QLabel(
            "Number of lateral pixels over which the image is calculated :"
        )
        self.nx_particle.setFixedWidth(150)

        self.r_ = QtWidgets.QLineEdit()
        self.r_.setToolTip("Number of lateral pixels over which the calculated image is cropped")
        self.r_.setPlaceholderText("pixels")
        self.r_label = QtWidgets.QLabel(
            "Number of lateral pixels over which the calculated image is cropped:"
        )
        self.r_.setFixedWidth(150)

        self.r_particle = QtWidgets.QLineEdit()
        self.r_particle.setToolTip(
            "Number of lateral pixels over which the calculated image is cropped"
        )
        self.r_particle.setPlaceholderText("pixels")
        self.r_label_particle = QtWidgets.QLabel(
            "Number of lateral pixels over which the calculated image is cropped:"
        )
        self.r_particle.setFixedWidth(150)

        self.generate_model = QtWidgets.QPushButton("Generate model")
        self.generate_model.clicked.connect(self.do_model)
        self.generate_model.setFixedWidth(150)

        self.display = QtWidgets.QPushButton("Display")
        self.display.clicked.connect(self.do_display)
        self.display.setFixedWidth(150)

        self.Initialization = QtWidgets.QPushButton("Initialization")
        self.Initialization.clicked.connect(self.do_Initialization)
        self.Initialization.setFixedWidth(150)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        self.grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        self.grid.addWidget(self.createThirdExclusiveGroup(), 2, 0)
        # self.grid.addWidget(self.createForthExclusiveGroup(), 3, 0)
        self.grid.addWidget(self.createFifthExclusiveGroup(), 4, 0)

        self.setLayout(self.grid)
        self.do_Initialization()

    def __del__(self):
        print("Destructor called, Employee deleted.")

    def createFirstExclusiveGroup(self):
        self.groupBox_Parameters = QtWidgets.QGroupBox("Imaging setup parameters")

        self.grid1 = QtWidgets.QGridLayout()

        self.grid1.addWidget(self.wavelength_label, 0, 0)
        self.grid1.addWidget(self.wavelength, 1, 0)
        self.grid1.addWidget(self.NA_label, 0, 1)
        self.grid1.addWidget(self.NA, 1, 1)
        self.grid1.addWidget(self.ti0_label, 2, 0)
        self.grid1.addWidget(self.ti0, 3, 0)
        self.grid1.addWidget(self.ni0_label, 2, 1)
        self.grid1.addWidget(self.ni0, 3, 1)
        self.grid1.addWidget(self.ni_label, 4, 0)
        self.grid1.addWidget(self.ni, 5, 0)

        self.grid1.addWidget(self.tg0_label, 4, 1)
        self.grid1.addWidget(self.tg0, 5, 1)
        self.grid1.addWidget(self.tg_label, 0, 2)
        self.grid1.addWidget(self.tg, 1, 2)

        self.grid1.addWidget(self.ng0_label, 2, 2)
        self.grid1.addWidget(self.ng0, 3, 2)
        self.grid1.addWidget(self.ng_label, 4, 2)
        self.grid1.addWidget(self.ng, 5, 2)
        self.grid1.addWidget(self.ns_label, 0, 3)
        self.grid1.addWidget(self.ns, 1, 3)
        self.grid1.addWidget(self.pixel_size_physical_label, 2, 3)
        self.grid1.addWidget(self.pixel_size_physical, 3, 3)
        self.grid1.addWidget(self.pixel_size_label, 4, 3)
        self.grid1.addWidget(self.pixel_size, 5, 3)

        self.groupBox_Parameters.setLayout(self.grid1)
        return self.groupBox_Parameters

    def createSecondExclusiveGroup(self):
        self.groupBox_iPSF_model_1 = QtWidgets.QGroupBox("Modelling iPSF images (I) ")
        self.groupBox_iPSF_model_1.setCheckable(True)
        self.groupBox_iPSF_model_1.setChecked(True)
        self.groupBox_iPSF_model_1.toggled.connect(
            lambda: self.groupBox_iPSF_model_2.setChecked(False)
        )

        self.grid2 = QtWidgets.QGridLayout()
        self.grid2.addWidget(self.s_range_label, 0, 0)
        self.grid2.addWidget(self.s_range, 1, 0)
        self.grid2.addWidget(self.e_range_label, 2, 0)
        self.grid2.addWidget(self.e_range, 3, 0)
        self.grid2.addWidget(self.step_range_label, 4, 0)
        self.grid2.addWidget(self.step_range, 5, 0)
        self.grid2.addWidget(self.position_particle3D_label, 0, 1)
        self.grid2.addWidget(self.position_particle3D, 1, 1)
        self.grid2.addWidget(self.nx_label, 2, 1)
        self.grid2.addWidget(self.nx, 3, 1)
        self.grid2.addWidget(self.r_label, 4, 1)
        self.grid2.addWidget(self.r_, 5, 1)

        self.groupBox_iPSF_model_1.setLayout(self.grid2)
        return self.groupBox_iPSF_model_1

    def createThirdExclusiveGroup(self):
        self.groupBox_iPSF_model_2 = QtWidgets.QGroupBox("Modelling iPSF images (II)")
        self.groupBox_iPSF_model_2.setCheckable(True)
        self.groupBox_iPSF_model_2.setChecked(False)
        self.groupBox_iPSF_model_2.toggled.connect(
            lambda: self.groupBox_iPSF_model_1.setChecked(False)
        )

        self.grid3 = QtWidgets.QGridLayout()
        self.grid3.addWidget(self.s_range_label_particle, 0, 0)
        self.grid3.addWidget(self.s_range_particle, 1, 0)
        self.grid3.addWidget(self.e_range_label_particle, 2, 0)
        self.grid3.addWidget(self.e_range_particle, 3, 0)
        self.grid3.addWidget(self.step_range_label_particle, 4, 0)
        self.grid3.addWidget(self.step_range_particle, 5, 0)
        self.grid3.addWidget(self.nx_label_particle, 0, 1)
        self.grid3.addWidget(self.nx_particle, 1, 1)
        self.grid3.addWidget(self.r_label_particle, 2, 1)
        self.grid3.addWidget(self.r_particle, 3, 1)
        self.grid3.addWidget(self.z_focus_label_particle, 4, 1)
        self.grid3.addWidget(self.z_focus_particle, 5, 1)

        self.groupBox_iPSF_model_2.setLayout(self.grid3)
        return self.groupBox_iPSF_model_2

    # def createForthExclusiveGroup(self):
    #     self.groupBox_Parameters = QtWidgets.QGroupBox("Imaging setup parameters")
    #
    #     self.grid4 = QtWidgets.QGridLayout()
    #     self.grid4.addWidget(self.wavelength_label, 0, 0)
    #     self.groupBox_Parameters.setLayout(self.grid4)
    #     return self.groupBox_Parameters

    def createFifthExclusiveGroup(self):
        self.groupBox_PushButton = QtWidgets.QGroupBox("")

        self.grid5 = QtWidgets.QGridLayout()

        self.grid5.addWidget(self.generate_model, 0, 0)
        self.grid5.addWidget(self.Initialization, 0, 1)
        self.grid5.addWidget(self.display, 0, 2)

        self.groupBox_PushButton.setLayout(self.grid5)
        return self.groupBox_PushButton

    def get_values_Imaging(self):
        try:
            self.wavelength_var = float(self.wavelength.text())
            self.NA_var = float(self.NA.text())
            self.ti0_var = float(self.ti0.text())
            self.ni0_var = float(self.ni0.text())
            self.ni_var = float(self.ni.text())
            self.tg0_var = float(self.tg0.text())
            self.tg_var = float(self.tg.text())
            self.ng0_var = float(self.ng0.text())
            self.ng_var = float(self.ng.text())
            self.ns_var = float(self.ns.text())
            self.pixel_size_physical_var = float(self.pixel_size_physical.text())
            self.pixel_size_var = float(self.pixel_size.text())

            self.empty_value_ImagingBox_flag = True
        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all setup parameters!")
            self.msg_box3.exec_()

            self.empty_value_ImagingBox_flag = False

    def get_values_model(self):
        try:
            if self.groupBox_iPSF_model_1.isChecked():
                self.s_range_var = float(self.s_range.text())
                self.e_range_var = float(self.e_range.text())
                self.step_range_var = float(self.step_range.text())
                self.position_particle3D_var = eval(self.position_particle3D.text())
                self.nx_var = int(self.nx.text())
                self.r_var = int(self.r_.text())

                self.empty_value_ModelBox_flag = True
            elif self.groupBox_iPSF_model_2.isChecked():
                self.s_range_var = float(self.s_range_particle.text())
                self.e_range_var = float(self.e_range_particle.text())
                self.step_range_var = float(self.step_range_particle.text())
                self.nx_var = int(self.nx_particle.text())
                self.r_var = int(self.r_particle.text())
                self.z_focus_var = eval(self.z_focus_particle.text())
                self.empty_value_ModelBox_flag = True

        except:
            self.msg_box3 = QtWidgets.QMessageBox()
            self.msg_box3.setWindowTitle("Warning!")
            self.msg_box3.setText("Please filled all Modelling iPSF!")
            self.msg_box3.exec_()

            self.empty_value_ModelBox_flag = False

    def initial_values_Imaging(self):
        self.wavelength.setText("540e-9")
        self.NA.setText("1.4")
        self.ti0.setText("180e-6")
        self.ni0.setText("1.5")
        self.ni.setText("1.5")
        self.tg0.setText("170e-6")
        self.tg.setText("170e-6")
        self.ng0.setText("1.5")
        self.ng.setText("1.5")
        self.ns.setText("1.33")
        self.pixel_size_physical.setText("5.8e-6")
        self.pixel_size.setText("38e-9")

    def initial_values_model_I(self):
        self.s_range.setText("0")
        self.e_range.setText("10")
        self.step_range.setText("1e-1")
        self.position_particle3D.setText("[0, 0, 3e-6]")
        self.nx.setText("513")
        self.r_.setText("50")

    def initial_values_model_II(self):
        self.s_range_particle.setText("1.0")
        self.e_range_particle.setText("5.0")
        self.step_range_particle.setText("25e-3")
        self.position_particle3D.setText("[0, 0, 3e-6]")
        self.nx_particle.setText("513")
        self.r_particle.setText("50")
        self.z_focus_particle.setText("[3e-6]")

    def do_Initialization(self):
        self.initial_values_Imaging()
        self.initial_values_model_I()
        self.initial_values_model_II()

    def do_model(self):
        if self.generate_model.clicked:
            self.get_values_Imaging()
            self.get_values_model()
            if self.empty_value_ModelBox_flag and self.empty_value_ImagingBox_flag:
                p = ImagingSetupParameters

                # Wavelength of the light source in meters
                p.wavelength = self.wavelength_var
                # Numerical Aperture (NA) of the objective lens
                p.NA = self.NA_var
                # Thickness of the immersion oil, nominal value in meters
                p.ti0 = self.ti0_var
                # Refractive index of the immersion oil, nominal value
                p.ni0 = self.ni0_var
                # Refractive index of the immersion oil, experimental value
                p.ni = self.ni_var
                # Thickness of the coverglass, nominal value in meters
                p.tg0 = self.tg0_var
                # Thickness of the coverglass, experimental value in meters
                p.tg = self.tg_var
                # Refractive index of the coverglass, nominal value
                p.ng0 = self.ng0_var
                # Refractive index of the coverglass, experimental value
                p.ng = self.ng_var
                # Refractive index of the sample/medium
                p.ns = self.ns_var
                # Physical size of the camera pixel in meters
                p.pixel_size_physical = self.pixel_size_physical_var
                # Imaging pixel size in meters, related to the physical pixel size through the magnification of the setup
                pixel_size = self.pixel_size_var
                # Magnification of the imaging system
                p.M = self.pixel_size_physical_var / self.pixel_size_var

                p.k0 = 2 * math.pi / self.wavelength_var  # Wavevector
                p.alpha = math.asin(
                    self.NA_var / self.ni_var
                )  # Largest angle collected by our Objective lens
                if self.groupBox_iPSF_model_1.isChecked():
                    z_focus_array = (
                        np.arange(self.s_range_var, self.e_range_var, self.step_range_var) * 1e-6
                    )
                    nz = np.size(z_focus_array)

                    scattered_field = ScatteredFieldDifferentialPhase(
                        p, self.position_particle3D_var, z_focus_array, nz, self.nx_var
                    )
                    (
                        scatteredFieldAmplitude_focalStack,
                        scatteredFieldPhase_focalStack,
                    ) = scattered_field.calculate(self.r_var)
                    iPSFs_focalStack = np.multiply(
                        scatteredFieldAmplitude_focalStack, np.cos(scatteredFieldPhase_focalStack)
                    )
                    self.iPSFs_focalStack = iPSFs_focalStack / np.max(
                        np.abs(iPSFs_focalStack[...])
                    )

                    list_titles = z_focus_array.tolist()
                    self.list_titles = [
                        "defocusing positions=" + str(z_ * 1e6) + " um" for z_ in list_titles
                    ]

                    self.iPSFs_focalStack_meirdonal = iPSFs_focalStack[:, 1 + self.r_var, :]
                    self.scatteredFieldAmplitude_focalStack_meirdonal = (
                        scatteredFieldAmplitude_focalStack[:, 1 + self.r_var, :]
                    )
                    self.scatteredFieldPhase_focalStack_meirdonal = (
                        scatteredFieldPhase_focalStack[:, 1 + self.r_var, :]
                    )

                    self.iPSFs_focalStack_meirdonal = (
                        self.iPSFs_focalStack_meirdonal / self.iPSFs_focalStack_meirdonal.max()
                    )
                    self.scatteredFieldAmplitude_focalStack_meirdonal = (
                        self.scatteredFieldAmplitude_focalStack_meirdonal
                        / self.scatteredFieldAmplitude_focalStack_meirdonal.max()
                    )
                    self.scatteredFieldPhase_focalStack_meirdonal = (
                        self.scatteredFieldPhase_focalStack_meirdonal
                        / self.scatteredFieldPhase_focalStack_meirdonal.max()
                    )

                elif self.groupBox_iPSF_model_2.isChecked():
                    nz = np.size(self.z_focus_var)
                    z_particle_array = (
                        np.arange(self.s_range_var, self.e_range_var, self.step_range_var) * 1e-6
                    )
                    self.scatteredFieldAmplitude_AxialStack = np.zeros(
                        (len(z_particle_array), 2 * self.r_var + 1, 2 * self.r_var + 1)
                    )
                    self.scatteredFieldPhase_AxialStack = np.zeros(
                        (len(z_particle_array), 2 * self.r_var + 1, 2 * self.r_var + 1)
                    )

                    for cnt, zp_ in enumerate(z_particle_array):
                        # The 3D position of the nanoparticle
                        Xp = [0, 0, zp_]
                        scattered_field = ScatteredFieldDifferentialPhase(
                            p, Xp, self.z_focus_var, nz, self.nx_var
                        )
                        scatteredFieldAmplitude, scatteredFieldPhase = scattered_field.calculate(
                            self.r_var
                        )
                        self.scatteredFieldAmplitude_AxialStack[
                            cnt, ...
                        ] = scatteredFieldAmplitude
                        self.scatteredFieldPhase_AxialStack[cnt, ...] = scatteredFieldPhase

                    iPSFs_AxialStack = np.multiply(
                        self.scatteredFieldAmplitude_AxialStack,
                        np.cos(self.scatteredFieldPhase_AxialStack),
                    )
                    self.iPSFs_AxialStack = iPSFs_AxialStack / np.max(
                        np.abs(iPSFs_AxialStack[...])
                    )

                    self.iPSFs_AxialStack_meirdonal = iPSFs_AxialStack[:, 1 + self.r_var, :]
                    self.scatteredFieldAmplitude_AxialStack_meirdonal = (
                        self.scatteredFieldAmplitude_AxialStack[:, 1 + self.r_var, :]
                    )
                    self.scatteredFieldPhase_AxialStack_meirdonal = (
                        self.scatteredFieldPhase_AxialStack[:, 1 + self.r_var, :]
                    )

                    self.iPSFs_AxialStack_meirdonal = (
                        self.iPSFs_AxialStack_meirdonal / self.iPSFs_AxialStack_meirdonal.max()
                    )
                    self.scatteredFieldAmplitude_AxialStack_meirdonal = (
                        self.scatteredFieldAmplitude_AxialStack_meirdonal
                        / self.scatteredFieldAmplitude_AxialStack_meirdonal.max()
                    )
                    self.scatteredFieldPhase_AxialStack_meirdonal = (
                        self.scatteredFieldPhase_AxialStack_meirdonal
                        / self.scatteredFieldPhase_AxialStack_meirdonal.max()
                    )

                    a = 1

    def do_display(self):
        if self.groupBox_iPSF_model_1.isChecked() and self.iPSFs_focalStack is not None:
            self.display_trigger.emit(["iPSF_Model_I", self.iPSFs_focalStack])
            self.pg = UpdatingPlots_Image(
                list_img=[
                    self.iPSFs_focalStack_meirdonal,
                    self.scatteredFieldAmplitude_focalStack_meirdonal,
                    self.scatteredFieldPhase_focalStack_meirdonal,
                ],
                list_titles=[
                    "Focal stack: iPSFs,",
                    "Scattered field amplitude,",
                    "Scattered field phase",
                ],
                x_axis_labels=[
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                ],
                y_axis_labels=[
                    "Focal stack: iPSFs,",
                    "Defocus axis (x" + str(1e2) + " nm)",
                    "Defocus axis (x" + str(1e2) + " nm)",
                ],
                title="iPSF_Model_I",
            )
        elif self.groupBox_iPSF_model_2.isChecked() and self.iPSFs_AxialStack is not None:
            self.display_trigger.emit(["iPSF_Model_II", self.iPSFs_AxialStack])
            self.pg = UpdatingPlots_Image(
                list_img=[
                    self.iPSFs_AxialStack_meirdonal,
                    self.scatteredFieldAmplitude_AxialStack_meirdonal,
                    self.scatteredFieldPhase_AxialStack_meirdonal,
                ],
                list_titles=[
                    "Axial stack: iPSFs,",
                    "Scattered field amplitude,",
                    "Scattered field phase",
                ],
                x_axis_labels=[
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                    "x-axis (x" + str(self.pixel_size_var * 1e9) + " nm)",
                ],
                y_axis_labels=[
                    "Focal stack: iPSFs,",
                    "Defocus axis (x" + str(1e2) + " nm)",
                    "Defocus axis (x" + str(1e2) + " nm)",
                ],
                title="iPSF_Model_II",
            )

    @QtCore.Slot(int)
    def get_sliceNumber(self, frame_number):
        if self.groupBox_iPSF_model_1.isChecked():
            self.frame_num = frame_number
            x = list(range(0, self.iPSFs_focalStack_meirdonal.shape[0]))
            y = [frame_number for _ in range(len(x))]
            self.pg.update_plot(x, y)
            self.pg.show()

        elif self.groupBox_iPSF_model_2.isChecked():
            self.frame_num = frame_number
            x = list(range(0, self.iPSFs_AxialStack_meirdonal.shape[0]))
            y = [frame_number for _ in range(len(x))]
            self.pg.update_plot(x, y)
            self.pg.show()

        # print(self.list_titles[self.frame_num])
