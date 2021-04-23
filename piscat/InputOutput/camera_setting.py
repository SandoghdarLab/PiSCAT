from __future__ import print_function

import pandas as pd
import json
import os
import sys


class CameraParameters:

    def __init__(self, name, quantum_efficiency=0.3, electron_well_depth=180e3, max_electron_well_depth=200e3, bit_depth=12, pixelSize=0.66):
        """
        Based on the camera features, this class generates a JSON file.
        This JSON was used by other functions and methods to set certain parameters, such as pixel size.

        Parameters
        ----------
        name: str
            Name of camera.

        quantum_efficiency: float
            Quantum efficiency of camera.

        electron_well_depth: float
             Electron well depth of camera.

        max_electron_well_depth: float
            Maximum electron well depth of camera

        bit_depth: float
            Bit depth of the camera.

        pixelSize: float
            Pixel size of camera.
        """
        try:
            self.read_camera_setting(name=name)
        except FileNotFoundError:
            self.quantum_efficiency = quantum_efficiency
            self.electron_well_depth = electron_well_depth
            self.max_electron_well_depth = max_electron_well_depth
            self.bit_depth = bit_depth
            self.pixelSize = pixelSize

            self.setting_dic = {'quantum_efficiency': [self.quantum_efficiency],
                                'electron_well_depth': [self.electron_well_depth],
                                'max_electron_well_depth': [self.max_electron_well_depth],
                                'bit_depth': [self.bit_depth],
                                'pixelSize': [self.pixelSize]}
            self.save_camera_setting(name)

    def save_camera_setting(self, name='Photonfocus.json'):

        here = os.path.dirname(os.getcwd())
        subdir = 'piscat_configuration'

        try:
            dr_mk = os.path.join(here, subdir)
            os.mkdir(dr_mk)
            print("Directory ", subdir, " Created ")
        except FileExistsError:
            print("Directory ", subdir, " already exists")

        filepath = os.path.join(here, subdir, name)
        df_configfile = pd.DataFrame(data=self.setting_dic)
        df_configfile.to_json(filepath)

    def read_camera_setting(self, name='Photonfocus.json'):
        subdir = 'piscat_configuration'
        here = os.path.dirname(os.getcwd())
        filepath = os.path.join(here, subdir, name)

        with open(filepath) as json_file:
            cam_setting = json.load(json_file)

        self.quantum_efficiency = cam_setting['quantum_efficiency']['0']
        self.electron_well_depth = cam_setting['electron_well_depth']['0']
        self.max_electron_well_depth = cam_setting['max_electron_well_depth']['0']
        self.bit_depth = cam_setting['bit_depth']['0']
        self.pixelSize = cam_setting['pixelSize']['0']






