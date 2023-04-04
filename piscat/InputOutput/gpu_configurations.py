import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import sys

import pandas as pd
from tensorflow.python.client import device_lib

from piscat.Visualization.print_colors import PrintColors


class GPUConfigurations(PrintColors):
    def __init__(self, gpu_device=None, gpu_active_flag=True, flag_report=False):
        """
        This class generates a JSON file for setting on the GPU that the user prefers.
        This JSON was used by other functions and methods to set hyperparameters in a gpu usage.
        For parallelization, PiSCAT used Tensorflow.

        Parameters
        ----------
        gpu_device: int
            Select the GPU device that will be used.

        gpu_active_flag: bool
            Turn on the GPU version of the code. Otherwise, code is executed on the CPU.

        flag_report: bool
            This flag is set if you need to see the values that will be used for CPU configuration.
        """

        PrintColors.__init__(self)

        try:
            self.read_gpu_setting(flag_report)

        except FileNotFoundError:
            list_gpu_cpu = self.get_available_devices()
            count_ = 0
            for gpu_ in list_gpu_cpu:
                if "GPU" in gpu_[0]:
                    count_ = count_ + 1
                    print(str(gpu_))

            if count_ == 0:
                self.gpu_active_flag = False
                self.gpu_device = None
            else:
                self.gpu_active_flag = True
                if gpu_device is None:
                    self.gpu_device = 0
                else:
                    self.gpu_device = gpu_device

            setting_dic = {"gpu_active": [self.gpu_active_flag], "gpu_device": [self.gpu_device]}

            self.save_gpu_setting(setting_dic)

    def save_gpu_setting(self, setting_dic):
        name = "gpu_configurations.json"
        here = os.path.dirname(os.getcwd())
        subdir = "piscat_configuration"

        try:
            dr_mk = os.path.join(here, subdir)
            os.mkdir(dr_mk)
            print("Directory ", subdir, " Created ")
        except FileExistsError:
            print("Directory ", subdir, " already exists")

        filepath = os.path.join(here, subdir, name)
        df_configfile = pd.DataFrame(data=setting_dic)
        df_configfile.to_json(filepath)

    def read_gpu_setting(self, flag_report=False):
        """
        Parameters
        ----------
         flag_report: bool
            This flag is set if you need to see the values that will be used for CPU configuration.

        """
        subdir = "piscat_configuration"
        here = os.path.dirname(os.getcwd())
        filepath = os.path.join(here, subdir, "gpu_configurations.json")

        with open(filepath) as json_file:
            gpu_setting = json.load(json_file)

        self.gpu_active_flag = gpu_setting["gpu_active"]["0"]
        self.gpu_device = gpu_setting["gpu_device"]["0"]

        if flag_report:
            print("PiSCAT's general parallel GPU flag is set to {}".format(self.gpu_active_flag))
            print("\nThe code is executed on the GPU device {}.".format(self.gpu_device))

    def get_available_devices(self):
        local_device_protos = device_lib.list_local_devices()
        return [
            [x.name, x.memory_limit]
            for x in local_device_protos
            if x.device_type == "GPU" or x.device_type == "CPU"
        ]

    def print_all_available_gpu(self):
        list_gpu_cpu = self.get_available_devices()
        count_ = 0

        for gpu_ in list_gpu_cpu:
            if "GPU" in gpu_[0]:
                count_ = count_ + 1
                print(str(gpu_[0]) + ", memory:" + str(gpu_[1]))

        if count_ == 0:
            print(
                f"{self.WARNING}\nPiSCAT cannot detect any GPU! TensorFlow code is entirely executed on the CPU!!{self.ENDC}"
            )
