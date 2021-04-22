from piscat.GUI.CPU_Configurations import Import_configuration
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from PySide2 import QtWidgets
from PySide2 import QtCore


class CPU_setting_wrapper(QtWidgets.QMainWindow):

    def __init__(self):
        super(CPU_setting_wrapper, self).__init__()
        self.change_cpu_setting()

    def change_cpu_setting(self):
        print('\nChange CPU setting-->', end='')
        self.cpu_setting = Import_configuration.CPU_Setting()
        self.cpu_setting.update_CPU_Setting.connect(self.save_CPU_setting)

    @QtCore.Slot()
    def save_CPU_setting(self):
        if self.cpu_setting.parallel_active is True:

            setting_dic = {'n_jobs': [self.cpu_setting.n_jobs], 'backend': [self.cpu_setting.backend], 'verbose': [self.cpu_setting.verbose],
                           'parallel_active': [self.cpu_setting.parallel_active],
                           'threshold_for_parallel_run': [self.cpu_setting.threshold_for_parallel_run]}

            cpu_setting = CPUConfigurations()
            cpu_setting.save_cpu_setting(setting_dic)
        else:
            setting_dic = {'n_jobs': [1], 'backend': ['None'],
                           'verbose': [0],
                           'parallel_active': [self.cpu_setting.parallel_active],
                           'threshold_for_parallel_run': [None]}
            cpu_setting = CPUConfigurations()
            cpu_setting.save_cpu_setting(setting_dic)
        print("Done!")