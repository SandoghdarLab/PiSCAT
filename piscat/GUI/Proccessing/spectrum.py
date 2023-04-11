from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QThreadPool

from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat.Preproccessing.filtering import FFT2D


class FFT2D_GUI_wrapper:
    def __init__(self):
        super(FFT2D_GUI_wrapper, self).__init__()
        self.flag_update_FFT_video = True
        self.original_video = None
        self.fft_v = None
        self.threadpool = QThreadPool()

    def spectrum_input(self, video):
        self.original_video = video

    def spectrum(self):
        if self.original_video is not None:
            self.flag_update_FFT_video = True
            worker = FFT2D(video=self.original_video)
            worker.signals.result.connect(self.update_FFT_video)
            self.threadpool.start(worker)
            while self.flag_update_FFT_video:
                QtCore.QCoreApplication.processEvents()

            self.flag_update_FFT_video = True

            visualization_ = Visulization_localization()
            visualization_.new_display(self.fft_v, self.fft_v, object=None, title="Spectrum")

        else:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Please load one video!")
            self.msg_box.exec_()

    def update_FFT_video(self, r_):
        self.fft_v = r_
        self.flag_update_FFT_video = False
