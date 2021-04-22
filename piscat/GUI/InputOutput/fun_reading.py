from piscat.GUI.InputOutput import import_im2vid, import_RAW, video_cropping
from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat.InputOutput import reading_videos, image_to_video, read_status_line

from PySide2 import QtCore
from PySide2 import QtWidgets

import os
import numpy as np


class Reading(QtWidgets.QMainWindow):

    update_output = QtCore.Signal(object)

    def __init__(self):
        super(Reading, self).__init__()
        self.original_video = None
        self.filename = None
        self.folder = None

    def askdir_file(self):
        self.filename = False
        dialog = QtWidgets.QFileDialog(self)
        dialog.setWindowTitle('Open File')
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.filename = dialog.selectedFiles()[0]
        if type(self.filename) is str:

            file_extention = os.path.splitext(self.filename)[1]
            if file_extention == ".pf" or file_extention == ".raw" or file_extention == ".bin" or file_extention == ".PF" or file_extention == ".RAW" or file_extention == ".BIN":
                return "Raw"
            elif file_extention == ".jpg" or file_extention == ".JPG" or file_extention == ".tiff" or file_extention == ".TIFF" or file_extention == ".png" or file_extention == ".PNG":
                return "PNG"
            elif file_extention == ".avi" or file_extention == ".AVI":
                return "AVI"
            elif file_extention == ".TIF" or file_extention == ".tif":
                return "TIF"
            elif file_extention == ".py" or file_extention == ".PY":
                return "Python"

    def askdir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', '',
                                                                 QtWidgets.QFileDialog.ShowDirsOnly)
        return folder

    def read_video(self):
        title = self.askdir_file()
        if self.filename:
            if title == "Raw":
                self.info_image = import_RAW.RawImage(self.filename)
                while self.info_image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()
                try:
                    video = reading_videos.read_binary(self.filename, img_width=self.info_image.width_size,
                                                        img_height=self.info_image.height_size,
                                                        image_type=self.info_image.set_bit_order + self.info_image.type,
                                                        s_frame=self.info_image.frame_s, e_frame=self.info_image.frame_e)

                    if self.info_image.groupBox_cropping.isChecked():

                        self.original_video = video[
                                         0:-1:self.info_image.frame_jump,
                                         self.info_image.width_size_s:self.info_image.width_size_e,
                                         self.info_image.height_size_s:self.info_image.height_size_e]

                    else:
                        self.original_video = video

                    self.original_video = self.status_line_remove(self.original_video)

                    if self.info_image.flag_display is True:
                        self.visualization_ = Visulization_localization()
                        self.visualization_.new_display(self.original_video, self.original_video, object=None, title='Raw video')

                    self.update_output.emit([self.original_video, title, self.filename])
                except:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Selected parameters are not correct!")
                    self.msg_box.exec_()

            elif title == "PNG":
                png_img = reading_videos.read_png(self.filename)
                self.original_video = png_img

                self.visualization_ = Visulization_localization(self.filename)
                self.visualization_.new_display(self.original_video, self.original_video, object=None, title='PNG')

                self.update_output.emit([self.original_video, title, self.filename])

            elif title == "AVI":
                avi_video = reading_videos.read_avi(self.filename)

                self.info_image = video_cropping.Cropping(self)

                while self.info_image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.info_image.frame_e is not None:

                    if self.info_image.frame_e != -1:
                        self.original_video = avi_video[
                                              self.info_image.frame_s:self.info_image.frame_e:self.info_image.frame_jump,
                                              self.info_image.width_size_s:self.info_image.width_size_e,
                                              self.info_image.height_size_s:self.info_image.height_size_e]
                    elif self.info_image.frame_e == -1:
                        self.original_video = avi_video[self.info_image.frame_s:-1:self.info_image.frame_jump,
                                              self.info_image.width_size_s:self.info_image.width_size_e,
                                              self.info_image.height_size_s:self.info_image.height_size_e]
                else:
                    self.original_video = avi_video

                self.original_video = self.original_video.copy(order='C')

                if self.info_image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(self.original_video, self.original_video, object=None, title='AVI')

                self.update_output.emit([self.original_video, title, self.filename])

            elif title == "TIF":
                tif_video = reading_videos.read_tif(self.filename)
                self.info_image = video_cropping.Cropping(self)

                while self.info_image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.info_image.frame_e is not None:

                    if self.info_image.frame_e != -1:
                        self.original_video = tif_video[
                                              self.info_image.frame_s:self.info_image.frame_e:self.info_image.frame_jump,
                                              self.info_image.width_size_s:self.info_image.width_size_e,
                                              self.info_image.height_size_s:self.info_image.height_size_e]
                    elif self.info_image.frame_e == -1:
                        self.original_video = tif_video[self.info_image.frame_s::self.info_image.frame_jump,
                                              self.info_image.width_size_s:self.info_image.width_size_e,
                                              self.info_image.height_size_s:self.info_image.height_size_e]
                else:
                    self.original_video = tif_video

                self.original_video = self.original_video.copy(order='C')

                if self.info_image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(self.original_video, self.original_video, object=None, title='TIF')

                self.update_output.emit([self.original_video, title, self.filename])

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("The video bin_type is not defined!")
                self.msg_box.exec_()

        elif self.filename is False:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Filename is not valid.!")
            self.msg_box.exec_()

    def im2video(self):

        title = 'image2video'
        folder = self.askdir()
        self.info_image = import_im2vid.Image2Video(self)
        while self.info_image.raw_data_update_flag:
            QtWidgets.QApplication.processEvents()
        try:

            if self.info_image.im_type is not None:

                im2vid = image_to_video.Image2Video(path=folder, file_format=self.info_image.im_type,
                                                    width_size=self.info_image.width_size,
                                                    height_size=self.info_image.height_size,
                                                    image_type=np.dtype(self.info_image.set_bit_order + self.info_image.type),
                                                    reader_type=self.info_image.video_reader_type)
                video = im2vid()

                video = self.status_line_remove(video)

                self.original_video = video

                if self.info_image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(self.original_video, self.original_video, object=None, title='im2video')

                self.update_output.emit([self.original_video, title, self.folder])
            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Type of image is not defined!")
                self.msg_box.exec_()
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Selected parameters are not correct!")
            self.msg_box.exec_()

    def run_py_script(self):
        title = self.askdir_file()

        if title == "Python":
            self.threadpool.start(self.run_py_script_thread(self.filename))
            self.set_plain_text(" Done")

        else:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("File bin_type is not valid.!")
            self.msg_box.exec_()

    def status_line_remove(self, video):
        statusLine = read_status_line.StatusLine(video)
        cut_frames, axis_status_line = statusLine.find_status_line()

        if axis_status_line == 'row' or axis_status_line == 'column':
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("status line remove" + "\nold shape:" + str(video.shape) + "\nnew shape:" + str(cut_frames.shape))
            self.msg_box.exec_()

        return cut_frames

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
        print("closing PlaySetting")



