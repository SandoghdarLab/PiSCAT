import os

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from piscat.GUI.InputOutput import import_im2vid, import_RAW, video_cropping
from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat.InputOutput import image_to_video, read_status_line, reading_videos


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
        dialog.setWindowTitle("Open File")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.filename = dialog.selectedFiles()[0]
        if type(self.filename) is str:
            file_extention = os.path.splitext(self.filename)[1]
            if (
                file_extention == ".pf"
                or file_extention == ".raw"
                or file_extention == ".bin"
                or file_extention == ".PF"
                or file_extention == ".RAW"
                or file_extention == ".BIN"
            ):
                return "Raw"
            elif (
                file_extention == ".jpg"
                or file_extention == ".JPG"
                or file_extention == ".png"
                or file_extention == ".PNG"
            ):
                return "PNG"
            elif file_extention == ".tiff" or file_extention == ".TIFF":
                return "TIF"
            elif file_extention == ".avi" or file_extention == ".AVI":
                return "AVI"
            elif file_extention == ".TIF" or file_extention == ".tif":
                return "TIF"
            elif file_extention == ".fits" or file_extention == ".fits":
                return "Fits"
            elif file_extention == ".fli" or file_extention == ".fli":
                return "Fli"
            elif file_extention == ".py" or file_extention == ".PY":
                return "Python"

    def askdir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select a folder:", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        return folder

    def read_video(self):
        title = self.askdir_file()
        if self.filename:
            if title == "Raw":
                self.image = import_RAW.RawImage(self.filename)
                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()
                try:
                    video = reading_videos.read_binary(
                        self.filename,
                        img_width=self.image.width_size,
                        img_height=self.image.height_size,
                        image_type=self.image.set_bit_order + self.image.type,
                    )

                    if self.image.groupBox_cropping.isChecked():
                        self.original_video = video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]

                    else:
                        self.original_video = video

                    self.original_video, status_line_info = self.status_line_remove(
                        self.original_video
                    )

                    if self.image.flag_display is True:
                        self.visualization_ = Visulization_localization()
                        self.visualization_.new_display(
                            self.original_video,
                            self.original_video,
                            object=None,
                            title="Raw video",
                        )

                    self.update_output.emit(
                        [self.original_video, title, self.filename, status_line_info]
                    )
                except:
                    self.msg_box = QtWidgets.QMessageBox()
                    self.msg_box.setWindowTitle("Warning!")
                    self.msg_box.setText("Selected parameters are not correct!")
                    self.msg_box.exec_()

            elif title == "PNG":
                png_img = reading_videos.read_png(self.filename)
                self.original_video = png_img

                self.visualization_ = Visulization_localization(self.filename)
                self.visualization_.new_display(
                    self.original_video, self.original_video, object=None, title="PNG"
                )

                self.update_output.emit([self.original_video, title, self.filename, None])

            elif title == "AVI":
                avi_video = reading_videos.read_avi(self.filename)

                self.image = video_cropping.Cropping(self)

                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.image.frame_e is not None:
                    if self.image.frame_e != -1:
                        self.original_video = avi_video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                    elif self.image.frame_e == -1:
                        self.original_video = avi_video[
                            self.image.frame_s : -1 : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                else:
                    self.original_video = avi_video

                self.original_video = self.original_video.copy(order="C")

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="AVI"
                    )

                self.update_output.emit([self.original_video, title, self.filename, None])

            elif title == "TIF":
                tif_video = reading_videos.read_tif(self.filename)
                self.image = video_cropping.Cropping(self)

                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.image.flag_RGB2GRAY:
                    if tif_video.shape[2] == 4:
                        tif_video = cv2.cvtColor(tif_video, cv2.COLOR_BGR2GRAY)
                        tif_video = np.expand_dims(tif_video, axis=0)

                if self.image.frame_e is not None:
                    if self.image.frame_e != -1:
                        self.original_video = tif_video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                    elif self.image.frame_e == -1:
                        self.original_video = tif_video[
                            self.image.frame_s :: self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                else:
                    self.original_video = tif_video

                self.original_video = self.original_video.copy(order="C")

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="TIF"
                    )

                self.update_output.emit([self.original_video, title, self.filename, None])

            elif title == "Fits":
                fits_video = reading_videos.read_fits(self.filename)
                self.image = video_cropping.Cropping(self)

                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.image.frame_e is not None:
                    if self.image.frame_e != -1:
                        self.original_video = fits_video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                    elif self.image.frame_e == -1:
                        self.original_video = fits_video[
                            self.image.frame_s :: self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                else:
                    self.original_video = fits_video

                self.original_video = self.original_video.copy(order="C")

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="TIF"
                    )

                self.update_output.emit([self.original_video, title, self.filename])

            elif title == "Fli":
                fli_video = reading_videos.read_fli(self.filename)
                self.image = video_cropping.Cropping(self)

                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.image.frame_e is not None:
                    if self.image.frame_e != -1:
                        self.original_video = fli_video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                    elif self.image.frame_e == -1:
                        self.original_video = fli_video[
                            self.image.frame_s :: self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                else:
                    self.original_video = fli_video

                self.original_video = self.original_video.copy(order="C")

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="TIF"
                    )

                self.update_output.emit([self.original_video, title, self.filename])

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("The video type is not defined!")
                self.msg_box.exec_()

        elif self.filename is False:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Filename is not valid.!")
            self.msg_box.exec_()

    def load_batch_data(self):
        data_setting = {}
        self.filename = self.askdir()
        self.image = import_RAW.RawImage(self.filename)
        while self.image.raw_data_update_flag:
            QtWidgets.QApplication.processEvents()
        try:
            data_setting["path"] = self.filename
            data_setting["title"] = "Raw"
            data_setting["img_width"] = self.image.width_size
            data_setting["img_height"] = self.image.height_size
            data_setting["image_type"] = self.image.set_bit_order + self.image.type
            data_setting["s_frame"] = self.image.frame_s
            data_setting["e_frame"] = self.image.frame_e

            if self.image.groupBox_cropping.isChecked():
                data_setting["frame_stride"] = self.image.frame_jump
                data_setting["width_size_s"] = self.image.width_size_s
                data_setting["width_size_e"] = self.image.width_size_e
                data_setting["height_size_s"] = self.image.height_size_s
                data_setting["height_size_e"] = self.image.height_size_e
            else:
                data_setting["frame_stride"] = 1
                data_setting["width_size_s"] = 0
                data_setting["width_size_e"] = -1
                data_setting["height_size_s"] = 0
                data_setting["height_size_e"] = -1
            self.update_output.emit([data_setting])
        except:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("Selected parameters are not correct!")
            self.msg_box.exec_()

    def im2video(self):
        title = "image2video"
        folder = self.askdir()
        self.image = import_im2vid.Image2Video(self)
        while self.image.raw_data_update_flag:
            QtWidgets.QApplication.processEvents()
        try:
            if self.image.im_type is not None:
                im2vid = image_to_video.Image2Video(
                    path=folder,
                    file_format=self.image.im_type,
                    width_size=self.image.width_size,
                    height_size=self.image.height_size,
                    image_type=np.dtype(self.image.set_bit_order + self.image.type),
                    reader_type=self.image.video_reader_type,
                )
                video = im2vid()

                video, status_line_info = self.status_line_remove(video)

                self.original_video = video

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="im2video"
                    )

                self.update_output.emit(
                    [self.original_video, title, self.folder, status_line_info]
                )
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

    def tiff_video(self):
        title = self.askdir_file()

        if self.filename:
            if title == "TIF":
                tif_video = reading_videos.read_tif_iterate(self.filename)
                self.image = video_cropping.Cropping(self)

                while self.image.raw_data_update_flag:
                    QtWidgets.QApplication.processEvents()

                if self.image.flag_RGB2GRAY:
                    if tif_video.shape[2] == 4:
                        tif_video = cv2.cvtColor(tif_video, cv2.COLOR_BGR2GRAY)
                        tif_video = np.expand_dims(tif_video, axis=0)

                if self.image.frame_e is not None:
                    if self.image.frame_e != -1:
                        self.original_video = tif_video[
                            self.image.frame_s : self.image.frame_e : self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                    elif self.image.frame_e == -1:
                        self.original_video = tif_video[
                            self.image.frame_s :: self.image.frame_jump,
                            self.image.width_size_s : self.image.width_size_e,
                            self.image.height_size_s : self.image.height_size_e,
                        ]
                else:
                    self.original_video = tif_video

                self.original_video = self.original_video.copy(order="C")

                if self.image.flag_display is True:
                    self.visualization_ = Visulization_localization()
                    self.visualization_.new_display(
                        self.original_video, self.original_video, object=None, title="TIF"
                    )

                self.update_output.emit([self.original_video, title, self.filename, None])

            else:
                self.msg_box = QtWidgets.QMessageBox()
                self.msg_box.setWindowTitle("Warning!")
                self.msg_box.setText("Type of image is not defined!")
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

        if axis_status_line == "row" or axis_status_line == "column":
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText(
                "status line remove"
                + "\nold shape:"
                + str(video.shape)
                + "\nnew shape:"
                + str(cut_frames.shape)
            )
            self.msg_box.exec_()

        return cut_frames, axis_status_line

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
        print("closing PlaySetting")
