import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QRunnable, Signal
from PySide6.QtGui import QPen, Qt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from piscat.GUI.Visualization.updating_plots import (
    UpdatingPlotsPyqtGraphSpatial,
    UpdatingPlotsPyqtGraphTemporal,
)
from piscat.Localization.directional_intensity import DirectionalIntensity
from piscat.Preproccessing.normalization import Normalization
from piscat.Visualization.contrast_adjustment import ContrastAdjustment


class SliceView(QtWidgets.QGraphicsView, QRunnable):
    photoClicked = QtCore.Signal(tuple)
    frameClicked = QtCore.Signal(int)
    pixmapClicked = QtCore.Signal(QtCore.QPoint)
    cursorMove = QtCore.Signal(object)
    annotation_s = QtCore.Signal()
    annotation_t = QtCore.Signal(object)
    finished = Signal()

    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag_paint = False

    def __init__(self, input_video, video_original, stride=1, mask=False, *args, **kwargs):
        self.scene = QtWidgets.QGraphicsScene()
        super(SliceView, self).__init__(self.scene, *args, **kwargs)

        self.updatePlot_spatial = UpdatingPlotsPyqtGraphSpatial()
        self.updatePlot_temporal = UpdatingPlotsPyqtGraphTemporal()

        self.RAW_Video = video_original

        if video_original.ndim == 3:
            self.video_width = video_original.shape[1]
            self.video_height = video_original.shape[2]
        elif video_original.ndim == 2:
            self.video_width = video_original.shape[0]
            self.video_height = video_original.shape[1]

        self.input_video = input_video
        self.con_adj = ContrastAdjustment(input_video)
        self.current_pixmap = None
        self.current_mask_pixmap = None
        self.maskArray = None
        self.mask_is_set = False
        self.live_spatial_paint_Flag = False
        self.live_temporal_paint_Flag = False
        self.medianFilterFlag = False
        self.ori_X0 = 0
        self.ori_Y0 = 0
        self.ori_X1 = 0
        self.ori_Y1 = 0
        self.pixel_value = 0
        self.slice_num = 0
        self.marker_size = 5

        self._zoom = 0
        self._empty = True

        self.annotation_s.connect(self.active_spatial_live_paint)
        self.annotation_t.connect(self.active_temporal_live_paint)

        self.setBackgroundBrush(QtWidgets.QColor(0, 0, 0))
        self.setScene(self.scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        if mask is True:
            self.initialze_mask()

    def __del__(self):
        print("Destructor called, Employee deleted.")

    def enterEvent(self, event):
        self.setMouseTracking(True)

    def leaveEvent(self, event):
        self.setMouseTracking(False)

    def pollCursor(self):
        pos = QtGui.QCursor.pos()
        if pos != self.cursor:
            self.cursor = pos
            self.cursorMove.emit(pos)

    def handleCursorMove(self, pos):
        print(pos)

    def hasPhoto(self):
        return not self._empty

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            sp = self.mapToScene(event.pos())
            lp = self._photo.mapFromScene(sp).toPoint()

            self.x0 = lp.x()
            self.y0 = lp.y()
            self.flag_paint = True

            self.ori_X0 = int(
                (self.RAW_Video.shape[1] / self.pixmap_video.size().toTuple()[0]) * self.x0
            )
            self.ori_Y0 = int(
                (self.RAW_Video.shape[2] / self.pixmap_video.size().toTuple()[1]) * self.y0
            )

            self.frameClicked.emit(self.slice_num)
            self.photoClicked.emit((self.ori_X0, self.ori_Y0))
            self.paint_point()

        super(SliceView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.flag_paint = False

    def mouseMoveEvent(self, event):
        if self.flag_paint:
            if self._photo.isUnderMouse():
                sp = self.mapToScene(event.pos())
                lp = self._photo.mapFromScene(sp).toPoint()

                self.x1 = lp.x()
                self.y1 = lp.y()

                self.ori_X1 = int(
                    (self.RAW_Video.shape[1] / self.pixmap_video.size().toTuple()[0]) * self.x1
                )
                self.ori_Y1 = int(
                    (self.RAW_Video.shape[2] / self.pixmap_video.size().toTuple()[1]) * self.y1
                )

                self.paint_line()

    def initialze_mask(self):
        if self.input_video.shape[1] >= self.input_video.shape[2]:
            self.mask_qimage = QtGui.QImage(
                QtCore.QSize(self.input_video.shape[2], self.input_video.shape[1]),
                QtGui.QImage.Format_ARGB32,
            )

            self.mask_qimage.fill(QtGui.QColor(250, 240, 0, 0))
            self.mask_qimage_view = np.frombuffer(
                self.mask_qimage.bits(), dtype=np.uint8
            ).reshape(self.input_video.shape[1], self.input_video.shape[2], 4)
        else:
            self.mask_qimage = QtGui.QImage(
                QtCore.QSize(self.input_video.shape[2], self.input_video.shape[1]),
                QtGui.QImage.Format_ARGB32,
            )

            self.mask_qimage.fill(QtGui.QColor(250, 240, 0, 0))
            self.mask_qimage_view = np.frombuffer(
                self.mask_qimage.bits(), dtype=np.uint8
            ).reshape(self.input_video.shape[1], self.input_video.shape[2], 4)

    def create_mask_pixmap(self, input_mask_slide):
        self.mask_qimage_view[..., 3] = input_mask_slide * int(1 * 255)
        self.current_mask_pixmap = QtGui.QPixmap.fromImage(self.mask_qimage)
        return self.current_mask_pixmap

    def create_pixmap(self, input_file):
        if isinstance(input_file, str):
            self.input_file = input_file

        else:
            input_file = Normalization(video=input_file).normalized_image_specific()
            if self.parent().parent().alpha is None:
                input_file = self.con_adj.auto_pixelTransforms(input_file)
                self.input_file = input_file
            else:
                alpha = self.parent().parent().alpha
                beta = self.parent().parent().beta
                min_intensity = self.parent().parent().min_intensity
                max_intensity = self.parent().parent().max_intensity
                input_file = self.con_adj.pixel_transforms(
                    input_file, alpha, beta, min_intensity, max_intensity
                )
                self.input_file = input_file

        if self.medianFilterFlag:
            input_file = median_filter(input_file, 3)

        if self.parent().parent().title != "PNG":
            if input_file.shape[0] >= input_file.shape[1]:
                widthStep = input_file.shape[1]
                img = QtGui.QImage(
                    input_file,
                    input_file.shape[1],
                    input_file.shape[0],
                    widthStep,
                    QtGui.QImage.Format_Indexed8,
                )
                img.scaledToWidth(input_file.shape[0])
            else:
                widthStep = input_file.shape[1]
                img = QtGui.QImage(
                    input_file,
                    input_file.shape[1],
                    input_file.shape[0],
                    widthStep,
                    QtGui.QImage.Format_Indexed8,
                )

            newimg = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
            self.current_pixmap = QtGui.QPixmap(newimg)
            self.update_slice(self.current_pixmap)
        elif self.parent().parent().title == "PNG":
            self.current_pixmap = QtGui.QPixmap(input_file)

        return self.current_pixmap

    def update_slice(self, input_pixmap):
        self.t = input_pixmap
        self._zoom = 0
        width = np.max([self.video_width, 400])
        height = np.max([self.video_height, 400])

        self.pixmap_video = input_pixmap.scaled(
            QtCore.QSize(width, height), QtCore.Qt.KeepAspectRatio
        )
        self._photo = self.scene.addPixmap(self.pixmap_video)
        if self.live_spatial_paint_Flag:
            self.paint_line()
        if self.live_temporal_paint_Flag:
            self.paint_point()

    def update_overlay(self, input_mask_pixmap, first_flag=True):
        width = np.max([self.video_width, 400])
        height = np.max([self.video_height, 400])
        self.scaled_mask_pixmap = input_mask_pixmap.scaled(
            QtCore.QSize(width, height), QtCore.Qt.KeepAspectRatio
        )

        self._photo.setPixmap(self.scaled_mask_pixmap)
        self._photo_mask = self.scene.addPixmap(self.scaled_mask_pixmap)

        if self.parent().parent().title != "PNG" and first_flag is True:
            self.parent().parent().slice_slider.setValue(self.slice_num)
            self.current_pixmap = self.create_pixmap(self.input_video[self.slice_num, :, :])
            self.update_slice(self.current_pixmap)
            if self.mask_is_set is True:
                self.current_mask_pixmap = self.create_mask_pixmap(
                    self.maskArray[self.slice_num, :, :]
                )
                self.update_overlay(self.current_mask_pixmap, first_flag=False)

    def wheelEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            elif event.angleDelta().y() < 0:
                factor = 0.75
                self._zoom -= 1
            else:
                factor = 0.8
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom < 0:
                self.scale(factor, factor)
            else:
                self._zoom = 0

        else:
            if self.parent().parent().title == "PNG":
                pass
            elif self.parent().parent().title != "PNG":
                steps = event.angleDelta().y() // 120
                vector = steps and steps // abs(steps)  # 0, 1, or -1
                for _ in range(1, abs(steps) + 1):
                    self.slice_num -= vector
                    if self.slice_num >= 0 and self.slice_num <= self.input_video.shape[0] - 1:
                        self.parent().parent().slice_slider.setValue(self.slice_num)
                        self.current_pixmap = self.create_pixmap(
                            self.input_video[self.slice_num, :, :]
                        )
                        self.update_slice(self.current_pixmap)
                        if self.mask_is_set is True:
                            self.current_mask_pixmap = self.create_mask_pixmap(
                                self.maskArray[self.slice_num, :, :]
                            )
                            self.update_overlay(self.current_mask_pixmap)

    @QtCore.Slot()
    def active_spatial_live_paint(self):
        if self.live_spatial_paint_Flag is False:
            self.live_spatial_paint_Flag = True
        elif self.live_spatial_paint_Flag is True:
            self.live_spatial_paint_Flag = False
            try:
                self.scene.removeItem(self.Line)
                self.scene.update()
            except:
                pass

    @QtCore.Slot()
    def active_temporal_live_paint(self, setting=None):
        if self.live_temporal_paint_Flag is False:
            self.live_temporal_paint_Flag = True
        elif self.live_temporal_paint_Flag is True:
            self.live_temporal_paint_Flag = False

        try:
            self.scene.removeItem(self.point_)
            self.scene.update()
        except:
            pass

        self.read_temporal_pixel_values(setting=setting)

    def paint_line(self):
        pen = QPen(Qt.red)
        pen.setCosmetic(True)
        pen.setWidth(1)

        try:
            self.scene.removeItem(self.Line)
        except AttributeError:
            pass

        self.Line = self.scene.addLine(self.x0, self.y0, self.x1, self.y1, pen)
        self.scene.addItem(self.Line)
        self.scene.update()
        self.read_spatial_pixel_values()

    def paint_point(self):
        pen = QPen(Qt.red)
        pen.setCosmetic(True)
        pen.setWidth(1)

        try:
            self.scene.removeItem(self.Line)
        except AttributeError:
            pass

        self.point_ = self.scene.addEllipse(
            self.x0, self.y0, self.marker_size, self.marker_size, pen
        )
        self.scene.addItem(self.point_)
        self.scene.update()
        self.read_temporal_pixel_values(setting=None)

    def read_spatial_pixel_values(self):
        try:
            DI = DirectionalIntensity()
            radial_index = DI.interpolate_pixels_along_line(
                x0=int(self.ori_X0), y0=int(self.ori_Y0), x1=int(self.ori_X1), y1=int(self.ori_Y1)
            )
            frame = self.RAW_Video[self.slice_num, ...]

            pixels_ = []
            for idx_ in radial_index:
                pixels_.append(frame[int(idx_[1]), int(idx_[0])])

            x = list(range(0, len(pixels_)))
            self.updatePlot_spatial.data_line.setData(x, pixels_)  # Update the data.
            self.updatePlot_spatial.show()
        except:
            pass

    def read_temporal_pixel_values(self, setting=None):
        if setting is not None:
            self.win_size = setting[0]
            self.flag_smooth = setting[1]
            self.marker_size = setting[2]

        try:
            min_temporal_range = int(np.max((0, self.slice_num - self.win_size)))
            max_temporal_range = int(
                np.min((self.RAW_Video.shape[0], self.slice_num + self.win_size))
            )

            temporal_intensity = self.RAW_Video[
                min_temporal_range:max_temporal_range, self.ori_Y0, self.ori_X0
            ]
            # current_intensity = self.RAW_Video[self.slice_num, self.ori_Y0, self.ori_X0]

            if self.flag_smooth:
                smooth_window_size = round(0.05 * len(temporal_intensity))
                if (smooth_window_size % 2) == 0:
                    smooth_window_size = smooth_window_size + 1

                if self.win_size > 3:
                    temporal_intensity = savgol_filter(temporal_intensity, smooth_window_size, 3)

            x = list(range(min_temporal_range, max_temporal_range))
            idx_ = np.abs((min_temporal_range - self.slice_num))
            self.updatePlot_temporal.data_line.setData(x, temporal_intensity)  # Update the data.
            self.updatePlot_temporal.data_point.setData(
                [x[idx_]], [temporal_intensity[idx_]]
            )  # Update the data.
            self.updatePlot_temporal.show()
        except:
            pass
