from PySide2.QtWidgets import QWidget, QApplication, QLabel
from PySide2.QtCore import QRect, Qt, QLine
from PySide2.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication

from piscat.Localization.directional_intensity import DirectionalIntensity


class Line_Annotated(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False

    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    def mouseReleaseEvent(self, event):
        self.flag = False
        # self.read_pixel_valus()

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QLine(self.x0, self.y0, self.x1, self.y1)

        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawLine(rect)

    def read_pixel_valus(self):
        DI = DirectionalIntensity()
        radial_index = DI.interpolate_pixels_along_line(x0=int(self.x0), y0=int(self.y0), x1=int(self.x1),
                                                        y1=int(self.y1))

        for idx_ in radial_index:
            self.ori_X = int((self.orginal_video.shape[1] / self.pixmap_video.size().toTuple()[0]) * idx_[0])
            self.ori_Y = int((self.orginal_video.shape[2] / self.pixmap_video.size().toTuple()[1]) * idx_[1])
