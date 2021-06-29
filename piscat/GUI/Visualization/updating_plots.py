from PyQt5 import QtCore, QtWidgets

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class UpdatingPlots(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(UpdatingPlots, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)
        self.setCentralWidget(self.canvas)

    def update_plot(self, ydata):
        self.canvas.axes.clear()  # Clear the canvas.
        self.canvas.axes.plot(ydata, 'r.')
        self.canvas.axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        self.canvas.axes.grid()
        self.canvas.axes.set_ylabel('Pixel intensity')
        self.canvas.draw()


class UpdatingPlotsPyqtGraph(QtWidgets.QWidget):

    def __init__(self):
        super(UpdatingPlotsPyqtGraph, self).__init__()
        self.layout = QtWidgets.QVBoxLayout(self)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        self.graphWidget.showGrid(x=True, y=True)
        styles = {"color": "#f00", "font-size": "20px"}
        self.graphWidget.setLabel("left", "Pixel intensity", **styles)

        self.layout.addWidget(self.graphWidget)

        pen = pg.mkPen(color=(255, 102, 0))
        self.data_line = self.graphWidget.plot([1], [1], pen=pen)

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()
        print("closing plot")
