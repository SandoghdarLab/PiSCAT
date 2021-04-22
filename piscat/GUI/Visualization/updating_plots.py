import matplotlib
from PyQt5 import QtCore, QtWidgets
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
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(ydata, 'r-')
        self.canvas.axes.grid()
        self.canvas.axes.set_ylabel('Pixel intensity')
        self.canvas.draw()
