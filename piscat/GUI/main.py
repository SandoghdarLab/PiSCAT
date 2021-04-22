import sys
import os
import multiprocessing
import warnings
import datetime
import time
import pkg_resources

from piscat.GUI.InputOutput import Reading
from piscat.GUI.CPU_Configurations import CPU_setting_wrapper
from piscat.GUI.Proccessing import FFT2D_GUI_wrapper
from piscat.GUI.VideoAnalysis import Analysis
from piscat.GUI.VideoAnalysis import AnalysisConstant
from piscat.GUI.ProgressBar.fun_progressBar import ProgressBar
from piscat.GUI.Projects.Protein import ProteinTabs
from piscat.GUI.Projects.Protein import Noise_Floor
from piscat.GUI.Memory.video_in_memory import VideoInMemory
from piscat.GUI.Visualization.fun_display_localization import Visulization_localization
from piscat._version import version

from PySide2 import QtGui
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2.QtCore import *
from PySide2.QtWidgets import QMenuBar
from functools import partial

warnings.filterwarnings('ignore')


class PiSCAT_GUI(QtWidgets.QMainWindow):

    progressChanged = QtCore.Signal(int)

    def __init__(self):
        super(PiSCAT_GUI, self).__init__()

        self.threadpool = QThreadPool()
        self.original_video = None
        self.dra_video = None

        # self.processor_setting = CPU_setting()
        self.FFT2D_GUI_wrapper = FFT2D_GUI_wrapper()

        self.initUI()

    def initUI(self):

        # defining the main window.
        window_icon = pkg_resources.resource_filename('piscat.GUI.icons', 'mpl.png')
        self.setWindowIcon(QtGui.QIcon(window_icon))
        self.setWindowTitle("PiSCAT")
        self.setStyleSheet('QMainWindow{background-color: darkgray;}')
        self.setGeometry(90, 90, 400, 1000)
        self.setFixedSize(340, 600)

        # defining the main frame as the central widget and give it a vertical layout.
        self.main_frame = QtWidgets.QFrame()
        self.setCentralWidget(self.main_frame)
        self.main_frame_layout = QtWidgets.QVBoxLayout()
        self.main_frame.setLayout(self.main_frame_layout)

        # ------------------------- Adding the main Logo ---------------------------------------------
        label = QtWidgets.QLabel(self)
        movie_path = pkg_resources.resource_filename('piscat.GUI.icons', 'movie.gif')
        loader_gif = QtGui.QMovie(movie_path)
        self.resize(loader_gif.currentPixmap().width(), loader_gif.currentPixmap().height())
        loader_gif.start()
        label.setMovie(loader_gif)
        self.main_frame_layout.addWidget(label)

        # ------------------defining the toolbox GroupBox and add it to the main frame.---------------
        self.toolbox_GB = QtWidgets.QGroupBox("Information")
        self.toolbox_GB.setFixedHeight(200)
        self.toolbox_GB_layout = QtWidgets.QVBoxLayout()
        self.toolbox_GB.setLayout(self.toolbox_GB_layout)
        self.main_frame_layout.addWidget(self.toolbox_GB)

        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.verticalScrollBar().minimum()
        # textEdit.setTextBackgroundColor(QtGui.QColor(140,140,0))
        self.toolbox_GB_layout.addWidget(self.textEdit)

        # adding different tab of "File" in menuBar.
        menubar = self.menuBar()

        #--------File------
        fileMenu = menubar.addMenu('&File')

        open_file = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'Open', self)
        # open_file.setShortcut('Ctrl+S')
        open_file.setStatusTip('Open new File')
        fileMenu.addAction(open_file)
        self.connect(open_file, QtCore.SIGNAL('triggered()'), self.video_loading_wrapper)

        imp = fileMenu.addMenu("Import")
        open_im2vid = imp.addAction("Image Sequence to video")
        self.connect(open_im2vid, QtCore.SIGNAL('triggered()'), self.image2video_loading_wrapper)

        load_python_script = imp.addAction("Run Python script")
        self.connect(load_python_script, QtCore.SIGNAL('triggered()'), Reading().run_py_script)
        # --------File------

        # --------Setting--------
        setting_menu = menubar.addMenu('&Setting')

        setting_cpu = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'CPU_Configurations', self)
        # setting_cpu.setShortcut('Ctrl+O')
        setting_cpu.setStatusTip('Open new File')
        setting_menu.addAction(setting_cpu)
        self.connect(setting_cpu, QtCore.SIGNAL('triggered()'), self.cpu_setting)
        # --------Setting--------

        # --------View --------
        self.list_available_video = {}
        view_menu = menubar.addMenu('&View')

        self.video_in_memory_flag = {'original_video': False, 'DRA_video': False, 'Protein_Tracking': False, 'GUV3D': False,
                                     'arc_cos': False, 'PSF Clustering': False, 'Template_matching': False}

        Display_video_in_memory = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'Load videos', self)
        # open_file.setShortcut('Ctrl+S')
        Display_video_in_memory.setStatusTip('Display')
        view_menu.addAction(Display_video_in_memory)
        self.connect(Display_video_in_memory, QtCore.SIGNAL('triggered()'), self.available_video_in_memory)
        # --------View --------

        # --------Process--------
        process_menu = menubar.addMenu('&Process')

        spectrum = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'FFT', self)
        # open_file.setShortcut('Ctrl+S')
        spectrum.setStatusTip('FFT')
        process_menu.addAction(spectrum)
        self.connect(spectrum, QtCore.SIGNAL('triggered()'), self.spectrum_init)

        image_calculator = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'Image calculator', self)
        # open_file.setShortcut('Ctrl+S')
        image_calculator.setStatusTip('Image calculator')
        process_menu.addAction(image_calculator)
        self.connect(image_calculator, QtCore.SIGNAL('triggered()'), self.analysis_wrapper)

        image_calculator_2 = QtWidgets.QAction(QtGui.QIcon('folder.ico'), 'Image calculator constant number', self)
        # open_file.setShortcut('Ctrl+S')
        image_calculator_2.setStatusTip('Image calculator constant number')
        process_menu.addAction(image_calculator_2)
        self.connect(image_calculator_2, QtCore.SIGNAL('triggered()'), self.analysis_wrapper_2)
        # --------Process--------

        # --------Analyze.--------
        analyze_menu = menubar.addMenu('&Analyze')

        noise_floor = QtWidgets.QAction('Noise Floor', self)
        # open_file.setShortcut('Ctrl+R')
        noise_floor.setStatusTip('Noise Floor.')
        self.connect(noise_floor, QtCore.SIGNAL('triggered()'), self.noise_floor)

        protein_track = QtWidgets.QAction("iSCAT Protein", self)
        protein_track.setStatusTip("iSCAT Protein")
        self.connect(protein_track, QtCore.SIGNAL('triggered()'), self.protein_projects)

        analyze_menu.addAction(noise_floor)
        analyze_menu.addAction(protein_track)
        # --------Analyze.--------

        # --------Information Box--------
        save_txt = QtWidgets.QAction('Report', self)
        # save_txt.setShortcut('Ctrl+Q')
        save_txt.setStatusTip('save history')
        save_txt.triggered.connect(self.save_text_history)
        toolbar_save = self.addToolBar('Save')
        toolbar_save.addAction(save_txt)

        clear_txt = QtWidgets.QAction('Clear', self)
        # exitAction.setShortcut('Ctrl+Q')
        clear_txt.setStatusTip('clear Information Box')
        clear_txt.triggered.connect(self.clear_plain_text)

        toolbar_clear = self.addToolBar('Clear')
        toolbar_clear.addAction(clear_txt)

        exitAction = QtWidgets.QAction('Exit', self)
        # exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.closeEvent)
        fileMenu.addAction(exitAction)
        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)
        # --------Information Box--------

        self.statusbar = self.statusBar()
        self.progressBar = QtWidgets.QProgressBar(self)
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.setValue(0)
        self.update_progressBar = ProgressBar(self.progressBar)

        self.show()

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def closeEvent(self, **kwargs):
        QtCore.QCoreApplication.instance().quit()
        print("closing PlaySetting")

    @QtCore.Slot()
    def set_new_text(self, input_text):
        if type(input_text) is str:
            self.textEdit.append(input_text)

    @QtCore.Slot()
    def set_plain_text(self, input_text):
        if type(input_text) is str:
            self.textEdit.insertPlainText(input_text)

    def clear_plain_text(self):
        self.textEdit.clear()

    def save_text_history(self):
        self.file_path = False
        self.file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "project_report",
                                                              QtCore.QDir.currentPath())
        timestr = time.strftime("%Y%m%d-%H%M%S")

        with open(self.file_path + '_' + timestr +'.txt', 'w') as outfile:
            outfile.write(str(self.textEdit.toPlainText()))

    def spectrum_init(self):
        try:
            self.FFT2D_GUI_wrapper.spectrum_input(self.reading.original_video)
            self.FFT2D_GUI_wrapper.spectrum()

        except AttributeError:
            self.msg_box = QtWidgets.QMessageBox()
            self.msg_box.setWindowTitle("Warning!")
            self.msg_box.setText("input video does not find!")
            self.msg_box.exec_()

    def analysis_wrapper(self):
        self.analysis_videos = Analysis()
        self.analysis_videos.update_output.connect(partial(self.updata_input_video, label='Video_analysis'))

    def analysis_wrapper_2(self):
        self.analysis_constant = AnalysisConstant()
        self.analysis_constant.update_output.connect(partial(self.updata_input_video, label='Video_analysis_constant'))

    def cpu_setting(self):
        self.processor_setting = CPU_setting_wrapper()

    def available_video_in_memory(self):
        self.vid_in_memory = VideoInMemory(self.list_available_video)
        self.vid_in_memory.display_trigger.connect(self.video_display)

    def video_loading_wrapper(self):
        self.reading = Reading()
        self.reading.update_output.connect(partial(self.updata_input_video, label='loading'))
        self.reading.read_video()

    def image2video_loading_wrapper(self):
        self.reading = Reading()
        self.reading.update_output.connect(partial(self.updata_input_video, label='loading'))
        self.reading.im2video()

    def noise_floor(self):
        self.noise_floor_ = Noise_Floor(video=self.original_video)

    def protein_projects(self):
        self.protein_gui = ProteinTabs(video_in=self.original_video, batch_size=None, object_update_progressBar=self.update_progressBar)
        self.protein_gui.new_update_DRA_video.connect(partial(self.updata_input_video, label='DRA', flag_DRA=True))
        self.protein_gui.show()

    def updata_input_video(self, data_in, label, flag_DRA=False):

        if flag_DRA:
            self.dra_video = data_in[0]
            self.batch_size = data_in[3]
            self.list_available_video['DRA_video'] = True
        else:
            self.original_video = data_in[0]
            self.list_available_video['original_video'] = True
        title = data_in[1]
        file_name = data_in[2]

        self.video_in_memory_flag[label] = True
        self.set_new_text('**** ' + str(datetime.datetime.now()) + ' ****')
        self.set_new_text(file_name)
        self.set_new_text(label + ' ' + title + " shape:" + str(self.original_video.shape))

    def video_display(self, data_in):
        if 'original_video' == data_in:
            self.visualization_ = Visulization_localization()
            self.visualization_.new_display(self.original_video, self.original_video, object=None, title='RAW', mask_status=False)

        elif 'DRA_video' == data_in:
            self.visualization_ = Visulization_localization()
            self.visualization_.new_display(self.dra_video, self.dra_video, object=None,
                                            title='DRA', mask_status=False)


def main():
    if sys.argv[0][-4:] == '.exe':
        setattr(sys, 'frozen', True)

    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    mpl_icon = pkg_resources.resource_filename('piscat.GUI.icons', 'mpl.png')
    app.setWindowIcon(QtGui.QIcon(mpl_icon))

    ex = PiSCAT_GUI()

    print('Starting PiSCAT version %s' % version)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
