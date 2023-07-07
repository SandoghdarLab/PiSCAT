import datetime
import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from functools import partial

import pkg_resources
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QThreadPool

from piscat import version
from piscat.GUI.BatchAnalysis.batch_data import BatchAnalysis
from piscat.GUI.CPU_Configurations import CPU_setting_wrapper
from piscat.GUI.InputOutput import Reading
from piscat.GUI.InputOutput.help import Help
from piscat.GUI.Memory.video_in_memory import VideoInMemory
from piscat.GUI.Proccessing import FFT2D_GUI_wrapper
from piscat.GUI.ProgressBar.fun_progressBar import ProgressBar
from piscat.GUI.Projects.iPSF.iPSF_model import GUI_iPSF
from piscat.GUI.Projects.Protein import Noise_Floor, ProteinTabs
from piscat.GUI.VideoAnalysis import Analysis, AnalysisConstant
from piscat.GUI.Visualization.fun_display_localization import Visulization_localization

warnings.filterwarnings("ignore")


class PiSCAT_GUI(QtWidgets.QMainWindow):
    progressChanged = QtCore.Signal(int)

    def __init__(self):
        super(PiSCAT_GUI, self).__init__()

        self.threadpool = QThreadPool()
        self.original_video = None
        self.dra_video = None

        self.FFT2D_GUI_wrapper = FFT2D_GUI_wrapper()

        self.initUI()

    def initUI(self):
        # defining the main window.
        window_icon = pkg_resources.resource_filename("piscat.GUI.icons", "mpl.png")
        main_icon = QtGui.QIcon(window_icon)
        self.setWindowIcon(main_icon)
        self.setWindowTitle("PiSCAT")
        self.setStyleSheet("QMainWindow{background-color: darkgray;}")
        self.setGeometry(90, 90, 400, 1000)
        self.setFixedSize(340, 600)

        # defining the main frame as the central widget and give it a vertical layout.
        self.main_frame = QtWidgets.QFrame()
        self.setCentralWidget(self.main_frame)
        self.main_frame_layout = QtWidgets.QVBoxLayout()
        self.main_frame.setLayout(self.main_frame_layout)

        # ---------- Adding the main Logo ----------
        label_logo = QtWidgets.QLabel(self)
        logo_piscat_path = pkg_resources.resource_filename(
            "piscat.GUI.icons", "PiSCAT_logo_bg.png"
        )
        loader_logo = QtGui.QPixmap(logo_piscat_path)
        loader_logo = loader_logo.scaled(320, 100)
        label_logo.setPixmap(loader_logo)
        self.main_frame_layout.addWidget(label_logo)

        label = QtWidgets.QLabel(self)
        movie_path = pkg_resources.resource_filename("piscat.GUI.icons", "movie.gif")
        loader_gif = QtGui.QMovie(movie_path)
        self.resize(loader_gif.currentPixmap().width(), loader_gif.currentPixmap().height())
        loader_gif.start()
        label.setMovie(loader_gif)
        self.main_frame_layout.addWidget(label)

        # ---------- defining the toolbox GroupBox and add it to the main frame. ----------
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

        # --------File------
        fileMenu = menubar.addMenu("&File")

        open_file = QtGui.QAction(QtGui.QIcon("folder.ico"), "Open", self)
        # open_file.setShortcut('Ctrl+S')
        open_file.setStatusTip("Open new File")
        fileMenu.addAction(open_file)
        self.connect(open_file, QtCore.SIGNAL("triggered()"), self.video_loading_wrapper)

        imp = fileMenu.addMenu("Import")
        open_im2vid = imp.addAction("Image Sequence to video")
        self.connect(open_im2vid, QtCore.SIGNAL("triggered()"), self.image2video_loading_wrapper)

        open_tiff_vid = imp.addAction("TIFF")
        self.connect(open_tiff_vid, QtCore.SIGNAL("triggered()"), self.tiff_video_loading_wrapper)

        load_python_script = imp.addAction("Run Python script")
        self.connect(load_python_script, QtCore.SIGNAL("triggered()"), Reading().run_py_script)

        Batch_analysis = imp.addAction("Run Batch analysis")
        self.connect(Batch_analysis, QtCore.SIGNAL("triggered()"), self.batch_analysis)
        # --------File------

        # --------Setting--------
        setting_menu = menubar.addMenu("&Setting")

        setting_cpu = QtGui.QAction(QtGui.QIcon("folder.ico"), "CPU_Configurations", self)
        # setting_cpu.setShortcut('Ctrl+O')
        setting_cpu.setStatusTip("Open new File")
        setting_menu.addAction(setting_cpu)
        self.connect(setting_cpu, QtCore.SIGNAL("triggered()"), self.cpu_setting)
        # --------Setting--------

        # --------View --------
        self.list_available_video = {}
        view_menu = menubar.addMenu("&View")

        self.video_in_memory_flag = {
            "original_video": False,
            "DRA_video": False,
            "Protein_Tracking": False,
            "GUV3D": False,
            "arc_cos": False,
            "PSF Clustering": False,
            "Template_matching": False,
        }

        Display_video_in_memory = QtGui.QAction(QtGui.QIcon("folder.ico"), "Loaded videos", self)
        # open_file.setShortcut('Ctrl+S')
        Display_video_in_memory.setStatusTip("Display")
        view_menu.addAction(Display_video_in_memory)
        self.connect(
            Display_video_in_memory, QtCore.SIGNAL("triggered()"), self.available_video_in_memory
        )
        # --------View --------

        # --------Process--------
        process_menu = menubar.addMenu("&Process")

        spectrum = QtGui.QAction(QtGui.QIcon("folder.ico"), "FFT", self)
        # open_file.setShortcut('Ctrl+S')
        spectrum.setStatusTip("FFT")
        process_menu.addAction(spectrum)
        self.connect(spectrum, QtCore.SIGNAL("triggered()"), self.spectrum_init)

        image_calculator = QtGui.QAction(QtGui.QIcon("folder.ico"), "Image calculator", self)
        # open_file.setShortcut('Ctrl+S')
        image_calculator.setStatusTip("Image calculator")
        process_menu.addAction(image_calculator)
        self.connect(image_calculator, QtCore.SIGNAL("triggered()"), self.analysis_wrapper)

        image_calculator_2 = QtGui.QAction(
            QtGui.QIcon("folder.ico"), "Image calculator constant number", self
        )
        # open_file.setShortcut('Ctrl+S')
        image_calculator_2.setStatusTip("Image calculator constant number")
        process_menu.addAction(image_calculator_2)
        self.connect(image_calculator_2, QtCore.SIGNAL("triggered()"), self.analysis_wrapper_2)
        # --------Process--------

        # --------Analyze.--------
        analyze_menu = menubar.addMenu("&Analyze")

        noise_floor = QtGui.QAction("Noise Floor", self)
        # open_file.setShortcut('Ctrl+R')
        noise_floor.setStatusTip("Noise Floor.")
        self.connect(noise_floor, QtCore.SIGNAL("triggered()"), self.noise_floor)

        protein_track = QtGui.QAction("iSCAT Protein", self)
        protein_track.setStatusTip("iSCAT Protein")
        self.connect(protein_track, QtCore.SIGNAL("triggered()"), self.protein_projects)

        iPSF_model = QtGui.QAction("iPSF model", self)
        iPSF_model.setStatusTip("iPSF model")
        self.connect(iPSF_model, QtCore.SIGNAL("triggered()"), self.iPSF_projects)

        analyze_menu.addAction(noise_floor)
        analyze_menu.addAction(protein_track)
        analyze_menu.addAction(iPSF_model)

        # --------Analyze.--------

        # --------Help-----------
        help_menu = menubar.addMenu("&Help")

        help = QtGui.QAction("Help", self)
        # open_file.setShortcut('Ctrl+R')
        help.setStatusTip("Help.")
        self.connect(help, QtCore.SIGNAL("triggered()"), self.help)

        tutorials = QtGui.QAction("tutorials", self)
        # open_file.setShortcut('Ctrl+R')
        tutorials.setStatusTip("tutorials.")
        self.connect(tutorials, QtCore.SIGNAL("triggered()"), self.tutorials)

        about = QtGui.QAction("About", self)
        # open_file.setShortcut('Ctrl+R')
        about.setStatusTip("About.")
        self.connect(about, QtCore.SIGNAL("triggered()"), self.about)

        help_menu.addAction(help)
        # help_menu.addAction(tutorials)
        help_menu.addAction(about)
        # --------About-----------

        # --------Information Box--------
        save_txt = QtGui.QAction("Report", self)
        # save_txt.setShortcut('Ctrl+Q')
        save_txt.setStatusTip("save history")
        save_txt.triggered.connect(self.save_text_history)
        toolbar_save = self.addToolBar("Save")
        toolbar_save.addAction(save_txt)

        clear_txt = QtGui.QAction("Clear", self)
        # exitAction.setShortcut('Ctrl+Q')
        clear_txt.setStatusTip("clear Information Box")
        clear_txt.triggered.connect(self.clear_plain_text)

        toolbar_clear = self.addToolBar("Clear")
        toolbar_clear.addAction(clear_txt)

        exitAction = QtGui.QAction("Exit", self)
        # exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(self.closeEvent)
        fileMenu.addAction(exitAction)
        toolbar = self.addToolBar("Exit")
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
        print("Destructor called, Employee deleted.")

    def closeEvent(self, event):
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
        self.file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "project_report", QtCore.QDir.currentPath()
        )
        timestr = time.strftime("%Y%m%d-%H%M%S")

        with open(self.file_path + "_" + timestr + ".txt", "w") as outfile:
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
        self.analysis_videos.update_output.connect(
            partial(self.updata_input_video, label="Video_analysis")
        )

    def analysis_wrapper_2(self):
        self.analysis_constant = AnalysisConstant()
        self.analysis_constant.update_output.connect(
            partial(self.updata_input_video, label="Video_analysis_constant")
        )

    def cpu_setting(self):
        self.processor_setting = CPU_setting_wrapper()

    def available_video_in_memory(self):
        self.vid_in_memory = VideoInMemory(self.list_available_video)
        self.vid_in_memory.display_trigger.connect(self.video_display)

    def video_loading_wrapper(self):
        self.reading = Reading()
        self.reading.update_output.connect(partial(self.updata_input_video, label="loading"))
        self.reading.read_video()

    def image2video_loading_wrapper(self):
        self.reading = Reading()
        self.reading.update_output.connect(partial(self.updata_input_video, label="loading"))
        self.reading.im2video()

    def tiff_video_loading_wrapper(self):
        self.reading = Reading()
        self.reading.update_output.connect(partial(self.updata_input_video, label="loading"))
        self.reading.tiff_video()

    def noise_floor(self):
        self.noise_floor_ = Noise_Floor(video=self.original_video)

    def protein_projects(self):
        self.protein_gui = ProteinTabs(
            video_in=self.original_video,
            batch_size=None,
            object_update_progressBar=self.update_progressBar,
        )
        self.protein_gui.new_update_DRA_video.connect(
            partial(self.updata_input_video, label="DRA", flag_DRA=True)
        )
        self.protein_gui.show()

    def iPSF_projects(self):
        self.iPSF_gui = GUI_iPSF()
        self.iPSF_gui.display_trigger.connect(self.video_display)
        self.iPSF_gui.show()

    def updata_input_video(self, data_in, label, flag_DRA=False):
        if flag_DRA:
            self.dra_video = data_in[0]
            self.batch_size = data_in[3]
            self.list_available_video["DRA_video"] = True
            status_line_info = None
        else:
            try:
                status_line_info = data_in[3]
            except:
                status_line_info = None
            self.original_video = data_in[0]
            self.list_available_video["original_video"] = True
        title = data_in[1]
        file_name = data_in[2]

        self.video_in_memory_flag[label] = True
        self.set_new_text("**** " + str(datetime.datetime.now()) + " ****")
        self.set_new_text(file_name)
        self.set_new_text(label + " " + title + " shape:" + str(self.original_video.shape))

        if status_line_info is not None and status_line_info["status_line_position"] != "":
            self.set_new_text(
                "---Status line detected in " + status_line_info["status_line_position"] + "---"
            )
        elif status_line_info is not None and status_line_info["status_line_position"] == "":
            self.set_new_text("---Status line does not detect---")

    def video_display(self, data_in):
        if "original_video" == data_in:
            self.visualization_ = Visulization_localization()
            self.visualization_.new_display(
                self.original_video,
                self.original_video,
                object=None,
                title="RAW",
                mask_status=False,
            )

        elif "DRA_video" == data_in:
            self.visualization_ = Visulization_localization()
            self.visualization_.new_display(
                self.dra_video, self.dra_video, object=None, title="DRA", mask_status=False
            )

        elif "iPSF_Model_I" == data_in[0] or "iPSF_Model_II" == data_in[0]:
            self.visualization_ = Visulization_localization()
            self.visualization_.new_display(
                data_in[1],
                data_in[1],
                object=self.iPSF_gui,
                title="iPSF Model",
                mask_status=False,
            )

    def tutorials(self):
        subprocess.run("python -m piscat.Tutorials", shell=True)

    def about(self):
        self.msg_box = QtWidgets.QMessageBox()
        self.msg_box.setWindowTitle("About PiSCAT")
        self.msg_box.setText("PiSCAT version %s" % version)
        self.msg_box.exec_()

    def help(self):
        self.help_windows = Help()

    def open_dialog_box(self):
        full_path, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Multi File", os.getcwd()
        )

        # Store files in list
        self.file_list = []
        [self.file_list.append(path) for path in full_path]

    def batch_analysis(self):
        self.batch_analysis_gui = BatchAnalysis(object_update_progressBar=self.update_progressBar)
        self.batch_analysis_gui.show()


def main():
    if sys.argv[0][-4:] == ".exe":
        sys.frozen = True

    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    mpl_icon = pkg_resources.resource_filename("piscat.GUI.icons", "mpl.png")
    app.setWindowIcon(QtGui.QIcon(mpl_icon))

    ex = PiSCAT_GUI()

    print("Starting PiSCAT version %s" % version)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
