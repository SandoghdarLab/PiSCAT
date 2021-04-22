import inspect

import piscat.Analysis
import piscat.BackgroundCorrection
import piscat.InputOutput
import piscat.Localization
import piscat.Preproccessing
import piscat.Trajectory
import piscat.Visualization
import piscat.GUI
import multiprocessing
from PyQt5 import QtWidgets
import sys

def test_can_import_piscat_subpackages():

    assert inspect.isclass(piscat.Analysis.PlotProteinHistogram)
    assert inspect.isclass(piscat.Analysis.ReadProteinAnalysis)
    assert inspect.isfunction(piscat.Analysis.protein_analysis)

    assert inspect.isclass(piscat.BackgroundCorrection.DifferentialRollingAverage)
    assert inspect.isclass(piscat.BackgroundCorrection.NoiseFloor)

    assert inspect.isclass(piscat.InputOutput.download_tutorial_data)
    assert inspect.isclass(piscat.InputOutput.CPUConfigurations)
    assert inspect.isclass(piscat.InputOutput.CameraParameters)
    assert inspect.isclass(piscat.InputOutput.DirectoryType)
    assert inspect.isclass(piscat.InputOutput.StatusLine)
    assert inspect.isclass(piscat.InputOutput.Image2Video)

    assert inspect.isclass(piscat.Localization.DirectionalIntensity)
    assert inspect.isclass(piscat.Localization.PSFsExtraction)
    assert inspect.isclass(piscat.Localization.SpatialFilter)
    assert inspect.isclass(piscat.Localization.RadialCenter)
    assert inspect.isclass(piscat.Localization.CPUConfigurations)

    assert inspect.isclass(piscat.Preproccessing.ImagePatching)
    assert inspect.isclass(piscat.Preproccessing.Normalization)
    assert inspect.isclass(piscat.Preproccessing.Filters)
    assert inspect.isclass(piscat.Preproccessing.FFT2D)
    assert inspect.isclass(piscat.Preproccessing.FastRadialSymmetryTransform)
    assert inspect.isclass(piscat.Preproccessing.RadialVarianceTransform)

    assert inspect.isclass(piscat.Trajectory.TemporalFilter)
    assert inspect.isclass(piscat.Trajectory.Linking)
    assert inspect.isfunction(piscat.Trajectory.protein_trajectories_list2dic)

    assert inspect.isclass(piscat.Visualization.Display)
    assert inspect.isclass(piscat.Visualization.JupyterDisplay)
    assert inspect.isclass(piscat.Visualization.JupyterPSFs_localizationPreviewDisplay)
    assert inspect.isclass(piscat.Visualization.DisplayPSFs_subplotLocalizationDisplay)
    assert inspect.isclass(piscat.Visualization.JupyterSelectedPSFs_localizationDisplay)
    assert inspect.isclass(piscat.Visualization.ContrastAdjustment)
    assert inspect.isclass(piscat.Visualization.PrintColors)


def test_gui_can_be_constructed():
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    piscat.GUI.PiSCAT_GUI()
    #sys.exit(app.exec_())  # for checking only
