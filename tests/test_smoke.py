import inspect
import piscat.Analysis
import piscat.BackgroundCorrection
import piscat.InputOutput
import piscat.Localization
import piscat.Preproccessing
import piscat.Trajectory
import piscat.Visualization
import piscat.GUI


def test_download_data():
    from piscat.InputOutput.read_write_data import download_url
    import os
    current_path = os.path.abspath(os.path.join('.'))
    dr_name = os.path.join(current_path, 'TestData')
    try:
        os.mkdir(dr_name)
        print("\nDirectory TestData Created ")
        print("\nStrats to download Test Data ")
        download_url(url='https://owncloud.gwdg.de/index.php/s/v0TJAYN52vtuXk0/download', save_path=current_path)
        print("\nTestData downloaded successfully")
    except FileExistsError:
        print("\nDirectory TestData already exists in package directory. If you are not sure Test "
              "Data download correctly please delete this directory")


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


