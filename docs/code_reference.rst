=============
API Reference
=============


piscat.Analysis
---------------

.. autoclass:: piscat.Analysis.ReadProteinAnalysis
    :members:

.. autoclass:: piscat.Analysis.PlotProteinHistogram
    :members:

.. autofunction:: piscat.Analysis.protein_analysis


piscat.BackgroundCorrection
---------------------------

.. autoclass:: piscat.BackgroundCorrection.DifferentialRollingAverage
    :members:

.. autoclass:: piscat.BackgroundCorrection.NoiseFloor
    :members:


piscat.InputOutput
------------------

.. autoclass:: piscat.InputOutput.CameraParameters
    :members:

.. autoclass:: piscat.InputOutput.CPUConfigurations
    :members:

.. autoclass:: piscat.InputOutput.Image2Video
    :members:

.. autoclass:: piscat.InputOutput.StatusLine
    :members:

.. autofunction:: piscat.InputOutput.save_mat

.. autofunction:: piscat.InputOutput.read_mat

.. autofunction:: piscat.InputOutput.save_dic_to_hdf5

.. autofunction:: piscat.InputOutput.save_list_to_hdf5

.. autofunction:: piscat.InputOutput.load_dict_from_hdf5

.. autofunction:: piscat.InputOutput.save_df2csv

.. autofunction:: piscat.InputOutput.save_dic2json

.. autofunction:: piscat.InputOutput.read_json2dic

.. autofunction:: piscat.InputOutput.video_reader

.. autofunction:: piscat.InputOutput.read_binary

.. autofunction:: piscat.InputOutput.read_tif

.. autofunction:: piscat.InputOutput.read_avi

.. autofunction:: piscat.InputOutput.read_png

.. autoclass:: piscat.InputOutput.DirectoryType
    :members:

.. autofunction:: piscat.InputOutput.write_binary

.. autofunction:: piscat.InputOutput.write_MP4

.. autofunction:: piscat.InputOutput.write_GIF


piscat.Localization
-------------------

.. autoclass:: piscat.Localization.RadialCenter
    :members:

.. autoclass:: piscat.Localization.PSFsExtraction
    :members:

.. autoclass:: piscat.Localization.SpatialFilter
    :members:

.. autoclass:: piscat.Localization.DirectionalIntensity
    :members:

.. autofunction:: piscat.Localization.gaussian_2d

.. autofunction:: piscat.Localization.fit_2D_Gaussian_varAmp

.. autofunction:: piscat.Localization.blob_frst

.. autofunction:: piscat.Localization.feature2df

.. autofunction:: piscat.Localization.list2dataframe

piscat.Preproccessing
---------------------

.. autoclass:: piscat.Preproccessing.FFT2D
    :members:

.. autoclass:: piscat.Preproccessing.Filters
    :members:

.. autoclass:: piscat.Preproccessing.RadialVarianceTransform
    :members:

.. autoclass:: piscat.Preproccessing.FastRadialSymmetryTransform
    :members:

piscat.Trajectory
-----------------

.. autoclass:: piscat.Trajectory.Linking
    :members:

.. autoclass:: piscat.Trajectory.TemporalFilter
    :members:

.. autofunction:: piscat.Trajectory.protein_trajectories_list2dic


piscat.Visualization
--------------------

.. autoclass:: piscat.Visualization.ContrastAdjustment
    :members:

.. autoclass:: piscat.Visualization.Display
    :members:

.. autoclass:: piscat.Visualization.DisplayDataFramePSFsLocalization
    :members:

.. autoclass:: piscat.Visualization.DisplayPSFs_subplotLocalizationDisplay
    :members:

.. autoclass:: piscat.Visualization.DisplaySubplot
    :members:

.. autoclass:: piscat.Visualization.JupyterDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterDisplay_StatusLine
    :members:

.. autoclass:: piscat.Visualization.JupyterPSFs_localizationDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterPSFs_localizationPreviewDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterPSFs_subplotLocalizationDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterPSFs_TrackingDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterSelectedPSFs_localizationDisplay
    :members:

.. autoclass:: piscat.Visualization.JupyterPSFs_2_modality_subplotLocalizationDisplay
    :members:

.. autofunction:: piscat.Visualization.plot2df

.. autofunction:: piscat.Visualization.plot3

.. autofunction:: piscat.Visualization.plot_histogram
