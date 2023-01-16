from piscat.Analysis.analysis_protein_histogram import ReadProteinAnalysis
import unittest
import os
from unittest.mock import patch

current_path = os.path.abspath(os.path.join('.'))


class ReadProteinAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.dir_name = os.path.join(current_path, 'TestData/Inject')
        self.name_dir = 'mFPN_batch_2500'

        self.his_setting = {'bins': 50, 'lower_limitation': -3e-3, 'upper_limitation': 3e-3,
                       'Flag_GMM_fit': True, 'max_n_components': 6, 'step_range': 1e-6, 'face': 'g', 'edge': 'k',
                       'scale': 1, 'external_GMM': False,
                       'radius': 29, 'flag_localization_filter': False, 'centerOfImage_X': 34, 'centerOfImage_Y': 34}

        self.hist_ = ReadProteinAnalysis()

    # def test_plots(self):
    #     self.hist_(dirName=self.dir_name, name_dir=self.name_dir,
    #           video_frame_num=None, MinPeakWidth=500, his_setting=self.his_setting, MinPeakProminence=0, type_file='h5')
    #     # plt.close()
    #     with patch("piscat.Analysis.plot_protein_histogram.plt.show") as show_patch:
    #         self.hist_.plot_hist(self.his_setting)
    #         assert show_patch.called
    #
    #     with patch("piscat.Analysis.plot_protein_histogram.plt.show") as show_patch:
    #         self.hist_.plot_localization_heatmap(pixelSize=1, unit='um', flag_in_time=False, time_delay=1, dir_name=None)
    #         assert show_patch.called




