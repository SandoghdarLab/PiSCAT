import os
import pickle
import unittest

from piscat.Analysis.analysis_protein_histogram import ReadProteinAnalysis

current_path = os.path.abspath(os.path.join("."))


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


class ReadProteinAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.dir_name = os.path.join(current_path, "TestData/Inject")
        self.name_dir = "mFPN_batch_2500"

        self.his_setting = {
            "bins": 50,
            "lower_limitation": -3e-3,
            "upper_limitation": 3e-3,
            "Flag_GMM_fit": True,
            "max_n_components": 6,
            "step_range": 1e-6,
            "face": "g",
            "edge": "k",
            "scale": 1,
            "external_GMM": False,
            "radius": 29,
            "flag_localization_filter": False,
            "centerOfImage_X": 34,
            "centerOfImage_Y": 34,
        }

        self.hist_ = ReadProteinAnalysis()

    def test_plots(self):
        self.hist_(
            dirName=self.dir_name,
            name_dir=self.name_dir,
            video_frame_num=None,
            MinPeakWidth=500,
            his_setting=self.his_setting,
            MinPeakProminence=0,
            type_file="h5",
        )
        dir_name = os.path.join(current_path, "TestData/Video")
        fixture_name = os.path.join(dir_name, "PlotProteinHistogram.pck")
        loaded_plot_protein_histogram_obj = load_fixture(fixture_name)
        self.assertTrue(
            len(loaded_plot_protein_histogram_obj.t_len_linking)
            == len(self.hist_.his_.t_len_linking)
        )
        self.assertTrue(
            len(loaded_plot_protein_histogram_obj.t_linking_len_bright)
            == len(self.hist_.his_.t_linking_len_bright)
        )
        self.assertTrue(
            len(loaded_plot_protein_histogram_obj.t_linking_len_dark)
            == len(self.hist_.his_.t_linking_len_dark)
        )
        self.assertTrue(
            len(loaded_plot_protein_histogram_obj.t_mean_x_center_bright)
            == len(self.hist_.his_.t_mean_x_center_bright)
        )
