import os

import pandas as pd

from piscat.Analysis.plot_protein_histogram import PlotProteinHistogram
from piscat.InputOutput import read_write_data, reading_videos
from piscat.InputOutput.camera_setting import CameraParameters
from piscat.InputOutput.read_write_data import load_dict_from_hdf5


class ReadProteinAnalysis(CameraParameters):
    def __init__(self, camera_Name="Photonfocus.json"):
        """
        This class is developed to read the results of different video analyses that were analyzed with the ``protein_analysis``
        function. In the end, this class has different methods to display histograms or save histogram results.

        Parameters
        ----------
        camera_Name: str
            Define the name of the camera configuration JSON file that will use for reading pixel size.
        """
        CameraParameters.__init__(self, name=camera_Name)

    def __call__(
        self,
        dirName,
        name_dir,
        video_frame_num,
        MinPeakWidth,
        MinPeakProminence=0,
        type_file="h5",
        his_setting=None,
    ):
        """
        By calling the object of class this function tries to read the result of the corresponding video it is defined
        with ``dirName`` and ``name_dir``. These results concatenated with previous results to use for plotting histogram.

        Parameters
        ----------
        dirName: str
            Path to the result of video analysis.

        name_dir: str
            The name of a folder that analysis of video was saved on it.

        video_frame_num: int
            The number of frames for the corresponding video.

        MinPeakWidth: int
            This is defined as the minimum V-shaped mouth that will use for prominence.

        MinPeakProminence: int
            This is defined as the minimum V-shape height that will use for prominence.

        type_file: str
            It defines the format of the file as the save file for analysis data ('HDF5', 'Matlab').

        his_setting: dict
            The dictionary is used to establish various parameters for localization-based filtering of PSF information
            in the histogram. The following gives an example of how this dictionary might be used:

            | his_setting = {'radius': 20, 'flag_localization_filter': True, 'centerOfImage_X': 34, 'centerOfImage_Y': 34}

        """

        df_histogram = reading_videos.DirectoryType(dirName, type_file=type_file).return_df()
        paths = df_histogram["Directory"].tolist()
        file_names = df_histogram["File"].tolist()

        self.his_ = PlotProteinHistogram(intersection_display_flag=False)
        if his_setting is None:
            self.his_.get_setting(
                radius=None,
                flag_localization_filter=False,
                centerOfImage_X=None,
                centerOfImage_Y=None,
            )
        else:
            self.his_.get_setting(
                radius=his_setting["radius"],
                flag_localization_filter=his_setting["flag_localization_filter"],
                centerOfImage_X=his_setting["centerOfImage_X"],
                centerOfImage_Y=his_setting["centerOfImage_X"],
            )

        self.dic_video_num_particle = {"Folder_name": [], "num_particles": []}

        num_particles = 0
        folder_cnt_ = 0

        for p_, n_ in zip(paths, file_names):
            if name_dir in p_:
                print(p_)
                hyperparameters = None
                PSFs_Particels_num = None
                df_json = reading_videos.DirectoryType(p_, type_file=".json").return_df()
                df_json["num_particles"] = None
                paths_json = df_json["Directory"].tolist()
                names_json = df_json["File"].tolist()
                for p_json, n_json in zip(paths_json, names_json):
                    if "flags" in n_json:
                        flags = read_write_data.read_json2dic(path=p_json, name=n_json)

                    elif "hyperparameters" in n_json:
                        hyperparameters = read_write_data.read_json2dic(path=p_json, name=n_json)

                    elif "PSFs_Particels_num" in n_json:
                        PSFs_Particels_num = read_write_data.read_json2dic(
                            path=p_json, name=n_json
                        )
                        num_particles = (
                            num_particles + PSFs_Particels_num["#Particles_after_V_shapeFilter"]
                        )
                        self.dic_video_num_particle["Folder_name"].append(p_)
                        self.dic_video_num_particle["num_particles"].append(
                            PSFs_Particels_num["#Particles_after_V_shapeFilter"]
                        )

                if hyperparameters is not None:
                    filename = os.path.join(p_, n_)
                    data_dic = load_dict_from_hdf5(filename)
                    if PSFs_Particels_num is not None:
                        if "#Totall_frame_num_DRA" in PSFs_Particels_num.keys():
                            video_frame_num = PSFs_Particels_num["#Totall_frame_num_DRA"]
                        else:
                            video_frame_num = video_frame_num

                        self.his_(
                            folder_name=n_,
                            particles=data_dic,
                            batch_size=hyperparameters["batch_size"],
                            video_frame_num=video_frame_num,
                            MinPeakWidth=MinPeakWidth,
                            MinPeakProminence=MinPeakProminence,
                            pixel_size=self.pixelSize,
                        )

                        folder_cnt_ += 1

        print("{} folders was read".format(folder_cnt_))
        print("{} PSFs discovered prior to trimming ".format(num_particles))

    def plot_localization_heatmap(
        self, pixelSize=None, unit="um", flag_in_time=False, time_delay=0.1, dir_name=None
    ):
        """
        This method plots heatmap of particle localization. The size of each disk depicts the movment of each particles during tracking.

        Parameters
        ----------
        pixelSize: float
            Camera pixel size.

        unit: str
            The axis unit.

        flag_in_time: bool
            In the case of True, show binding and unbinding events in time.

        time_delay: float
            Define the time delay between binding and unbinding events frames. This only works when `flag_in_time` is set to True.

        dir_name: str
            You can save time slap frames if you specify a save path.
        """
        if pixelSize is not None:
            self.pixelSize = pixelSize
        self.his_.plot_localization_heatmap(
            self.pixelSize,
            unit=unit,
            flag_in_time=flag_in_time,
            time_delay=time_delay,
            dir_name=dir_name,
        )

    def plot_hist(self, his_setting):
        """
        This method plots histograms for different contrast extraction methods for black PSFs, white PSFs and all together.

        Parameters
        ----------
        his_setting: dic
             This dictionary defines a histogram plotting setting. In the following you can see the example for it:

                | his_setting = {'bins': None, 'lower_limitation': -7e-4, 'upper_limitation': 7e-4,
                                   'Flag_GMM_fit': True, 'max_n_components': 3, 'step_range': 1e-6,
                                   'face': 'g', 'edge': 'k', 'scale': 1e1, 'external_GMM': False}
        """
        self.his_.plot_histogram(
            bins=his_setting["bins"],
            upper_limitation=his_setting["upper_limitation"],
            lower_limitation=his_setting["lower_limitation"],
            step_range=his_setting["step_range"],
            face=his_setting["face"],
            edge=his_setting["edge"],
            Flag_GMM_fit=his_setting["Flag_GMM_fit"],
            max_n_components=his_setting["max_n_components"],
            scale=his_setting["scale"],
            external_GMM=his_setting["external_GMM"],
        )

    def plot_hist_2Dfit(self, his_setting):
        """
        This method plots histograms for 2D Gaussian fitting contrast for black PSFs, white PSFs and all together.

        Parameters
        ----------
        his_setting: dic
            This dictionary defines a histogram plotting setting. In the following you can see the example for it:

               | his_setting = {'bins': None, 'lower_limitation': -7e-4, 'upper_limitation': 7e-4,
                                  'Flag_GMM_fit': True, 'max_n_components': 3, 'step_range': 1e-6, 'face': 'g', 'edge': 'k', }
        """
        self.his_.plot_fit_histogram(
            bins=his_setting["bins"],
            upper_limitation=his_setting["upper_limitation"],
            lower_limitation=his_setting["lower_limitation"],
            step_range=his_setting["step_range"],
            face=his_setting["face"],
            edge=his_setting["edge"],
            Flag_GMM_fit=his_setting["Flag_GMM_fit"],
            max_n_components=his_setting["max_n_components"],
        )

    def save_hist_data(self, dirName, name_dir, his_setting):
        """
        This function save the histogram data with HDF5 format.

        Parameters
        ----------
        dirName: str
            Path for saving data.

        name_dir: str
            Name that use for saving data.

        his_setting: dic
            This dictionary defines a histogram plotting setting. In the following you can see the example for it:

               | his_setting = {'lower_limitation': -7e-4, 'upper_limitation': 7e-4,
                                  'Flag_GMM_fit': True, 'max_n_components': 3}
        """
        self.his_.save_hist_data(
            dirName=dirName,
            name=name_dir,
            upper_limitation=his_setting["upper_limitation"],
            lower_limitation=his_setting["lower_limitation"],
            Flag_GMM_fit=his_setting["Flag_GMM_fit"],
            max_n_components=his_setting["max_n_components"],
        )
        df_num = pd.DataFrame.from_dict(self.dic_video_num_particle)
        read_write_data.save_df2csv(df=df_num, path=dirName, name="list_video_num_particles")
