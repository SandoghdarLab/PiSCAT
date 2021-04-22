import os
from piscat.Analysis.plot_protein_histogram import PlotProteinHistogram
from piscat.InputOutput import reading_videos
from piscat.InputOutput import read_write_data
from piscat.InputOutput.read_write_data import load_dict_from_hdf5
from piscat.InputOutput.camera_setting import CameraParameters


class ReadProteinAnalysis(CameraParameters):

    def __init__(self, camera_Name='Photonfocus.json'):
        """
        This class is developed to read the results of different video analyses that were analyzed with the ``protein_analysis``
        function. In the end, this class has different methods to display histograms or save histogram results.

        Parameters
        ----------
        camera_Name: str
            Define the name of the camera configuration JSON file that will use for reading pixel size.
        """
        CameraParameters.__init__(self, name=camera_Name)

    def __call__(self, dirName, name_dir, video_frame_num, MinPeakWidth, MinPeakProminence=0, type_file='h5'):
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

        """

        df_histogram = reading_videos.DirectoryType(dirName, type_file=type_file).return_df()
        paths = df_histogram['Directory'].tolist()
        file_names = df_histogram['File'].tolist()

        self.his_ = PlotProteinHistogram(intersection_display_flag=False)

        num_particles = 0
        folder_cnt_ = 0

        for p_, n_ in zip(paths, file_names):

            if name_dir in p_:
                hyperparameters = None
                PSFs_Particels_num = None
                df_json = reading_videos.DirectoryType(p_, type_file='.json').return_df()
                paths_json = df_json['Directory'].tolist()
                names_json = df_json['File'].tolist()
                for p_json, n_json in zip(paths_json, names_json):
                    if 'flags' in n_json:
                        flags = read_write_data.read_json2dic(path=p_json, name=n_json)

                    elif 'hyperparameters' in n_json:
                        hyperparameters = read_write_data.read_json2dic(path=p_json, name=n_json)

                    elif 'PSFs_Particels_num' in n_json:
                        PSFs_Particels_num = read_write_data.read_json2dic(path=p_json, name=n_json)
                        num_particles = num_particles + PSFs_Particels_num['#Particles_after_V_shapeFilter']

                if hyperparameters is not None:
                    filename = os.path.join(p_, n_)
                    data_dic = load_dict_from_hdf5(filename)
                    if PSFs_Particels_num is not None:
                        if '#Totall_frame_num_DRA' in PSFs_Particels_num.keys():
                            video_frame_num = PSFs_Particels_num['#Totall_frame_num_DRA']
                        else:
                            video_frame_num = video_frame_num

                        self.his_(folder_name=n_, particles=data_dic, batch_size=hyperparameters["batch_size"], video_frame_num=video_frame_num, MinPeakWidth=MinPeakWidth,
                             MinPeakProminence=MinPeakProminence,
                             pixel_size=self.pixelSize)

                        folder_cnt_ += 1

        print('{} folders was read'.format(folder_cnt_))
        print('{} df_PSFs should find in histogram '.format(num_particles))


    def plot_hist(self, his_setting):
        """
        This method plots histograms for different contrast extraction methods for black PSFs, white PSFs and all together.

        Parameters
        ----------
        his_setting: dic
             This dictionary defines a histogram plotting setting. In the following you can see the example for it:

                | his_setting = {'bins': None, 'lower_limitation': -7e-4, 'upper_limitation': 7e-4,
                                   'Flag_GMM_fit': True, 'max_n_components': 3, 'step_range': 1e-6, 'face': 'g', 'edge': 'k', }
        """
        self.his_.plot_histogram(bins=his_setting['bins'], upper_limitation=his_setting['upper_limitation'],
                                lower_limitation=his_setting['lower_limitation'], step_range=his_setting['step_range'],
                                face=his_setting['face'], edge=his_setting['edge'], Flag_GMM_fit=his_setting['Flag_GMM_fit'],
                                max_n_components=his_setting['max_n_components'])

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
        self.his_.plot_fit_histogram(bins=his_setting['bins'], upper_limitation=his_setting['upper_limitation'],
                            lower_limitation=his_setting['lower_limitation'], step_range=his_setting['step_range'],
                            face=his_setting['face'], edge=his_setting['edge'], Flag_GMM_fit=his_setting['Flag_GMM_fit'],
                            max_n_components=his_setting['max_n_components'])

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
        self.his_.save_hist_data(dirName=dirName, name=name_dir, upper_limitation=his_setting['upper_limitation'],
                                 lower_limitation=his_setting['lower_limitation'],
                                 Flag_GMM_fit=his_setting['Flag_GMM_fit'], max_n_components=his_setting['max_n_components'])

