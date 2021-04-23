import numpy as np
import scipy.optimize
from ipywidgets import widgets
from ipywidgets import Layout, interact
from matplotlib import pyplot as plt, cm as cm
from matplotlib.patches import Circle, Arrow, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from skimage.filters import median
from skimage.morphology import disk

from piscat.Localization import gaussian_2D_fit
from piscat.Localization import directional_intensity
from piscat.Trajectory.data_handeling import protein_trajectories_list2dic
from piscat.InputOutput import read_status_line


class JupyterDisplay():

    def __init__(self, video, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=1):
        """
        This class displays the video in jupyter notebook.

        Parameters
        ----------
        video:(NDArray)
            Input video.

        median_filter_flag: bool
            In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
            It defines the colormap for visualization.

        imgSizex: int
            Image length size.

        imgSizey: int
            Image width size.

        IntSlider_width: str
            Size of slider

        step: int
            Stride between visualization frames.

        """
        self.color = color
        self.video = video
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.median_filter_flag = median_filter_flag

        interact(self.display, frame=widgets.IntSlider(min=0, max=self.video.shape[0] - 1, step=step, value=10,
                                                       layout=Layout(width=IntSlider_width),
                                                       readout_format='10', continuous_update=False,
                                                       description='Frame:'))

    def display(self, frame):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        ax = fig.add_axes([0, 0, 1, 1])

        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(frame), :, :], 3)
        else:
            frame_v = self.video[int(frame), :, :]

        myplot = ax.imshow(frame_v, cmap=self.color)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(myplot, cax=cax)

        plt.show()


class JupyterDisplay_StatusLine():

    def __init__(self, video, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=1):
        """
        This class displays the video in the Jupyter notebook interactively while highlight status line

        Parameters
        ----------
        video: NDArray
            Input video.

        median_filter_flag: bool
            In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
            It defines the colormap for visualization.

        imgSizex: int
            Image length size.

        imgSizey: int
            Image width size.

        IntSlider_width: str
            Size of slider.

        step: int
            Stride between visualization frames.
        """
        self.color = color
        self.video = video
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.median_filter_flag = median_filter_flag

        status_ = read_status_line.StatusLine(video)
        self.video_remove, status_information = status_.find_status_line()
        self.statusLine_position = status_information['status_line_position']

        interact(self.display, frame=widgets.IntSlider(min=0, max=self.video.shape[0] - 1, step=step, value=10,
                                                       layout=Layout(width=IntSlider_width),
                                                       readout_format='10', continuous_update=False,
                                                       description='Frame:'))

    def display(self, frame):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        ax = fig.add_axes([0, 0, 1, 1])

        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(frame), :, :], 3)
            frame_v_rm = median_filter(self.video_remove[int(frame), :, :], 3)

        else:
            frame_v = self.video[int(frame), :, :]
            frame_v_rm = self.video_remove[int(frame), :, :]


        myplot_rm = ax.imshow(frame_v_rm, cmap=self.color)
        myplot = ax.imshow(frame_v, cmap=self.color)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(myplot_rm, cax=cax)

        if self.statusLine_position == 'column':
            rect = Rectangle((0, self.video.shape[2]), self.video.shape[2], 1, linewidth=10, edgecolor='r',
                                  facecolor='none')
            ax.add_patch(rect)
        elif self.statusLine_position == 'row':
            rect = Rectangle((self.video.shape[1], 0), 1, self.video.shape[1], linewidth=10, edgecolor='r',
                                  facecolor='none')
            ax.add_patch(rect)
        plt.show()


class JupyterPSFs_localizationDisplay():

    def __init__(self, video, df_PSFs, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=1,  value=0):
        """
        This class displays the video in the Jupyter notebook interactively while highlight PSFs.

        Parameters
        ----------
        video: NDArray
           Input video.

        df_PSFs: panda data_frame
            Data Frames that contains the location of PSFs.

        median_filter_flag: bool
           In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
           It defines the colormap for visualization.

        imgSizex: int
           Image length size.

        imgSizey: int
           Image width size.

        IntSlider_width: str
           Size of slider.

        step: int
           Stride between visualization frames.

        value: int
           Initial frame value for visualization
       """
        self.video = video
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.df_PSFs = df_PSFs
        self.median_filter_flag = median_filter_flag
        self.color = color

        interact(self.show_psf, frame_number=widgets.IntSlider(min=0, max=self.video.shape[0] - 1, step=step, value=value,
                                                               readout_format='1', continuous_update=False, layout=Layout(width=IntSlider_width),
                                                               description='Frame:'))

    def show_psf(self, frame_number):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        ax = fig.add_axes([0, 0, 1, 1])
        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(frame_number), :, :], 3)
        else:
            frame_v = self.video[int(frame_number), :, :]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        myplot = ax.imshow(frame_v, cmap=self.color)
        plt.colorbar(myplot, cax=cax)

        particle = self.df_PSFs.loc[self.df_PSFs['frame'] == frame_number]
        particle_X = particle['x'].tolist()
        particle_Y = particle['y'].tolist()
        particle_sigma = particle['sigma'].tolist()

        for j_ in range(len(particle_X)):
            y = int(particle_Y[j_])
            x = int(particle_X[j_])
            sigma = particle_sigma[j_]
            ax.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))
        plt.show()


class JupyterPSFs_localizationPreviewDisplay():

    def __init__(self, video, df_PSFs, title='', frame_num=None, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=1):
        """
        This class displays the video in the Jupyter notebook interactively while highlight PSFs.

        Parameters
        ----------
        video: NDArray
           Input video.

        df_PSFs: panda data_frame
            Data Frames that contains the location of PSFs.

        title: str
            It defines title of plot.

        frame_num: list
            list of frame that we want to see preview of localization.

        median_filter_flag: bool
           In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
           It defines the colormap for visualization.

        imgSizex: int
           Image length size.

        imgSizey: int
           Image width size.

        IntSlider_width: str
           Size of slider.

        step: int
           Stride between visualization frames.
       """
        self.video = video
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.df_PSFs = df_PSFs
        self.median_filter_flag = median_filter_flag
        self.color = color
        if self.df_PSFs is not None:
            self.frame_num = np.sort(np.unique(df_PSFs['frame'].tolist()))
        else:
            self.frame_num = frame_num
        self.currentFrame = 0
        self.IntSlider_width = IntSlider_width
        self.step = step
        self.title = title

    def display_run(self):
        interact(self.show_psf,
                 index=widgets.IntSlider(min=0, max=len(self.frame_num) - 1, step=self.step, value=self.currentFrame,
                                         readout_format='1', continuous_update=False,
                                         layout=Layout(width=self.IntSlider_width),
                                         description='index:'))

    def show_psf(self, index):
        frame_number = self.frame_num[index]
        self.currentFrame = frame_number
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        ax = fig.add_axes([0, 0, 1, 1])
        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(frame_number), :, :], 3)
        else:
            frame_v = self.video[int(frame_number), :, :]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        myplot = ax.imshow(frame_v, cmap=self.color)
        ax.set_title(self.title)
        plt.colorbar(myplot, cax=cax)

        if self.df_PSFs is not None:
            particle = self.df_PSFs.loc[self.df_PSFs['frame'] == frame_number]
            particle_X = particle['x'].tolist()
            particle_Y = particle['y'].tolist()
            particle_sigma = particle['sigma'].tolist()

            for j_ in range(len(particle_X)):
                y = int(particle_Y[j_])
                x = int(particle_X[j_])
                sigma = particle_sigma[j_]
                ax.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))
        plt.show()

    def set_df(self, new_df):
        self.df_PSFs = new_df


class JupyterPSFs_subplotLocalizationDisplay():

    def __init__(self, list_videos, list_df_PSFs, numRows, numColumns, list_titles=None,
                 median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, IntSlider_width='500px', step=1, value=0):
        """
        This class shows several videos (with the same number of frames) at once in the Jupyter notebook interactively while highlight localize PSFs.

        Parameters
        ----------
        list_videos: list of NDArray
          List of videos.

        list_df_PSFs: list panda data_frame
           List Data Frames that contains the location of PSFs for each video.

        numRows: int
            It defines number of rows in sub-display.

        numColumns: int
            It defines number of columns in sub-display.

        list_titles: list str
            List of titles for each sub plot.

        median_filter_flag: bool
          In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
          It defines the colormap for visualization.

        imgSizex: int
          Image length size.

        imgSizey: int
          Image width size.

        IntSlider_width: str
          Size of slider

        step: int
          Stride between visualization frames.

        value: int
            Initial frame value for visualization

        """
        self.list_video = list_videos
        self.numVideos = len(list_videos)
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.list_df_PSFs = list_df_PSFs
        self.median_filter_flag = median_filter_flag
        self.color = color
        self.numRows = numRows
        self.numColumns = numColumns

        if list_titles is None:
            self.list_titles = [None for _ in range(self.numVideos)]
        else:
            self.list_titles = list_titles

        max_numberFrames = np.max([vid_.shape[0] for vid_ in list_videos])
        interact(self.show_psf, frame_number=widgets.IntSlider(min=0, max=max_numberFrames, step=step, value=value,
                                                               readout_format='1', continuous_update=False, layout=Layout(width=IntSlider_width),
                                                               description='Frame:'))

    def show_psf(self, frame_number):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        grid = plt.GridSpec(self.numRows, self.numColumns, hspace=0.3, wspace=0.3)

        imgGrid_list = []
        for i in range(self.numRows):
            for j in range(self.numColumns):
                imgGrid_list.append(fig.add_subplot(grid[i, j]))

        for img_, tit_, img_grid_, df_PSFs in zip(self.list_video, self.list_titles, imgGrid_list, self.list_df_PSFs):

            if self.median_filter_flag:

                frame_v = median_filter(img_[int(frame_number), :, :], 3)
            else:
                frame_v = img_[int(frame_number), :, :]

            divider = make_axes_locatable(img_grid_)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            myplot = img_grid_.imshow(frame_v, cmap=self.color)
            img_grid_.axis('off')
            plt.colorbar(myplot, cax=cax)
            if tit_ is not None:
                img_grid_.set_title(tit_)

            particle = df_PSFs.loc[df_PSFs['frame'] == frame_number]
            particle_X = particle['x'].tolist()
            particle_Y = particle['y'].tolist()
            particle_sigma = particle['sigma'].tolist()

            for j_ in range(len(particle_X)):
                y = int(particle_Y[j_])
                x = int(particle_X[j_])
                sigma = particle_sigma[j_]
                img_grid_.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))
        plt.show()


class JupyterPSFs_TrackingDisplay():

    def __init__(self, video, df_PSFs, median_filter_flag=False, step=1, color='gray', imgSizex=5, imgSizey=5,):
        """
        This class displays video in the Jupyter notebook interactively while highlighting PSFs with trajectories.

        Parameters
        ----------
        video: NDArray
           Input video.

        df_PSFs: panda data_frame
            Data Frames that contains the location of PSFs.

        median_filter_flag: bool
           In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
           It defines the colormap for visualization.

        imgSizex: int
           Image length size.

        imgSizey: int
           Image width size.

        IntSlider_width: str
           Size of slider.

        step: int
           Stride between visualization frames.
        """

        self.video = video
        self.df_PSFs = df_PSFs
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.df_PSFs = df_PSFs
        self.median_filter_flag = median_filter_flag
        self.color = color
        self.median_filter_flag = median_filter_flag
        if self.df_PSFs.particle.isnull().any():
            self.df_PSFs['particle'] = 0
            self.list_particles_idx = self.df_PSFs.particle.unique()
        else:
            self.list_particles_idx = self.df_PSFs.particle.unique()

        colors_ = cm.autumn(np.linspace(0, 1, len(self.list_particles_idx)))

        self.colors = colors_[0:len(self.list_particles_idx), :]
        interact(self.show_psf, frame_number=widgets.IntSlider(min=0, max=self.video.shape[0] - 1, step=step, value=0,
                    readout_format='1', continuous_update=False, description='Frame:'))

    def show_psf(self, frame_number):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))
        ax = fig.add_axes([0, 0, 1, 1])

        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(frame_number), :, :], 3)
        else:
            frame_v = self.video[int(frame_number), :, :]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        myplot = ax.imshow(frame_v, cmap=self.color)
        plt.colorbar(myplot, cax=cax)

        particle = self.df_PSFs.loc[self.df_PSFs['frame'] == frame_number]
        particle_X = particle['x'].tolist()
        particle_Y = particle['y'].tolist()
        particle_sigma = particle['sigma'].tolist()
        particle_labels = particle['particle'].tolist()

        if len(particle_labels) != 0:

            for j_, p_l in zip(range(len(particle_X)), particle_labels):
                y = int(particle_Y[j_])
                x = int(particle_X[j_])
                sigma = particle_sigma[j_]
                ax = plt.gca()
                ax.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor=self.colors[int(p_l)], facecolor='none', linewidth=2))

                all_particle_ = self.df_PSFs.loc[self.df_PSFs['particle'] == p_l]
                particle_f_ = all_particle_['frame'].tolist()
                particle_X_= all_particle_['x'].tolist()
                particle_Y_ = all_particle_['y'].tolist()
                particle_sigma_ = all_particle_['sigma'].tolist()
                for f_, x_, y_, sigma_ in zip(particle_f_, particle_X_, particle_Y_, particle_sigma_):
                    if f_ <= frame_number:
                        ax.add_patch(Circle((int(x_), int(y_)), radius=0.1, edgecolor=self.colors[int(p_l)], facecolor='none', linewidth=1))
                        # ax.add_collection((x_, y_), autolim=True, cmap=self.colors [p_l])
        plt.show()


class JupyterSelectedPSFs_localizationDisplay():

    def __init__(self, video, particles, particles_num='#0', frame_extend=0, median_filter_flag=False, flag_fit2D=False,
                 color='gray', imgSizex=10, imgSizey=10, IntSlider_width='500px', step=1):
        """
        This class interactively shows video in a Jupyter notebook while highlighting PSFs based on ID.

        Parameters
        ----------
        video: NDArray
           Input video.

        particles: dic
             Dictionary similar to the following structures.:

            | {"#0": {'intensity_horizontal': ..., 'intensity_vertical': ..., ..., 'particle_ID': ...},
                "#1": {}, ...}

        particles_num: str
            Choose the corresponding key in the particles dictionary.

        frame_extend: int
            Display particle for ``frame_extend`` before and after segmented ones. In case there are not enough frames before/after
            , it shows only for the number of existing frames.

        flag_fit2D: bool
            It activate 2D-Gaussian fit to extract fitting information of selected PSF.

        median_filter_flag: bool
           In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
           It defines the colormap for visualization.

        imgSizex: int
           Image length size.

        imgSizey: int
           Image width size.

        IntSlider_width: str
           Size of slider.

        step: int
           Stride between visualization frames.
        """

        if type(particles) is list:
            particles = protein_trajectories_list2dic(particles)

        if type(particles) is dict:
            for key, item in particles.items():
                if key == particles_num:
                    intensity_horizontal = item['intensity_horizontal']
                    intensity_vertical = item['intensity_vertical']
                    center_int = item['center_int']
                    center_int_flow = item['center_int_flow']
                    self.frame_number_ = item['frame_number']
                    self.sigma = item['sigma']
                    self.x_center = item['x_center']
                    self.y_center = item['y_center']
                    self.particle_ID = item['particle_ID']

        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.video = video
        self.color = color
        self.flag_fit2D = flag_fit2D
        self.median_filter_flag = median_filter_flag
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))

        video_frameNum = video.shape[0]
        max_particle_frame = np.max(self.frame_number_)
        min_particle_frame = np.min(self.frame_number_)

        max_extend_particle_frame = np.min([max_particle_frame+frame_extend, video_frameNum])
        min_extend_particle_frame = np.max([min_particle_frame-frame_extend, 0])

        self.frame_number = list(range(min_extend_particle_frame, max_extend_particle_frame, 1))
        interact(self.show_psf,
                 f_num=widgets.IntSlider(min=self.frame_number[0], max=self.frame_number[-1] - 1, step=step, value=10,
                                         readout_format='1', continuous_update=False, layout=Layout(width=IntSlider_width),
                                         description='Frame:'))

    def show_psf(self, f_num):

        if self.flag_fit2D:
            fit_params = self.fit_2D_gussian(frame_num=f_num, scale=5)

        if self.median_filter_flag:
            frame_v = median_filter(self.video[int(f_num), :, :], 3)
        else:
            frame_v = self.video[int(f_num), :, :]

        ax = plt.gca()
        myplot = ax.imshow(frame_v, cmap=self.color)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(myplot, cax=cax)

        if f_num <= np.max(self.frame_number_) and f_num >= np.min(self.frame_number_):
            idx_ = np.where(self.frame_number_ == f_num)

            particle_X = self.x_center[idx_]
            particle_Y = self.y_center[idx_]
            particle_sigma = self.sigma[idx_]
            particle_labels = self.particle_ID[idx_]

            y = int(particle_Y)
            x = int(particle_X)
            sigma = particle_sigma
            ax.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))

        plt.show()

    def fit_2D_gussian(self, frame_num, scale=5):

        particle = self.list_psf.loc[self.list_psf['frame'] == frame_num]
        particle_X = particle['x'].tolist()
        particle_Y = particle['y'].tolist()
        particle_sigma = particle['sigma'].tolist()
        particle_center_intensity = particle['center_intensity'].tolist()

        index_list = [index for index in particle.index]

        list_one_frame_fit = []
        if particle.shape[0] > 0:
            for p_x, p_y, sigma_0, c_0, i_ in zip(particle_X, particle_Y, particle_sigma, particle_center_intensity,
                                                  index_list):
                window_size = scale * np.sqrt(2) * sigma_0
                start_sigma = sigma_0
                if p_x > window_size and p_y > window_size:
                    window_frame = self.video[int(frame_num), int(p_y - window_size) + 1:int(p_y + window_size),
                                   int(p_x - window_size) + 1:int(p_x + window_size)]
                    w_s = window_size
                else:
                    window_size_temp = window_size
                    while p_x < window_size_temp or p_y < window_size_temp:
                        window_size_temp = window_size_temp - 2
                    window_frame = self.video[int(frame_num),
                                   int(p_y - window_size_temp) + 1:int(p_y + window_size_temp),
                                   int(p_x - window_size_temp) + 1:int(p_x + window_size_temp)]
                    w_s = window_size_temp

                fit_params_ = gaussian_2D_fit.fit_2D_Gaussian_varAmp(window_frame, sigma_0=start_sigma,
                                                                     sigma_1=start_sigma,
                                                                     display_flag=False)
                fit_params = fit_params_[1]
                fit_errors = fit_params_[2]

                print('Fit Amplitude:', fit_params[0], '\u00b1', fit_errors[0])
                print('Fit X-Center: ', fit_params[1], '\u00b1', fit_errors[1])
                print('Fit Y-Center: ', fit_params[2], '\u00b1', fit_errors[2])
                print('Fit X-Sigma:  ', fit_params[3], '\u00b1', fit_errors[3])
                print('Fit Y-Sigma:  ', fit_params[4], '\u00b1', fit_errors[4])
                print('Fit Bias:  ', fit_params[5], '\u00b1', fit_errors[5])

                fit_params = [frame_num, i_, fit_params_, p_x, p_y, w_s]
                return fit_params


class JupyterSubplotDisplay():

    def __init__(self, list_videos, numRows, numColumns, list_titles=None, imgSizex=20, imgSizey=20, IntSlider_width='500px',
                 median_filter_flag=False, color='gray', step=1):

        """
        This class interactively displays several videos (with the same number of frames) in a Jupyter notebook.

        Parameters
        ----------
        list_videos: list of NDArray
          List of videos

        numRows: int
            It defines number of rows in sub-display

        numColumns: int
            It defines number of columns in sub-display

        list_titles: list str
            List of titles for each sub plot

        median_filter_flag: bool
          In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
          It defines the colormap for visualization.

        imgSizex: int
          Image length size.

        imgSizey: int
          Image width size.

        IntSlider_width: str
          Size of slider

        step: int
          Stride between visualization frames.
        """

        self.color = color
        self.video = list_videos
        self.median_filter_flag = median_filter_flag
        self.numRows = numRows
        self.numColumns = numColumns
        self.numVideos = len(list_videos)
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey

        if list_titles is None:
            self.list_titles = [None for _ in range(self.numVideos)]
        else:
            self.list_titles = list_titles

        max_numberFrames = np.max([vid_.shape[0] for vid_ in list_videos])

        interact(self.display, frame=widgets.IntSlider(min=0, max=max_numberFrames, step=step, value=0, layout=Layout(width=IntSlider_width),
                                                       readout_format='100', continuous_update=False, description='Frame:'))

    def display(self, frame):
        fig = plt.figure(figsize=(self.imgSizex, self.imgSizey))

        grid = plt.GridSpec(self.numRows, self.numColumns, hspace=0.3, wspace=0.3)

        imgGrid_list = []
        for i in range(self.numRows):
            for j in range(self.numColumns):
                imgGrid_list.append(fig.add_subplot(grid[i, j]))

        for img_, tit_, img_grid_ in zip(self.video, self.list_titles, imgGrid_list):

            if self.median_filter_flag:
                frame_v = median_filter(img_[int(frame), :, :], 3)
            else:
                frame_v = img_[int(frame), :, :]

            # img_grid_.imshow(frame_v, cmap=self.color)
            # img_grid_.axis('off')

            divider = make_axes_locatable(img_grid_)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            myplot = img_grid_.imshow(frame_v, cmap=self.color)
            img_grid_.axis('off')
            plt.colorbar(myplot, cax=cax)
            if tit_ is not None:
                img_grid_.set_title(tit_)

        plt.show()

