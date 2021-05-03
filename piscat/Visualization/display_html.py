"""
__author__ = "Houman Mirzaalian D., xxxx"
"""

from __future__ import print_function

import numpy as np
import matplotlib.ticker as ticker
import pylab as pl
from PySide2 import QtGui
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2.QtCore import *
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import animation
from matplotlib.patches import Circle, Arrow, Rectangle
from piscat.InputOutput import read_status_line
from piscat.Trajectory.data_handeling import protein_trajectories_list2dic
from piscat.Visualization.print_colors import PrintColors


class WorkerSignals(QObject):

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class HTML_Display():

    def __init__(self, video, color='gray', time_delay=0.5, median_filter_flag=False, imgSizex=5, imgSizey=5):
        """
        This class display the video for HTML.

        Parameters
        ----------
        video:(NDArray)
           Input video

        median_filter_flag: bool
           In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
           It defines the colormap for visualization.

        time_delay: float
           Delay between frames in milliseconds.

        imgSizex: int
          Image length size.

        imgSizey: int
          Image width size.
        """
        self.video = video
        self.median_filter_flag = median_filter_flag

        self.fig = plt.figure()
        self.fig.set_size_inches(imgSizex, imgSizey, True)
        self.ax = self.fig.add_subplot(111)
        self.pressed_key = {}
        plt.axis('off')

        self.div = make_axes_locatable(self.ax)
        self.cax = self.div.append_axes('right', '5%', '5%')

        self.cv0 = self.video[0, :, :]
        self.im = self.ax.imshow(self.cv0, origin='lower', cmap=color)
        self.cb = self.fig.colorbar(self.im, cax=self.cax)

        self.tx = self.ax.set_title('Frame 0')

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.video.shape[0],
                                           blit=False, interval=time_delay, repeat=False, cache_frame_data=False)

    def animate(self, i_):

        arr = self.video[i_, :, :]
        vmax = np.max(arr)
        vmin = np.min(arr)
        if self.median_filter_flag:
            frame_v = median_filter(arr, 3)
        else:
            frame_v = arr
        self.im.set_data(frame_v)
        self.im.set_clim(vmin, vmax)
        self.tx.set_text('Frame {0}'.format(i_))


class HTML_Display_StatusLine():

    def __init__(self, video, median_filter_flag=False, color='gray', imgSizex=5, imgSizey=5, time_delay=0.5):
        """
        This class displays the video in the HTML while highlight status line

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

        time_delay: float
           Delay between frames in milliseconds.
       """
        self.color = color
        self.video = video
        self.imgSizex = imgSizex
        self.imgSizey = imgSizey
        self.median_filter_flag = median_filter_flag

        status_ = read_status_line.StatusLine(video)
        _, status_information = status_.find_status_line()
        self.statusLine_position = status_information['status_line_position']

        self.fig = plt.figure()
        self.fig.set_size_inches(imgSizex, imgSizey, True)
        self.ax = self.fig.add_subplot(111)
        self.pressed_key = {}
        plt.axis('off')

        self.div = make_axes_locatable(self.ax)
        self.cax = self.div.append_axes('right', '5%', '5%')

        self.cv0 = self.video[0, :, :]
        self.im = self.ax.imshow(self.cv0, origin='lower', cmap=self.color)
        self.cb = self.fig.colorbar(self.im, cax=self.cax)
        self.tx = self.ax.set_title('Frame 0')

        if self.statusLine_position == 'column':
            rect = Rectangle((0, self.video.shape[2]), self.video.shape[2], 1, linewidth=10, edgecolor='r',
                                  facecolor='none')
            self.ax.add_patch(rect)
        elif self.statusLine_position == 'row':
            rect = Rectangle((self.video.shape[1], 0), 1, self.video.shape[1], linewidth=10, edgecolor='r',
                                  facecolor='none')
            self.ax.add_patch(rect)

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.video.shape[0],
                                           blit=False, interval=time_delay, repeat=False, cache_frame_data=False)

    def animate(self, i_):
        [p.remove() for p in reversed(self.ax.patches)]
        arr = self.video[i_, :, :]
        vmax = np.max(arr)
        vmin = np.min(arr)
        if self.median_filter_flag:
            frame_v = median_filter(arr, 3)
        else:
            frame_v = arr
        self.im.set_data(frame_v)
        self.im.set_clim(vmin, vmax)
        self.tx.set_text('Frame {0}'.format(i_))

        if self.statusLine_position == 'column':
            rect = Rectangle((0, self.video.shape[2]), self.video.shape[2], 1, linewidth=10, edgecolor='r',
                                  facecolor='none')
            self.ax.add_patch(rect)

        elif self.statusLine_position == 'row':
            rect = Rectangle((self.video.shape[1], 0), 1, self.video.shape[1], linewidth=10, edgecolor='r',
                                  facecolor='none')
            self.ax.add_patch(rect)


class HTML_PSFs_subplotLocalizationDisplay(QRunnable, PrintColors):

    def __init__(self, list_videos, list_df_PSFs, list_titles, numRows, numColumns, color='gray',
                 median_filter_flag=False, imgSizex=5, imgSizey=5, time_delay=0.5, save_path=None, fps=10):
        """
        This class displays the videos in the HTML while highlight PSFs.

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

        time_delay: float
                   Delay between frames in milliseconds.
        """
        super(HTML_PSFs_subplotLocalizationDisplay, self).__init__()
        PrintColors.__init__(self)

        self.list_video = list_videos
        self.list_df_PSFs = list_df_PSFs
        self.save_path = save_path
        self.fps = fps
        self.signals = WorkerSignals()
        self.median_filter_flag = median_filter_flag
        self.memory = 0
        self.numRows = numRows
        self.numColumns = numColumns
        self.numVideos = len(list_videos)
        if list_titles is None:
            self.list_titles = [None for _ in range(self.numVideos)]
        else:
            self.list_titles = list_titles

        self.fig = plt.figure(figsize=(imgSizex, imgSizey))
        self.imgGrid_list = []
        for i_ in range(1, self.numRows*self.numColumns+1):
            img_grid_ = self.fig.add_subplot(self.numRows, self.numColumns, i_)
            img_grid_.axis('off')
            self.imgGrid_list.append(img_grid_)

        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.18)

        self.div_0 = []
        self.cax_0 = []
        self.im_0 = []
        self.cb_0 = []
        self.tx_0 = []

        for ax_, vid_, df_PSFs, tit_ in zip(self.imgGrid_list, self.list_video, self.list_df_PSFs, self.list_titles):
            div_ = make_axes_locatable(ax_)
            self.div_0.append(div_)

            cax_ = div_.append_axes('right', '5%', '5%')
            self.cax_0.append(cax_)

            im_ = ax_.imshow(vid_[0, :, :], origin='lower', cmap=color)
            self.im_0.append(im_)
            self.cb_0.append(self.fig.colorbar(im_, cax=cax_))

            if tit_ is not None:
                self.tx_0.append(ax_.set_title(tit_ + ', Frame 0'))
            else:
                self.tx_0.append(ax_.set_title('Frame 0'))

            particle = df_PSFs.loc[df_PSFs['frame'] == 0]
            particle_X = particle['x'].tolist()
            particle_Y = particle['y'].tolist()
            particle_sigma = particle['sigma'].tolist()

            for j_ in range(len(particle_X)):
                y = int(particle_Y[j_])
                x = int(particle_X[j_])
                sigma = particle_sigma[j_]
                ax_.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.list_video[0].shape[0],
                                           blit=False, interval=time_delay, repeat=False, cache_frame_data=False)

    def animate(self, i_):
        if i_ < self.list_video[0].shape[0]:
            for idx_, (ax_, vid_, df_PSFs, tit_) in enumerate(zip(self.imgGrid_list, self.list_video, self.list_df_PSFs, self.list_titles)):
                [p.remove() for p in reversed(ax_.patches)]

                arr = vid_[i_, :, :]
                vmax = np.max(arr)
                vmin = np.min(arr)
                if self.median_filter_flag:
                    frame_v = median_filter(arr, 3)
                else:
                    frame_v = arr

                self.im_0[idx_].set_data(frame_v)
                self.im_0[idx_].set_clim(vmin, vmax)

                if tit_ is not None:
                    self.tx_0[idx_].set_text(tit_ + ', Frame {}'.format(i_))
                else:
                    self.tx_0[idx_].set_text('Frame {}'.format(i_))

                particle = df_PSFs.loc[df_PSFs['frame'] == i_]
                particle_X = particle['x'].tolist()
                particle_Y = particle['y'].tolist()
                particle_sigma = particle['sigma'].tolist()
                for j_ in range(len(particle_X)):

                    y = int(particle_Y[j_])
                    x = int(particle_X[j_])
                    sigma = particle_sigma[j_]
                    ax_.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))

    @Slot()
    def run(self, *args, **kwargs):
        if self.save_path is not None:
            dpi = 300
            writer = animation.writers['ffmpeg'](fps=self.fps, bitrate=-1, extra_args=['-vcodec', 'libx264'])
            self.ani.save(self.save_path, writer=writer, dpi=dpi)

            self.signals.finished.emit()
        else:
            print(f"{self.WARNING}\nError occurred in MP4!{self.ENDC}")
            self.signals.finished.emit()


class HTML_subplotDisplay():

    def __init__(self, list_videos, list_titles, numRows, numColumns, color='gray',
                 median_filter_flag=False, imgSizex=5, imgSizey=5, time_delay=0.5):
        """
        This class displays the videos in the HTML.

        Parameters
        ----------
        list_videos: list of NDArray
            List of videos.

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

        time_delay: float
            Delay between frames in milliseconds.
        """

        self.list_video = list_videos
        self.median_filter_flag = median_filter_flag
        self.memory = 0
        self.numRows = numRows
        self.numColumns = numColumns
        self.numVideos = len(list_videos)
        if list_titles is None:
            self.list_titles = [None for _ in range(self.numVideos)]
        else:
            self.list_titles = list_titles

        self.fig = plt.figure(figsize=(imgSizex, imgSizey))
        self.imgGrid_list = []
        for i_ in range(1, self.numRows*self.numColumns+1):
            img_grid_ = self.fig.add_subplot(self.numRows, self.numColumns, i_)
            img_grid_.axis('off')
            self.imgGrid_list.append(img_grid_)

        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0.18)

        self.div_0 = []
        self.cax_0 = []
        self.im_0 = []
        self.cb_0 = []
        self.tx_0 = []

        for ax_, vid_, tit_ in zip(self.imgGrid_list, self.list_video, self.list_titles):
            div_ = make_axes_locatable(ax_)
            self.div_0.append(div_)

            cax_ = div_.append_axes('right', '5%', '5%')
            self.cax_0.append(cax_)

            im_ = ax_.imshow(vid_[0, :, :], origin='lower', cmap=color)
            self.im_0.append(im_)
            self.cb_0.append(self.fig.colorbar(im_, cax=cax_))

            if tit_ is not None:
                self.tx_0.append(ax_.set_title(tit_ + ', Frame 0'))
            else:
                self.tx_0.append(ax_.set_title('Frame 0'))

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.list_video[0].shape[0],
                                           blit=False, interval=time_delay, repeat=False, cache_frame_data=False)

    def animate(self, i_):
        if i_ < self.list_video[0].shape[0]:
            for idx_, (ax_, vid_, tit_) in enumerate(zip(self.imgGrid_list, self.list_video, self.list_titles)):

                arr = vid_[i_, :, :]
                vmax = np.max(arr)
                vmin = np.min(arr)
                if self.median_filter_flag:
                    frame_v = median_filter(arr, 3)
                else:
                    frame_v = arr

                self.im_0[idx_].set_data(frame_v)
                self.im_0[idx_].set_clim(vmin, vmax)

                if tit_ is not None:
                    self.tx_0[idx_].set_text(tit_ + ', Frame {}'.format(i_))
                else:
                    self.tx_0[idx_].set_text('Frame {}'.format(i_))


class HTMLSelectedPSFs_localizationDisplay():

    def __init__(self, video, particles, particles_num='#0', frame_extend=0, color='gray',
                 median_filter_flag=False, imgSizex=5, imgSizey=5, time_delay=0.5):
        """
        This class displays video in the HTML while highlighting selected PSF.

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

        time_delay: float
            Delay between frames in milliseconds.
        """

        self.video = video
        self.particles = particles
        self.median_filter_flag = median_filter_flag
        self.memory = 0

        if type(particles) is list:
            self.particles = protein_trajectories_list2dic(particles)

        if type(self.particles) is dict:
            for key, item in self.particles.items():
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

        video_frameNum = video.shape[0]
        max_particle_frame = np.max(self.frame_number_)
        min_particle_frame = np.min(self.frame_number_)

        max_extend_particle_frame = np.min([max_particle_frame + frame_extend, video_frameNum])
        min_extend_particle_frame = np.max([min_particle_frame - frame_extend, 0])

        self.frame_number = list(range(min_extend_particle_frame, max_extend_particle_frame, 1))

        self.fig = plt.figure(figsize=(imgSizex, imgSizey))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.imgGrid_list = []

        img_grid_ = self.fig.add_subplot(1, 1, 1)
        img_grid_.axis('off')
        self.imgGrid_list = img_grid_

        self.fig.tight_layout()
        # self.fig.subplots_adjust(wspace=0.18)

        self.div_0 = []
        self.cax_0 = []
        self.im_0 = []
        self.cb_0 = []
        self.tx_0 = []

        div_ = make_axes_locatable(self.imgGrid_list)
        self.div_0.append(div_)

        cax_ = div_.append_axes('right', '5%', '5%')
        self.cax_0.append(cax_)

        selected_frame = self.frame_number[0]
        im_ = self.imgGrid_list.imshow(video[selected_frame, :, :], origin='lower', cmap=color)
        self.im_0 = im_
        self.cb_0 = self.fig.colorbar(im_, cax=cax_)

        self.tx_0 = self.imgGrid_list.set_title('Frame 0')

        if 0 <= np.max(self.frame_number_) and 0 >= np.min(self.frame_number_):
            idx_ = np.where(self.frame_number_ == 0)

            particle_X = self.x_center[idx_]
            particle_Y = self.y_center[idx_]
            particle_sigma = self.sigma[idx_]
            particle_labels = self.particle_ID[idx_]

            y = int(particle_Y)
            x = int(particle_X)
            sigma = particle_sigma
            self.imgGrid_list.add_patch(Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.frame_number,
                                           blit=False, interval=time_delay, repeat=False, cache_frame_data=False)

    def animate(self, i_):
        if i_ <= np.max(self.frame_number) and i_ >= np.min(self.frame_number):
            [p.remove() for p in reversed(self.imgGrid_list.patches)]

            arr = self.video[i_, :, :]
            vmax = np.max(arr)
            vmin = np.min(arr)
            if self.median_filter_flag:
                frame_v = median_filter(arr, 3)
            else:
                frame_v = arr

            self.im_0.set_data(frame_v)
            self.im_0.set_clim(vmin, vmax)
            self.tx_0.set_text('Frame {}'.format(i_))

            if i_ <= np.max(self.frame_number_) and i_ >= np.min(self.frame_number_):
                idx_ = np.where(self.frame_number_ == i_)

                particle_X = self.x_center[idx_]
                particle_Y = self.y_center[idx_]
                particle_sigma = self.sigma[idx_]
                particle_labels = self.particle_ID[idx_]

                y = int(particle_Y)
                x = int(particle_X)
                sigma = particle_sigma
                self.imgGrid_list.add_patch(
                    Circle((x, y), radius=np.sqrt(2) * sigma, edgecolor='r', facecolor='none', linewidth=2))



