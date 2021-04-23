from __future__ import print_function
from PySide2 import QtCore
from PySide2.QtCore import *

import imageio
import numpy as np
import matplotlib

from piscat.Preproccessing.normalization import Normalization
from piscat.InputOutput.cpu_configurations import CPUConfigurations

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import matplotlib.animation as manimation
from matplotlib.patches import Circle, Arrow, Rectangle

import pylab as pl
import matplotlib.cm as cm
import matplotlib.patches as patches
from skimage.draw import circle_perimeter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import median_filter


class SignalConnection(QtCore.QObject):
    updateProgress_ = QtCore.Signal(int)
    finished = Signal()


class DisplaySubplot():

    def __init__(self, list_videos, numRows, numColumns, step=0, median_filter_flag=False, color='gray'):

        """
        This class shows several videos (with the same number of frames) at once.

        Parameters
        ----------
        list_videos: list of NDArray
            List of videos.

        numRows: int
           It defines number of rows in sub-display.

        numColumns: int
           It defines number of columns in sub-display.

        median_filter_flag: bool
            In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
            It defines the colormap for visualization.

        step: int
            Stride between visualization frames.
        """

        self.color = color
        self.jump = step
        self.pressed_key = {}
        self.memory = 0
        self.video = list_videos
        self.median_filter_flag = median_filter_flag
        self.numRows = numRows
        self.numColumns = numColumns
        self.numVideos = len(list_videos)

        self.max_numberFrames = np.max([vid_.shape[0] for vid_ in list_videos])

        self.fig, self.axes = plt.subplots(numRows, numColumns)

        self.im = []
        self.cb = []
        self.tx = []
        for vid_, ax in zip(self.video, self.axes):
            self.div = make_axes_locatable(ax)
            self.cax = self.div.append_axes('right', '5%', '5%')
            self.cv0 = vid_[0, :, :]
            im_ = ax.imshow(self.cv0, origin='lower', cmap=self.color)
            self.im.append(im_)
            self.cb.append(self.fig.colorbar(im_, cax=self.cax, format=ticker.FuncFormatter(self.fmt)))
            self.tx.append(ax.set_title('Frame 0'))

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=np.arange(0, self.max_numberFrames), blit=False)
        plt.show()

    def animate(self, i_):
        print(i_)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        if self.pressed_key.get('key') == 'q':
            pl.close('all')

        i_ = i_ + self.memory
        if i_ < self.max_numberFrames:
            for vid_, im, tx in zip(self.video, self.im, self.tx):

                if self.median_filter_flag:
                    arr = median_filter(vid_[int(i_), :, :], 3)
                else:
                    arr = median_filter(vid_[int(i_), :, :], 3)
                vmax = np.max(arr)
                vmin = np.min(arr)
                im.set_data(arr)
                im.set_clim(vmin, vmax)
                tx.set_text('Frame {0}'.format(i_))

                self.memory = i_ + self.jump
        else:
            self.ani.event_source.stop()

    def fmt(self, x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    def press(self, event):
        print('press', event.key)
        if event.key == 'q':
            # close the current figure
            plt.close(event.canvas.figure)
            self.pressed_key['key'] = event.key


class Display():

    def __init__(self, video, step=0, color='gray', time_delay=0, median_filter_flag=False):
        """
        This class display the video.

        Parameters
        ----------
        video:(NDArray)
            Input video.

        median_filter_flag: bool
            In case it defines as True, a median filter is applied with size 3 to remove hot pixel effect.

        color: str
            It defines the colormap for visualization.

        time_delay: float
            Delay between frames in milliseconds.

        step: int
            Stride between visualization frames.

        """
        self.video = video
        self.median_filter_flag = median_filter_flag
        self.jump = step
        self.time_delay = time_delay
        self.memory = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.pressed_key = {}

        self.div = make_axes_locatable(self.ax)
        self.cax = self.div.append_axes('right', '5%', '5%')

        self.cv0 = self.video[0, :, :]
        self.im = self.ax.imshow(self.cv0, origin='lower', cmap=color) # Here make an AxesImage rather than contour
        self.cb = self.fig.colorbar(self.im, cax=self.cax, format=ticker.FuncFormatter(self.fmt))
        self.tx = self.ax.set_title('Frame 0')

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.video.shape[0],
                                           blit=False, interval=self.time_delay, repeat=False, cache_frame_data=False)
        plt.show()

    def animate(self, i_):
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        if self.pressed_key.get('key') == 'q':
            pl.close('all')

        i_ = i_ + self.memory
        if i_ < self.video.shape[0]:
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
            self.memory = i_ + self.jump
        else:
            self.ani.event_source.stop()

    def fmt(self, x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    def press(self, event):
        print('press', event.key)
        if event.key == 'q':
            # close the current figure
            plt.close(event.canvas.figure)
            self.pressed_key['key'] = event.key


class DisplayPSFs_subplotLocalizationDisplay():

    def __init__(self, list_videos, list_df_PSFs, list_titles, numRows, numColumns, color='gray',
                 median_filter_flag=False, imgSizex=5, imgSizey=5, time_delay=0.1):
        """
        This class shows several videos (with the same number of frames) at once while highlight localize PSFs.

        Parameters
        ----------
        list_videos: list of NDArray
            List of videos

        list_df_PSFs: list panda data_frame
            List Data Frames that contains the location of PSFs for each video.

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

        time_delay: float
            Delay between frames in milliseconds.
        """

        self.list_video = list_videos
        self.list_df_PSFs = list_df_PSFs
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
        self.pressed_key = {}

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
            self.cb_0.append(self.fig.colorbar(im_, cax=cax_, format=ticker.FuncFormatter(self.fmt)))
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
                                           blit=False, interval=time_delay, repeat=True, cache_frame_data=False)
        plt.show()

    def animate(self, i_):
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        if self.pressed_key.get('key') == 'q':
            pl.close('all')
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
        else:
            self.ani.event_source.stop()

    def fmt(self, x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    def press(self, event):
        print('press', event.key)
        if event.key == 'q':
            # close the current figure
            plt.close(event.canvas.figure)
            self.pressed_key['key'] = event.key


class DisplayDataFramePSFsLocalization(QRunnable):

    def __init__(self, video, df_PSFs, time_delay=0.1, GUI_progressbar=False, *args, **kwargs):
        """
        This class displays video while highlighting PSFs.

        Parameters
        ----------
        video: NDArray
            Input video.

        df_PSFs: panda data_frame
            Data Frames that contains the location of PSFs.

        time_delay: float
            Delay between frames in milliseconds.

        GUI_progressbar: bool
            This actives GUI progress bar

        """
        self.cpu = CPUConfigurations()
        super(DisplayDataFramePSFsLocalization, self).__init__()

        self.video = video
        self.time_delay = time_delay
        self.df_PSFs = df_PSFs
        self.pressed_key = {}
        self.list_line = []
        if 'particle' in self.df_PSFs.keys():
            self.list_particles_idx = self.df_PSFs.particle.unique()

        else:
            self.df_PSFs['particle'] = 0
            self.list_particles_idx = self.df_PSFs.particle.unique()

        self.GUI_progressbar = GUI_progressbar
        self.args = args
        self.kwargs = kwargs

        colors_ = cm.autumn(np.linspace(0, 1, len(self.list_particles_idx)))
        self.colors = colors_[0:len(self.list_particles_idx), :]
        self.obj_connection = SignalConnection()

    @Slot()
    def run(self):
        self.gif_genrator(*self.args, **self.kwargs)
        if self.GUI_progressbar is True:
            self.obj_connection.finished.emit()

    def show_psf(self, jump=1, display_history=True, color_map='gray', save_flag=False, save_path=None):
        self.norm_vid = Normalization(self.video).normalized_image_specific()
        img = None
        pl.ion()
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        if len(self.norm_vid.shape) == 3 and self.norm_vid.shape[0] > 0:
            for frame_number in range(0, self.norm_vid.shape[0]-jump, jump):

                if self.GUI_progressbar:

                    fig.canvas.mpl_connect('close_event', self.closeEvent)

                else:
                    fig.canvas.mpl_connect('key_press_event', self.press)

                if self.pressed_key.get('key') == 'q':
                    pl.close('all')
                    break

                im = self.norm_vid[frame_number, :, :]
                pl.pause(self.time_delay)
                [p.remove() for p in reversed(ax.patches)]
                if img is None:
                    img = pl.imshow(im, cmap=color_map)
                    self.draw_circles(self.df_PSFs, ax, frame_number, color=self.colors)
                    if display_history:
                        [ln_.remove() for ln_ in self.list_line]
                        self.draw_trajectory(self.df_PSFs, ax, frame_number, color=self.colors)
                else:
                    img.set_data(im)
                    pl.draw()
                    pl.title("Press Q for exit\n Frame: "+str(frame_number))
                    self.draw_circles(self.df_PSFs, ax, frame_number, color=self.colors)
                    if display_history:
                        [ln_.remove() for ln_ in self.list_line]
                        self.draw_trajectory(self.df_PSFs, ax, frame_number, color=self.colors)
                    if save_flag:
                        pl.savefig(save_path + 'fig' + str(frame_number) +'.png')

        else:
            im = self.norm_vid
            pl.pause(self.time_delay)
            [p.remove() for p in reversed(ax.patches)]
            if img is None:
                img = pl.imshow(im)
                self.draw_circles(self.df_PSFs, ax, frame_number=0, color=self.colors)
                if display_history:
                    [ln_.remove() for ln_ in self.list_line]
                    self.draw_trajectory(self.df_PSFs, ax, frame_number=0, color=self.colors)
            else:
                img.set_data(im)
                pl.draw()
                self.draw_circles(self.df_PSFs, ax, frame_number=0, color=self.colors)
                if display_history:
                    [ln_.remove() for ln_ in self.list_line]
                    self.draw_trajectory(self.df_PSFs, ax, frame_number=0, color=self.colors)

    def draw_circles(self, list_psf, ax, frame_number, color='red'):
        particle = list_psf.loc[list_psf['frame'] == frame_number]
        particle_X = particle['x'].tolist()
        particle_Y = particle['y'].tolist()
        particle_sigma = particle['sigma'].tolist()
        particle_labels = particle['particle'].tolist()
        for j_, p_label in zip(range(len(particle_X)), particle_labels):
            y = particle_Y[j_]
            x = particle_X[j_]
            sigma = particle_sigma[j_]
            color_idx = list(self.list_particles_idx).index(p_label)
            c = patches.Circle((x, y), np.sqrt(2) * sigma, color=self.colors[color_idx], alpha=1, fill=False, linewidth=2)

            ax.add_patch(c)
            pl.draw()

    def draw_trajectory(self, list_psf, ax, frame_number, color='red'):
        particle = list_psf.loc[list_psf['frame'] == frame_number]
        particle_labels = particle['particle'].tolist()

        self.list_line = []

        for label in particle_labels:
            all_particle_ = list_psf.loc[list_psf['particle'] == label]
            particle_f = all_particle_['frame'].tolist()
            particle_X = all_particle_['x'].tolist()
            particle_Y = all_particle_['y'].tolist()
            particle_sigma = all_particle_['sigma'].tolist()
            list_center = None
            for f, x, y, sigma in zip(particle_f, particle_X, particle_Y, particle_sigma):
                if list_center is None:
                    center_position = [[x, y]]
                    list_center = center_position
                    list_center_array = np.asarray(list_center)

                else:
                    center_position = [x, y]
                    list_center.append(center_position)
                    list_center_array = np.asarray(list_center)

                if f <= frame_number:
                    color_idx = list(self.list_particles_idx).index(label)
                    ln, = ax.plot(list_center_array[:, 0], list_center_array[:, 1], color=self.colors[color_idx], linewidth=1)

                    self.list_line.append(ln)
                    pl.draw()

    def draw_trajectory_1(self, list_psf, ax, frame_number, color='red'):
        particle = list_psf.loc[list_psf['frame'] == frame_number]
        particle_labels = particle['particle'].tolist()
        list_line = []

        for label in particle_labels:
            all_particle_ = list_psf.loc[list_psf['particle'] == label]
            particle_f = all_particle_['frame'].tolist()
            particle_X = all_particle_['x'].tolist()
            particle_Y = all_particle_['y'].tolist()
            particle_sigma = all_particle_['sigma'].tolist()
            list_center = None
            flag_loop = True
            ln, = ax.plot(particle_X, particle_Y, color=self.colors[label], linewidth=1, alpha=0.7)
            list_line.append(ln)
        pl.draw()
        plt.pause(self.time_delay)
        [ln_.remove() for ln_ in list_line]

    def draw_trajectory_2(self, list_psf, ax, frame_number, color='red'):
        particle = list_psf.loc[list_psf['frame'] == frame_number]
        particle_labels = particle['particle'].tolist()
        list_line = []

        for label in particle_labels:
            all_particle_ = list_psf.loc[list_psf['particle'] == label]
            particle_f = all_particle_['frame'].tolist()
            particle_X = all_particle_['x'].tolist()
            particle_Y = all_particle_['y'].tolist()
            particle_sigma = all_particle_['sigma'].tolist()
            list_center = None
            flag_loop = True

            while flag_loop:
                list_line = []
                for f, x, y, sigma in zip(particle_f, particle_X, particle_Y, particle_sigma):
                    if list_center is None:
                        center_position = [[x, y]]
                        list_center = center_position
                        list_center_array = np.asarray(list_center)
                    else:
                        center_position = [x, y]
                        list_center.append(center_position)
                        list_center_array = np.asarray(list_center)
                    if f <= frame_number:
                        ln, = ax.plot(list_center_array[:, 0], list_center_array[:, 1], color=self.colors[label], linewidth=1, alpha=0.55)
                        list_line.append(ln)
                        pl.draw()
                        plt.pause(self.time_delay)
                    else:
                        flag_loop = False

                pl.draw()
                plt.pause(self.time_delay)
                [ln_.remove() for ln_ in list_line]

    def press(self, event):
        print('press', event.key)
        if event.key == 'q':
            plt.close(event.canvas.figure)
            self.pressed_key['key'] = event.key

    def closeEvent(self, event):
        plt.close(event.canvas.figure)
        self.pressed_key['key'] = 'q'

    def make_gif(self, frame_number, display_history=True, color_map='gray'):
        matplotlib.use('Agg')
        img = None
        pl.ioff()
        plt.ioff()

        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        if len(self.norm_vid.shape) == 3 and self.norm_vid.shape[0] > 0:

            im = self.norm_vid[frame_number, :, :]
            pl.pause(self.time_delay)
            [p.remove() for p in reversed(ax.patches)]
            if img is None:
                img = pl.imshow(im, cmap=color_map)
                self.draw_circles(self.df_PSFs, ax, frame_number, color=self.colors)
                if display_history:
                    [ln_.remove() for ln_ in self.list_line]
                    self.draw_trajectory(self.df_PSFs, ax, frame_number, color=self.colors)
            else:
                img.set_data(im)
                pl.draw()
                pl.title(str(frame_number))
                self.draw_circles(self.df_PSFs, ax, frame_number, color=self.colors)
                if display_history:
                    [ln_.remove() for ln_ in self.list_line]
                    self.draw_trajectory(self.df_PSFs, ax, frame_number, color=self.colors)

            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            pl.close(fig)

            return image

    def gif_genrator(self, save_path, jump=5, fps=1.0):
        self.norm_vid = Normalization(self.video).normalized_image_specific()
        image_ = []
        if self.cpu.parallel_active is True:
            image_ = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                delayed(self.make_gif)(frame_number) for frame_number in range(0, self.norm_vid.shape[0] - jump, jump))
            imageio.mimsave(save_path, image_, fps=fps)

        else:

            for frame_number in tqdm(range(0, self.norm_vid.shape[0] - jump, jump)):
                image_.append(self.make_gif(frame_number))
                if self.GUI_progressbar:
                    self.obj_connection.updateProgress_.emit(frame_number)
            imageio.mimsave(save_path, image_, fps=fps)


def histogram_of_each_frames(frame, bins='auto', range=None, normed=None, weights=None, density=None):
    return np.histogram(frame, bins=bins, range=range, normed=normed, weights=weights, density=density)


def histogram_1D_signal(signal, bins='auto'):
    if bins == 'auto':
        plt.hist(signal, bins=bins)  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()
    else:
        plt.hist(signal, bins=bins)  # arguments are passed to np.histogram
        plt.title('Histogram with'+str(bins)+'bins')
        plt.show()


