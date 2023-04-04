from __future__ import print_function

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot3(X, Y, Z, scale="(um)", title=""):
    """
    3D localization plotting

    Parameters
    ----------
    X: list
        X-position of PSFs.

    Y: list
        Y-position of PSFs.

    Z: list
        Z-position of PSFs.

    scale: str
        Measurement scale. e.g ('nm')

    title: str
        The title of the figure.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X, Y, Z)

    ax.set_xlabel("X" + scale)
    ax.set_ylabel("Y" + scale)
    ax.set_zlabel("Z" + scale)

    plt.title(title)
    plt.show()


def plot2df(df_PSFs, pixel_size=1, scale="(um)", title="", flag_label=False):
    """
    2D localization plotting with color code for each particle.

    Parameters
    ----------
    df_PSFs: panda data frame
        Data frame contains PSFs localization and ID.

    pixel_size:
        The camera's pixel size is used to scale the results of the localization.

    scale: str
        Measurement scale. e.g ('nm')

    title: str
        The title of the figure.

    flag_label: bool


    """
    groups = df_PSFs.groupby("particle")

    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(
            group.x * pixel_size, group.y * pixel_size, marker=".", linestyle="", ms=3, label=name
        )
    if flag_label:
        ax.legend()
    ax.set_xlabel(scale)
    ax.set_ylabel(scale)
    plt.title(title)
    plt.show()


def plot_bright_dark_psf(df_bright, df_dark, unit="nm"):
    """
    Plot heatmap of particle localization.

    Parameters
    ----------
    df_bright: pandas dataframe
        Bright PSF positions are stored in the data frame.

    df_dark: pandas dataframe
        Dark PSF positions are stored in the data frame.

    unit: str
        The axis unit.
    """
    particle_ID_bright = df_bright["particle"].tolist()
    particle_x_bright = df_bright["x"].tolist()
    particle_y_bright = df_bright["y"].tolist()
    particle_size_bright = df_bright["bubble_size"].tolist()

    particle_ID_dark = df_dark["particle"].tolist()
    particle_x_dark = df_dark["x"].tolist()
    particle_y_dark = df_dark["y"].tolist()
    particle_size_dark = df_dark["bubble_size"].tolist()

    particle_ID_bright_ = [str(l_b) for l_b in particle_ID_bright]
    particle_ID_dark_ = [str(l_b) for l_b in particle_ID_dark]

    particle_ID_all = particle_ID_bright_ + particle_ID_dark_
    particle_x_all = particle_x_bright + particle_x_dark
    particle_y_all = particle_y_bright + particle_y_dark
    particle_size_all = particle_size_bright + particle_size_dark

    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    (line1,) = plt.plot(particle_x_bright, particle_y_bright, "r.", markersize=15)

    ax2 = ax1.twinx()
    (line2,) = ax2.plot(particle_x_dark, particle_y_dark, "b.", markersize=15)
    ax2.tick_params(axis="y", labelcolor="black")

    ax1.set_xlabel("X-Position(" + unit + ")")
    ax1.set_ylabel("Y-Position(" + unit + ")")
    ax2.set_ylabel("Y-Position(" + unit + ")")

    annots = []
    for ax in [ax1, ax2]:
        annot = ax1.annotate(
            "",
            xy=(0, 0),
            xytext=(-20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.4),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(True)
        annots.append(annot)

    annot_dic = dict(zip([ax1, ax2], annots))
    line_dic = dict(zip([ax1, ax2], [line1, line2]))

    def update_annot(line, annot, ind):
        x, y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "ID= {}".format(particle_ID_all[ind["ind"][0]])
        annot.set_text(text)

    def hover(event):
        if event.inaxes in [ax1, ax2]:
            for ax in [ax1, ax2]:
                cont, ind = line_dic[ax].contains(event)
                annot = annot_dic[ax]
                if cont:
                    update_annot(line_dic[ax], annot, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    plt.show()


def plot_bright_dark_psf_inTime(df_bright, df_dark, time_delay=0.1, dir_name=None):
    """
    Showing binding and unbinding events in time.

    Parameters
    ----------
    df_bright: pandas dataframe
        Bright PSF positions are stored in the data frame.

    df_dark: pandas dataframe
        Dark PSF positions are stored in the data frame.

    time_delay: float
        Define the time delay between binding and unbinding events frames. This only works when `flag_in_time` is set to True.

    dir_name: str
        You can save time slap frames if you specify a save path.
    """
    particle_ID_bright = df_bright["particle"].tolist()
    particle_x_bright = df_bright["x"].tolist()
    particle_y_bright = df_bright["y"].tolist()
    particle_size_bright = df_bright["bubble_size"].tolist()

    particle_ID_dark = df_dark["particle"].tolist()
    particle_x_dark = df_dark["x"].tolist()
    particle_y_dark = df_dark["y"].tolist()
    particle_size_dark = df_dark["bubble_size"].tolist()

    if len(particle_ID_bright) > len(particle_ID_dark):
        diff_len = len(particle_ID_bright) - len(particle_ID_dark)
        for _ in range(diff_len):
            particle_ID_dark.append(None)
            particle_x_dark.append(None)
            particle_y_dark.append(None)
            particle_size_dark.append(None)
    elif len(particle_ID_bright) < len(particle_ID_dark):
        diff_len = len(particle_ID_dark) - len(particle_ID_bright)
        for _ in range(diff_len):
            particle_ID_bright.append(None)
            particle_x_bright.append(None)
            particle_y_bright.append(None)
            particle_size_bright.append(None)

    fig, ax = plt.subplots(figsize=(5, 5))
    for i_ in range(len(particle_size_bright)):
        ax.scatter(
            particle_x_bright[i_],
            particle_y_bright[i_],
            label="1",
            color="blue",
            s=100,
            alpha=0.5,
            marker="o",
        )
        ax.scatter(
            particle_x_dark[i_],
            particle_y_dark[i_],
            label="2",
            color="red",
            s=100,
            alpha=0.5,
            marker="o",
        )

        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-0.1, 5.1)

        if dir_name is not None:
            s_path = os.path.join(dir_name, "bright_dark_psfs_frame" + str(i_) + ".png")
            plt.savefig(s_path)
        else:
            plt.pause(time_delay)
    plt.show()


def plot_histogram(data, title, fc="C1", ec="k"):
    plt.figure()
    plt.hist(data, alpha=0.7, bins=None, density=False, stacked=True, fc=fc, ec=ec)
    plt.title(title + "\nMedian: " + str(np.median(data)))
    plt.ylabel("#Counts", fontsize=18)
    plt.xlabel("Length of linking", fontsize=18)
    plt.tight_layout()
    plt.show()
