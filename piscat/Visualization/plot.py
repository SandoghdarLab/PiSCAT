from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def plot3(X, Y, Z, scale='(um)', title=''):
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
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z)

    ax.set_xlabel('X' + scale)
    ax.set_ylabel('Y' + scale)
    ax.set_zlabel('Z' + scale)

    plt.title(title)
    plt.show()


def plot2df(df_PSFs, pixel_size=1, scale='(um)', title='', flag_label=False):
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
    groups = df_PSFs.groupby('particle')

    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.x*pixel_size, group.y*pixel_size, marker='.', linestyle='', ms=3, label=name)
    if flag_label:
        ax.legend()
    ax.set_xlabel(scale)
    ax.set_ylabel(scale)
    plt.title(title)
    plt.show()


def plot_histogram(data, title, fc='C1', ec='k'):
    """
    Plotting a histogram out of the data.

    Parameters
    ----------
    data: list
        Input data.

    title: str
        The title of the figure.

    fc: str
        Face color of histogram.

    ec: str
        Edge color of histogram.

    """

    plt.figure()
    plt.hist(data, alpha=0.7, bins=None, density=False, stacked=True, fc=fc, ec=ec)
    plt.title(title + '\nMedian: ' + str(np.median(data)))
    plt.ylabel('#Counts', fontsize=18)
    plt.xlabel('Length of linking', fontsize=18)
    plt.tight_layout()
    plt.show()

