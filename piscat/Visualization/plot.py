from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def plot3(X, Y, Z, scale='(um)', title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z)

    ax.set_xlabel('X' + scale)
    ax.set_ylabel('Y' + scale)
    ax.set_zlabel('Z' + scale)

    plt.title(title)
    plt.show()


def plot2df(df_PSFs, pixel_size=1, scale='(um)', title='', falg_label=False ):
    groups = df_PSFs.groupby('particle')

    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.x*pixel_size, group.y*pixel_size, marker='.', linestyle='', ms=3, label=name)
    if falg_label:
        ax.legend()
    ax.set_xlabel(scale)
    ax.set_ylabel(scale)
    plt.title(title)
    plt.show()


def plot_histogram(data, title, fc='C1', ec='k'):

    plt.figure()
    plt.hist(data, alpha=0.7, bins=None, density=False, stacked=True, fc=fc, ec=ec)
    plt.title(title + '\nMedian: ' + str(np.median(data)))
    plt.ylabel('#Counts', fontsize=18)
    plt.xlabel('Length of linking', fontsize=18)
    plt.tight_layout()
    plt.show()

