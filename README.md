![](https://github.com/SandoghdarLab/PiSCAT/blob/28eafb06ea4f6b468dde70e33fa970d1699974cf/docs/Fig/PiSCAT_logo_bg.png)

# PiSCAT: An open source package in Python for interferometric Scattering Microscopy ([Homepage](https://piscat.readthedocs.io))

iSCAT microscopy was introduced about two decades ago ([1](https://link.aps.org/doi/10.1103/PhysRevLett.93.037401)) and demonstrated to be the method of choice for label-free imaging and tracking of matter at nanometric scale ([2](https://doi.org/10.1021/acs.nanolett.9b01822)), with a wide range of applications such as detection of gold nanoparticles, single dye molecules, viruses, and small proteins ([3](https://en.wikipedia.org/wiki/Interferometric_scattering_microscopy)).
The image of a nanoparticle in iSCAT microscopy is formed via the interference between the light scattered from the particle and a reference field which is a part of the incident laser light. The photostable scattering signal from nanoparticles allows for very long measurements at high speeds, all the way up to megahertz, limited only by the available technology, e.g. of cameras or scanners. Recording fast and long videos however, produces a large volume of data which needs to undergo several stages of computationally demanding analysis. We introduce **PiSCAT** as a python-based package for the analysis of variuos iSCAT measurements and related experiments. PiSCAT aims to facilitate high-performance quantitative analysis of big data and provide a generally open-access platform to enable and speed up the research in iSCAT and related communities. To facilitate the use of PiSCAT, we offer tutorials with live-code features in which we present state-of-the-art algorithms for iSCAT microscopy. These cover important educative materials in [jupyter notebooks](https://jupyter.org/), supported with a web-based [documentation page](https://piscat.readthedocs.io).

In this first release, we provide analysis tools for the sensitive detection of single unlabelled proteins via wide-field iSCAT microscopy. Proteins are only a few nanometers in size with a molecular weight of a few to several hundred kDa. They were detected via iSCAT already in 2014 for small proteins down to the Bovines Serumalbumin (BSA) protein with a mass of 65 kDa ([4](https://doi.org/10.1038/ncomms5495)). iSCAT microscopy is since employed in several more advanced applications such as real-time investigation of cellular secretion ([5](https://doi.org/10.3791/58486),[6](https://doi.org/10.1021/acs.nanolett.7b04494)) and quantitative mass spectrometry of single proteins ([7](https://doi.org/10.1126/science.aar5839)).

## Documentation

The documentation webpage of PiSCAT modules can be found
[here](https://piscat.readthedocs.io).

The outputs from most of the PiSCAT localization and tracking methods are of [Panda data frame type](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html). This data structure has the ability to be easily appended/extended with more information based on different levels of analysis. The data structures containing the results of localization and tracking routines can be saved as csv, mat and HDF5 files. This helps users to work with the analyzed information using different softwares namely, MATLAB and Microsoft Excel. HDF5 is a well-known format that is readable in different programming languages and supports large, complex, heterogeneous data. HDF5 uses a "file directory" like structure that allows users to organize data within the file in structured ways and to embed metadata as well, making it self-describing. 


## Installation

### From PyPi

To install PiSCAT using PyPi, enter the following command in the console:

```
pip install piscat
```

### Local installation of PiSCAT

Clone/download this repository and unzip it. In the project directory enter the following command

```
pip install -e .
```

## Running PiSCAT GUI

Once the installation is done and the python environment is activated, enter the following command in the console:

```
python -m piscat
```

## Running PiSCAT Tutorials

Once the installation is done and the python environment is activated, enter the following command in the console:

```
python -m piscat.Tutorials
```

## Citing PiSCAT

TODO

## Testing

To run the tests, please activate the PiSCAT virtual environment. In the project directory, in the console, enter the following command:

```
python setup.py test
```

## Installation of PiSCAT virtual environment in the PyCharm IDE:

1.	Follow the hyper links and the install [ Python 3.8](https://www.python.org/downloads/) and [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows).
2.	Create a virtual environment based on the instructions provided [here](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html).
3.  Follow [this link](https://www.jetbrains.com/help/pycharm/creating-and-running-setup-py.html) to select PiSCAT venv as the interpreter, to install the setup.py file and then to run a setup.py task. 


