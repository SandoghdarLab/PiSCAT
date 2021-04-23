.. PiSCAT documentation master file, created by
   sphinx-quickstart on Mon Jan 25 16:37:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PiSCAT: An open source package in Python for interferometric Scattering Microscopy
==================================================================================

iSCAT microscopy was introduced in our laboratory about two decades ago
(`[1] <https://link.aps.org/doi/10.1103/PhysRevLett.93.037401>`_) and demonstrated to be the method of choice for label-free
imaging and tracking of matter at nanometric scale (`[2] <https://doi.org/10.1021/acs.nanolett.9b01822>`_), with a wide range
of applications such as detection of gold nanoparticles, single dye molecules, viruses, and small proteins
(`[3] <https://en.wikipedia.org/wiki/Interferometric_scattering_microscopy>`_).
The image of a nanoparticle in iSCAT microscopy is formed via the interference between the light scattered from the
particle and a reference field which is a part of the incident laser light. The photostable scattering signal from
nanoparticles allows for very long measurements at high speeds, all the way up to megahertz, limited only by the available
technology, e.g. of cameras or scanners. Recording fast and long videos however, produces a large volume of data
which needs to undergo several stages of computationally demanding analysis. We introduce **PiSCAT** as a python-based
package for the analysis of variuos iSCAT measurements and related experiments.
PiSCAT aims to facilitate high-performance quantitative analysis of big data and provide a generally open-access platform
to enable and speed up the research in iSCAT and related communities. To facilitate the use of PiSCAT, we offer tutorials
with live-code features in which we present state-of-the-art algorithms for iSCAT microscopy. These cover important educative materials
in `jupyter notebooks <https://jupyter.org/>`_, supported with a web-based
`documentation page <https://piscat.readthedocs.io/en/latest/index.html>`_.

In this first release, we provide analysis tools for the sensitive detection of single unlabelled proteins via .
wide-field iSCAT microscopy. Proteins are only a few nanometers in size with a molecular weight of a few to several hundred
kDa. They were detected via iSCAT already in 2014 for small proteins down to the Bovines Serumalbumin (BSA) protein with
a mass of 65 kDa (`[4] <https://doi.org/10.1038/ncomms5495>`_). iSCAT microscopy is since employed in several more
advanced applications such as real-time investigation of cellular secretion (`[5] <https://doi.org/10.3791/58486>`_,
`[6] <https://doi.org/10.1021/acs.nanolett.7b04494>`_) and quantitative mass spectrometry of single proteins (`[7] <https://doi.org/10.1126/science.aar5839>`_).

Documentation
-------------

The documentation webpage of PiSCAT modules can be found
`here <https://piscat.readthedocs.io/en/latest/#>`_.

The outputs from most of the PiSCAT localization and tracking methods are of `Panda data frame type <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_.
This data structure has the ability to be easily appended/extended with more information based on different levels of analysis.
The data structures containing the results of localization and tracking routines can be saved as csv,
mat and HDF5 files. This helps users to work with the analyzed information using different softwares namely,
MATLAB and Microsoft Excel. HDF5 is a well-known format that is readable in different programming languages and supports
large, complex, heterogeneous data. HDF5 uses a "file directory" like structure that allows users to organize data within
the file in structured ways and to embed metadata as well, making it self-describing.

PiSCAT:
----------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   tutorials
   code_reference
   bibliography

Core Contributors
-----------------

1. **Houman Mirzaalian Dastjerdi**: The main contributor and developer of this package (including GUI) is Houman Mirzaalian Dastjerdi. PiSCAT is part of his Ph.D. thesis in collaboration between the Sandoghdar group at Max Planck Institute for the science of light and the Chair for Computer Science 10 (LSS).Houman can be reached at houman.mirzaalian-dastjerdi@mpl.mpg.de.

2. **Reza Gholami Mahmoodabadi**: Reza contributed in the design and development of the following modules: Analysis, Background Correction, InputOutput, Localization, Preprocessing and trajectory. He can be reached at reza.gholami@mpl.mpg.de.


Bibliography
------------
1. Lindfors, Kalkbrenner, et al. "Detection and spectroscopy of gold nanoparticles using supercontinuum white light confocal microscopy." Physical review letters 93.3 (2004): 037401.

2. Taylor, Richard W., and Vahid Sandoghdar. "Interferometric scattering microscopy: seeing single nanoparticles and molecules via rayleigh scattering." Nano letters 19.8 (2019): 4827-4835.

3. https://en.wikipedia.org/wiki/Interferometric_scattering_microscopy

4. Piliarik, Marek, and Vahid Sandoghdar. "Direct optical sensing of single unlabelled proteins and super-resolution imaging of their binding sites." Nature communications 5.1 (2014): 1-8.

5. Gemeinhardt, André, et al. "Label-free imaging of single proteins secreted from living cells via iSCAT microscopy." JoVE (Journal of Visualized Experiments) 141 (2018): e58486.

6. McDonald, Matthew P., et al. "Visualizing single-cell secretion dynamics with single-protein sensitivity." Nano letters 18.1 (2018): 513-519.

7. Young, Gavin, et al. "Quantitative mass imaging of single biological macromolecules." Science 360.6387 (2018): 423-427.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

