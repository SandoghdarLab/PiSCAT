PiSCAT UAI plugin
=================

The new PiSCAT plugin, unsupervised anomaly identification (UAI), uses a deep neural network and classical machine
learning to combine spatial-temporal features to track particles with low SNR in a background with noise structures
similar to real iSCAT PSF (iPSF). The UAI deep neural networks are robust and fully automated algorithms that manipulate
features of iPSFs and background speckles to the domain that can separate the footprint of the iPSF from background
speckles. Since this method is unsupervised, it can be used for different applications with only fine-tuning of the
hyperparameters. This method illustrates remarkable results in pushing the sensitivity limit in label-free detection
of single proteins below 10 kDa.

Existing algorithms for detecting biomolecules using a label-free optical technique cannot detect single proteins
with a molecular mass of fewer than 66 KDa with high precision
`[1] <https://joss.theoj.org/papers/10.21105/joss.04024>`_, `[2] <https://iopscience.iop.org/article/10.1088/1361-6463/ac2f68>`_. Although
prior PiSCAT pipeline algorithms included spatial-temporal information, they operated in two steps with equal
weights `[2] <https://iopscience.iop.org/article/10.1088/1361-6463/ac2f68>`_. The modified version of this technique
changed the localization part to reduce false positive detection by utilizing the capabilities of ML/DNN to mix
spatial-temporal information with unequal weights in a simultaneous fashion. UAI demonstrates that an unsupervised
machine learning approach for anomaly detection increases the mass sensitivity limit to below 10 kDa, i.e., by a
factor of four compared to the state-of-the-art from other methods. Thus, this plugin enables the identification of
tiny nanoparticles by optical methods. The details and findings of this pipeline are published in the journal
Nature Methods with the title  "Self-supervised machine learning pushes the sensitivity limit in label-free detection
of single proteins below 10 kDa" `[3] <https://www.nature.com/articles/s41592-023-01778-2>`_.



Installation
------------

Since this is a plugin for PiSCAT, first, you should install
PiSCAT (see `PiSCAT installation <https://piscat.readthedocs.io/installation.html>`_); after that, this
plugging can add to PiSCAT.

1. To install PiSCAT using PyPi, enter the following command in the console (you need the last version of piscat):

```
pip install piscat
```

2. To add PiSCAT Plugins, enter the following command in the console in the same python environment you install piscat:

```
python -m piscat.Plugins UAI
```


Running PiSCAT plugin Tutorials
--------------------------------

Once the installation is done and the python environment is activated, enter the following command in the console:

```
python -m piscat.Tutorials
```

This plugin has two tutorials.

[1] `Tutorial for anomaly detection (Hand crafted feature matrix) <https://piscat.readthedocs.io/Tutorial_UAI_1/Tutorial_UAI_1.html>`_

[2] `Tutorial for anomaly detection (DNN feature matrix) <https://piscat.readthedocs.io/Tutorial_UAI_2/Tutorial_UAI_2.html>`_

license
-------

This plugin has a separate license from the general PiSCAT. The license will be added very soon. Meanwhile, if you have any questions regarding the
license, please contact with the following emails:

[1] houman.mirzaalian-dastjerdi@mpl.mpg.de

[2] matthias.baer@mpl.mpg.de


References
----------

[1] `Mirzaalian Dastjerdi, H., Gholami Mahmoodabadi, R., Bär, M., Sandoghdar, V., & Köstler, H. (2022). PiSCAT: A Python Package for Interferometric Scattering Microscopy. The Journal of Open Source Software, 7. <https://joss.theoj.org/papers/10.21105/joss.04024>`_

[2] `Dastjerdi, H. M., Dahmardeh, M., Gemeinhardt, A., Mahmoodabadi, R. G., Köstler, H., & Sandoghdar, V. (2021). Optimized analysis for sensitive detection and analysis of single proteins via interferometric scattering microscopy. Journal of Physics D: Applied Physics, 55, 054002. <https://iopscience.iop.org/article/10.1088/1361-6463/ac2f68>`_

[3] `Dahmardeh, M., Mirzaalian Dastjerdi, H., Mazal, H. et al. Self-supervised machine learning pushes the sensitivity limit in label-free detection of single proteins below 10kDa. Nat Methods (2023). <https://www.nature.com/articles/s41592-023-01778-2>`_















