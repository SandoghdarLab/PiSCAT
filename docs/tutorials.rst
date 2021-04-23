Introduction to the tutorials for protein detection analysis
============================================================

In this set of tutorials, we pick out few exemplary iSCAT videos and demonstrate how PiSCAT functionalities can be used in order to detect single unlabelled proteins via wide-field iSCAT microscopy to begin with and continue all the way to performing quantitative analysis on the detected particles. In doing so, we also present a number of necessary preprocessing tools together with various insightful visualization modules in PiSCAT.

The demo iSCAT videos that we use here, have been recorded with a `Photonfocus <https://www.photonfocus.com/products/camerafinder/camera/mv1-d1024e-160-cl/>`_  CMOS camera running at 4 kHz and imaging a Region Of Interest (ROI) of 128x128 pixels which is about 5.6x5.6 µm^2 given a magnification of 242X, i.e., the pixel size on the object side is 43.7 nm. The laser operates at a wavelength of 420 nm and the Numerical Aperture (NA) of the objective lens is 1.4. At the acquisition time, every recorded frame is the average of 10 consecutive frames, so effectively we record 400 fps. As a case study, we detect gold nanoparticles (GNP) as small as 5nm with a concentration of 25 nM which were injected into 0.1 M SA buffer onto an uncoated coverslip. Prior to such measurements, we also record a few calibration videos. For example, before injecting nanoparticles, we record iSCAT images of the empty cuvette filled only with the medium. These videos are called blank or control measurements. We also record camera frames while its shutter is closed, just to see how much signal we get in such dark frames under no incident light condition. These demo videos are however quite large and therefore we give the option to the users to download them when they are needed in the tutorials.

The first processing step tackles Intensity fluctuations as they play a role in the formation of very tiny signals in iSCAT. Mechanical vibrations or thermal drifts, for example, can cause temporal instability in the power of a laser beam. In sensitive iSCAT applications as in single protein detection experiments, a small change in the incident light is picked up by the microscope and can limit the extent to which we could average frames. The well depth of pixels in CMOS cameras is finite and this puts an upper limit to the number of photons one can integrate in a single frame. Thus, averaging high number of frames is necessary in order to achieve a detectable signal for very weak scattering objects with contrast less than 1%.

In the first tutorial, we work with an iSCAT demo video and normalize the pixel values in each frame to the sum of all pixels in the same frame to create a power normalized video in which the temporal instability of the laser light is suppressed. We import PiSCAT modules, run some basic checks on the acquisition process, suppress the temporal instability of the laser light and use some of the basic data visualization tools in PiSCAT.

We discuss about the fluctuations in the recorded signal (e.g. shot noise) whose origins lies in optics which can be smoothed for example through frame averaging. PiSCAT contains a variety of efficient functionalities to tackle noises of such nature and in the second tutorial these methods will be introduced and applied to iSCAT videos.

In the last tutorial we go through the entire analysis pipeline for the detection of single proteins. After performing the above mentioned preprocessing analysis on the 5nm GNP measurements, DLSs are localized in each frame of the video and then linked together to build up trajectories. We later examine the spatio-temporal characteristics of the candidate particles and filter out the outliers. In the next step, the landing profile of a protein is extracted from the intensity trace of the central pixel of the protein in its entire trajectory. The particle contrast can then be estimated via several methods available in PiSCAT. In case the injected sample consists of more than one type of protein, there could exist a number of sub-populations in the distribution of the detected particles. We finally show how to use clustering routines to distinguish multiple modes in the contrast histograms.




---------
Tutorials
---------

.. toctree::
    :maxdepth: 1

    Tutorial1/Tutorial1.md
    Tutorial2/Tutorial2.md
    Tutorial3/Tutorial3.md
