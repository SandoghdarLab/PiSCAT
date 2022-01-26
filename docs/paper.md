---
title: 'PiSCAT: A Python Package for Interferometric Scattering Microscopy'
tags:
  - Python
  - microscopy
  - interferometric microscopy
authors:
  - name: Houman Mirzaalian Dastjerdi
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2, 3"
  - name: Reza Gholami Mahmoodabadi 
    affiliation: "1, 2" 
  - name: Matthias Bär
    affiliation: "1, 2"  
  - name: Vahid Sandoghdar
    affiliation: "1, 2, 4"
  - name: Harald Köstler
    affiliation: "3, 5"
affiliations:
 - name: Max Planck Institute for the Science of Light, 91058 Erlangen, Germany.
   index: 1
 - name: Max-Planck-Zentrum für Physik und Medizin, 91058 Erlangen, Germany.
   index: 2
 - name: Department of Computer Science, Friedrich-Alexander University Erlangen-Nürnberg, 91058 Erlangen, Germany.
   index: 3
 - name: Department of Physics, Friedrich-Alexander University Erlangen-Nürnberg, 91058 Erlangen, Germany.
   index: 4
 - name: Erlangen National High Performance Computing Center (NHR@FAU).
   index: 5
date: 26 November 2021
bibliography: paper.bib


---

# Summary

Interferometric scattering (iSCAT) microscopy allows one to image and track nano-objects with a nanometer spatial 
and microsecond temporal resolution over arbitrarily long measurement 
times [@lindfors2004PRL], [@taylor2019interferometric], [@RichardBook]. A key advantage of this technique over 
the well-established fluorescence methods is the indefinite photostability of the scattering phenomenon in 
contrast to the photobleaching of fluorophores. This means that one can perform very long measurements. Moreover, 
scattering processes are linear and thus do not saturate. This leads to larger signals than is possible from a 
single fluorophore. As a result, one can image at a much faster rate than in fluorescence microscopy. Furthermore, 
the higher signal makes it possible to localize a nano-object with much better spatial precision. The remarkable 
sensitivity of iSCAT, however, also brings about the drawback that one obtains a rich speckle-like background 
from other nano-objects in the field of view.

# Statement of need

The aim of this project is to explore the potential of powerful image processing tools <span style="color: blue;">(such as computer vision, 
background correction, and object tracking)</span> to address challenges in the analysis of iSCAT images. The use of 
existing libraries and packages turns out not to be satisfactory. Here, we present a software platform implemented 
in Python for processing the iSCAT images, named PiSCAT. In order to make this framework accessible to a wide range 
of applications, we have developed a GUI (Graphical User Interface), which enables the users to conduct the analysis 
of interest, regardless of their programming skills. Reading different image 
or video formats, background correction (e.g removing fixed pattern noise (FPN), differential rolling average (DRA)), 
point spread function (PSF) detection, linking and extraction of trajectories of target PSFs are amongst the 
key features of PiSCAT. PiSCAT is also optimized for parallel coding to decrease the analysis time. 


# Concept and Structure of PiSCAT

PiSCAT is a unified python-based package that presents classical computer vision and machine learning analysis for 
various applications such as unlabeled protein detection or three-dimensional (3D) tracking of nanoparticles.
Protein detection based on the linear relationship of contrast and particle mass is one of the popular iSCAT 
applications [@Dastjerdi2021], [@piliarik2014direct]. But due to camera hardware 
limitations, a protein cannot be visualized in a single frame (see \autoref{fig:piscat}a-(I)). To remove the 
static background, this method subtracts the averages of two consecutive batches (see \autoref{fig:piscat}a-(II)). 
Additionally, we need algorithms to localize and track particles in spatial and temporal domains to correctly read 
the contrasts and provide a histogram from them (see \autoref{fig:piscat}a-(III)).
Another application of iSCAT is to monitor the diffusion of single nanoparticles such as lipid vesicles across the surface of 
giant unilamellar vesicles (GUVs) [@spindler2018high](see \autoref{fig:piscat}a-(IV)). As in most iSCAT signal processing 
steps, we first separate the nanoparticle signals from the background signal present in the raw videos. In comparison 
to diffusing nanoparticles, the slowly varying GUV signal can be estimated using temporal median and then removed during 
background correction. Although this method partially removes background rings and can reveal nanoparticles as small 
as 20 nm, one is still left with a residual background signal that creates some difficulties for quantitative tracking. To address 
this issue, we developed a background correction routine based on clustering of the similar frames in a video to get around small 
fluctuations in the background signal. In order to compensate slightly stronger fluctuations in the background signal, we 
additionally perform affine image registration [@spindler2018high] (see \autoref{fig:piscat}a-(V)). High frame rate 
imaging in iSCAT provides us enough signal redundancy in time so that we can afford to select the frames in a 
video sparsely. We therefore call this routine Sparse In Time Affine Registration (SITAR).

![ (a) Some iSCAT protein images and the results of PiSCAT analysis: 
(I) The raw camera image of an iSCAT video frame from BSA protein samples. 
(II) After background correction, the frame contains iPSFs of a single BSA protein (highlighted with yellow arrow). 
In (I) and (II), the scale bars correspond to 1.5 $\mu$m. 
(III) The corresponding iPSF's temporal contrast in (a-II).  
(IV) iSCAT image of a part of GUV which shows a Newton ring-like pattern. 
The GUV signal is treated as the background. After its removal one obtains (V) in which a yellow arrow points to the 
iPSF of a gold nanoparticle. 
In (IV) and (V), the scale bars correspond to 4 $\mu$m. Such iPSFs can be localized across several frames and linked 
together to obtain a 3D trajectory as shown in (VI). Inset image illustrates 3D position of several particles that are 
color coded differently. 
(b) Modelling iPSF images of a nanoparticle travelling in the axial direction using iPSF module: 
(I) 2D image of the iPSF of a  particle positioned at 3 $\mu$m with the focus positioned at 3 $\mu$m. The particle 
height is swept axially to form an axial stack made of 2D iPSF images. This stack is then sliced along its meridional 
plane and plotted in (II). The amplitude and differential phase of the same plane is shown in (III) and (IV) respectively. 
\label{fig:piscat}](Fig1.png){ width=100% }

PiSCAT provides several algorithms for localization and tracking of nanoparticles (see \autoref{fig:piscat}a-(VI)). The 
current version offers several image processing techniques to mainly accomplish the following: 1) Background correction 
that improves the distinction of the signal of interest from the background, 2) Localizing particles in (X,Y,Z axis and time). 
To achieve this, we need the combination of different computer vision and machine learning algorithms with physical 
models such as the iSCAT point-spread function (iPSF) model (see \autoref{fig:piscat}b-(I-IV)) [@gholami2020iPSF]. 
\autoref{fig:piscat_structure} depicts the structure of the PiSCAT package. 
<font color="blue">PiSCAT uses a set of dependencies that assist us in achieving various goals. The PiSCAT 
repository contains this list.</font>

![Diagram of the PiSCAT structure: There are nine main modules in the PiSCAT package (blue boxes). Green and yellow 
represent classes and python scripts (functions), respectively. 
For the rest of the package modules, ``CPUConfiguration`` and ``CameraParameters`` are used to 
configure CPU parallel 
and camera parameters. \label{fig:piscat_structure}](Fig2.png){ width=100% }


# Ongoing Research and Dissemination

Our future work aims to add GPU kernels by using code generation technologies. This kind of design helps us to add 
real-time processing and different deep neural networks (DNN) <font color="blue">and machine learning</font> to this package in order to facilitate feature extraction 
and distinguish target signals from the noisy background even at low SNR.


# Acknowledgements

We are grateful to Mahyar Dahmardeh and André Gemeinhardt for assisting with test data preparation and determining 
camera features and parameters. We also would like to thank Katharina König for her insightful comments on the 
documentation, as well as Richard Taylor for his help in designing the GUI logo, and Susann Spindler for providing the 
GUV video. This work was supported by the Max Planck Society and NHR@FAU.

# References
