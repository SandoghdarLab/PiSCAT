from piscat.iPSF_model.VectorialDiffractionIntegral import VectorialDiffractionIntegral
import math
import numpy as np


class ScatteredFieldDifferentialPhase:

    def __init__(self, p, Xp, z_focus, nz, nx):
        """
        To calculate the Electric field scattered from a nanoparticle using a vectorial diffraction formulation.

        References
        ----------
        "Point spread function in interferometric scattering microscopy (iSCAT). Part I: aberrations in defocusing and axial localization.",
        Reza Gholami Mahmoodabadi, Richard W. Taylor, Martin Kaller, Susann Spindler, Mahdi Mazaheri, Kiarash Kasaian, and Vahid Sandoghdar
        Optics Express Vol. 28, Issue 18, pp. 25969-25988 (2020) https://doi.org/10.1364/OE.401374

        Parameters
        ----------
        p : Imaging setup parameters which is an object from the class imagingSetupParameters
        Xp : The 3D position of the scatterer in the sample, in unit of meters
        z_focus : The position of the focus. This could be an array as well.
        nz : The number of focus positions to calculate.
        nx : The lateral extent of the image given in pixels

        """
        self.p = p
        self.Xp = Xp
        self.z_focus = z_focus
        self.nz = nz
        self.nx = nx

        pass

    def calculate(self, r_):

        p = self.p
        Xp = self.Xp
        z_focus = self.z_focus
        nz = self.nz
        nx = self.nx

        vec_diffraction_integral = VectorialDiffractionIntegral(p, Xp, z_focus, nz, nx)
        vec_diffraction_integral.calculate()

        I0_re = vec_diffraction_integral.I0_int_re_op_3D
        I1_re = vec_diffraction_integral.I1_int_re_op_3D
        I2_re = vec_diffraction_integral.I2_int_re_op_3D
        I0_im = vec_diffraction_integral.I0_int_im_op_3D
        I1_im = vec_diffraction_integral.I1_int_im_op_3D
        I2_im = vec_diffraction_integral.I2_int_im_op_3D

        I0 = I0_re+1j*I0_im
        I1 = I1_re+1j*I1_im
        I2 = I2_re+1j*I2_im

        nxi = np.shape(I0)[2]     #size(I0,2)
        xpMesh = (nxi+1)/2
        nyi = np.shape(I0)[1]     #size(I0,1)
        ypMesh = (nyi+1)/2

        Y, X = np.meshgrid(np.arange(1, 1+nyi), np.arange(1, 1+nxi))
        pixel_size = p.pixel_size_physical/p.M
        xp_ = Xp[0]
        yp_ = Xp[1]
        Xnm = (X - xpMesh) * pixel_size - xp_
        Ynm = (Y - ypMesh) * pixel_size - yp_
        eps = np.finfo(float).eps
        PhiD_ = np.arctan(Xnm/(Ynm+eps))
        PhiD = np.tile(PhiD_, (np.shape(I0_re)[0], 1, 1)) #size(I0_re,3))
        #EFTppol = -1j *(I0 +  I2 .* cos(2*PhiD));
        # xpread = round(xpMesh+xp_/pixel_size)
        # ypread = round(ypMesh+yp_/pixel_size)
        xpread = round(xpMesh)
        ypread = round(ypMesh)

        M11 = (I0 + I2 * np.cos(2*PhiD))
        M12 = I2 * np.sin(2*PhiD)
        M13 = -2j * I1 * np.cos(PhiD)


        # M21 = (I2 .* sin(2*PhiD));
        # M22 = (I0 -  I2 .* cos(2*PhiD));
        # M23 = -2i*I1.*sin(PhiD);

        # dipole orientation
        phiP = 0       #azimuth
        thetP = math.pi/2   #zenith
        P11 = np.sin(thetP) * np.cos(phiP)
        P21 = np.sin(thetP) * math.sin(phiP)
        P31 = np.cos(thetP)
        EE11 = -1j*(M11*P11 + M12*P21 + M13*P31)
        #EE21 = -1j*(M21.*P11 + M22.*P21 + M23.*P31);

        angsEE11 = np.angle(np.squeeze(EE11[:, (xpread-r_-1):(xpread+r_) , (ypread-r_-1):(ypread+r_)]))
        ampsEE11 = np.abs(np.squeeze(EE11[:, (xpread-r_-1):(xpread+r_), (ypread-r_-1):(ypread+r_)]))

        return ampsEE11, angsEE11