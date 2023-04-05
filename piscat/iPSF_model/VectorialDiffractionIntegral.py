import cmath
import math

import numpy as np
import scipy.special

from piscat.iPSF_model.L_theta import L_theta


class VectorialDiffractionIntegral:
    def __init__(self, p, Xp=[0, 0, 0], z_=0, nz=1, nx=513):
        """A vectorial diffraction formulation to model the scattered light
        from a nanoparticle.

        References
        ----------
        "Point spread function in interferometric scattering microscopy
        (iSCAT). Part I: aberrations in defocusing and axial localization.",
        Reza Gholami Mahmoodabadi, Richard W. Taylor, Martin Kaller, Susann
        Spindler, Mahdi Mazaheri, Kiarash Kasaian, and Vahid Sandoghdar Optics
        Express Vol. 28, Issue 18, pp. 25969-25988 (2020)
        https://doi.org/10.1364/OE.401374

        Parameters
        ----------
        p : Imaging setup parameters which is an object from the class imagingSetupParameters.
        Xp : The 3D position of the scatterer in the sample, in unit of meters.
        z_ : The position of the focus. This could be an array as well.
        nz : The number of focus positions to calculate.
        nx : The lateral extent of the image given in pixels.

        """
        self.p = p
        self.xp_ = Xp[0]
        self.yp_ = Xp[1]
        self.zp_ = Xp[2]
        self.z_ = z_
        self.nz_ = nz
        self.nx = nx
        xystep_ = p.pixel_size_physical / p.M
        self.xystep_ = xystep_
        xymax_ = (nx - 1) / 2
        xymax_ = int(xymax_)
        self.xymax_ = xymax_

        self.xp_ *= 1 / xystep_
        self.yp_ *= 1 / xystep_

        rn = 1 + int(math.sqrt(self.xp_**2 + self.yp_**2))
        rn = int(rn)

        rmax_ = math.ceil(math.sqrt(2.0) * xymax_) + rn + 1
        self.rmax_ = rmax_
        npx_ = (2 * xymax_ + 1) ** 2
        npx_ = int(npx_)

        N = nx * nx * nz
        self.I0_int_re_op_1D = np.zeros((N))
        self.I1_int_re_op_1D = np.zeros((N))
        self.I2_int_re_op_1D = np.zeros((N))
        self.I0_int_im_op_1D = np.zeros((N))
        self.I1_int_im_op_1D = np.zeros((N))
        self.I2_int_im_op_1D = np.zeros((N))

        self.I0_int_re_op_3D = np.zeros((nz, nx, nx))
        self.I1_int_re_op_3D = np.zeros((nz, nx, nx))
        self.I2_int_re_op_3D = np.zeros((nz, nx, nx))
        self.I0_int_im_op_3D = np.zeros((nz, nx, nx))
        self.I1_int_im_op_3D = np.zeros((nz, nx, nx))
        self.I2_int_im_op_3D = np.zeros((nz, nx, nx))

        self.I0_int_re = np.zeros((nz, rmax_))
        self.I1_int_re = np.zeros((nz, rmax_))
        self.I2_int_re = np.zeros((nz, rmax_))
        self.I0_int_im = np.zeros((nz, rmax_))
        self.I1_int_im = np.zeros((nz, rmax_))
        self.I2_int_im = np.zeros((nz, rmax_))

    def calculate(self):
        p = self.p
        phC = 0

        ci = self.zp_ * (1 - p.ni / p.ns) + p.ni * (p.tg0 / p.ng0 + p.ti0 / p.ni0 - p.tg / p.ng)
        ud = 3
        L_th = np.zeros((2), dtype=complex)

        for k in range(0, self.nz_):  # defocusing loop
            L_th = L_theta(L_th, p.alpha, p, ci, self.z_[k], self.zp_)
            w_exp = np.abs(L_th[1])  # missing p.k0, multiply it into it later
            cst = 0.975
            while cst >= 0.9:
                L_th = L_theta(L_th, cst * p.alpha, p, ci, self.z_[k], self.zp_)
                if np.abs(L_th[1]) > w_exp:
                    w_exp = np.abs(L_th[1])
                cst -= 0.025

            w_exp *= p.k0

            for ri in range(0, self.rmax_):
                r = self.xystep_ * float(ri)
                constJ = p.k0 * r * p.ni

                if w_exp > constJ:
                    nSamples = 4 * int(1 + p.alpha * w_exp / math.pi)
                else:
                    nSamples = 4 * int(1 + p.alpha * constJ / math.pi)

                if nSamples < 20:
                    nSamples = 20

                step = p.alpha / float(nSamples)
                iconst = step / ud

                # Odd and evens of Simpson's rule
                sum_I0_even, sum_I1_even, sum_I2_even = self.simpsonsRuleVec(
                    ci, constJ, k, nSamples, p, phC, step, 0
                )
                sum_I0_odd, sum_I1_odd, sum_I2_odd = self.simpsonsRuleVec(
                    ci, constJ, k, nSamples, p, phC, step, 1
                )

                sum_I0_edge, sum_I1_edge, sum_I2_edge = self.theta_equal_to_alpha(
                    ci, constJ, k, p, phC
                )

                sum_I0 = sum_I0_even + sum_I0_odd + sum_I0_edge
                sum_I1 = sum_I1_even + sum_I1_odd + sum_I1_edge
                sum_I2 = sum_I2_even + sum_I2_odd + sum_I2_edge

                # Integrals complexValue
                self.I0_int_re[k, ri] = (sum_I0 * iconst).real
                self.I1_int_re[k, ri] = (sum_I1 * iconst).real
                self.I2_int_re[k, ri] = (sum_I2 * iconst).real
                self.I0_int_im[k, ri] = (sum_I0 * iconst).imag
                self.I1_int_im[k, ri] = (sum_I1 * iconst).imag
                self.I2_int_im[k, ri] = (sum_I2 * iconst).imag

            # Interpolate and fill up output arrays
            self.interpolation_xy_vec(k)

        self.I0_int_re_op_3D = np.reshape(self.I0_int_re_op_1D, (self.nz_, self.nx, self.nx))
        self.I1_int_re_op_3D = np.reshape(self.I1_int_re_op_1D, (self.nz_, self.nx, self.nx))
        self.I2_int_re_op_3D = np.reshape(self.I2_int_re_op_1D, (self.nz_, self.nx, self.nx))
        self.I0_int_im_op_3D = np.reshape(self.I0_int_im_op_1D, (self.nz_, self.nx, self.nx))
        self.I1_int_im_op_3D = np.reshape(self.I1_int_im_op_1D, (self.nz_, self.nx, self.nx))
        self.I2_int_im_op_3D = np.reshape(self.I2_int_im_op_1D, (self.nz_, self.nx, self.nx))
        pass

    def interpolation_xy_vec(self, k):
        x = np.arange(-self.xymax_, 1 + self.xymax_)
        y = np.arange(-self.xymax_, 1 + self.xymax_)
        X, Y = np.meshgrid(x, y)
        xis = X - self.xp_
        yis = Y - self.yp_
        rxs = np.sqrt(xis * xis + yis * yis)
        r0s = np.int64(rxs)
        drs = rxs - r0s
        drs[((r0s + 1) >= self.rmax_)] = 0

        I0_int_re_op_1Ds = drs * self.I0_int_re[k, r0s + 1] + (1.0 - drs) * self.I0_int_re[k, r0s]

        I1_int_re_op_1Ds = drs * self.I1_int_re[k, r0s + 1] + (1.0 - drs) * self.I1_int_re[k, r0s]
        I2_int_re_op_1Ds = drs * self.I2_int_re[k, r0s + 1] + (1.0 - drs) * self.I2_int_re[k, r0s]
        I0_int_im_op_1Ds = drs * self.I0_int_im[k, r0s + 1] + (1.0 - drs) * self.I0_int_im[k, r0s]
        I1_int_im_op_1Ds = drs * self.I1_int_im[k, r0s + 1] + (1.0 - drs) * self.I1_int_im[k, r0s]
        I2_int_im_op_1Ds = drs * self.I2_int_im[k, r0s + 1] + (1.0 - drs) * self.I2_int_im[k, r0s]

        dims = np.shape(I0_int_re_op_1Ds)
        self.I0_int_re_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I0_int_re_op_1Ds
        )
        self.I1_int_re_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I1_int_re_op_1Ds
        )
        self.I2_int_re_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I2_int_re_op_1Ds
        )
        self.I0_int_im_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I0_int_im_op_1Ds
        )
        self.I1_int_im_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I1_int_im_op_1Ds
        )
        self.I2_int_im_op_1D[(k * dims[0] * dims[1]) : ((k + 1) * dims[0] * dims[1])] += np.ravel(
            I2_int_im_op_1Ds
        )

    def theta_equal_to_alpha(self, ci, constJ, k, p, phC):
        # theta = alpha;
        sintheta = math.sin(p.alpha)
        costheta = math.cos(p.alpha)
        sqrtcostheta = math.sqrt(costheta)
        nsroot = cmath.sqrt((p.ns**2 - p.NA**2))
        ngroot = cmath.sqrt((p.ng**2 - p.NA**2))
        ts1ts2 = 4.0 * p.ni * costheta * ngroot
        tp1tp2 = ts1ts2
        tp1tp2 /= (p.ng * costheta + p.ni / p.ng * ngroot) * (
            p.ns / p.ng * ngroot + p.ng / p.ns * nsroot
        )
        ts1ts2 /= (p.ni * costheta + ngroot) * (ngroot + nsroot)
        bessel_0 = scipy.special.j0(constJ * sintheta) * sintheta * sqrtcostheta
        bessel_1 = scipy.special.j1(constJ * sintheta) * sintheta * sqrtcostheta
        if constJ != 0.0:
            bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0
        else:
            bessel_2 = 0.0
        bessel_0 *= ts1ts2 + tp1tp2 / p.ns * nsroot
        bessel_1 *= tp1tp2 * p.ni / p.ns * sintheta
        bessel_2 *= ts1ts2 - tp1tp2 / p.ns * nsroot

        con_ti_up = p.tg0 / p.ng0 - p.tg / p.ng + p.ti0 / p.ni0 - self.zp_ / p.ns
        con_refOPD_up = p.ng0 * p.tg0 + p.ni0 * p.ti0
        ti_up = self.zp_ - self.z_[k] + p.ni * con_ti_up
        refOPD_up = -1 * ((p.ng * p.tg + p.ni * ti_up) - con_refOPD_up)
        expW = cmath.exp(
            1j * p.k0 * p.ns * self.zp_
            + 1j * phC
            + 1j * p.k0 * refOPD_up
            + 1j
            * p.k0
            * (
                (ci - self.z_[k]) * cmath.sqrt((p.ni**2 - p.NA**2))
                + self.zp_ * nsroot
                + p.tg * ngroot
                - p.tg0 * math.sqrt((p.ng0**2 - p.NA**2))
                - p.ti0 * cmath.sqrt((p.ni0**2 - p.NA**2))
            )
        )
        sum_I0 = expW * bessel_0
        sum_I1 = expW * bessel_1
        sum_I2 = expW * bessel_2
        return sum_I0, sum_I1, sum_I2

    def simpsonsRuleVec(self, ci, constJ, k, nSamples, p, phC, step, oddVal):
        ran_ = np.arange(1, int(nSamples / 2) + oddVal)
        thetas = (2 * ran_ - oddVal) * step
        sinthetas = np.sin(thetas)
        costhetas = np.cos(thetas)
        sqrtcosthetas = np.sqrt(costhetas)
        ni2sin2thetas = p.ni**2 * sinthetas**2
        nsroots = np.sqrt(p.ns**2 - ni2sin2thetas + 0j)
        ngroots = np.sqrt(p.ng**2 - ni2sin2thetas + 0j)

        ts1ts2s = 4 * p.ni * costhetas * ngroots
        tp1tp2s = ts1ts2s
        tp1tp2s = np.divide(
            tp1tp2s,
            (p.ng * costhetas + p.ni / p.ng * ngroots)
            * (p.ns / p.ng * ngroots + p.ng / p.ns * nsroots),
        )
        ts1ts2s = np.divide(ts1ts2s, (p.ni * costhetas + ngroots) * (ngroots + nsroots))

        bessel_0s = (
            2 ** (1 + oddVal) * scipy.special.j0(constJ * sinthetas) * sinthetas * sqrtcosthetas
        )
        bessel_1s = (
            2 ** (1 + oddVal) * scipy.special.j1(constJ * sinthetas) * sinthetas * sqrtcosthetas
        )
        if constJ != 0:
            bessel_2s = 2 * bessel_1s / (constJ * sinthetas) - bessel_0s
        else:
            bessel_2s = 0 * bessel_1s

        bessel_0s = np.multiply(bessel_0s, (ts1ts2s + tp1tp2s / p.ns * nsroots))
        bessel_1s = np.multiply(bessel_1s, (tp1tp2s * p.ni / p.ns * sinthetas))
        bessel_2s = np.multiply(bessel_2s, (ts1ts2s - tp1tp2s / p.ns * nsroots))

        con_ti_up = p.tg0 / p.ng0 - p.tg / p.ng + p.ti0 / p.ni0 - self.zp_ / p.ns
        con_refOPD_up = p.ng0 * p.tg0 + p.ni0 * p.ti0
        ti_up = self.zp_ - self.z_[k] + p.ni * con_ti_up
        refOPD_up = -1 * ((p.ng * p.tg + p.ni * ti_up) - con_refOPD_up)

        expWs = np.exp(
            1j * p.k0 * p.ns * self.zp_
            + 1j * phC
            + 1j * p.k0 * refOPD_up
            + 1j
            * p.k0
            * (
                (ci - self.z_[k]) * p.ni * costhetas
                + self.zp_ * nsroots
                + p.tg * ngroots
                - p.tg0 * np.sqrt(p.ng0**2 - ni2sin2thetas + 0j)
                - p.ti0 * np.sqrt(p.ni0**2 - ni2sin2thetas + 0j)
            )
        )

        sum_I0 = np.sum(expWs * bessel_0s)
        sum_I1 = np.sum(expWs * bessel_1s)
        sum_I2 = np.sum(expWs * bessel_2s)

        return sum_I0, sum_I1, sum_I2
