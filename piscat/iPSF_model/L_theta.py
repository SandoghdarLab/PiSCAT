import cmath

import numpy as np


def L_theta(L, theta, p, ci, z, z_p):
    ni2sin2theta = p.ni**2 * cmath.sin(theta) ** 2
    sroot = cmath.sqrt(p.ns**2 - ni2sin2theta)
    groot = cmath.sqrt(p.ng**2 - ni2sin2theta)
    g0root = cmath.sqrt(p.ng0**2 - ni2sin2theta)
    i0root = cmath.sqrt(p.ni0**2 - ni2sin2theta)
    L[0] = (
        p.ni * (ci - z) * cmath.cos(theta)
        + z_p * sroot
        + p.tg * groot
        - p.tg0 * g0root
        - p.ti0 * i0root
    )
    L[1] = (
        p.ni
        * cmath.sin(theta)
        * (
            z
            - ci
            + p.ni
            * cmath.cos(theta)
            * (p.tg0 / g0root + p.ti0 / i0root - p.tg / groot - z_p / sroot)
        )
    )

    return L
