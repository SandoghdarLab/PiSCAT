from __future__ import print_function

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy import signal
from tqdm.autonotebook import tqdm

from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Preproccessing import patch_genrator


class RadialCenter():

    def __init__(self):

        """
        The RadialCenter localization algorithm is implemented in Python. 

        References
        ----------
        Parthasarathy, R. Rapid, accurate particle tracking by calculation of radial symmetry centers. Nat Methods 9, 724–726 (2012).
        https://doi.org/10.1038/nmeth.2071
        """

        self.cpu = CPUConfigurations()

        self.patch_gen = None
        self.patch = None
        self.video_shape = None
        self.probility_map = None

    def psf_center_all_frames(self, video):
        p_center_all = []

        if self.cpu.parallel_active is True:
            print('\n---Center localization with parallel loop---')

            result = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(
                delayed(self.radialcenter)(video[i_, :, :]) for i_ in range(video.shape[0]))
            columns_name = ['x', 'y', 'sigma']
            df_psf = pd.DataFrame(result, columns=columns_name)
        else:
            print('\n---Center localization without parallel loop---')
            for i_ in range(video.shape[0]):
                xc, yc, sigma = self.radialcenter(video[i_, :, :])
                p_center_all.append([xc, yc, sigma])

            columns_name = ['x', 'y', 'sigma']
            df_psf = pd.DataFrame(p_center_all, columns=columns_name)
        return df_psf

    def patch_genrator(self, original_video, patch_size=16, strides=4):
        self.video_shape = original_video.shape
        self.patch_gen = patch_genrator.ImagePatching()
        self.patch = self.patch_gen.split_weight_matrix(original_video, patch_size=patch_size, strides=strides)

    def radialcenter_probility_map(self):
        if self.patch is not None and self.patch_gen is not None and self.patch_gen is not None:
            patch_probility = []
            for p_ in tqdm(self.patch):
                tmp_ = np.zeros(p_.shape)
                self.parallel_active = False
                df_psf = self.psf_center_all_frames(p_)
                y_list = df_psf['y'].to_list()
                x_list = df_psf['y'].to_list()
                f_list = list(range(0, df_psf.shape[0], 1))

                y_list_ = [int(v_) for v_ in y_list]
                x_list_ = [int(v_) for v_ in x_list]

                try:
                    tmp_[f_list, y_list_, x_list_] = 1
                    patch_probility.append(tmp_)

                except:
                    patch_probility.append(tmp_)

            self.probility_map = self.patch_gen.reconstruction_weight_matrix(self.video_shape, new_patch=patch_probility)
            return self.probility_map
        else:
            print('you need to call patch_genrator method!')

    def radialcenter(self, Image):
        Image = Image.astype(np.float64)
        Nx = Image.shape[0]
        Ny = Image.shape[1]

        xm_onerow = np.arange(-((Nx-1)/2.0)+0.5, ((Nx-1)/2.0)+0.5, 1, dtype=np.int)
        xm = np.broadcast_to(xm_onerow, (Nx-1, Ny-1))

        ym_onecol = np.arange(-((Ny-1)/2.0)+0.5, ((Ny-1)/2.0)+0.5,  dtype=np.int)
        ym = np.transpose(np.broadcast_to(ym_onecol, (Nx-1, Ny-1)))

        dIdu = Image[0:Ny-1, 1:Nx] - Image[1:Ny, 0:Nx-1]
        dIdv = Image[0:Ny-1, 0:Nx-1] - Image[1:Ny, 1:Nx]

        h = (1/9) * np.ones((3, 3))
        fdu = signal.convolve2d(dIdu, h, mode='same')
        fdv = signal.convolve2d(dIdv, h, mode='same')
        dImag2 = np.multiply(fdu, fdu) + np.multiply(fdv, fdv)

        m = -np.divide((fdv + fdu), (fdu - fdv))

        NNanm = np.sum(np.isnan(m))
        if NNanm > 0:
            unsmoothm = np.divide((dIdv + dIdu), (dIdu - dIdv))
            m[np.isnan(m)] = unsmoothm[np.isnan(m)]

        NNanm = np.sum(np.isnan(m))
        if NNanm > 0:
            m[np.isnan(m)] = 0

        try:
            m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])
        except:
            m = np.divide((dIdv + dIdu), (dIdu-dIdv))

        b = ym - np.multiply(m, xm)
        sdI2 = np.sum(dImag2)
        xcentroid = np.sum(np.sum(np.multiply(dImag2, xm))) / sdI2
        ycentroid = np.sum(np.sum(np.multiply(dImag2, ym))) / sdI2
        w = np.divide(dImag2, np.sqrt(np.multiply((xm - xcentroid), (xm - xcentroid)) + np.multiply((ym - ycentroid), (ym - ycentroid))))

        xc, yc = self.lsradialcenterfit(m, b, w)

        xc = xc + (Nx + 1) / 2.0
        yc = yc + (Ny + 1) / 2.0

        Isub = Image - np.min(Image)
        x = np.linspace(0, Nx - 1, Nx, dtype=np.int)
        y = np.linspace(0, Ny - 1, Ny, dtype=np.int)
        px, py = np.meshgrid(x, y)
        xoffset = px - xc
        yoffset = py - yc
        r2 = np.multiply(xoffset, xoffset) + np.multiply(yoffset, yoffset)
        sigma = np.sqrt(np.sum(np.sum(np.multiply(Isub, r2))) / np.sum(Isub)) / 2

        return xc, yc, sigma

    def lsradialcenterfit(self, m, b, w):

        wm2p1 = np.divide(w, np.multiply(m, m) + 1)
        sw = np.sum(wm2p1)
        smmw = np.sum(np.multiply(m, np.multiply(m, wm2p1)))
        smw = np.sum(np.multiply(m, wm2p1))
        smbw = np.sum(np.multiply(m, np.multiply(b, wm2p1)))
        sbw = np.sum(np.multiply(b, wm2p1))
        det = smw * smw - smmw * sw
        xc = (smbw * sw - smw * sbw) / det
        yc = (smbw * smw - smmw * sbw) / det
        return xc, yc
