from __future__ import print_function

from glob import glob
from joblib import Parallel, delayed
from piscat.InputOutput.reading_videos import video_reader
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from tqdm.autonotebook import tqdm

import numpy as np
import os


class Image2Video():

    def __init__(self, path, file_format, width_size, height_size, image_type, reader_type):
        """
        This class reads images of a particular kind from a folder and concatenates them into a single NumPy array.

        Parameters
        ----------
        path: str
            The directory path that includes images.

        file_format: str
            Postfix of image names.

        width_size: int
            For binary images, it specifies the image width.

        height_size: int
            For binary images, it specifies the image height.

        image_type: str

            * "i"  (signed) integer, "u" unsigned integer, "f" floating-point.
            * "<" active little-endian.
            * "1" 8-bit, "2" 16-bit, "4" 32-bit, "8" 64-bit.

        reader_type: str
            Specify the video/image format to be loaded.

            * `'binary'`: use this flag to load binary
            * `'tif'`: use this flag to load tif
            * `'avi`': use this flag to load avi
            * `'png'`: use this flag to load png
        """
        self.cpu = CPUConfigurations()

        self.reader_type = reader_type

        self.path = os.path.join(path, file_format)
        self.width_size = width_size
        self.height_size = height_size
        self.type = image_type

        self.path_list = glob(self.path)
        self.img_bin = []
        temp = Parallel(n_jobs=self.cpu.n_jobs, backend=self.cpu.backend, verbose=self.cpu.verbose)(delayed(
            self.parallel_read_img)(x) for x in tqdm(self.path_list))
        self.video = np.asarray(temp)

    def __call__(self):
        return self.video

    def parallel_read_img(self, x):
        tmp = video_reader(file_name=x, type=self.reader_type,
                            img_width=self.width_size, img_height=self.height_size,
                            image_type=self.type, s_frame=0, e_frame=-1)
        if len(tmp.shape) == 3:
            return tmp[0]
        elif len(tmp.shape) == 0:
            return tmp
        else:
            raise ValueError('The shape {} does not correct'.format(tmp.shape))




