from __future__ import print_function
from skimage import io
import os
import pandas as pd
import numpy as np
import cv2


def video_reader(file_name, type='binary', img_width=128, img_height=128, image_type=np.dtype('<f8'), s_frame=0, e_frame=-1):
    """
    This is a wrapper that can be used to call various video/image readers.

    Parameters
    ----------
    file_name: str
        Path of video and file name, e.g. test.jpg.

    type: str
        Define the video/image format to be loaded.

            * 'binary': use this flag to load binary
            * 'tif': use this flag to load tif
            * 'avi': use this flag to load avi
            * 'png': use this flag to load png

    optional_parameters:
        These parameters are used when video 'bin_type' define as binary.

        img_width: int
             For binary images, it specifies the image width.

        img_height: int
            For binary images, it specifies the image height.

        image_type: str
            Numpy.dtype('<u2') --> video with uint16 pixels data type

            * "i"  (signed) integer, "u" unsigned integer, "f" floating-point
            * "<" active little-endian
            * "1" 8-bit, "2" 16-bit, "4" 32-bit, "8" 64-bit

        s_frame: int
            Video reads from this frame. This is used for cropping a video.

        e_frame: int
            Video reads until this frame. This is used for cropping a video.

    Returns
    -------
    @returns: NDArray
        The video/image
    """
    if type == 'binary':
        video = read_binary(file_name, img_width=img_width, img_height=img_height, image_type=image_type, s_frame=s_frame, e_frame=e_frame)
    elif type == 'tif':
        video = read_tif(file_name)
    elif type == 'avi':
        video = read_avi(file_name)
    elif type == 'png':
        video = read_png(file_name)
    return video


def read_binary(file_name, img_width=128, img_height=128, image_type=np.dtype('<f8'), s_frame=0, e_frame=-1):
    """
    This function reads binary video.

    Parameters
    ----------
    file_name: str
        Path and name of binary video.

    img_width: int
         It specifies the image width.

    img_height: int
        It specifies the image height.

    image_type: str

        * "i"  (signed) integer, "u" unsigned integer, "f" floating-point
        * "<" active little-endian
        * "1" 8-bit, "2" 16-bit, "4" 32-bit, "8" 64-bit

    s_frame: int
        Video reads from this frame. This is used for cropping a video.

    e_frame: int
        Video reads until this frame. This is used for cropping a video.

    Returns
    -------
    @returns: NDArray
        The video is 3D-numpy (number of frames, width, height).
    """
    if e_frame == -1:
        num_selected_frames = -1
    else:
        num_selected_frames = (e_frame - s_frame) * (img_width * img_height)
    offset = s_frame * (img_width * img_height)

    img_bin = np.fromfile(file_name, image_type, count=num_selected_frames, sep='', offset=offset)
    number_of_frame = int(img_bin.size / (img_width * img_height))
    return np.reshape(img_bin, (number_of_frame, img_width, img_height))


def read_tif(filename):
    """
    Reading image/video with TIF format.

    Parameters
    ----------
    file_name: str
        Path and name of TIF image/video.

    Returns
    -------
    @returns: NDArray
        The video is 3D-numpy (number of frames, width, height).

    """
    return io.imread(filename)


def read_avi(filename):
    """
    Reading video with AVI format.

    Parameters
    ----------
    file_name: str
        Path and name of AVI video.

    Returns
    -------
    video: NDArray
        The video is 3D-numpy (number of frames, width, height).
    """

    cap = cv2.VideoCapture(filename)
    frames_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_list.append(gray)
        else:
            break

    cap.release()
    video = np.asarray(frames_list)
    return video


def read_png(filename):
    """
    Reading image with PNG format.

    Parameters
    ----------
    file_name: str
        Path and name of PNG image.

    Returns
    -------
    @returns: NDArray
        The video is 2D-numpy (width, height).
    """

    return io.imread(filename)


class DirectoryType:

    def __init__(self, dirName, type_file):
        """
        Based on the type file description, this class generates a dataframe
        containing 'Directory,' 'Folder,' and 'File' from all files below the define directory.

        Parameters
        ----------
        dirName: str
            A directory that is used to look for files of a particular type file.

        type_file: str
            The type of files that the user is looking for.
        """
        self.dirName = dirName
        self.type = type_file

        # Get the list of all files in directory tree at given path
        fileNames = []
        dirPaths = []
        dirnames = []
        for (dirpath, dirname, filenames) in os.walk(dirName):
            for file in filenames:
                if file.endswith(self.type):
                    dirPaths.append(dirpath)
                    fileNames.append(file)
                    dirnames.append(os.path.basename(dirpath))

        self.df = pd.DataFrame(list(zip(dirPaths, dirnames, fileNames)), columns=['Directory', 'Folder', 'File'])

    def return_df(self):
        """
        Based on the type file specification, this function returns a pandas data frame containing
        'Directory,"Folder,' and 'File' from all files below the define directory.

        Returns
        -------
            The data frame contains ('Directory', 'Folder', 'File')
        """
        return self.df

    def get_list_of_files(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.get_list_of_files(fullPath)
            else:
                allFiles.append(fullPath)

        return allFiles





