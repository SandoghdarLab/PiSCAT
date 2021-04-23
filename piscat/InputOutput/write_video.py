import os
import time
import cv2
import imageio
from tqdm.autonotebook import tqdm


def write_binary(dir_path, file_name, data, type='original'):
    """
    This function writes video as a binary.

    Parameters
    ----------
    dir_path: str
        Path of the directory that video save on it.

    file_name: str
       Name of the save file.

    data: NDArray
        Video with numpy format.

    type: str or bin_type
        The video bin type is not changed by 'original,' but the user can convert it (e.g. float --> int16).
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    try:
        dr_mk = os.path.join(dir_path, timestr)
        os.mkdir(dr_mk)
        print("Directory ", timestr, " Created ")
    except FileExistsError:
        print("Directory ", timestr, " already exists")

    save_path = os.path.join(dir_path, timestr, file_name)

    if type == 'original':
        data = data
    else:
        data = data.astype(type)

    data = data.copy(order='C')
    with open(save_path, 'wb') as outfile:
        outfile.write(data)


def write_MP4(dir_path, file_name, data, fps=10):
    """
    This function writes video as a MP4.

    Parameters
    ----------
    dir_path: str
        Path of the directory that video save on it.

    file_name: str
       Name of the save file.

    data: NDArray
        Video with numpy format.

    fps: int
        Number of frame per seconds.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    try:
        dr_mk = os.path.join(dir_path, timestr)
        os.mkdir(dr_mk)
        print("Directory ", timestr, " Created ")
    except FileExistsError:
        print("Directory ", timestr, " already exists")

    save_path = os.path.join(dir_path, timestr, file_name)
    data = data.copy(order='C')
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (data.shape[1], data.shape[2]), False)

    for fr_ in tqdm(range(data.shape[0])):
        frame = data[fr_, ...]
        out.write(frame)
    out.release()


def write_GIF(dir_path, file_name, data, jump=0, fps=10):
    """
    This function writes video as a GIF.

    Parameters
    ----------
    dir_path: str
        Path of the directory that video save on it.

    file_name: str
       Name of the save file

    data: NDArray
        Video with numpy format.

    jump: int
        Define stride between frames

    fps: int
        Number of frame per seconds
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    try:
        dr_mk = os.path.join(dir_path, timestr)
        os.mkdir(dr_mk)
        print("Directory ", timestr, " Created ")
    except FileExistsError:
        print("Directory ", timestr, " already exists")

    save_path = os.path.join(dir_path, timestr, file_name)
    data = data.copy(order='C')
    image_ = []
    for frame_number in tqdm(range(0, data.shape[0] - jump, jump)):
        image_.append(data[frame_number, ...])

    imageio.mimsave(save_path, image_, fps=fps)
