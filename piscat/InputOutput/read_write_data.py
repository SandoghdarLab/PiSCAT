from piscat.Trajectory.data_handeling import protein_trajectories_list2dic
import scipy.io
import h5py
import json
import time
import os
import numpy as np
import zipfile
import wget
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display


def save_mat(data, path, name=''):
    """
    This function saves the array as matlab format.

    Parameters
    ----------
    data: list
        List or array.

    path: str
        Path of the directory that data saves on it.

    name: str
        Name of the save file.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = name + '_' + timestr + '.mat'
    filepath = os.path.join(path, name)
    scipy.io.savemat(filepath, {'data': data}, do_compression=True)


def read_mat(path, name=''):
    """
    This function reads the array with matlab format.

    Parameters
    ----------
    path: str
       Path of the directory that data reads from it.

    name: str
       Name of the file.
    """
    filepath = os.path.join(path, name)
    particles = scipy.io.loadmat(filepath)

    if particles['data'].shape[1] != 1:
        p_ = particles['data']
    else:
        p_ = particles['data']
    return p_


def save_dic_to_hdf5(dic_data, path, name):
    """
    This function writes the dictionary data as hdf5 format.

    Parameters
    ----------
    data: dic
        Dictionary data.

    path: str
        Path of the directory that data saves on it.

    name: str
        Name of the save file
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = name + '_' + timestr + '.h5'
    filepath = os.path.join(path, name)
    with h5py.File(filepath, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic_data)


def save_list_to_hdf5(list_data, path, name):
    """
    This function writes the list data as hdf5 format.

    Parameters
    ----------
    data: list
       List data.

    path: str
       Path of the directory that data saves on it.

    name: str
       Name of the save file.
    """
    dic_data = protein_trajectories_list2dic(list_data)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = name + '_' + timestr + '.h5'
    filepath = os.path.join(path, name)
    with h5py.File(filepath, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic_data)


def recursively_save_dict_contents_to_group(h5file, path, dic_):
    for key, item in dic_.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s bin_type' % type(item))


def load_dict_from_hdf5(filename):
    """
    This function reads the hdf5 file and convert it to dictionary.

    Parameters
    ----------
    filename: str
       Path and name of the hdf5 file.
    """

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def save_df2csv(df, path, name=''):
    """
    This function writes the panda data frame to CSV file

    Parameters
    ----------
    data: data frame
        Panda data frame.

    path: str
        Path of the directory that data save on it.

    name: str
        Name of the save file
   """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = name + '_' + timestr + '.csv'
    filepath = os.path.join(path, name)
    df.to_csv(filepath)


def save_dic2json(data_dictionary, path, name=''):
    """
    This function writes the dictionary data to JSON file

    Parameters
    ----------
    data: dic
        Dictionary data.

    path: str
        Path of the directory that data save on it.

    name: str
        Name of the save file.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = name + '_' + timestr + '.json'
    filepath = os.path.join(path, name)
    with open(filepath, 'w') as file:
        file.write(json.dumps(data_dictionary))


def read_json2dic(path, name=''):
    """
    This function reads the JSON file and converts it to dictionary.

    Parameters
    ----------
    path: str
        Path of the directory that data load from it.
        
    name: str
        Name of the JSON file.
    """
    filepath = os.path.join(path, name)
    if os.path.exists(filepath):
        with open(filepath) as json_file:
            history_setting = json.load(json_file)
        return history_setting


def download_url(url, save_path):
    extension = ".zip"
    # change directory from working dir to dir with files
    os.chdir(save_path)
    filename = wget.download(url)
    print('\nStart unzip files --->', end='')
    for item in os.listdir(save_path):  # loop through items in dir
        if item.endswith(extension):  # check for ".zip" extension
            file_name = os.path.abspath(item)  # get full path of files
            zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
            zip_ref.extractall(save_path)  # extract file to dir
            zip_ref.close()  # close file
            os.remove(file_name)  # delete zipped file
    print('Done')


class download_tutorial_data():

    def __init__(self, tutorial_id, flag_status='JPY'):
        self.flag_status = flag_status
        current_path = os.path.abspath(os.path.join('..'))
        save_path = os.path.dirname(current_path)
        save_path = os.path.join(save_path, 'Tutorials')

        try:
            os.mkdir(save_path)
            print("\nDirectory ", save_path, " Created ")
        except FileExistsError:
            print("\nDirectory ", save_path, " already exists")

        try:
            name_mkdir_1 = 'Demo data'
            dr_mk = os.path.join(save_path, name_mkdir_1)
            os.mkdir(dr_mk)
            print("\nThe directory with the name ", name_mkdir_1, " is created in the following path:", save_path)
        except FileExistsError:
            print("\nThe directory with the name ", name_mkdir_1, " already exists in the following path:", save_path)

        if tutorial_id == 'control_video':
            name_mkdir_1 = 'Demo data'
            name_mkdir_2 = 'Control'
            dr_ = os.path.join(save_path, name_mkdir_1, name_mkdir_2)
            if os.path.isdir(dr_):
                print("\nThe data file named ", name_mkdir_2, " already exists in the following path:", os.path.join(save_path, name_mkdir_1))
            else:
                dr_ = os.path.join(save_path, name_mkdir_1)
                self.run_download(url='https://owncloud.gwdg.de/index.php/s/tzRZ7ytBd1weNDl/download', save_path=dr_)

        elif tutorial_id == 'Tutorial4_video':
            name_mkdir_1 = 'Demo data'
            name_mkdir_2 = 'Tutorial4'
            name_mkdir_3 = 'Histogram'

            dr_ = os.path.join(save_path, name_mkdir_1, name_mkdir_2)
            if os.path.isdir(dr_):
                print("Directory ", name_mkdir_2, " already exists!")
            else:
                dr_ = os.path.join(save_path, name_mkdir_1)
                self.run_download(url='https://owncloud.gwdg.de/index.php/s/Cq5vU8qIAFIWwEh/download', save_path=dr_)

            dr_mk = os.path.join(save_path, name_mkdir_1, name_mkdir_3)

            try:
                os.mkdir(dr_mk)
                print("\nDirectory ", name_mkdir_3, " Created ")
            except FileExistsError:
                print("\nDirectory ", name_mkdir_3, " already exists")

    def run_download(self, url, save_path):
        self.url = url
        self.save_path = save_path

        if "JPY_PARENT_PID" in os.environ and self.flag_status == 'JPY':
            self.button = widgets.Button(description='Download', disabled=False, icon='fa-cloud-download')
            self.out = widgets.Output()

            self.button.on_click(self.on_button_clicked)
            display(self.button)
        elif self.flag_status == 'HTML':
            download_url(self.url, self.save_path)

    def on_button_clicked(self, _):
        download_url(self.url, self.save_path)







