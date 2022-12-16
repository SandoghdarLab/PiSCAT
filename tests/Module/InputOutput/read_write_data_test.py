import unittest
import os
import shutil
import numpy as np
from piscat.InputOutput import download_tutorial_data
from piscat.InputOutput.read_write_data import download_url
from piscat.InputOutput.read_write_data import read_json2dic, save_dic2json, save_mat, read_mat, save_df2csv, \
    save_dic_to_hdf5, load_dict_from_hdf5
import pandas as pd


def delete_directory(path):
    try:
        shutil.rmtree(path)
        print("\nDirectory ", path, " deleted")
    except FileNotFoundError:
        print("\nDirectory ", path, " dose not exist")


current_path = os.path.abspath(os.path.join('..'))


class TestDownloadTutorialDataConstructor(unittest.TestCase):
    def setUp(self):
        self.current_path = current_path
        save_path = os.path.dirname(self.current_path)
        self.save_path = os.path.join(save_path, 'Tutorials')
        delete_directory(self.save_path)

    def test_control_folder_created(self):
        self.test_obj_control = download_tutorial_data('control_video')
        name_mkdir_1 = 'Demo data'
        dr_mk = os.path.join(self.save_path, name_mkdir_1)
        self.assertTrue(os.path.isdir(self.save_path), 'download_tutorial_data did not create the folders')
        self.assertTrue(os.path.isdir(dr_mk), 'download_tutorial_data did not create the folders')
        delete_directory(self.save_path)

    def test_Tutorial3_folder_created(self):
        self.test_obj_Tutorial3 = download_tutorial_data('Tutorial3_video')
        name_mkdir_1 = 'Demo data'
        dr_mk = os.path.join(self.save_path, name_mkdir_1)
        self.assertTrue(os.path.isdir(self.save_path), 'download_tutorial_data did not create the folders')
        self.assertTrue(os.path.isdir(dr_mk), 'download_tutorial_data did not create the folders')
        delete_directory(self.save_path)


class TestDownloadUrl(unittest.TestCase):
    def setUp(self):
        self.current_path = current_path
        save_path = os.path.dirname(self.current_path)
        self.save_path = os.path.join(save_path, 'Tutorials')
        self.download_path = os.path.join(self.save_path, 'Demo data')

    def test_control_download_method(self):

        test_obj = download_tutorial_data('control_video')
        download_url(test_obj.url, self.download_path)
        dir_list = os.listdir(self.download_path)
        self.assertTrue(len(dir_list) > 0, 'Downloaded folder did not create')
        file_dir = os.path.join(self.download_path, dir_list[0])
        file_list = os.listdir(file_dir)
        self.assertTrue(len(file_list) > 0, 'downloaded folder is empty')

    def test_Tutorial3_download_method(self):
        test_obj = download_tutorial_data('Tutorial3_video')
        download_url(test_obj.url, self.download_path)
        dir_list = os.listdir(self.download_path)
        self.assertTrue(len(dir_list) > 0, 'Downloaded folder did not create')
        file_dir = os.path.join(self.download_path, 'Tutorial3')
        file_list = os.listdir(file_dir)
        self.assertTrue(len(file_list) > 0, 'downloaded folder is empty')

    def test_delete_directory(self):
        delete_directory(self.save_path)


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.save_path = current_path
        self.save_path = os.path.join(self.save_path, 'Module/TestData')
        self.file_name = 'sample'
        self.saved_file_name = ''

    def test_save_read_json2dic(self):
        sample_dict = {'item1': 'A', 'item2': 'B', 'item3': 'C'}
        save_dic2json(sample_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith('.json'):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), 'file is not saved')
        read_dict = read_json2dic(self.save_path, self.saved_file_name)
        self.assertTrue(len(read_dict) > 0, 'imported file is incorrect')
        self.assertTrue(read_dict.get('item1') == 'A', 'imported file is incorrect')
        os.remove(file_path)

    def test_save_read_mat(self):
        sample_array = np.ndarray(shape=(2, 2))
        sample_array[0, 0] = 1
        save_mat(sample_array, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith('.mat'):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), 'file is not saved')
        read_array = read_mat(self.save_path, self.saved_file_name)
        self.assertTrue(len(read_array) > 0, 'imported file is incorrect')
        self.assertTrue(sample_array[0, 0] == 1, 'imported file is incorrect')
        os.remove(file_path)

    def test_save_df2csv(self):
        sample_df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        save_df2csv(sample_df, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith('.csv'):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), 'file is not saved')
        os.remove(file_path)

    def test_save_load_dic_to_hdf5_simple_dict(self):
        sample_dict = {'item1': 'A', 'item2': 'B', 'item3': 'C'}
        save_dic_to_hdf5(sample_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith('.h5'):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), 'file is not saved')
        read_dict = load_dict_from_hdf5(file_path)
        self.assertTrue(len(read_dict) > 0, 'imported file is incorrect')
        os.remove(file_path)

    def test_save_load_dic_to_hdf5_complex_dict(self):
        sample_dict = {'item1': 'A', 'item2': 'B', 'item3': 'C'}
        sample_dict_in_dict = {'item1': 'A', 'item2': 'B', 'item3': sample_dict}
        save_dic_to_hdf5(sample_dict_in_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith('.h5'):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), 'file is not saved')
        read_dict = load_dict_from_hdf5(file_path)
        self.assertTrue(len(read_dict) > 0, 'imported file is incorrect')
        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
