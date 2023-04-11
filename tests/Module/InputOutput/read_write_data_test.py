import os
import shutil
import unittest

import numpy as np
import pandas as pd

from piscat.InputOutput.read_write_data import (
    load_dict_from_hdf5,
    read_json2dic,
    read_mat,
    save_df2csv,
    save_dic2json,
    save_dic_to_hdf5,
    save_mat,
)


def delete_directory(path):
    try:
        shutil.rmtree(path)
        print("\nDirectory ", path, " deleted")
    except FileNotFoundError:
        print("\nDirectory ", path, " dose not exist")


current_path = os.path.abspath(os.path.join(".."))


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.save_path = current_path
        self.save_path = os.path.join(self.save_path, "PiSCAT/TestData")
        self.file_name = "sample"
        self.saved_file_name = ""

    def test_save_read_json2dic(self):
        sample_dict = {"item1": "A", "item2": "B", "item3": "C"}
        save_dic2json(sample_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith(".json"):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), "file is not saved")
        read_dict = read_json2dic(self.save_path, self.saved_file_name)
        self.assertTrue(len(read_dict) > 0, "imported file is incorrect")
        self.assertTrue(read_dict.get("item1") == "A", "imported file is incorrect")
        os.remove(file_path)

    def test_save_read_mat(self):
        sample_array = np.ndarray(shape=(2, 2))
        sample_array[0, 0] = 1
        save_mat(sample_array, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith(".mat"):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), "file is not saved")
        read_array = read_mat(self.save_path, self.saved_file_name)
        self.assertTrue(len(read_array) > 0, "imported file is incorrect")
        self.assertTrue(sample_array[0, 0] == 1, "imported file is incorrect")
        os.remove(file_path)

    def test_save_df2csv(self):
        sample_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
        save_df2csv(sample_df, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith(".csv"):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), "file is not saved")
        os.remove(file_path)

    def test_save_load_dic_to_hdf5_simple_dict(self):
        sample_dict = {"item1": "A", "item2": "B", "item3": "C"}
        save_dic_to_hdf5(sample_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith(".h5"):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), "file is not saved")
        read_dict = load_dict_from_hdf5(file_path)
        self.assertTrue(len(read_dict) > 0, "imported file is incorrect")
        os.remove(file_path)

    def test_save_load_dic_to_hdf5_complex_dict(self):
        sample_dict = {"item1": "A", "item2": "B", "item3": "C"}
        sample_dict_in_dict = {"item1": "A", "item2": "B", "item3": sample_dict}
        save_dic_to_hdf5(sample_dict_in_dict, self.save_path, self.file_name)
        for item in os.listdir(self.save_path):  # loop through items in dir
            if item.endswith(".h5"):
                self.saved_file_name = item
        file_path = os.path.join(self.save_path, self.saved_file_name)
        self.assertTrue(os.path.exists(file_path), "file is not saved")
        read_dict = load_dict_from_hdf5(file_path)
        self.assertTrue(len(read_dict) > 0, "imported file is incorrect")
        os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
