import pickle
import functools
import os
import inspect


def dump_array(file_name, arr):
    if os.path.exists(file_name):
        raise Exception('File already exists!')

    with open(file_name, 'wb') as f:
        pickle.dump(arr, f)
