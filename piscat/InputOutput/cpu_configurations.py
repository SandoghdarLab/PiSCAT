from __future__ import print_function

import pandas as pd
import json
import os
import sys


class CPUConfigurations():

    def __init__(self, n_jobs=-1, backend='multiprocessing', verbose=0, parallel_active=True, threshold_for_parallel_run=None):
        """
        This class generates a JSON file based on the parallel loop setting on the CPU that the user prefers.
        This JSON was used by other functions and methods to set hyperparameters in a parallel loop.
        For parallelization, PiSCAT used Joblib.

        | [1]. https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

        Parameters
        ----------
        n_jobs: int
            The maximum number of workers that can work at the same time.
            If -1, all CPU cores are available for use.

        backend: str
            Specify the implementation of the parallelization backend.
            The following backends are supported:

            * `“loky”`:
                It can induce some communication and Memory overhead when exchanging
                input and output data with the worker Python processes.

            * `“multiprocessing”`:
                It previous process-based backend based on multiprocessing.Pool. Less robust than loky.

            * `“threading”`:
                It is a very low-overhead backend but it suffers from the Python Global Interpreter.
                Lock if the called function relies a lot on Python objects. “threading” is mostly useful when
                the execution bottleneck is a compiled extension that explicitly releases
                the GIL (for instance a Cython loop wrapped in a “with nogil” block or an expensive call to a library such as NumPy).

        verbose: int, optional
            The verbosity level, if non zero, progress messages are printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported.

        parallel_active: bool
            Functions will run the parallel implementation if it is True.

        threshold_for_parallel_run: float
            It reserved for next generation of PiSCAT.
        """
        try:
            self.read_cpu_setting()

        except FileNotFoundError:
            self.n_jobs = n_jobs
            self.backend = backend
            self.verbose = verbose
            self.parallel_active = parallel_active
            self.threshold_for_parallel_run = threshold_for_parallel_run

            setting_dic = {'n_jobs': [self.n_jobs], 'backend': [self.backend], 'verbose': [self.verbose],
                            'parallel_active': [self.parallel_active],
                            'threshold_for_parallel_run': [self.threshold_for_parallel_run]}

            self.save_cpu_setting(setting_dic)

    def save_cpu_setting(self, setting_dic):
        name = 'cpu_configurations.json'
        here = os.path.dirname(os.getcwd())
        subdir = 'piscat_configuration'

        try:
            dr_mk = os.path.join(here, subdir)
            os.mkdir(dr_mk)
            print("Directory ", subdir, " Created ")
        except FileExistsError:
            print("Directory ", subdir, " already exists")

        filepath = os.path.join(here, subdir, name)
        df_configfile = pd.DataFrame(data=setting_dic)
        df_configfile.to_json(filepath)

    def read_cpu_setting(self):
        subdir = "piscat_configuration"
        here = os.path.dirname(os.getcwd())
        filepath = os.path.join(here, subdir, 'cpu_configurations.json')

        with open(filepath) as json_file:
            cpu_setting = json.load(json_file)

        self.n_jobs = cpu_setting['n_jobs']['0']
        self.backend = cpu_setting['backend']['0']
        self.verbose = cpu_setting['verbose']['0']
        self.parallel_active = cpu_setting['parallel_active']['0']
        self.threshold_for_parallel_run = cpu_setting['threshold_for_parallel_run']['0']


