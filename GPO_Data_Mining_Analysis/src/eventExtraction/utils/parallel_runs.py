# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:25:30 2024

@author: mfixlz
"""

import os
import pandas as pd
import numpy as np
from copy import deepcopy
import more_itertools as mit
from joblib import Parallel, delayed, parallel_backend, parallel_config
from tqdm import tqdm
import contextlib
import joblib
# from ray.util.joblib import register_ray


class parallelWrapper:

    def __init__(self, parent_class):

        self.parent_class = parent_class
        self.back_end = 'loky'
        self.slice_length = 800

    @contextlib.contextmanager
    def tqdm_joblib(self, tqdm_object):
        """Context manager to patch joblib to report into tqdm 
        progress bar given as argument"""
        os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):

                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()

    def _create_dict_template(self, results_list):

        req_len_list = [len(result) for result in results_list]

        req_idx = np.argmax(req_len_list)

        req_dict = results_list[req_idx]

        dict_template = {}

        for key, val in req_dict.items():

            if isinstance(val, pd.DataFrame):
                dict_template[key] = [pd.DataFrame(columns=val.columns)]
            elif isinstance(val, np.ndarray):
                req_shape = np.shape(val)
                req_shape = (0, req_shape[1])
                dict_template[key] = [np.zeros(req_shape)]

        return dict_template

    def _concat_dict(self, dict_template):

        for key, val in dict_template.items():

            val_type = type(val[0])

            if val_type == pd.DataFrame:
                dict_template[key] = pd.concat(dict_template[key], )
            elif val_type == np.ndarray:
                dict_template[key] = np.concatenate(dict_template[key],
                                                    axis=0)

        return dict_template

    def _call_helper(self, results_list):

        dict_template_ = self._create_dict_template(results_list)
        dict_template = deepcopy(dict_template_)

        for result in results_list:

            for key, val in result.items():

                if isinstance(val, pd.DataFrame):
                    dict_template[key].append(val)
                elif isinstance(val, np.ndarray):
                    dict_template[key].append(val)

        return_val = self._concat_dict(dict_template)

        return return_val

    def __call__(self, num_jobs, kwargs_list):

        self.run = self.parent_class.run

        # kwargs_sliced = list(mit.chunked(kwargs_list,
        #                                  self.slice_length))

        with parallel_config(backend=self.back_end):

            with self.tqdm_joblib(tqdm(desc="My calculation",
                                       total=len(kwargs_list)
                                       )) as progress_bar:

                results_list = Parallel(n_jobs=num_jobs,
                                        prefer='processes',
                                        # return_as="generator",
                                        )(delayed(
                                            self.parallel_wrapper_run)(kwargs)
                                          for kwargs in kwargs_list)

        results_list = [result for result in results_list
                        if isinstance(result, dict)
                        for key, val in result.items()
                        if (type(val) == np.ndarray
                            or type(val) == pd.DataFrame)]
        return_val = self._call_helper(results_list)

        return return_val

    def parallel_wrapper_run(self, kwargs_tuple):

        file_name, run_kwargs = kwargs_tuple[0], kwargs_tuple[1]

        return self.run(file_name,
                        **run_kwargs
                        )
