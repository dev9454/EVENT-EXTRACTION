# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:35:19 2024

@author: mfixlz
"""
from pathlib import Path
import os
import yaml
import pandas as pd
import sys

if __package__ is None:
    from os import path
    print('Here at none package')
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(to_change_path)
    print(f'Current dir: {os.getcwd()}, to change : {to_change_path}')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    from utils.utils_generic import (read_platform,
                                     loadmat,
                                     merge_pandas_df,
                                     stream_check,
                                     transform_df)
else:

    # from .. import utils
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

    os.chdir(to_change_path)
    print(f'Current dir *: {os.getcwd()}, \n to change *: {to_change_path}')
    try:
        from eventExtraction.utils.utils_generic import (read_platform,
                                                         loadmat,
                                                         merge_pandas_df,
                                                         stream_check,
                                                         transform_df)
    except:
        from utils.utils_generic import (read_platform,
                                         loadmat,
                                         merge_pandas_df,
                                         stream_check,
                                         transform_df)


class signalMapping:

    def __init__(self, raw_data) -> None:

        self.raw_data = raw_data

        self.mapping_names_polarion_to_generic = {'host_long_accel_flag':
                                                  'f_CSCSA_36417',
                                                  'target_lat_vel_flag':
                                                  'f_CSCSA_36292',
                                                  'target_lat_accel_flag':
                                                  'f_CSCSA_36295',
                                                  'target_closing_in_to_host_lane_flag':
                                                  'f_CSCSA_87591',
                                                  }
        self.keys_to_adjust = {'target_ID': ['reduced_id', 'reducedID'],
                               }

        # Might need to be tuned for future programs
        self.curvature_limit_straight_path = 1/(2*2000)
        self.abs_yaw_rate_limit_straight_path = 0.2

    def _get_signal_paths(self, config_path):

        with open(os.path.join(config_path, )) as f:
            # use safe_load instead load
            signal_dict = yaml.safe_load(f)

        return signal_dict

    def _get_signals(self, signal_dict_vals):

        signal_dict_vals = {key: val
                            for key, val in signal_dict_vals.items()
                            if bool(val.strip())
                            }

        signal_dict_extracted = {key: stream_check(self.raw_data, val)
                                 for key, val in signal_dict_vals.items()
                                 }
        for key_adjust, replace_val in self.keys_to_adjust.items():

            if (key_adjust in signal_dict_extracted
                    and
                    isinstance(signal_dict_extracted[key_adjust], str)
                ):

                signal_dict_extracted[key_adjust] = \
                    stream_check(self.raw_data,
                                 signal_dict_vals[key_adjust].replace(
                                     replace_val[0],
                                     replace_val[1])
                                 )

        return signal_dict_extracted

    def _signal_mapping(self, config_path):

        signal_dict = self._get_signal_paths(config_path)

        if 'cTime_multiplier' in signal_dict:

            cTime_multiplier = float(signal_dict.pop("cTime_multiplier"))
        else:
            cTime_multiplier = 1.0

        df_list = [transform_df(
            self._get_signals(val)) for val in signal_dict.values()]
        df_list.sort(key=len, reverse=True)

        df_list_2 = []
        for df_iter in df_list:

            if df_iter['cTime'].isnull().any():

                df_iter['cTime'] = df_iter['cTime'].fillna(method='ffill')

            df_list_2.append(df_iter)

        df = merge_pandas_df(df_list_2, merge_key_list='cTime')

        df['cTime'] = df['cTime']*cTime_multiplier

        return df


if __name__ == '__main__':

    import warnings
    import os
    from pathlib import Path
    warnings.filterwarnings("ignore")

    import time
    from functools import reduce
    import psutil

    def secondsToStr(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(t*1000,), 1000, 60, 60])

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss, mem_info.vms

    start_time = time.time()
    mem_before_phy, mem_before_virtual = process_memory()
    program = 'Northstar'  # 'Thunder'
    config_file = 'config_northstar_v1_cut_in.yaml'  # 'config_thunder_cut_in.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        '20240424_M14_cs219195_2_HKMC_CL_RBL_Route1_MAPv8_RPv0_run2_065606_040.mat')

    config_path = os.path.join(
        Path(os.getcwd()).parent.parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        config_file,

    )

    mat_file_data = loadmat(file_name)
    CUTIIN_signal_map_obj = signalMapping(mat_file_data)

    df = CUTIIN_signal_map_obj._signal_mapping(config_path)

    mem_after_phy, mem_after_virtual = process_memory()

    end_time = time.time()

    elapsed_time = secondsToStr(end_time-start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy)*1E-6
    consumed_memory_virtual = (
        mem_after_virtual - mem_before_virtual)*1E-6

    print(
        f'&&&&&&&&&&&& Elapsed time is {elapsed_time} %%%%%%%%%%%%%%%%')
    print(
        f'&&&&&&&&&&&& Consumed physical memory MB is {consumed_memory_phy} %%%%%%%%%%%%%%%%')

    print(
        f'&&&&&&&&&&&& Consumed virtual memory MB is {consumed_memory_virtual} %%%%%%%%%%%%%%%%')
