# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:35:19 2025

@author: mfixlz
"""
from pathlib import Path
import os
import yaml
import pandas as pd
import sys
from itertools import chain

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

        self.keys_to_adjust = {'target_ID': ['reduced_id', 'reducedID'],
                               }

        self.readff_map_dict = {'Thunder': ['_dma', '_v01'],
                                'GPO-IFV7XX': ['_p01', '_v01']
                                }

        self.program_name_readff_map = None

        self._default_val_overtake_ff_20ms = 0

        self.avi_feature_mapping = {0: 2,
                                    1: 4,
                                    2: 6,
                                    3: 8,
                                    4: 10,
                                    5: 12,
                                    6: 14,
                                    7: 16,
                                    8: 18,
                                    9: 20,
                                    10: 22,
                                    11: 24,
                                    12: 26,
                                    13: 28,
                                    28: 2,
                                    29: 4,
                                    30: 6,
                                    31: 8,
                                    32: 10,
                                    33: 12,
                                    34: 14,
                                    35: 16,
                                    36: 18,
                                    37: 20,
                                    38: 22,
                                    39: 24,
                                    40: 26,
                                    41: 28,
                                    85: 30,
                                    86: 32,
                                    100: 1,
                                    101: 3,
                                    102: 5,
                                    103: 7,
                                    104: 9,
                                    105: 11,
                                    106: 13,
                                    107: 15,
                                    108: 17,
                                    109: 19,
                                    110: 21,
                                    111: 23,
                                    112: 25,
                                    113: 27,
                                    114: 29,
                                    115: 1,
                                    116: 3,
                                    117: 5,
                                    118: 7,
                                    119: 9,
                                    120: 11,
                                    121: 13,
                                    122: 15,
                                    123: 17,
                                    124: 19,
                                    125: 21,
                                    126: 23,
                                    127: 25
                                    }
        self.feature_avi_mapping = {key: []
                                    for key in
                                    self.avi_feature_mapping.values()}
        for key, val in self.avi_feature_mapping.items():

            self.feature_avi_mapping[val].append(key)

        self.feature_avi_mapping[0] = [65535]
        # self.feature_avi_mapping = {val : key
        #                             for key, val in
        #                             self.avi_feature_mapping.items()
        #                             }

        self.feature_can_telematics_mapping = {
            0: 0,
            1: 1,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
            10: 11,
            11: 12,
            12: 13,
            13: 14,
            14: 15,
            15: 16,
            16: 17,
            17: 18,
            18: 19,
            19: 20,
            20: 21,
            21: 22,
            22: 23,
            23: 24,
            24: 25,
            26: 26,
            28: 27,
            30: 28,
            32: 29,
            62: 30,
        }

        self.df_GT = pd.DataFrame()

        # Might need to be tuned for future programs

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

    def _get_enums(self, signal_dict):

        req_enums_keys = [key for key in signal_dict.keys()
                          if 'enums' in key]

        windows_enums = [key for key in req_enums_keys
                         if 'windows' in key]
        linux_enums = [key for key in req_enums_keys
                       if 'linux' in key]

        if "lin" in sys.platform:

            enums_dict_paths = {key.replace('path_linux', 'dict'):
                                signal_dict.pop(key)
                                for key in linux_enums
                                }
            _ = [signal_dict.pop(key)
                 for key in windows_enums]

        elif "win" in sys.platform:

            enums_dict_paths = {key.replace('path_windows', 'dict'):
                                signal_dict.pop(key)
                                for key in windows_enums
                                }

            _ = [signal_dict.pop(key)
                 for key in linux_enums]

        enums_dict = {}
        for key, path_ in enums_dict_paths.items():

            if os.path.isfile(path_):

                with open(path_) as f:

                    enums_dict[key] = yaml.safe_load(f)

        return enums_dict, signal_dict

    def _signal_mapping(self, config_path, log_name=None):

        signal_dict = self._get_signal_paths(config_path)

        if 'cTime_multiplier' in signal_dict:

            cTime_multiplier = float(signal_dict.pop("cTime_multiplier"))
        else:
            cTime_multiplier = 1.0

        (enums_dict,
         signal_dict) = self._get_enums(signal_dict)

        df_list = [transform_df(
            self._get_signals(val)) for val in signal_dict.values()]

        missing_columns_list = [
            item for item in df_list if isinstance(item, list)
            and bool(item) and all(isinstance(elem, str) for elem in item)
        ]
        missing_columns_list = list(
            chain.from_iterable(
                missing_columns_list
            )
        )

        df_list = [item for item in df_list if not isinstance(item, list)]

        df_list.sort(key=len, reverse=True)

        df_list_2 = []
        for df_iter in df_list:

            if df_iter['cTime'].isnull().any():

                df_iter['cTime'] = df_iter['cTime'].fillna(method='ffill')

            df_list_2.append(df_iter)

        df = merge_pandas_df(df_list_2, merge_key_list='cTime')

        df['cTime'] = df['cTime']*cTime_multiplier

        df['log_name_only'] = log_name

        for item in missing_columns_list:

            if 'cTime' not in item:

                df[item] = -999180618999

        return df, enums_dict


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
    program = 'MCIP'  # 'Thunder'  # 'Northstar'
    # 'config_thunder_v1_tsi.yaml'  # 'config_northstar_v1_tsi.yaml'
    config_file = 'config_mcip_v1_tsi.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent.parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'ThunderMCIP_WS11656_20250819_180535_0002_p01.mat')

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
    TSI_signal_map_obj = signalMapping(mat_file_data)

    df = TSI_signal_map_obj._signal_mapping(config_path)

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
