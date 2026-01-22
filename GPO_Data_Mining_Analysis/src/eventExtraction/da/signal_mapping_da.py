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
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
from itertools import chain

# --- Dynamic Path Resolution ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_SRC = str(CURRENT_DIR.parents[1]) # Points to 'src'

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# --- Standardized Imports ---
from eventExtraction.utils.utils_generic import (
    read_platform,
    loadmat,
    merge_pandas_df,
    stream_check,
    transform_df,
    _create_flat_file_list,
    _create_base_name_list_from_file_list,
)


class signalMapping:

    def __init__(self, raw_data, file_path) -> None:

        self.raw_data = raw_data
        self.file_path = file_path

        self.keys_to_adjust = {'target_ID': ['reduced_id', 'reducedID'],
                               }

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

        # df['log_name_only'] = log_name

        log_path, log_name = os.path.split(self.file_path)

        (log_path_list, base_name_list,
         rTag_list,
         seq_path_list,
         original_log_name_list) = _create_base_name_list_from_file_list(
            [self.file_path])
        basename = base_name_list[0]
        rTag = rTag_list[0]
        seq_path = seq_path_list[0]
        orig_log_name = original_log_name_list[0]

        seq_name = '_'.join(basename.split('_')[:-1])

        df['seq_name'] = seq_name
        df['log_path_flat'] = log_path

        df['log_name_flat'] = log_name
        df['orig_log_name_flat'] = orig_log_name

        df['base_logname'] = basename
        df['rTag'] = rTag

        df['frame_ID'] = (
            df['vision_avi_tsr_camera_stream_ref_index'] -
            df['vision_avi_tsr_camera_stream_ref_index'].min(
                skipna=True)
        )

        for item in missing_columns_list:

            if 'cTime' not in item:

                df[item] = -999180618999

        return df, enums_dict

if __name__ == '__main__':

    import warnings
    import os
    import sys
    import time
    from pathlib import Path
    from functools import reduce
    import psutil
    import pandas as pd

    warnings.filterwarnings("ignore")

    # --- Helper Functions ---
    def secondsToStr(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(t*1000,), 1000, 60, 60])

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss, mem_info.vms

    # --- Start Execution ---
    start_time = time.time()
    mem_before_phy, mem_before_virtual = process_memory()

    program = 'E2E'
    config_file = 'config_e2e_v1_da_basic.yaml'

    # Dynamically resolve project structure
    # CURRENT_DIR should be defined at the top of your file as:
    # CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parents[2] 
    
    # Path to local data folder (GPO_Data_Mining_Analysis/data/...)
    file_name = str(PROJECT_ROOT / 'data' / program / 'extracted_data' / 'SDV_E2EML_M16_20251229_111748_0000_p01.mat')

    # Path to the config file (GPO_Data_Mining_Analysis/src/eventExtraction/data/...)
    config_path = str(PROJECT_ROOT / 'src' / 'eventExtraction' / 'data' / program / config_file)

    print(f"Executing with Config: {config_path}")
    print(f"Loading Mat File: {file_name}")

    # Initialize and Map Signals
    mat_file_data = loadmat(file_name)
    DA_signal_map_obj = signalMapping(mat_file_data, file_name)

    df, enums_dict = DA_signal_map_obj._signal_mapping(config_path)

    # Output results for verification
    if "win" in sys.platform:
        print("\nSuccessfully mapped signals. Preview of DataFrame:")
        print(df.head())

    # --- Performance Metrics ---
    mem_after_phy, mem_after_virtual = process_memory()
    end_time = time.time()

    elapsed_time = secondsToStr(end_time - start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy) * 1E-6
    consumed_memory_virtual = (mem_after_virtual - mem_before_virtual) * 1E-6

    print('\n' + '#' * 50)
    print(f'Elapsed time: {elapsed_time}')
    print(f'Consumed physical memory: {consumed_memory_phy:.2f} MB')
    print(f'Consumed virtual memory: {consumed_memory_virtual:.2f} MB')
    print('#' * 50)