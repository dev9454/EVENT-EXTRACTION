# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:42:58 2024

@author: mfixlz
"""
import sys
from os import path
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
from collections.abc import Iterable
if __package__ is None:

    sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.insert(1, path.dirname(path.abspath(__file__)))
    from utils.utils_generic import (read_platform,
                                     loadmat,
                                     stream_check,
                                     transform_df,
                                     merge_pandas_df,
                                     get_all_sw_ver_no_mid_map,
                                     get_tavi_path,
                                     )
    from signal_mapping_aeb import signalMapping

else:

    # from .. import utils
    from utils.utils_generic import (read_platform, loadmat,
                                     stream_check, transform_df,
                                     merge_pandas_df,
                                     get_all_sw_ver_no_mid_map,
                                     get_tavi_path,
                                     )
    # from . import signal_mapping_aeb
    from signal_mapping_aeb import signalMapping


class signalData:

    def __init__(self, raw_data) -> None:

        self.raw_data = raw_data

    def get_signals_dict(self,
                         signal_mapping_dict):

        df_dict = {key.replace('map', 'df'): pd.DataFrame
                   for key in signal_mapping_dict.keys()
                   }

        for i, ((key, val), df_key) in enumerate(
            zip(signal_mapping_dict.items(),
                df_dict.keys())):

            data_dict = {key_type: stream_check(self.raw_data, val_type)
                         for key_type, val_type in val.items()
                         }
            if 'FDCAN' in df_key:
                cTime_check = False
            else:
                cTime_check = True
            df_iter = pd.DataFrame(transform_df(data_dict, cTime_check))
            df_dict[df_key] = df_iter.sort_values('cTime')

        df_dict = OrderedDict(sorted(df_dict.items(),
                              key=lambda item: len(item[1]), reverse=True)
                              )

        return df_dict

    def get_signal_mapping(self, ):

        signal_map, enum_map, target_type_list = \
            signalMapping(self.raw_data)._signal_mapping()

        return signal_map, enum_map, target_type_list

    def main(self, ):

        # signal_data_obj = signalData(mat_file_data)
        signal_mapping_dict, enum_mapping_dict, target_type_list = \
            self.get_signal_mapping()
        df_dict = self.get_signals_dict(signal_mapping_dict)

        df_merged = merge_pandas_df(list(df_dict.values()))

        software_version_dict = get_all_sw_ver_no_mid_map(self.raw_data)

        # software_version_dict = {key : val[0] if isinstance(val, Iterable) else val
        #                          for key, val in software_version_dict.items()
        #                          }
        for key, val in software_version_dict.items():
            df_merged[key] = val

        return df_merged, enum_mapping_dict, target_type_list


if __name__ == '__main__':

    import warnings
    import os
    from pathlib import Path
    warnings.filterwarnings("ignore")

    file_name_resim = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data', 'WL',
        'FCAWL_20210826_VPCSkipper_RWUP_HWY_PAR_PAR_MS_AD_165229_094.mat')
    mat_file_data = loadmat(file_name_resim)

    signal_data_obj = signalData(mat_file_data)
    # signal_mapping_dict, enum_mapping_dict, target_type_list = \
    #     signal_data_obj.get_signal_mapping()
    # df_dict = signal_data_obj.get_signals_dict(signal_mapping_dict)

    # df_merged = merge_pandas_df(list(df_dict.values()))
    df_merged, enum_mapping_dict, target_type_list = signal_data_obj.main()
