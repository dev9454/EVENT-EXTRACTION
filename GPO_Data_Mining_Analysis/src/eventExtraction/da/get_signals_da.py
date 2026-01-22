# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:42:58 2024

@author: mfixlz
"""
import sys
from os import path
from pathlib import Path
import os
from collections import OrderedDict
from operator import itemgetter
from collections.abc import Iterable
import itertools
import math
import pandas as pd
import numpy as np

from geopy.distance import geodesic, lonlat, distance


import sys
import os
from pathlib import Path
from collections import OrderedDict
from operator import itemgetter
from collections.abc import Iterable
import itertools
import math
import pandas as pd
import numpy as np
from geopy.distance import geodesic, lonlat, distance

# --- Dynamic Path Resolution ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_SRC = str(CURRENT_DIR.parents[1])

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Import sibling class and project utilities
from signal_mapping_da import signalMapping
from eventExtraction.utils.utils_generic import (
    loadmat,
    _calc_derivative,
    _get_bearing_from_lat_long,
)

class signalData(signalMapping):

    def __init__(self, raw_data, file_path) -> None:

        # super().__init__(self, raw_data)
        signalMapping.__init__(self, raw_data, file_path)

        self.can_acc_status_mapping = {
            0: 'Disabled',
            1: 'Enabled',
        }

        self.can_alc_status_enabled_enum = 1

        self.udp_drive_mode_status_mapping = {
            -1: 'Undefined',
            0: 'Disabled',
            1: 'Enabled',
            2: 'Longitudinal',
            3: 'Longitudinal + Lateral',
        }

        self.udp_da_status_full_control_enum = 3

        self.udp_alc_status_mapping = {
            -1: 'Undefined',
            0: 'Disabled',
            1: 'Enabled',
            # 2: 'Unknown 1',
            # 3: 'Unknown 2',
        }

        self.alc_status_mapping = {
            0: 'disengaged',
            1: 'engaged'
        }

        self.udp_alc_status_enabled_enum = 0

        self.can_turn_indicator_mapping = {
            0: 'No Turn',
            1: 'left turn',
            2: 'right turn',
        }

        self.udp_turn_indicator_mapping = {
            0: 'No Turn',
            1: 'left turn',
            2: 'right turn',
        }

        self.can_steering_override_mapping = {
            0: 'No Override',
            1: 'Override',
        }

        self.can_braking_override_mapping = {
            0: 'No Override',
            1: 'Override',
        }
        self.can_steering_angle_mapping = {
            32768: 'Unknown'
        }

        self.kwargs_processing = {}

        self.da_status_col_name = 'can_steering_engaged'
        self.da_status_mapping = self.alc_status_mapping

        self.turn_status_col_name = 'host_turn_description'
        self.turn_indicator_mapping = self.can_turn_indicator_mapping
        self.steering_override_col_name = 'can_steering_override'
        self.brake_override_col_name = 'can_brake_override'

        self.UDP_signals_dict = {
            'udp_drive_mode': 'udp_drive_mode',
            'udp_ALC_status': 'udp_ALC_status',
        }

        self.CAN_signals_dict = {
            'can_steering_engaged': 'can_steering_engaged',

        }

        self.udp_alc_status_mapping_orig = {
            -1: 'undefined',
            0: 'disabled',
            1: 'enabled',
            2: 'disengaged',
            3: 'engaged',
            # 4: 'unknown 04'#'event completed'
        }

        self.udp_alc_status_aditya = {
            0: 'unknown',
            1: 'Lane change',
        }

        self.steering_angle_threshold_deg = 90  # [deg]
        self.yaw_rate_threshold_rps = 0.1  # [rps]

    def _time_duration_to_indices_len(self, df, time_duration_sec, ):

        start_index = 0
        start_cTime = float(df.loc[start_index, 'cTime'])
        delta_cTime = time_duration_sec
        end_cTime = start_cTime + delta_cTime
        end_index = int(df[df['cTime']
                           <= end_cTime]['cTime'].idxmax())

        indices_len = end_index - start_index

        return indices_len

    def _host_turn_status_helper(self,
                                 df,
                                 ):
        # negative is left, positive is right

        df['steering_angle_deg_based_turn'] = 0
        df['yaw_rate_rps_based_turn'] = 0

        right_turn_indices_steering_angle = df.query(
            'host_steering_angle_deg >= '
            + f'{self.steering_angle_threshold_deg}').index
        left_turn_indices_steering_angle = df.query(
            'host_steering_angle_deg <= '
            + f'-{self.steering_angle_threshold_deg}').index

        df.loc[right_turn_indices_steering_angle,
               'steering_angle_deg_based_turn'] = 2
        df.loc[left_turn_indices_steering_angle,
               'steering_angle_deg_based_turn'] = 1

        right_turn_indices_yaw_rate = df.query(
            f'host_yaw_rate_rps >= {self.yaw_rate_threshold_rps}').index
        left_turn_indices_yaw_rate = df.query(
            f'host_yaw_rate_rps <= -{self.yaw_rate_threshold_rps}').index

        df.loc[right_turn_indices_yaw_rate,
               'yaw_rate_rps_based_turn'] = 2
        df.loc[left_turn_indices_yaw_rate,
               'yaw_rate_rps_based_turn'] = 1

        df['host_turn_description'] = 0
        right_turn_indices = df.query(
            'steering_angle_deg_based_turn == 2 '
            # + 'or yaw_rate_rps_based_turn == 2'
        ).index
        left_turn_indices = df.query(
            'steering_angle_deg_based_turn == 1 '
            # + 'or yaw_rate_rps_based_turn == 1'
        ).index

        df.loc[right_turn_indices,
               'host_turn_description'] = 2
        df.loc[left_turn_indices,
               'host_turn_description'] = 1

        return df

    def _alc_status_udp_helper2(self,
                                df,):

        df['alc_event_type_udp_numeric'] = 0

        event_indices = df.query(
            # 'udp_drive_mode == 3 and udp_ALC_status > 0' #aditya
            'udp_drive_mode == 3 and udp_ALC_status >=3 '  # Koustav 07012025_1345
            # + 'and udp_ALC_turn_indicator_status in [1, 2]' # Koustav 08012025_1535
        ).index

        df.loc[event_indices, 'alc_event_type_udp_numeric'] = 1

        return df

    def _alc_status_udp_helper(self,
                               df,
                               UDP_signals_dict: dict,
                               CAN_signals_dict: dict,
                               ):

        alc_engaged_dict_udp = {
            'alc_event_type_udp': 'engaged',
            'udp_drive_mode': [3],
            'udp_steering_override': [0],
            'udp_ALC_status': [0],
        }

        alc_engaged_df_udp = pd.DataFrame(alc_engaged_dict_udp)

        alc_disengaged_dict_udp = {
            'alc_event_type_udp': 'disengaged',
            'udp_drive_mode': [3, 3, 1, 1],
            'udp_steering_override': [1, 1, 1, 1],
            'udp_ALC_status': [0, -1, 0, -1],
        }

        alc_disengaged_df_udp = pd.DataFrame(alc_disengaged_dict_udp)

        alc_enabled_dict_udp = {
            'alc_event_type_udp': 'enabled',
            'udp_drive_mode': [3, 1, 1],
            'udp_steering_override': [0, 0, 0],
            'udp_ALC_status': [-1, 0, -1],
        }

        alc_enabled_df_udp = pd.DataFrame(alc_enabled_dict_udp)

        alc_disabled_dict_udp = {
            'alc_event_type_udp': 'disabled',
            'udp_drive_mode': [2, 2, 0, 0, 2, 2, 0, 0],
            'udp_steering_override': [0, 0, 0, 0, 1, 1, 1, 1],
            'udp_ALC_status': [0, -1, 0, -1, 0, -1, 0, -1],
        }

        alc_disabled_df_udp = pd.DataFrame(alc_disabled_dict_udp)

        alc_undefined_dict_udp = {
            'alc_event_type_udp': 'undefined',
            'udp_drive_mode': [-1, -1, -1, -1],
            'udp_steering_override': [0, 0, 1, 1],
            'udp_ALC_status': [0, -1, 0, -1],
        }

        alc_undefined_df_udp = pd.DataFrame(alc_undefined_dict_udp)

        alc_udp_mappings_df = pd.concat([
                                        alc_engaged_df_udp,
                                        alc_disengaged_df_udp,
                                        alc_enabled_df_udp,
                                        alc_disabled_df_udp,
                                        alc_undefined_df_udp
                                        ], axis=0,
                                        ignore_index=True)
        # Mapping multiple columns to get a multi-column lookup
        df = pd.merge(left=df, right=alc_udp_mappings_df, how='left')

        self.udp_alc_status_mapping = {val: key
                                       for key, val in
                                       self.udp_alc_status_mapping_orig.items()
                                       }

        df['alc_event_type_udp_numeric'] = \
            df['alc_event_type_udp'].replace(self.udp_alc_status_mapping)

        return df

    def _process_signals(self, out_df, enums_dict, ):

        optional_cols = ['seq_name',
                         'log_name_flat',
                         'base_logname',
                         'rTag',
                         ]

        # drive_mode == 3 & alc_status > 0 -->>> event
        # extract alc_lane_side, alc_status unique values in the event duration
        # out_df = self._alc_status_udp_helper(out_df,
        #                                      self.UDP_signals_dict,
        #                                      self.CAN_signals_dict,
        #                                      )

        out_df = self._alc_status_udp_helper2(out_df,

                                              )

        out_df = self._host_turn_status_helper(out_df,

                                               )
        for col in optional_cols:

            if not col in out_df.columns:

                out_df[col] = 'Not available'

        return out_df, enums_dict

    def get_signal_mapping(self, config_path, ):

        out_df, enums_dict = self._signal_mapping(config_path)

        return out_df, enums_dict

    def main_get_signals(self, config_path, ):

        # signal_data_obj = signalData(mat_file_data)
        out_df, enums_dict = self.get_signal_mapping(config_path)

        out_df, misc_out_dict = self._process_signals(out_df, enums_dict)

        return out_df, misc_out_dict


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
    program = 'E2E'  # 'MCIP'  # 'Thunder'  # 'Northstar'
    # 'config_thunder_v1_tsi.yaml'  # 'config_northstar_v1_tsi.yaml'
    config_file = 'config_e2e_v1_da_basic.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'SDV_E2EML_M16_20251229_111748_0001_p01.mat')

    config_path = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        config_file,

    )

    mat_file_data = loadmat(file_name)
    DA_signal_data_obj = signalData(mat_file_data, file_name)

    df, misc_out_dict = DA_signal_data_obj.main_get_signals(config_path)

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
