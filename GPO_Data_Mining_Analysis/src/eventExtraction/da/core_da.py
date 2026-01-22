# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:08:59 2024

@author: mfixlz
"""
import sys
import os
from pathlib import Path
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
import copy
import scipy as sp
import pickle
import datetime
import re
from collections import Counter
import itertools
from itertools import groupby, accumulate, chain
from geopy.distance import geodesic, lonlat, distance

# --- Dynamic Path Resolution ---
# Get the absolute path of the directory containing this file
CURRENT_DIR = Path(__file__).resolve().parent
# Navigate up to the 'src' folder (assumes structure: src/eventExtraction/da/core_da.py)
PROJECT_SRC = str(CURRENT_DIR.parents[1])

# Add the 'src' directory to sys.path to enable package-style imports
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Add the local directory to path to support sibling module imports
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(1, str(CURRENT_DIR))

# --- Standardized Imports ---
from signal_mapping_da import signalMapping
from get_signals_da import signalData
from eventExtraction.utils.utils_generic import (
    read_platform,
    loadmat,
    stream_check,
    transform_df,
    merge_pandas_df,
    sort_list,
    patch_asscalar,
    _resim_path_to_orig_path,
    merge_dicts,
    is_monotonic,
    find_ranges_in_iterable,
    ndix_unique,
    _time_duration_to_indices_length,
)
    # from . import signal_mapping_aeb

# class dummy(object):
#     pass


class coreEventExtractionDA(signalData):

    def __init__(self, raw_data, file_path) -> None:

        # super().__init__(self, raw_data)
        signalData.__init__(self, raw_data, file_path)

        self._headers = dict()

        self._headers['output_overview'] = ['log_path',
                                            'log_name',

                                            ]

        self._headers['output_data'] = ['log_path',
                                        'log_name',

                                        ]

        self.flickering_threshold_sec = 0.2  # [s]

        self.da_status_pre_event_time_considered = 5  # [s]
        self.da_status_post_event_time_considered = 5  # [s]

    def _grouped_data(self,
                      column,
                      sign_unique_vals_idx,
                      sign_indices_dict,
                      df,
                      ):

        req_cols = [col
                    for col in df.columns
                    if column in col]
        test_array = df[req_cols].values

        test_array_grouped = {}

        for (ID, indices), item in zip(sign_indices_dict.items(),
                                       sign_unique_vals_idx):

            # print('*********************', ID, '&&&&&&&&&&&&&&&')

            if isinstance(indices, dict):
                test_array_grouped[ID] = []
                # for _, sub_indices in indices.items():
                for sub_indices in list(chain.from_iterable(indices.values())):

                    # indices_2d = np.array([[col, idx] for idx in sub_indices],
                    #                       dtype=int)

                    # data_req = test_array[sub_indices, col]

                    if test_array.size == len(test_array):

                        data_req = test_array.flatten()[sub_indices[:, 0]]
                    else:

                        data_req = test_array[sub_indices[:,
                                                          0], sub_indices[:, 1]]

                    group_iter = pd.Series(index=sub_indices[:, 0],
                                           data=data_req)

                    group_iter = group_iter.where(group_iter < pow(2, 15),
                                                  np.nan,
                                                  inplace=False)

                    test_array_grouped[ID].append(group_iter)
            else:

                if test_array.size == len(test_array):

                    data_iter = pd.Series(
                        index=indices,
                        data=test_array.flatten()[item[:, 0]])
                else:

                    data_iter = pd.Series(
                        index=indices,
                        data=test_array[item[:, 0], item[:, 1]])

                test_array_grouped[ID] = data_iter.where(data_iter < pow(2, 15),
                                                         np.nan,
                                                         inplace=False)
        # test_array_grouped = {
        #     ID:
        #         pd.Series(index=indices,
        #                   data=test_array[item[:, 0],
        #                                   item[:, 1]])
        #     for (ID, indices), item in zip(sign_indices_dict.items(),
        #                                    sign_unique_vals_idx)
        # }

        return test_array_grouped

    def _sign_extraction_helper(self,
                                df,
                                sign_indices_dict,
                                sign_unique_vals_idx,
                                event_start_end_groups_dict,
                                event_type: str = None,
                                event_comment: str = None,
                                ):

        da_status_array_grouped = self._grouped_data(
            self.da_status_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        turn_status_array_grouped = self._grouped_data(
            self.turn_status_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        event_start_cTime_list = []
        event_end_cTime_list = []

        event_duration_list = []
        # sequence_start_cTime_list = []
        log_path_event_start_list = []
        sequence_name_event_start_list = []
        log_name_event_start_list = []
        log_name_event_end_list = []
        base_logname_event_start_list = []
        rTag_event_start_list = []

        da_status_event_list = []
        turn_status_event_list = []
        vehicle_turn_status_event_list = []

        event_type_event_start_list = []
        event_comment_event_start_list = []

        vision_frame_ID_event_start_list = []
        vision_frame_ID_event_end_list = []

        min_median_max_yaw_rate_event_list = []

        alc_status_event_list = []
        turn_indicator_presence_event_list = []

        pre_event_drive_mode_status_list = []
        event_drive_mode_status_list = []
        post_event_drive_mode_status_list = []
        nexus_event_type_list = []

        for enum, event_start_end_groups_array_list in \
                event_start_end_groups_dict.items():

            for enumerated_idx, event_start_end_groups in \
                    enumerate(event_start_end_groups_array_list):

                req_indices_values = np.array(sign_indices_dict[enum],
                                              dtype=int)
                series_da_status = da_status_array_grouped[enum]

                series_turn_status = turn_status_array_grouped[enum]

                event_start_cTime = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'cTime'])
                event_end_cTime = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'cTime'])

                event_duration = np.array(
                    [
                        str(datetime.timedelta(seconds=end-start))
                        for start, end in
                        zip(event_start_cTime,
                            event_end_cTime)
                    ]
                )

                sequence_name_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'seq_name'])
                log_name_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'log_name_flat'])
                log_name_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'log_name_flat'])

                log_path_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'log_path_flat'])

                # event_description_event_start_end =

                base_logname_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'base_logname'])

                rTag_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'rTag'])

                vision_frame_ID_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'frame_ID'])

                vision_frame_ID_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'frame_ID'])

                da_status_event_pre = [
                    series_da_status[req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]]
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                da_status_event = np.array(
                    [
                        str(self.da_status_mapping.get(
                            item.mode(dropna=True)[0],
                            item.mode(dropna=True)[0]))
                        if len(item.mode(dropna=True)) > 0
                        else 'not available'
                        for item in da_status_event_pre
                    ])

                # Koustav 09012025_1223
                req_indices_len_pre_drive_mode_event = \
                    _time_duration_to_indices_length(
                        df['cTime'].to_frame(),
                        self.da_status_pre_event_time_considered, )

                pre_event_drive_mode_status_01 = [
                    df.loc[
                        max(start-req_indices_len_pre_drive_mode_event,
                            df['cTime'].idxmin()
                            ): start,
                        'udp_drive_mode']
                    for start in event_start_end_groups[:, 0]
                ]

                pre_event_drive_mode_status = [
                    ', '.join(
                        item.dropna()
                        .replace(
                            self.udp_drive_mode_status_mapping,)
                        .astype("string").unique()
                    )
                    for item in pre_event_drive_mode_status_01
                ]

                event_drive_mode_status_01 = [
                    df.loc[start:end,
                           'udp_drive_mode'][req_indices_values[
                               (req_indices_values >= start)
                               & (req_indices_values <= end)]]

                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                event_drive_mode_status = [
                    ', '.join(
                        item.dropna()
                        .replace(
                            self.udp_drive_mode_status_mapping,)
                        .astype("string").unique()
                    )
                    for item in event_drive_mode_status_01
                ]

                req_indices_len_post_drive_mode_event = \
                    _time_duration_to_indices_length(
                        df['cTime'].to_frame(),
                        self.da_status_post_event_time_considered, )

                post_event_drive_mode_status_01 = [
                    df.loc[
                        end:
                        min(end+req_indices_len_post_drive_mode_event,
                            df['cTime'].idxmax()
                            ),
                        'udp_drive_mode']
                    for end in event_start_end_groups[:, 1]
                ]

                post_event_drive_mode_status = [
                    ', '.join(
                        item.dropna()
                        .replace(
                            self.udp_drive_mode_status_mapping,)
                        .astype("string").unique()

                    )
                    for item in post_event_drive_mode_status_01
                ]

                alc_status_event_pre = [
                    df.loc[start:end,
                           'alc_event_type_udp_numeric'][req_indices_values[
                               (req_indices_values >= start)
                               & (req_indices_values <= end)]]

                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                alc_status_event = np.array(
                    [
                        str(self.udp_alc_status_aditya.get(
                            item.mode(dropna=True)[0],
                            item.mode(dropna=True)[0]))
                        if len(item.mode(dropna=True)) > 0
                        else 'not available'
                        for item in alc_status_event_pre
                    ])

                turn_status_event_pre = [
                    series_turn_status[req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]]
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                turn_status_event = np.array(
                    [
                        str(self.turn_indicator_mapping.get(
                            item.mode(dropna=True)[0],
                            item.mode(dropna=True)[0]))
                        if len(item.mode(dropna=True)) > 0
                        else 'not available'
                        for item in turn_status_event_pre
                    ])

                vehicle_turn_status_pre = [
                    df.loc[start:end,
                           self.turn_indicator_col][req_indices_values[
                               (req_indices_values >= start)
                               & (req_indices_values <= end)]]
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                vehicle_turn_status_event = np.array(
                    [
                        str(self.turn_indicator_mapping.get(
                            item.mode(dropna=True)[0],
                            item.mode(dropna=True)[0]))
                        if len(item.mode(dropna=True)) > 0
                        else 'not available'
                        for item in vehicle_turn_status_pre
                    ])

                event_type_event_start = np.array(
                    [
                        event_type for start in
                        event_start_end_groups[:, 0]
                    ])

                # event_comment_event_start = np.array(
                #     [
                #         event_comment for start in
                #         event_start_end_groups[:, 0]
                #     ])

                event_comment_event_start = np.array(
                    [
                        f'Turn indicator (mode): {item}'
                        for item in vehicle_turn_status_event
                    ])

                # Koustav 08012025_1535
                turn_indicator_presence_event = [
                    ', '.join(
                        item.dropna()
                        .replace(
                            self.turn_indicator_mapping,)
                        .astype("string").unique()
                        # item.mode(dropna=True).replace(
                        # self.turn_indicator_mapping,)
                        # .dropna().astype("string")
                    )
                    for item in vehicle_turn_status_pre
                ]

                yaw_rate_steering_angle_event_pre = [
                    [
                        df.loc[start:end,
                               'host_yaw_rate_rps'][req_indices_values[
                                   (req_indices_values >= start)
                                   & (req_indices_values <= end)]],
                        df.loc[start:end,
                               'host_steering_angle_deg'][req_indices_values[
                                   (req_indices_values >= start)
                                   & (req_indices_values <= end)]],

                    ]
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                min_median_max_yaw_rate_event = [
                    'yaw rate min : {}, \nyaw rate median : {}, \nyaw rate max : {}'
                    .format(*[item[0].min(skipna=True),
                              item[0].median(skipna=True),
                              item[0].max(skipna=True),])
                    +
                    ('\nsteering angle min : {}, \nsteering angle median : {},'
                     + ' \nsteering angle max : {}')
                    .format(*[item[1].min(skipna=True),
                              item[1].median(skipna=True),
                              item[1].max(skipna=True),])
                    for item in yaw_rate_steering_angle_event_pre
                ]

                alc_status_enum_share = [
                    df.loc[start:end,
                           'udp_ALC_status'][req_indices_values[
                               (req_indices_values >= start)
                               & (req_indices_values <= end)]]
                    .value_counts(normalize=True)*100
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                alc_status_str_list = []
                for event_iter in alc_status_enum_share:

                    alc_status_str_list.append(
                        '\n'.join([
                            f'ALC enum {key} : {val:.2f}% time of total event'
                            for key, val in event_iter.items()
                        ]
                        ))

                turn_indicator_status_enum_share = [
                    df.loc[start:end,
                           'udp_ALC_turn_indicator_status'][req_indices_values[
                               (req_indices_values >= start)
                               & (req_indices_values <= end)]]
                    .value_counts(normalize=True)*100
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                turn_indicator_status_str_list = []
                for event_iter in turn_indicator_status_enum_share:

                    turn_indicator_status_str_list.append(
                        '\n'.join([
                            'turn indicator ' +
                            f'{self.turn_indicator_mapping.get(key, key)} : ' +
                            f'{val:.2f}% time of total event'
                            for key, val in event_iter.items()
                        ]
                        ))

                min_median_max_yaw_rate_event = [
                    item1 + '\n' + item2 + '\n' + item3
                    for item1, item2, item3 in
                    zip(min_median_max_yaw_rate_event,
                        alc_status_str_list,
                        turn_indicator_status_str_list
                        )
                ]

                # .value_counts(normalize=True)*100

                event_start_cTime_list.append(event_start_cTime)
                event_end_cTime_list.append(event_end_cTime)

                event_duration_list.append(event_duration)
                sequence_name_event_start_list.append(
                    sequence_name_event_start)

                log_path_event_start_list.append(log_path_event_start)

                log_name_event_start_list.append(log_name_event_start)
                log_name_event_end_list.append(log_name_event_end)
                base_logname_event_start_list.append(base_logname_event_start)
                rTag_event_start_list.append(rTag_event_start)

                da_status_event_list.append(da_status_event)
                turn_status_event_list.append(
                    turn_status_event)
                vehicle_turn_status_event_list.append(
                    vehicle_turn_status_event)

                event_type_event_start_list.append(event_type_event_start)
                event_comment_event_start_list.append(
                    event_comment_event_start)

                vision_frame_ID_event_start_list.append(
                    vision_frame_ID_event_start)
                vision_frame_ID_event_end_list.append(
                    vision_frame_ID_event_end)

                min_median_max_yaw_rate_event_list.append(
                    min_median_max_yaw_rate_event)

                alc_status_event_list.append(alc_status_event)
                turn_indicator_presence_event_list.append(
                    turn_indicator_presence_event)

                pre_event_drive_mode_status_list.append(
                    pre_event_drive_mode_status)
                event_drive_mode_status_list.append(event_drive_mode_status)
                post_event_drive_mode_status_list.append(
                    post_event_drive_mode_status)
                # --- Nexus Event Type Mapping Logic ---
                # Determine the Nexus type for each event in this group
                nexus_types = []
                for i in range(len(event_type_event_start)):
                    e_type = event_type_event_start[i]
                    a_status = alc_status_event[i]
                    v_turn = vehicle_turn_status_event[i]
                    
                    if e_type == "DA_status" and a_status == "Lane change":
                        nexus_types.append("DMA_LANE_CHANGE")
                    elif e_type == "Turn_status" and v_turn == "left turn":
                        nexus_types.append("DMA_LEFT_TURN")
                    elif e_type == "Turn_status" and v_turn == "right turn":
                        nexus_types.append("DMA_RIGHT_TURN")
                    else:
                        nexus_types.append("unknown")
                
                nexus_event_type_list.append(np.array(nexus_types))

        event_dict = {
            'log_path':
            np.hstack(log_path_event_start_list)
            if bool(log_path_event_start_list)
            else np.array([]),

            'sequence_name':
            np.hstack(sequence_name_event_start_list)
            if bool(sequence_name_event_start_list)
            else np.array([]),

            'log_name':
            np.hstack(log_name_event_start_list)
            if bool(log_name_event_start_list)
            else np.array([]),

            'base_logname':
            np.hstack(base_logname_event_start_list)
            if bool(base_logname_event_start_list)
            else np.array([]),

            'rTag':
            np.hstack(rTag_event_start_list)
            if bool(rTag_event_start_list)
            else np.array([]),

            'event_start_cTime':
            np.hstack(event_start_cTime_list)
            if bool(event_start_cTime_list)
            else np.array([]),
            'event_end_cTime':
            np.hstack(event_end_cTime_list)
            if bool(event_end_cTime_list)
            else np.array([]),

            'event_duration':
            np.hstack(event_duration_list)
            if bool(event_duration_list)
            else np.array([]),

            'da_status':
            np.hstack(da_status_event_list)
            if bool(da_status_event_list)
            else np.array([]),

            'drive_mode_pre_event':
            np.hstack(pre_event_drive_mode_status_list)
            if bool(pre_event_drive_mode_status_list)
            else np.array([]),

            'drive_mode_event':
            np.hstack(event_drive_mode_status_list)
            if bool(event_drive_mode_status_list)
            else np.array([]),

            'drive_mode_post_event':
            np.hstack(post_event_drive_mode_status_list)
            if bool(post_event_drive_mode_status_list)
            else np.array([]),

            'alc_status':
            np.hstack(alc_status_event_list)
            if bool(alc_status_event_list)
            else np.array([]),

            'turn_indicator_presence':
            np.hstack(turn_indicator_presence_event_list)
            if bool(turn_indicator_presence_event_list)
            else np.array([]),

            'vehicle_turn_status':
            np.hstack(turn_status_event_list)
            if bool(turn_status_event_list)
            else np.array([]),

            'event_type':
            np.hstack(event_type_event_start_list)
            if bool(event_type_event_start_list)
            else np.array([]),

            'event_comment':
            np.hstack(event_comment_event_start_list)
            if bool(event_comment_event_start_list)
            else np.array([]),

            'vision_frame_ID_event_start':
                np.hstack(vision_frame_ID_event_start_list)
                if bool(vision_frame_ID_event_start_list)
                else np.array([]),
            'vision_frame_ID_event_end':
                np.hstack(vision_frame_ID_event_end_list)
                if bool(vision_frame_ID_event_end_list)
                else np.array([]),


            'log_name_event_start':
            np.hstack(log_name_event_start_list)
            if bool(log_name_event_start_list)
            else np.array([]),
            'log_name_event_end':
            np.hstack(log_name_event_end_list)
            if bool(log_name_event_end_list)
            else np.array([]),

            'remarks':
            np.hstack(min_median_max_yaw_rate_event_list)
            if bool(min_median_max_yaw_rate_event_list)
            else np.array([]),
            
            'nexus_event_type':
            np.hstack(nexus_event_type_list)
            if bool(nexus_event_type_list)
            else np.array([]),
        }

        return event_dict

    def _sign_extraction(self,
                         df,
                         da_status_col_name,
                         turn_status_col_name,
                         steering_override_col_name,
                         brake_override_col_name,
                         **kwargs):

        self.req_indices_len = _time_duration_to_indices_length(
            df['cTime'].to_frame(),
            self.flickering_threshold_sec, )

        event_dict = {}
        print('************** Event Extraction Start *******************')

        self.turn_indicator_col = 'can_vehicle_turn_indicator_status'

        (event_start_end_groups_dict_da,
         sign_data_dict_da,
         sign_indices_dict_da,
         sign_unique_vals_idx_da) = self._events_from_col(
            da_status_col_name,
            df)

        event_dict_da_status_based = self._sign_extraction_helper(
            df,
            sign_indices_dict_da,
            sign_unique_vals_idx_da,
            event_start_end_groups_dict_da,
            event_type='DA_status',

        )

        (event_start_end_groups_dict_ti,
         sign_data_dict_ti,
         sign_indices_dict_ti,
         sign_unique_vals_idx_ti) = self._events_from_col(
            turn_status_col_name,
            df)

        event_dict_ti_status_based = self._sign_extraction_helper(
            df,
            sign_indices_dict_ti,
            sign_unique_vals_idx_ti,
            event_start_end_groups_dict_ti,
            event_type='Turn_status',

        )

        event_dict_CAN = {
            'da_status_based_CAN': event_dict_da_status_based,
            'turn_indicator_based_CAN': event_dict_ti_status_based,

        }

        overall_dict_CAN = pd.concat([pd.DataFrame(item)
                                      for item in event_dict_CAN.values()],
                                     axis=0).to_dict(orient='list')

        event_dict['overall_CAN'] = overall_dict_CAN

        # UDP

        # self.da_status_col_name = da_status_col_name = \
        #     'alc_event_type_udp_numeric'

        da_status_col_name = 'alc_event_type_udp_numeric'
        self.da_status_col_name = 'udp_ALC_status'

        self.turn_indicator_col = 'udp_ALC_turn_indicator_status'

        self.da_status_mapping = self.udp_alc_status_mapping_orig
        # self.da_status_mapping = self.udp_alc_status_aditya
        self.turn_indicator_mapping = self.udp_turn_indicator_mapping

        (event_start_end_groups_dict_da_udp,
         sign_data_dict_da_udp,
         sign_indices_dict_da_udp,
         sign_unique_vals_idx_da_udp) = self._events_from_col(
            da_status_col_name,
            df)

        event_dict_da_status_based_UDP = self._sign_extraction_helper(
            df,
            sign_indices_dict_da_udp,
            sign_unique_vals_idx_da_udp,
            event_start_end_groups_dict_da_udp,
            event_type='DA_status',

        )

        (event_start_end_groups_dict_ti_udp,
         sign_data_dict_ti_udp,
         sign_indices_dict_ti_udp,
         sign_unique_vals_idx_ti_udp) = self._events_from_col(
            turn_status_col_name,
            df)

        event_dict_ti_status_based_UDP = self._sign_extraction_helper(
            df,
            sign_indices_dict_ti_udp,
            sign_unique_vals_idx_ti_udp,
            event_start_end_groups_dict_ti_udp,
            event_type='Turn_status',

        )

        event_dict_UDP = {
            'da_status_based_UDP': event_dict_da_status_based_UDP,
            'turn_indicator_based_UDP': event_dict_ti_status_based_UDP,

        }

        # --- Fixed Mapping Logic at the end of _sign_extraction ---
        overall_df_UDP = pd.concat([pd.DataFrame(item)
                                      for item in event_dict_UDP.values()],
                                     axis=0, ignore_index=True)

        # Initialize the column
        overall_df_UDP['nexus_event_type'] = 'unknown'

        # 1. Map Lane Changes (DA_status events where alc_status is 'Lane change')
        lane_change_indices = overall_df_UDP.query(
            'event_type == "DA_status" and alc_status == "Lane change"').index
        overall_df_UDP.loc[lane_change_indices, 'nexus_event_type'] = 'DMA_LANE_CHANGE'

        # 2. Map Left Turns (Turn_status events where vehicle_turn_status is 'left turn')
        left_turn_indices = overall_df_UDP.query(
            'event_type == "Turn_status" and vehicle_turn_status == "left turn"').index
        overall_df_UDP.loc[left_turn_indices, 'nexus_event_type'] = 'DMA_LEFT_TURN'

        # 3. Map Right Turns (Turn_status events where vehicle_turn_status is 'right turn')
        right_turn_indices = overall_df_UDP.query(
            'event_type == "Turn_status" and vehicle_turn_status == "right turn"').index
        overall_df_UDP.loc[right_turn_indices, 'nexus_event_type'] = 'DMA_RIGHT_TURN'

        # Convert back to list format for the final dictionary
        overall_dict_UDP = overall_df_UDP.to_dict(orient='list')
        event_dict['overall_UDP'] = overall_dict_UDP

        return event_dict
    
    def _events_from_col(self, column_name, df, is_feature: bool = False):

        # self.req_indices_len

        req_cols = [col
                    for col in df.columns
                    if column_name in col]

        req_df = df[req_cols]

        req_cols_test_array = req_df.values

        # if is_feature:

        #     req_cols_test_array = req_df.where((req_df < pow(2, 15)) &
        #                                        (req_df > self.default_vals_ff_20ms),
        #                                        np.nan,
        #                                        inplace=False).values
        # else:
        #     req_cols_test_array = req_df.where(req_df < pow(2, 15),
        #                                        np.nan,
        #                                        inplace=False).values

        unique_vals_1, unique_vals_idx_1 = ndix_unique(req_cols_test_array)

        unique_vals_idx_2 = []
        unique_vals_2 = []

        req_cols_indices_dict = {}
        # print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     {unique_vals_1}')
        for col_identifier, indices in zip(unique_vals_1,
                                           unique_vals_idx_1):

            if np.isnan(col_identifier):
                continue
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',
            #       col_identifier)
            unique_vals_idx_2.append(indices)
            unique_vals_2.append(col_identifier)
            # creates an array of indices, sorted by unique element
            idx_sort = np.argsort(indices[:, 0], kind='mergesort')

            # sorts records array so all unique elements are together
            indices_sorted = indices[idx_sort]

            # returns the unique values, the index of the first occurrence of
            # a value, and the count for each element
            (unique_indices,
             repeated_indices_idx,
             repeated_indices_count) = np.unique(
                 indices_sorted[:, 0], return_counts=True, return_index=True)

            # splits the indices into separate arrays
            res = np.split(idx_sort, repeated_indices_idx[1:])
            res.sort(key=lambda s: len(s))
            # res = filter(lambda x: x.size > 1, res)
            repeated_groups = unique_indices[repeated_indices_count > 1]

            non_repeated_groups = unique_indices[repeated_indices_count == 1]
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # non_repeating_idx = (indices[:, 0] ==
            #                      non_repeated_groups[:, None]).any(0).nonzero()[0]
            # print('*****************************************')
            # non_repeating_all = indices[non_repeating_idx]

            # repeating_indices_only = list(set(range(len(indices)))
            #                               - set(non_repeating_idx))

            # repeating_all = indices[repeating_indices_only]

            # print('*****************************************')

            if len(repeated_groups) == 0:

                req_cols_indices_dict[col_identifier] = df[req_cols].index[
                    (df[req_cols]
                     == col_identifier).any(axis=1)].tolist()
            else:

                unique_repeat_counts = np.unique(repeated_indices_count)
                indices_collection = {}
                for repeat_count in unique_repeat_counts:

                    req_res = list(filter(lambda x: x.size == repeat_count,
                                          res))

                    # for _ in range(repeat_count):

                    indices_collection[repeat_count] = \
                        [indices[item] for item in np.transpose(req_res)]

                req_cols_indices_dict[col_identifier] = indices_collection

        req_col_data_dict = {col_identifier: pd.Series(req_cols_test_array[
            idx_array[:, 0],
            idx_array[:, 1]], index=idx_array[:, 0])
            for idx_array, col_identifier in
            zip(unique_vals_idx_2, unique_vals_2)
            if not np.isnan(col_identifier)
        }

        req_col_indices_processed = {}

        for ID, series in req_cols_indices_dict.items():

            # print('*********************', ID, '^^^^^^^^^^^^^^^^^')

            if isinstance(series, dict):
                req_col_indices_processed[ID] = []
                for sub_series in list(chain.from_iterable(series.values())):

                    # sub_series =
                    sub_series = pd.Series(sub_series[:, 0])
                    # print(np.shape(sub_series))
                    calc_values_event_start_end = np.array(sub_series.groupby(
                        # series.diff().ne(1)
                        sub_series.diff().gt(self.req_indices_len)
                        .cumsum()).apply(
                        lambda x:
                        [x.iloc[0], x.iloc[-1]]
                        if len(x) >= 2
                        else [x.iloc[0], x.iloc[-1]]).tolist())
                    req_col_indices_processed[ID].append(
                        calc_values_event_start_end)
                # req_col_indices_processed[ID] = np.array(
                #     req_col_indices_processed[ID])

            else:

                series = pd.Series(series)

                req_col_indices_processed[ID] = [np.array(series.groupby(
                    # series.diff().ne(1)
                    series.diff().gt(self.req_indices_len)
                    .cumsum()).apply(
                    lambda x:
                    [x.iloc[0], x.iloc[-1]]
                    if len(x) >= 2
                    else [x.iloc[0], x.iloc[-1]]).tolist())]
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        return (req_col_indices_processed,
                req_col_data_dict,
                req_cols_indices_dict,
                unique_vals_idx_2)

    def event_extraction(self, df, **kwargs):

        event_dict = self._sign_extraction(
            df,
            self.da_status_col_name,
            self.turn_status_col_name,
            self.steering_override_col_name,
            self.brake_override_col_name,
            **kwargs
        )

        return event_dict

    def main(self, config_path, ):

        out_df, misc_out_dict = self.main_get_signals(config_path)

        self.misc_out_dict = misc_out_dict

        print('************** Signal Extraction Done *******************')

        return_val = self.event_extraction(out_df,
                                           **self.kwargs_processing)

        return return_val


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

    # Dynamically find the project root (GPO_Data_Mining_Analysis/)
    # CURRENT_DIR was defined at the top as Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parents[2] 
    
    # Define your local data and config paths relative to project root
    # This assumes your .mat file is in: GPO_Data_Mining_Analysis/data/E2E/extracted_data/
    file_name = str(PROJECT_ROOT / 'data' / program / 'extracted_data' / 'SDV_E2EML_M16_20251229_111748_0001_p01.mat')

    # This assumes your config is in: GPO_Data_Mining_Analysis/src/eventExtraction/data/E2E/
    config_path = str(PROJECT_ROOT / 'src' / 'eventExtraction' / 'data' / program / config_file)

    print(f"Loading data from: {file_name}")
    print(f"Using config from: {config_path}")

    # Core logic execution
    mat_file_data = loadmat(file_name)
    DA_core_logic_obj = coreEventExtractionDA(mat_file_data, file_name)

    return_val_dict = DA_core_logic_obj.main(config_path)

    if "win" in sys.platform:
        return_val_dict_df = {key: pd.DataFrame(val)
                              for key, val in return_val_dict.items()}
        print("Extraction complete. Results converted to DataFrames.")

    # --- Performance Metrics ---
    mem_after_phy, mem_after_virtual = process_memory()
    end_time = time.time()

    elapsed_time = secondsToStr(end_time - start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy) * 1E-6
    consumed_memory_virtual = (mem_after_virtual - mem_before_virtual) * 1E-6

    print(f'\n' + '#' * 40)
    print(f'Elapsed time: {elapsed_time}')
    print(f'Consumed physical memory: {consumed_memory_phy:.2f} MB')
    print(f'Consumed virtual memory: {consumed_memory_virtual:.2f} MB')
    print('#' * 40)
