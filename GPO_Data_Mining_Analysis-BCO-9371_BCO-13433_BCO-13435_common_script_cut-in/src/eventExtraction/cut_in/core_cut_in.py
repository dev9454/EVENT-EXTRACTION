# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:08:59 2024

@author: mfixlz
"""
import sys
import os
from os import path
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
import copy
import itertools
import re
from pathlib import Path
from collections import Counter
if __package__ is None:

    print('Here at none package 1')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, \n to change 1: {to_change_path}')
    from signal_mapping_cut_in import signalMapping
    from get_signals_cut_in import signalData
    print('Here at none package 2')
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(to_change_path)
    print(f'Current dir 2: {os.getcwd()}, to change 2: {to_change_path}')
    from utils.utils_generic import (read_platform,
                                     loadmat,
                                     stream_check,
                                     transform_df,
                                     merge_pandas_df,
                                     sort_list,
                                     patch_asscalar,
                                     _resim_path_to_orig_path,

                                     )
    os.chdir(actual_package_path)


else:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, to change 1: {to_change_path}')

    from signal_mapping_cut_in import signalMapping
    from get_signals_cut_in import signalData
    # from .. import utils
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

    os.chdir(to_change_path)
    print(f'Current dir 2: {os.getcwd()}, to change 2: {to_change_path}')
    from eventExtraction.utils.utils_generic import (read_platform, loadmat,
                                                     stream_check, transform_df,
                                                     merge_pandas_df, sort_list,
                                                     patch_asscalar,
                                                     _resim_path_to_orig_path,
                                                     )
    os.chdir(actual_package_path)
    # from . import signal_mapping_aeb


class coreEventExtractionCUTIN(signalData):

    def __init__(self, raw_data) -> None:

        # super().__init__(self, raw_data)
        signalData.__init__(self, raw_data)

        self.target_longitudinal_position_limit = 100  # [m]
        self.target_heading_threshold = 1.0  # [deg]
        self._time_percent_target_heading_threshold = 0.75

        self.target_across_path_heading_threshold = 20  # [deg]
        self.time_percent_plausiblity_check = 0.35
        self.host_velocity_threshold_check = 1  # [m/s]

        self.look_back_time_sec = 3
        self.look_forward_time_sec = 2
        self._min_gap_between_events_sec = 1.0
        self.stabilisation_time_sec = 1
        self.rt1_pos_check_needed = True

        self._time_percent_rt1_distance_less_than_100 = 0.75
        self._time_percent_alc_type = 1.0
        self.percent_change_limits = [-0.01, 0.01]
        self._time_percent_duration_generic_condition = 0.9

        self._time_percent_duration_vague_condition = 0.5

        self.full_output = True

        self._headers = dict()

        self._headers['output_overview'] = ['log_path',
                                            'log_name',
                                            'cutin_detections',
                                            'CSCSA_36417_events',
                                            'dma_events',
                                            'Polarion_only_f_CSCSA_87591',
                                            'Polarion_only_f_CSCSA_36292',
                                            'Polarion_only_f_CSCSA_36295',
                                            'Polarion_only_event_start_hard_stop',
                                            'Polarion_only_event_end_hard_stop',
                                            'DMA_f_CSCSA_87591',
                                            'DMA_f_CSCSA_36292',
                                            'DMA_f_CSCSA_36295',
                                            'DMA_event_start_hard_stop',
                                            'DMA_event_end_hard_stop'
                                            ]

        self._headers['output_data'] = ['log_path',
                                        'log_name',
                                        'is_event_CSCSA_36417',
                                        'is_event_dma',
                                        'is_host_decelerating_flag',
                                        'event_f_CSCSA_87591',
                                        'event_f_CSCSA_36292',
                                        'event_f_CSCSA_36295',
                                        'event_is_alc_disabled',
                                        'event_chance_of_host_turning',
                                        'is_host_not_lane_changing',
                                        'is_host_vel_stabilised',
                                        'is_host_vel_target_diff_vel_stabilised',
                                        'is_event_end_hard_stop',
                                        'is_event_start_hard_stop',
                                        'type_of_requirements',
                                        'time_to_cross',
                                        'base_name',
                                        'Vehicle',
                                        'event_start_cTime',
                                        'event_end_cTime',
                                        # 'event_start_index',
                                        'event_start_fusion_index',
                                        # 'event_end_index',
                                        'event_end_fusion_index',
                                        # 'event_detected_index',
                                        'Host_Speed',
                                        'Time_to_collision_event_start',
                                        'min_Time_to_collision_during_event',
                                        'Time_to_collision_event_end',
                                        'cTimes_min_TTC_2_order',
                                        'headway_2_order',
                                        'Time_to_collision_interpolated_event_start',
                                        # add cTimes corresponding to this
                                        'min_Time_to_collision_interpolated_during_event',
                                        'Time_to_collision_interpolated_event_end',
                                        'cTimes_min_TTC_interpolated',
                                        'headway_interpolated',
                                        'Host_long_accln',
                                        'CSCSA-36496_Max_Host_long_deceleration',  # action item 04122023
                                        'Max_Host_acceleration_during_event',  # action item 04122023
                                        'Host_Long_Jerk',
                                        'CSCSA-36497_Max_Neg_Host_Long_Jerk',
                                        # 'Minimum_TTC',
                                        'CSCSA-36499_Manoeuvre_Latency',
                                        'CSCSA-47468_PFS',
                                        'CSCSA-47469_CFS',
                                        'CSCSA-36505_missed_proximity_warning',
                                        'video_link',
                                        'image_link',
                                        'min_gap_host_to_left_marker',
                                        'max_gap_host_to_left_marker',
                                        'min_gap_host_to_right_marker',
                                        'max_gap_host_to_right_marker',
                                        'target_rel_long_position_event_start',
                                        'target_long_velocity_event_start',
                                        'target_rel_long_position_event_end',
                                        'target_long_velocity_event_end',
                                        'DA(ALC)_status_event_start',
                                        'DA(ALC)_status_event_end',
                                        'DA(ALC)_status_vals_during_event',
                                        'target_id_event_start',
                                        'target_id_during_event',
                                        'target_id_event_end',
                                        'host_steering_angle_event_start',
                                        'host_steering_angle_event_end',
                                        'target_rel_long_position_event_start_target_id_based',
                                        'target_long_velocity_event_start_target_id_based',
                                        'target_long_acceleration_event_start',
                                        'target_heading_angle_event_start',
                                        'target_rel_lat_position_event_start',
                                        'target_rel_lat_position_above_threshold_flag_event_start',
                                        'target_rel_lat_velocity_event_start',
                                        'target_rel_lat_acceleration_event_start',
                                        'target_rel_long_position_event_end_target_id_based',
                                        'target_long_velocity_event_end_target_id_based',
                                        'target_long_acceleration_event_end',
                                        'target_heading_angle_event_end',
                                        'target_rel_lat_position_event_end',
                                        'target_rel_lat_position_below_threshold_flag_event_end',
                                        'target_rel_lat_velocity_event_end',
                                        'target_rel_lat_acceleration_event_end',
                                        'orig_log_path',
                                        'orig_log_name',
                                        ]

    def find_cutin_indices_polarion(self, out_df):

        df = out_df.query('host_long_accel_flag == True '
                          + 'and '
                          + '(target_lat_vel_flag == True '
                          + 'or '
                          + 'target_lat_accel_flag == True) '
                          + 'and '
                          + '(is_host_not_lane_changing == True '
                          + 'or '
                          + 'host_veh_cut_out_initiated == False) '
                          #  + 'and '
                          #  + 'target_longitudinal_position_m < '
                          #  + f'{self.target_longitudinal_position_limit}'
                          )

        event_indices = list(df.index)
        s = pd.Series(event_indices)
        event_start_end_groups = s.groupby(s.diff().ne(1)
                                           .cumsum()).apply(lambda x:
                                                            [x.iloc[0], x.iloc[-1]]
                                                            if len(x) >= 2
                                                            else [x.iloc[0]]).tolist()

        return out_df, event_start_end_groups

    def _find_events_based_on_target_heading(self,
                                             df,
                                             rt3_target_unique_ids,
                                             rt4_target_unique_ids,
                                             rt1_target_unique_ids,
                                             target_ID_change_array,
                                             ):

        # FIXME
        unique_target_ids = list(
            set(rt3_target_unique_ids + rt4_target_unique_ids
                ).intersection(set(rt1_target_unique_ids)) - {0})
        unique_target_ids = [id-1 for id in unique_target_ids]

        target_indices_length_lookback = \
            self._time_duration_to_indices_length(df, self.look_back_time_sec)

        req_length_lookback = min(target_indices_length_lookback,
                                  len(df))

        # tentative_start_indices, tentative_detection_indices = [], []

        tentative_start_indices_target_id = []

        for target_id in unique_target_ids:

            for index_target_cutin in target_ID_change_array[:, 0]:

                tentative_start_index_list = list(
                    df.loc[int(index_target_cutin
                               - req_length_lookback):
                           index_target_cutin, :].query(
                               'target_heading_rad_VCS'
                        + '_'
                        + str(int(target_id)) + '.abs()'
                        + f' >= {np.deg2rad(self.target_heading_threshold)}').index
                )  # place an upper bound too

                if not bool(tentative_start_index_list):

                    continue

                max_min_diff = np.max(tentative_start_index_list) - \
                    np.min(tentative_start_index_list) + 1

                if (len(tentative_start_index_list)
                    >= self._time_percent_target_heading_threshold
                        * max_min_diff):

                    tentative_start_index = np.min(tentative_start_index_list)
                    tentative_start_indices_target_id.append(
                        [tentative_start_index, target_id+1])

        tentative_start_indices_target_id = np.unique(
            np.array(
                tentative_start_indices_target_id, dtype=int),
            axis=0)

        return tentative_start_indices_target_id

    def _target_cutin_params_helper(self, df):

        indices_target_cutin = np.array(list(df.query('RT3_to_RT1_left_check == True '
                                             + 'or '
                                             + 'RT4_to_RT1_right_check == True').index))

        cTimes_target_cutin = df.loc[indices_target_cutin,
                                     'cTime'].to_numpy()

        # target_ID_req = df.loc[indices_target_cutin,
        #                              'target_ID'].to_numpy()

        # target_ID_change_from

        target_ID_change_from = df.loc[indices_target_cutin,
                                       'target_ID_change_from'].values

        target_ID_change_array = np.array((indices_target_cutin,
                                           cTimes_target_cutin,
                                           target_ID_change_from)).T

        return target_ID_change_array

    def _event_extraction_helper(self, df, **kwargs):

        host_intiated_lane_change_event_starts = list(
            itertools.chain(*kwargs['host_initiated_cutout_start_list']))

        df['events_dma'] = False

        df['is_host_vel_stabilised'] = False
        df['is_host_to_target_relative_vel_stabilised'] = False

        df['event_start_hard_stop'] = False
        df['event_end_hard_stop'] = False

        df, event_start_end_groups = self.find_cutin_indices_polarion(df)

        # host_duration_to_maintain = kwargs['host_duration_to_maintain']
        self.host_duration_to_maintain = kwargs['host_duration_to_maintain']

        target_ID_change_array = self._target_cutin_params_helper(df, )
        # print('&&&&&&&&&&&&&&&&&&&& target_ID_change_array')
        # print(target_ID_change_array)
        tentative_event_start_indices_target_id = \
            self._find_events_based_on_target_heading(df,
                                                      kwargs['rt3_target_unique_ids'],
                                                      kwargs['rt4_target_unique_ids'],
                                                      kwargs['rt1_target_unique_ids'],
                                                      target_ID_change_array,
                                                      )

        if len(tentative_event_start_indices_target_id) > 0:
            event_start_indices_list_heading_based = \
                tentative_event_start_indices_target_id[:, 0]
        else:
            event_start_indices_list_heading_based = []

        detected_events_count = len(target_ID_change_array)

        event_detection_list = ['polarion', 'dma']

        self.stabilisation_indices_length = self._time_duration_to_indices_length(df,
                                                                                  self.stabilisation_time_sec)
        self.end_index_length_hypothetical = self._time_duration_to_indices_length(
            df, 0.5)
        (event_start_indices_list_, event_detected_indices_,
         event_end_indices_list_, cTimes_array_start_,
         cTimes_array_detection_, cTimes_array_end_) = [], [], [], [], [], []

        event_detection_type_list_ = []

        type_det = {}

        for event_detection in event_detection_list:
            event_detection_type_list = []

            type_det[event_detection] = {}

            if (detected_events_count) > 0:

                start_end_indices_arr = np.array([item
                                                  for item in event_start_end_groups
                                                  if len(item) > 1])
                if len(start_end_indices_arr) > 0:
                    event_start_indices_list_polarion_based = start_end_indices_arr[:, 0]
                else:
                    event_start_indices_list_polarion_based = []

                event_start_indices_list_raw = np.array(list(set.union(
                    set(event_start_indices_list_heading_based),
                    set(event_start_indices_list_polarion_based))
                ))
                # Checking if tentative event occurs within min look back time.
                # else, set all params to the corresponding to min look back time
                # req_change_lane_3_4_to_1 = \
                #         (np.array(target_ID_change_array)[:, 2] - 1).astype(int)
                req_change_lane_3_4_to_1 = np.array(
                    target_ID_change_array)[:, 2]
                req_change_indices = np.array(
                    target_ID_change_array[:, 0], dtype=int)
                req_change_cTimes = np.array(
                    target_ID_change_array[:, 1], dtype=float) - self.look_back_time_sec
                req_change_cTimes_series = pd.Series(req_change_cTimes,
                                                     name='req_cTime').sort_values(ascending=True)

                req_df_copy = df.copy(deep=True).reset_index()[
                    ['index', 'cTime']]

                merged_df = pd.merge_asof(req_change_cTimes_series,
                                          req_df_copy.sort_values(by='cTime'),
                                          left_on='req_cTime', right_on='cTime',
                                          direction='nearest')

                req_change_indices_offset_arr = merged_df['index'].values

                differences = (req_change_indices_offset_arr.reshape(1, -1) -
                               event_start_indices_list_raw.reshape(-1, 1))

                req_change_cTimes_offset_arr = merged_df['cTime'].values

                event_start_cTimes_list_raw = \
                    df.loc[event_start_indices_list_raw, 'cTime'].values

                differences_cTimes = np.abs(req_change_cTimes_offset_arr.reshape(1, -1) -
                                            event_start_cTimes_list_raw.reshape(-1, 1))

                differences_cTimes[differences_cTimes >
                                   self.look_back_time_sec] = 1E15
                differences[differences > 0] = 1E15
                delete_cols_indices = np.array([all(col == 1E15)
                                                for col in differences.T]).nonzero()
                delete_cols_ctimes = np.array([all(col == 1E15)
                                               for col in differences_cTimes.T]).nonzero()

                delete_cols = np.union1d(
                    delete_cols_ctimes, delete_cols_indices)

                differences = np.delete(differences, delete_cols, axis=1)

                try:
                    if differences.shape[0] == 1 and differences.shape[1] > 0:
                        indices, idx_of_indices = \
                            np.unique(np.abs(differences).argmin(),
                                      return_index=True)

                    else:

                        indices, idx_of_indices = \
                            np.unique(np.abs(differences).argmin(axis=0),
                                      return_index=True)

                    if len(indices) < len(req_change_cTimes_offset_arr):
                        print('\nevent start times length '  # by christys method
                              + 'not matching event detections, exiting iteration\n')
                        raise ValueError('event start times length '  # by christys method
                                         + 'not matching event detections')

                    # ???? eliminate false positives, which fall beyond look back time
                    event_start_indices_list_unsorted = \
                        event_start_indices_list_raw[indices]

                    idx_of_event_start_indices_list = np.argsort(
                        event_start_indices_list_unsorted)
                    event_detected_indices_unsorted = \
                        req_change_indices[idx_of_indices]
                    event_detected_indices = event_detected_indices_unsorted[
                        idx_of_event_start_indices_list]
                    req_change_lane_3_4_to_1 = req_change_lane_3_4_to_1[
                        idx_of_indices][idx_of_event_start_indices_list]
                    req_change_indices_offset_arr_ = \
                        np.sort(req_change_indices_offset_arr[idx_of_indices])

                except:

                    if differences.shape[0] == 1 and differences.shape[1] > 0:

                        indices, idx_of_indices = \
                            np.unique(np.abs(differences).argmin(),
                                      return_index=True)
                        event_start_indices_list_unsorted = req_change_indices_offset_arr[
                            indices]

                        idx_of_event_start_indices_list = np.argsort(
                            event_start_indices_list_unsorted)
                        event_detected_indices_unsorted = req_change_indices[indices]
                        event_detected_indices = event_detected_indices_unsorted[
                            idx_of_event_start_indices_list]
                        req_change_lane_3_4_to_1 = req_change_lane_3_4_to_1[
                            indices][idx_of_event_start_indices_list]
                        req_change_indices_offset_arr_ = \
                            np.sort(req_change_indices_offset_arr[indices])

                    else:

                        indices, idx_of_indices = \
                            np.unique(req_change_indices_offset_arr,
                                      return_index=True)
                        event_start_indices_list_unsorted = req_change_indices_offset_arr[
                            idx_of_indices]

                        idx_of_event_start_indices_list = np.argsort(
                            event_start_indices_list_unsorted)
                        event_detected_indices_unsorted = \
                            req_change_indices[idx_of_indices]
                        event_detected_indices = event_detected_indices_unsorted[
                            idx_of_event_start_indices_list]
                        req_change_lane_3_4_to_1 = req_change_lane_3_4_to_1[
                            idx_of_indices][idx_of_event_start_indices_list]
                        req_change_indices_offset_arr_ = \
                            np.sort(
                                req_change_indices_offset_arr[idx_of_indices])

                event_start_indices_list = event_start_indices_list_unsorted[
                    idx_of_event_start_indices_list]

                ############
                cTimes_event_start = df.loc[event_start_indices_list,
                                            'cTime'].values
                cTimes_array_detection = df.loc[event_detected_indices,
                                                'cTime'].values
                req_max_cTimes_event_end = cTimes_array_detection + self.look_forward_time_sec

                req_change_cTimes_series_2 = pd.Series(req_max_cTimes_event_end,
                                                       name='req_cTime').sort_values(ascending=True)

                req_df_copy_2 = df.copy(deep=True).reset_index()[
                    ['index', 'cTime']]

                merged_df = pd.merge_asof(req_change_cTimes_series_2,
                                          req_df_copy_2.sort_values(
                                              by='cTime'),
                                          left_on='req_cTime', right_on='cTime',
                                          direction='nearest')
                req_change_indices_offset_arr_2 = merged_df['index'].values

                ###################

                df['host_target_long_vel_diff'] = -df['host_longitudinal_velocity_mps'] + \
                    df['target_longitudinal_velocity_mps']

                df['host_long_vel_diff'] = df['host_longitudinal_velocity_mps'].diff(
                    periods=1).fillna(value=0)

                df['host_target_long_vel_diff_percent_change'] = \
                    df['host_target_long_vel_diff'].pct_change(
                        periods=1, fill_method=None).bfill()

                df['percent_change_host_long_vel'] = \
                    df['host_longitudinal_velocity_mps'].pct_change(
                        periods=1, fill_method=None).bfill()

                idx_of_improbable_start_indices = \
                    np.asarray(cTimes_event_start <
                               cTimes_array_detection - self.look_back_time_sec).nonzero()

                event_start_indices_list[idx_of_improbable_start_indices] = \
                    req_change_indices_offset_arr_[
                        idx_of_improbable_start_indices]

                idx_of_impossible_start_indices = \
                    np.asarray(cTimes_event_start >
                               cTimes_array_detection).nonzero()

                event_start_indices_list = np.delete(event_start_indices_list,
                                                     idx_of_impossible_start_indices,
                                                     axis=0)
                event_detected_indices = np.delete(event_detected_indices,
                                                   idx_of_impossible_start_indices,
                                                   axis=0).astype(int)

                if len(event_start_indices_list) > 0:
                    event_end_indices_list = []

                    redundant_indices = []
                    improbable_indices = []

                    for i, (start_index,
                            detected_index,
                            target_lane_description) in enumerate(
                        zip(event_start_indices_list,
                            event_detected_indices,
                            req_change_lane_3_4_to_1)):

                        event_df = df.loc[detected_index+1:, ['host_target_long_vel_diff',
                                                              'target_longitudinal_velocity_mps',
                                                              'host_longitudinal_velocity_mps',
                                                              'vision_host_lane_change',
                                                              'target_longitudinal_position_m',
                                                              'ALC_type_feature_enum',
                                                              'host_long_vel_diff',
                                                              'host_longitudinal_acceleration_mps2',
                                                              'percent_change_host_long_vel',
                                                              'host_target_long_vel_diff_percent_change',
                                                              'is_alc_disabled',
                                                              'is_host_decelerating',

                                                              ]
                                          ]

                        if event_detection == 'polarion':

                            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&& HERE 21')

                            event_end_idx, \
                                host_vel_stabilisation_indices, \
                                indices_of_possible_TPs = \
                                self._check_convergence(event_df,
                                                        shift_count=self.stabilisation_indices_length,
                                                        polarion=True)

                            if not event_end_idx is None:

                                df.loc[event_end_idx,
                                       'host_long_accel_flag'] = True
                            df.loc[host_vel_stabilisation_indices,
                                   'is_host_vel_stabilised'] = True

                        elif event_detection == 'dma':

                            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&& HERE 22')

                            event_end_idx, \
                                host_vel_target_rel_vel_diff_stabilisation_indices, \
                                indices_of_possible_TPs = \
                                self._check_convergence(event_df,
                                                        shift_count=self.stabilisation_indices_length,
                                                        polarion=False)

                            # df.loc[row_indices, 'events_dma'] = True
                            df.loc[host_vel_target_rel_vel_diff_stabilisation_indices,
                                   'is_host_to_target_relative_vel_stabilised'] = True

                        start_cTime = float(df.loc[start_index, 'cTime'])
                        detect_cTime = float(df.loc[detected_index, 'cTime'])

                        if not event_end_idx is None:
                            end_cTime = float(df.loc[event_end_idx, 'cTime'])
                        else:
                            end_cTime = detect_cTime  # FIXME

                        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&& HERE 3')

                        if ((event_end_idx is None)
                                or (event_end_idx <= start_index)
                                or (end_cTime <= start_cTime + 0.75)
                                or (event_end_idx <= detected_index)
                                or (event_end_idx <= detected_index+1)
                                ):

                            event_end_idx = max(event_detected_indices[i] + 1,
                                                req_change_indices_offset_arr_2[i])  # ???

                            df.loc[start_index:event_end_idx,
                                   'event_end_hard_stop'] = True

                            if event_end_idx >= len(df):

                                event_end_idx = np.max(df.index)  # FIXME

                        end_cTime = float(df.loc[event_end_idx, 'cTime'])
                        if end_cTime > detect_cTime + self.look_forward_time_sec:

                            tentative_end_cTime = detect_cTime + self.look_forward_time_sec
                            event_end_idx = max(list(
                                df.query(f'cTime <= {tentative_end_cTime}').index))
                            improbable_indices.append(i)

                        if i > 0:
                            prev_end_idx = event_end_indices_list[i-1]
                            prev_end_time = float(
                                df.loc[prev_end_idx, 'cTime'])

                            if not (start_cTime > prev_end_time +
                                    self._min_gap_between_events_sec):
                                redundant_indices.append(i)

                        event_end_indices_list.append(event_end_idx)

                    if len(redundant_indices) > 0:
                        event_start_indices_list = np.delete(event_start_indices_list,
                                                             redundant_indices, axis=0)
                        event_end_indices_list = np.delete(event_end_indices_list,
                                                           redundant_indices, axis=0)
                        event_detected_indices = np.delete(event_detected_indices,
                                                           redundant_indices, axis=0)

                    event_start_indices_list, \
                        event_detected_indices, \
                        event_end_indices_list = \
                        self._values_in_between(host_intiated_lane_change_event_starts,
                                                np.array((event_start_indices_list,
                                                          event_detected_indices,
                                                          event_end_indices_list)).T)  # ???
                    if self.rt1_pos_check_needed:
                        event_start_indices_list, \
                            event_detected_indices, \
                            event_end_indices_list = \
                            self._sanity_checks(df,
                                                np.array((event_start_indices_list,
                                                          event_detected_indices,
                                                          event_end_indices_list)).T)

                    cTimes_array_start = df.loc[event_start_indices_list,
                                                'cTime'].values
                    cTimes_array_detection = df.loc[event_detected_indices,
                                                    'cTime'].values
                    cTimes_array_end = df.loc[event_end_indices_list,
                                              'cTime'].values

                    is_start_indices_changed = not set(
                        event_start_indices_list_polarion_based).issubset(
                            set(event_start_indices_list)) and bool(
                                list(event_start_indices_list_polarion_based)) and bool(
                                    event_start_indices_list)

                    if is_start_indices_changed:

                        for start_index_feature, det_index_feature in \
                                zip(event_start_indices_list, event_detected_indices):

                            df.loc[start_index_feature:det_index_feature,
                                   'event_start_hard_stop'] = True
                            if event_detection == 'dma':
                                df.loc[start_index_feature:det_index_feature,
                                       'events_dma'] = True
                            elif event_detection == 'polarion':
                                df.loc[start_index_feature:det_index_feature,
                                       'host_long_accel_flag'] = True

                    if event_detection == 'dma':

                        for det_index_feature, end_index_feature in \
                                zip(event_detected_indices, event_end_indices_list):

                            df.loc[det_index_feature:end_index_feature,
                                   'events_dma'] = True

                            df.loc[det_index_feature:end_index_feature,
                                   'is_host_to_target_relative_vel_stabilised'] = True

                    elif event_detection == 'polarion':

                        for det_index_feature, end_index_feature in \
                                zip(event_detected_indices, event_end_indices_list):

                            df.loc[det_index_feature:end_index_feature,
                                   'host_long_accel_flag'] = True

                            df.loc[det_index_feature:end_index_feature,
                                   'is_host_vel_stabilised'] = True

                    type_det_event_detection = {str(det_index) + '_' + str(end_index):
                                                event_detection
                                                for det_index, end_index in
                                                zip(event_detected_indices,
                                                    event_end_indices_list)
                                                }

                    type_det[event_detection] = {**type_det[event_detection],
                                                 **type_det_event_detection}

                    event_detection_type_list = [event_detection
                                                 ]*len(event_start_indices_list)
                else:

                    cTimes_array_start = np.array([])
                    cTimes_array_detection = np.array([])
                    cTimes_array_end_ = np.array([])

                    event_start_indices_list = np.array([])
                    event_detected_indices = np.array([])
                    event_end_indices_list = np.array([])
                    event_detection_type_list = np.array([])

            else:

                cTimes_array_start = np.array([])
                cTimes_array_detection = np.array([])
                cTimes_array_end = np.array([])

                event_start_indices_list = np.array([])
                event_detected_indices = np.array([])
                event_end_indices_list = np.array([])
                event_detection_type_list = np.array([])

                # event_start_indices_list_polarion_based = np.array([])
            if bool(event_start_indices_list
                    ) and bool(event_detected_indices
                               ) and bool(event_end_indices_list):
                event_start_indices_list_.append(event_start_indices_list)
                event_detected_indices_.append(event_detected_indices)
                event_end_indices_list_.append(event_end_indices_list)

                cTimes_array_start_.append(cTimes_array_start)
                cTimes_array_detection_.append(cTimes_array_detection)
                cTimes_array_end_.append(cTimes_array_end)

                event_detection_type_list_.append(event_detection_type_list)

        event_indices_array = np.array((event_start_indices_list_,
                                        event_detected_indices_,
                                        event_end_indices_list_)).T
        event_cTimes_array = np.array((cTimes_array_start_,
                                       cTimes_array_detection_,
                                       cTimes_array_end_)).T
        event_detection_type_array = np.array(event_detection_type_list_)

        shape_indices = event_indices_array.shape
        shape_cTimes = event_cTimes_array.shape

        if shape_indices[0]*shape_indices[1] >= 1:

            event_indices_array = event_indices_array.reshape(
                shape_indices[0]*shape_indices[1], -1)
            event_cTimes_array = event_cTimes_array.reshape(
                shape_cTimes[0]*shape_cTimes[1], -1)
            # event_detection_type_array = event_detection_type_array.reshape(
            #     shape_indices[0]*shape_indices[1], -1)

            if (len(event_indices_array) > 2
                and
                    len(event_cTimes_array) > 2):

                event_detection_type_array = \
                    event_detection_type_array.flatten(order='F')

                if (len(event_indices_array) % 2 != 0
                        and
                        len(event_cTimes_array) % 2 != 0):

                    event_indices_array, indices = np.unique(event_indices_array,
                                                             axis=0,
                                                             return_index=True)
                    event_cTimes_array = np.unique(event_cTimes_array, axis=0)

                    event_detection_type_array = event_detection_type_array[indices]

        host_stationary_indices = df.query(
            f'(-{self.host_velocity_threshold_check} <= ' +
            'host_longitudinal_velocity_mps <= ' +
            f'{self.host_velocity_threshold_check}) '
            # +
            # 'or ' +
            # 'host_longitudinal_acceleration_mps2 < 0'
        ).index

        host_turning_or_cutout_indices = df.query(
            'is_host_turning == True '
            + 'or '
            + 'host_veh_cut_out_initiated == True ').index

        # self.target_across_path_heading_threshold
        target_across_path_indices = df.query(
            'target_heading_rad.abs() >=  '
            + f'{np.deg2rad(self.target_across_path_heading_threshold)} '
            # +
            # 'or ' +
            # 'host_longitudinal_acceleration_mps2 < 0 '
            # + 'and '
            # + 'target_lateral_velocity_mps.abs() >= '
            # + 'target_longitudinal_velocity_mps.abs() '
        ).index

        event_indices_ranges = [list(range(item[0], item[2]))
                                for item in event_indices_array]
        event_detect_end_range = [list(range(item[1], item[2]))
                                  for item in event_indices_array]
        event_host_turn_range = [list(range(item[0], item[2]))
                                 for item in event_indices_array]

        # is_event_while_host_stationary_list = \
        #     [set(list(host_stationary_indices)).issubset(set(item))
        #      or set(item).issubset(set(list(host_stationary_indices)))
        #      if len(item) > self.end_index_length_hypothetical
        #      else True
        #      for item in
        #      # event_indices_ranges
        #      event_detect_end_range
        #      ]

        event_while_host_stationary_percentage_list = [len(set(
            [i
             for i in host_stationary_indices
             if min(item) <= i <= max(item)]).intersection(set(item)))
            / len(set([i
                       for i in host_stationary_indices
                       if min(item) <= i <= max(item)]).union(set(item)))
            for item in event_detect_end_range]

        is_event_while_host_stationary_list = \
            [True if item >= self.time_percent_plausiblity_check
             else False
             for item in
             event_while_host_stationary_percentage_list]

        # if not bool(is_event_while_host_stationary_list):
        #     is_event_while_host_stationary_list = [
        #         False]*len(event_indices_array)

        plausible_percentage_list = [
            len(set(
                [i
                 for i in indices_of_possible_TPs
                 if min(item) <= i <= max(item)]).intersection(set(item)))
            / len(set([i
                       for i in indices_of_possible_TPs
                       if min(item) <= i <= max(item)]).union(set(item)))
            for item in event_detect_end_range]

        is_event_plausible_list = [True
                                   if item >= self.time_percent_plausiblity_check
                                   else False
                                   for item in plausible_percentage_list]

        # if not bool(is_event_plausible_list):
        #     is_event_plausible_list = [
        #         False]*len(event_indices_array)

        # is_event_while_host_turning_or_cutout = \
        #     [set(list(host_turning_or_cutout_indices)).issubset(set(item))
        #      or set(item).issubset(set(list(host_turning_or_cutout_indices)))
        #      for item in event_host_turn_range]

        event_while_host_turning_or_cutout_percentage_list = [len(set(
            [i
             for i in host_turning_or_cutout_indices
             if min(item) <= i <= max(item)]).intersection(set(item)))
            / len(set([i
                       for i in host_turning_or_cutout_indices
                       if min(item) <= i <= max(item)]).union(set(item)))
            for item in event_host_turn_range]

        is_event_while_host_turning_or_cutout = \
            [True if item >= self.time_percent_plausiblity_check
             else False
             for item in
             event_while_host_turning_or_cutout_percentage_list]

        # if not bool(is_event_while_host_turning_or_cutout):
        #     is_event_while_host_turning_or_cutout = [
        #         False]*len(event_indices_array)

        # is_event_while_TAP = \
        #     [set(list(target_across_path_indices)).issubset(set(item))
        #      or set(item).issubset(set(list(target_across_path_indices)))
        #      for item in event_detect_end_range]

        # event_while_TAP_percentage_list = [len(set(
        #     target_across_path_indices).intersection(set(item)))
        #     / len(set(target_across_path_indices).union(set(item)))
        #     if len(item) > self.end_index_length_hypothetical
        #     else self.time_percent_plausiblity_check
        #     for item in event_detect_end_range]

        event_while_TAP_percentage_list = [len(set(
            [i
             for i in target_across_path_indices
             if min(item) <= i <= max(item)]).intersection(set(item)))
            / len(set([i
                       for i in target_across_path_indices
                       if min(item) <= i <= max(item)]).union(set(item)))
            # if len(item) > self.end_index_length_hypothetical
            # else self.time_percent_plausiblity_check
            for item in event_detect_end_range]

        is_event_while_TAP = [True
                              if item >= self.time_percent_plausiblity_check
                              else False
                              for item in event_while_TAP_percentage_list]

        # if not bool(is_event_while_TAP):
        #     is_event_while_TAP = [False]*len(event_indices_array)

        # is_event_while_TAP = [False]*len(event_indices_array)

        # is_event_plausible_list = \
        #     [set(list(indices_of_possible_TPs)).issubset(set(item))
        #      or set(item).issubset(set(list(indices_of_possible_TPs)))
        #      for item in event_detect_end_range]

        #############################################################
        event_indices_array = np.array([item

                                        for item,
                                        bool_val_stationary,
                                        bool_val_plausible,
                                        bool_val_host_turn_or_cutout,
                                        bool_val_TAP
                                        in
                                        zip(event_indices_array,
                                            is_event_while_host_stationary_list,
                                            is_event_plausible_list,
                                            is_event_while_host_turning_or_cutout,
                                            is_event_while_TAP)
                                        if ((not bool_val_stationary
                                            and not bool_val_host_turn_or_cutout
                                            and not bool_val_TAP)
                                            or bool_val_plausible
                                            )
                                        ])  # event_cTimes_array
        event_cTimes_array = np.array([item

                                       for item,
                                       bool_val_stationary,
                                       bool_val_plausible,
                                       bool_val_host_turn_or_cutout,
                                       bool_val_TAP
                                       in
                                       zip(event_cTimes_array,
                                           is_event_while_host_stationary_list,
                                           is_event_plausible_list,
                                           is_event_while_host_turning_or_cutout,
                                           is_event_while_TAP)
                                       if ((not bool_val_stationary
                                            and not bool_val_host_turn_or_cutout
                                            and not bool_val_TAP)
                                           or bool_val_plausible
                                           )
                                       ])
        #############################################################

        return df, event_indices_array, \
            event_cTimes_array, detected_events_count, event_detection_type_array

    def _values_in_between(self, vals_to_check, start_end_array, ):

        vals_to_check = np.array(vals_to_check)

        event_start_corr, event_detect_arr, event_end_corr = [], [], []
        for start, detect, end in start_end_array:

            truth_val = np.where((vals_to_check +
                                  self.host_duration_to_maintain <= end) &
                                 (vals_to_check >=
                                  start + self.host_duration_to_maintain))

            if isinstance(truth_val, tuple):

                truth_val = truth_val[0]

            if not len(truth_val) > 0:

                event_start_corr.append(start)
                event_detect_arr.append(detect)
                event_end_corr.append(end)

        return event_start_corr, event_detect_arr, event_end_corr

    def _sanity_checks(self,
                       df,
                       start_end_array,
                       ):
        event_start_corr, event_detect_arr, event_end_corr = [], [], []
        for start, detect, end in start_end_array:

            req_array = df.loc[detect:end+1,
                               'target_longitudinal_position_m'].values

            req_array_2 = df.loc[detect:end+1,
                                 'ALC_type_feature_enum'].values
            req_truth_table = ((req_array < 100)
                               ).nonzero()[0]

            req_truth_table_2 = (
                (req_array_2 == 0)).nonzero()[0]

            if ((len(req_truth_table) >=
                 self._time_percent_rt1_distance_less_than_100
                 * len(req_array)) or
                    (len(req_truth_table_2) >=
                     self._time_percent_alc_type
                     * len(req_array_2))
                ):

                event_start_corr.append(start)
                event_detect_arr.append(detect)
                event_end_corr.append(end)

        return event_start_corr, event_detect_arr, event_end_corr

    def _check_convergence(self,
                           event_df,
                           shift_count: int = 25,  # ??? Assumed,
                           polarion: bool = True
                           ):

        except_data_iter = event_df.copy(deep=True)

        lower_limit = self.percent_change_limits[0]
        upper_limit = self.percent_change_limits[1]

        except_data_iter['intermediate_truth_col'] = False
        if polarion:

            series_data_indices = except_data_iter.query(

                f'percent_change_host_long_vel <= {upper_limit} and '
                + f'percent_change_host_long_vel >= {lower_limit} '

                + 'and (vision_host_lane_change == False or '
                # + 'ALC_type_feature_enum == '
                # + f'{self.host_ALC_type_feature_disabled_enum}) '
                + 'is_alc_disabled == True) '
                # + 'and host_longitudinal_acceleration_mps2 < 0 '
                + 'and is_host_decelerating == True'

            ).index

            host_vel_stabilisation_indices = \
                except_data_iter.query(
                    f'percent_change_host_long_vel <= {upper_limit} and '
                    + f'percent_change_host_long_vel >= {lower_limit} ').index
        else:
            series_data_indices = except_data_iter.query(
                f'(host_target_long_vel_diff_percent_change <= {upper_limit} and ' +
                f'host_target_long_vel_diff_percent_change >= {lower_limit}) or '
                + f'(percent_change_host_long_vel <= {upper_limit} and '
                + f'percent_change_host_long_vel >= {lower_limit}) '
                + 'and (vision_host_lane_change == False or '
                # + 'ALC_type_feature_enum == '
                # + f'{self.host_ALC_type_feature_disabled_enum}) '
                + 'is_alc_disabled == True) '
                + 'or target_longitudinal_position_m < '
                + f'{self.target_longitudinal_position_limit} '
                # + 'and is_host_turning == False '

            ).index

            host_vel_stabilisation_indices = \
                except_data_iter.query(
                    f'host_target_long_vel_diff_percent_change <= {upper_limit} and ' +
                    f'host_target_long_vel_diff_percent_change >= {lower_limit} and '
                    + f'percent_change_host_long_vel <= {upper_limit} and '
                    + f'percent_change_host_long_vel >= {lower_limit} ').index

        except_data_iter.loc[series_data_indices,
                             'intermediate_truth_col'] = True

        # series_data = ~except_data_iter['intermediate_truth_col']
        series_data = except_data_iter['intermediate_truth_col']

        data1 = series_data.astype(int)
        data1_shifted = data1.shift(shift_count).fillna(0)

        except_data_iter['truth_table'] = \
            np.logical_and(np.array(data1), np.array(
                data1_shifted))  # FIXME

        if except_data_iter['truth_table'].empty:

            return None, host_vel_stabilisation_indices, list(series_data_indices)

        return_index = except_data_iter['truth_table'].idxmax(
        ) - shift_count  # .idxmin()

        return int(return_index), host_vel_stabilisation_indices, list(series_data_indices)

    def find_manoeuver_latency_val(self, df,
                                   start_cTime, end_cTime):

        time_to_look_back = 1.25

        try:
            start_index_event = start_index = int(df[df['cTime']
                                                     <= start_cTime]['cTime'].idxmax())
            start_index = int(df[df['cTime']
                                 <= start_cTime
                                 - time_to_look_back]['cTime'].idxmax())
            end_index = int(df[df['cTime']
                               <= end_cTime]['cTime'].idxmax())

            jerk_vals = df.loc[start_index:end_index,
                               'host_longitudinal_jerk_mps3'].values

            acceleration_vals = df.loc[start_index:end_index,
                                       'host_longitudinal_acceleration_mps2'].values

            mean_jerk = np.mean(jerk_vals)
            std_jerk = np.std(jerk_vals, ddof=1)

            mean_accel = np.mean(acceleration_vals)
            std_accel = np.std(acceleration_vals, ddof=1)

            truth_table = np.logical_or(
                np.logical_and((mean_jerk - 1.5*std_jerk <=
                                jerk_vals),
                               (jerk_vals <= mean_jerk + 1.5*std_jerk)
                               ),
                np.logical_and((mean_accel - 1.5*std_accel <=
                                acceleration_vals),
                               (acceleration_vals <=
                                mean_accel + 1.5*std_accel)
                               )
            )

            idx_req = np.min(np.where(~truth_table))

            c_time_manoeuver = df.loc[start_index_event + idx_req, 'cTime']

            manoeuver_latency = c_time_manoeuver - start_cTime

        except:
            manoeuver_latency = 0

        return manoeuver_latency

    def compute_CFS_PFS_vector(self, df, detect_indices, type='CFS'):

        metric_vals = []

        if type == 'CFS':

            req_method = self.compute_CFS_2
        elif type == 'PFS':
            req_method = self.compute_PFS_2

        for detect_index in detect_indices:

            metric = req_method(df, detect_index)
            metric_vals.append(metric)

        metric_vals = np.array(metric_vals)
        return np.max(metric_vals[metric_vals != np.array(None)])

    def compute_CFS_2(self, df,
                      detect_index,):

        d_safety_distance_complete_stop = 2
        b_cut_in_max = 7  # m/s2
        b_ego_comf = 4  # m/s2
        b_ego_max = 6  # m/s2
        T_host_reaction_time = 0.75  # s

        detect_indices = detect_index

        Uego_long = df.loc[detect_indices,
                           'host_longitudinal_velocity_mps']

        Ucutin_long = df.loc[detect_indices,
                             'target_longitudinal_velocity_mps']  # ???

        Ucutin_lat = df.loc[detect_indices,
                            'target_lateral_velocity_mps']  # ???
        Distance = df.loc[detect_indices,
                          'target_longitudinal_position_m']  # ???
        Aego = df.loc[detect_indices,
                      'host_longitudinal_acceleration_mps2']  # ???

        Aa_ego = max(Aego, -b_ego_comf)
        Uego_long_NEXT = Uego_long+(Aa_ego*T_host_reaction_time)
        d_new = ((((Uego_long+Uego_long_NEXT)/2)-Ucutin_long)
                 * T_host_reaction_time)

        if Uego_long_NEXT <= Ucutin_long:
            dsafe = (((Uego_long-Ucutin_lat)**2)/(2*Aa_ego))
        elif Uego_long_NEXT > Ucutin_long:
            dsafe = (d_new+(((Uego_long_NEXT-Ucutin_long)**2)/(2*b_ego_comf)))
        else:  # RB 30112023
            dsafe = np.nan

        if Uego_long_NEXT <= Ucutin_long:
            dunsafe = (((Uego_long-Ucutin_long)**2)/(2*Aa_ego))
        elif Uego_long_NEXT > Ucutin_long:
            dunsafe = (((Uego_long_NEXT-Ucutin_long)**2)/(2*b_ego_max))
        else:  # RB 30112023
            dunsafe = np.nan

        if 0 < Distance < dunsafe:
            return 1
        elif Distance >= dsafe:
            return 0
        elif dunsafe <= Distance <= dsafe:
            CFS = ((Distance-dsafe)/(dunsafe-dsafe))
            return CFS
        else:
            return None

    def compute_PFS_2(self, df,
                      detect_index,
                      ):

        d_safety_distance_complete_stop = 2
        b_cut_in_max = 7  # m/s2
        b_ego_comf = 4  # m/s2
        b_ego_max = 6  # m/s2
        T_host_reaction_time = 0.75  # s

        detect_indices = detect_index

        Uego_long = df.loc[detect_indices,
                           'host_longitudinal_velocity_mps']

        Ucutin_long = df.loc[detect_indices,
                             'target_longitudinal_velocity_mps']  # ???
        Distance = df.loc[detect_indices,
                          'target_longitudinal_position_m']  # ???

        dsafe = Uego_long*T_host_reaction_time+((Uego_long*Uego_long)/(2*b_ego_comf))-(
            (Ucutin_long*Ucutin_long)/(2*b_cut_in_max))+d_safety_distance_complete_stop
        dunsafe = Uego_long*T_host_reaction_time + \
            ((Uego_long*Uego_long)/(2*b_ego_max)) - \
            ((Ucutin_long*Ucutin_long)/(2*b_cut_in_max))

        if 0 < (Distance-d_safety_distance_complete_stop) < dunsafe:
            return 1

        elif (Distance-d_safety_distance_complete_stop) > dsafe:
            return 0

        elif dunsafe < (Distance-d_safety_distance_complete_stop) < dsafe:
            PFS = ((Distance-dsafe-d_safety_distance_complete_stop)/(dunsafe-dsafe))
            return PFS
        else:
            return None

    def event_extraction(self, df, file_name, **kwargs):

        complete_events = list()
        log_path, log_name = os.path.split(file_name)

        regex_split = re.split("(?<=[0-9]{6}_[0-9]{3}_)(.*)", log_name)
        if (len(regex_split) > 2):
            base_name = regex_split[0][:-1]
        elif (len(regex_split) == 1):
            base_name = regex_split[0].split('.')[0]

        try:
            search = re.split(r'(\d{6}_\d{3})', log_name)
            split = ''.join(search[:2]).split('_')

            Vehicle = split[2]
        except Exception as e:
            Vehicle = 'log name issue'
            print(e)

        df, event_start_end_groups_2, \
            event_start_end_cTimes, \
            detected_events_count, type_det = self._event_extraction_helper(
                df, **kwargs
            )
        if len(event_start_end_groups_2) > 0:
            start_end_and_detection_indices = event_start_end_groups_2[:, [
                0, 2, 1]]
            event_start_end_cTimes = event_start_end_cTimes[:, [0, 2, 1]]
        else:
            start_end_and_detection_indices = np.array([[], [], []]).T
            event_start_end_cTimes = np.array([[], [], []]).T
        event_start_end_groups_2 = start_end_and_detection_indices
        # # in pandas index is included, removing it now
        # start_end_and_detection_indices[:, 0] = \
        #     start_end_and_detection_indices[:, 0] - 1

        dma_type_idx = []
        host_long_accel_flag_type_idx = []

        if len(start_end_and_detection_indices) > 0:
            # print(start_end_and_detection_indices)
            # print('******************')
            # print(event_start_end_groups_2)
            # start_rows, end_rows = start_and_end_event_finder(event_rows)
            for i, type_of_iteration in zip(
                    range(len(start_end_and_detection_indices)), type_det):

                start_cTime = event_start_end_cTimes[i, 0]
                end_cTime = event_start_end_cTimes[i, 1]
                host_change_cTime = event_start_end_cTimes[i, 2]
                video_link, image_link = None, None

                Time_to_collision_event_start = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 0],
                           'target_time_to_collison_2_order']))
                Time_to_collision_event_duration_min = \
                    float(patch_asscalar(df.loc[event_start_end_groups_2[i, 0]:
                                                event_start_end_groups_2[i, 1],
                                                'target_time_to_collison_2_order'].min()))
                Time_to_collision_event_end = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 1],
                           'target_time_to_collison_2_order']))

                index_min_ttc = df.loc[event_start_end_groups_2[i, 0]:
                                       event_start_end_groups_2[i, 1],
                                       'target_time_to_collison_2_order'].idxmin()
                cTime_min_ttc = float(patch_asscalar(
                    df.loc[index_min_ttc, 'cTime']))
                target_headway_2_order = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 1],
                           'target_headway_2_order']))

                Time_to_collision_event_start_interpolated = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 0],
                           'target_time_to_collison_interpolated']))
                Time_to_collision_event_duration_min_interpolated = \
                    float(patch_asscalar(df.loc[event_start_end_groups_2[i, 0]:
                                                event_start_end_groups_2[i, 1],
                                                'target_time_to_collison_interpolated'].min()))
                Time_to_collision_event_end_interpolated = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 1],
                           'target_time_to_collison_interpolated']))
                index_min_ttc_interp = df.loc[event_start_end_groups_2[i, 0]:
                                              event_start_end_groups_2[i, 1],
                                              'target_time_to_collison_interpolated'].idxmin()
                cTime_min_ttc_interp = float(
                    patch_asscalar(df.loc[index_min_ttc_interp, 'cTime']))
                target_headway_interpolated = float(patch_asscalar(
                    df.loc[event_start_end_groups_2[i, 1],
                           'target_headway_interpolated']))

                #

                # Time_to_collision_event_start = np.nan  # FIXME

                max_host_long_deceleration = float(df.loc[event_start_end_groups_2[i, 0]:
                                                          event_start_end_groups_2[i, 1],
                                                          'host_longitudinal_acceleration_mps2'].clip(lower=None,
                                                                                                      upper=0).min())

                host_decel_history_during_event = df.loc[event_start_end_groups_2[i, 2]:
                                                         event_start_end_groups_2[i, 1],
                                                         'host_longitudinal_acceleration_mps2']
                host_decelerating_sum = np.sum(
                    host_decel_history_during_event < 0)
                if host_decelerating_sum >= self._time_percent_duration_generic_condition*len(host_decel_history_during_event):
                    is_host_decelerating = True
                else:
                    is_host_decelerating = False

                max_host_long_accn_during_event = float(df.loc[event_start_end_groups_2[i, 0]:
                                                        event_start_end_groups_2[i, 1],
                                                        'host_longitudinal_acceleration_mps2'].clip(lower=0,
                                                                                                    upper=None).max())
                host_long_accn = float(df.loc[event_start_end_groups_2[i, 0],
                                              'host_longitudinal_acceleration_mps2'])
                # Discuss with Aravind about the below
                max_neg_long_jerk = float(df.loc[event_start_end_groups_2[i, 0]:
                                                 event_start_end_groups_2[i, 1],
                                                 'host_longitudinal_jerk_mps3'].clip(lower=None, upper=0).min())
                host_longitudinal_jerk_mps3 = float(df.loc[event_start_end_groups_2[i, 2],
                                                    'host_longitudinal_jerk_mps3'])

                host_manoeuvre_latency = self.find_manoeuver_latency_val(
                    df,
                    start_cTime, end_cTime)
                # host_manoeuvre_latency = np.nan  # FIXME

                # PFS = self.compute_PFS_2(df,
                #                          int(event_start_end_groups_2[i, 2]))
                # CFS = self.compute_CFS_2(df,
                #                          int(event_start_end_groups_2[i, 2]))

                # based on BCO-11264
                CFS = self.compute_CFS_PFS_vector(df,
                                                  list(range(int(event_start_end_groups_2[i, 0]),
                                                             int(event_start_end_groups_2[i, 1])+1)),
                                                  type='CFS')
                PFS = self.compute_CFS_PFS_vector(df,
                                                  list(range(int(event_start_end_groups_2[i, 0]),
                                                             int(event_start_end_groups_2[i, 1])+1)),
                                                  type='PFS')
                missed_proximity_warning = np.nan

                min_left_host_marker_with_conf = float(df.loc[event_start_end_groups_2[i, 0]:
                                                              event_start_end_groups_2[i, 1],
                                                              'host_lateral_distance_to_left_lane'].min())
                max_left_host_marker_with_conf = float(df.loc[event_start_end_groups_2[i, 0]:
                                                              event_start_end_groups_2[i, 1],
                                                              'host_lateral_distance_to_left_lane'].max())

                min_right_host_marker_with_conf = float(df.loc[event_start_end_groups_2[i, 0]:
                                                               event_start_end_groups_2[i, 1],
                                                               'host_lateral_distance_to_right_lane'].min())
                max_right_host_marker_with_conf = float(df.loc[event_start_end_groups_2[i, 0]:
                                                               event_start_end_groups_2[i, 1],
                                                               'host_lateral_distance_to_right_lane'].max())

                rt1_rel_long_position_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                 'target_longitudinal_position_m'])
                rt1_rel_long_vel_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                            'target_longitudinal_velocity_mps'])

                rt1_rel_long_position_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                               'target_longitudinal_position_m'])
                rt1_rel_long_vel_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                          'target_longitudinal_velocity_mps'])

                if rt1_rel_long_position_event_start == 0:

                    continue
                DA_status_event_start = int(df.loc[event_start_end_groups_2[i, 0],
                                                   'ALC_type_feature_enum'])

                DA_status_event_end = int(df.loc[event_start_end_groups_2[i, 1],
                                                 'ALC_type_feature_enum'])

                DA_status_vals_during_event = list(np.unique(df.loc[event_start_end_groups_2[i, 0]:
                                                                    event_start_end_groups_2[i, 1],
                                                                    'ALC_type_feature_enum'].to_list()))

                target_id_event_start = int(np.nan_to_num(df.loc[event_start_end_groups_2[i, 0],
                                                                 'target_ID'],
                                                          nan=-999).astype('int'))
                target_id_during_event = list(np.unique(df.loc[event_start_end_groups_2[i, 0]:
                                                               event_start_end_groups_2[i, 1],
                                                               'target_ID']
                                                        .fillna(-999)
                                                        .astype("Int64").to_list()))
                target_id_event_end = int(np.nan_to_num(df.loc[event_start_end_groups_2[i, 1],
                                                               'target_ID'],
                                                        nan=-999).astype('int'))
                host_steering_angle_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                               'host_comp_steering_angle_deg'])
                host_steering_angle_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                             'host_comp_steering_angle_deg'])
                # FIXME
                # TAke the complete start to end or 90% of start to end duration
                req_key = str(event_start_end_groups_2[i, 2]) + '_' + \
                    str(event_start_end_groups_2[i, 1])

                # if req_key in type_det:
                #     type_of_iteration = type_det[req_key]
                # else:
                #     type_of_iteration = '0312'

                # print(i)
                # if patch_asscalar(type_of_iteration) == 'dma':
                #     dma_type_idx.append(i)
                # elif patch_asscalar(type_of_iteration) == 'polarion':
                #     host_long_accel_flag_type_idx.append(i)

                event_polarion_during = df.loc[event_start_end_groups_2[i, 0]:
                                               event_start_end_groups_2[i, 1],
                                               'host_long_accel_flag']
                event_polarion_sum = np.sum(event_polarion_during)

                if ((event_polarion_sum
                         >= self._time_percent_duration_generic_condition*len(event_polarion_during))
                        # and (type_of_iteration in 'host_long_accel_flag')
                        ):
                    event_host_long_accel_flag = True
                else:
                    event_host_long_accel_flag = False

                event_dma_during = df.loc[event_start_end_groups_2[i, 0]:
                                          event_start_end_groups_2[i, 1],
                                          'events_dma']
                event_dma_sum = np.sum(event_dma_during)

                if ((event_dma_sum >= self._time_percent_duration_generic_condition*len(event_dma_during))
                        # and (type_of_iteration in 'events_dma')
                        ):
                    event_dem_only = True
                else:
                    event_dem_only = False

                event_is_host_turning_det_till_end = df.loc[event_start_end_groups_2[i, 2]:
                                                            event_start_end_groups_2[i, 1],
                                                            'is_host_turning']
                event_is_host_turning_sum = np.sum(
                    event_is_host_turning_det_till_end)

                if ((event_is_host_turning_sum >=
                         self._time_percent_duration_generic_condition
                         # self._time_percent_duration_vague_condition
                         * len(event_is_host_turning_det_till_end))
                        # and (type_of_iteration in 'events_dma')
                        ):
                    event_is_host_turning = True
                else:
                    event_is_host_turning = False

                event_target_closing_in_to_host_lane_flag_during = df.loc[event_start_end_groups_2[i, 0]:
                                                                          event_start_end_groups_2[i, 2],
                                                                          'target_closing_in_to_host_lane_flag']
                event_target_closing_in_to_host_lane_flag_sum = np.sum(
                    event_target_closing_in_to_host_lane_flag_during)

                if event_target_closing_in_to_host_lane_flag_sum >= self._time_percent_duration_generic_condition*len(event_target_closing_in_to_host_lane_flag_during):
                    event_target_closing_in_to_host_lane_flag = True
                else:
                    event_target_closing_in_to_host_lane_flag = False

                event_target_lat_vel_flag_during = df.loc[event_start_end_groups_2[i, 0]:
                                                          event_start_end_groups_2[i, 2],
                                                          'target_lat_vel_flag']
                event_target_lat_vel_flag_sum = np.sum(
                    event_target_lat_vel_flag_during)

                if event_target_lat_vel_flag_sum >= self._time_percent_duration_generic_condition*len(event_target_lat_vel_flag_during):
                    event_target_lat_vel_flag = True
                else:
                    event_target_lat_vel_flag = False

                event_target_lat_accel_flag_during = df.loc[event_start_end_groups_2[i, 0]:
                                                            event_start_end_groups_2[i, 2],
                                                            'target_lat_accel_flag']
                event_target_lat_accel_flag_sum = np.sum(
                    event_target_lat_accel_flag_during)

                if event_target_lat_accel_flag_sum >= self._time_percent_duration_generic_condition*len(event_target_lat_accel_flag_during):
                    event_target_lat_accel_flag = True
                else:
                    event_target_lat_accel_flag = False

                event_is_alc_disabled_during = df.loc[event_start_end_groups_2[i, 0]:
                                                      event_start_end_groups_2[i, 1],
                                                      'is_alc_disabled']
                event_is_alc_disabled_sum = np.sum(
                    event_is_alc_disabled_during)

                if event_is_alc_disabled_sum >= self._time_percent_duration_generic_condition*len(event_is_alc_disabled_during):
                    event_is_alc_disabled = True
                else:
                    event_is_alc_disabled = False

                event_is_host_not_lane_changing_during = df.loc[event_start_end_groups_2[i, 0]:
                                                                event_start_end_groups_2[i, 1],
                                                                'is_host_not_lane_changing']
                event_is_host_not_lane_changing_sum = np.sum(
                    event_is_host_not_lane_changing_during)

                if event_is_host_not_lane_changing_sum >= self._time_percent_duration_generic_condition*len(event_is_host_not_lane_changing_during):
                    is_host_not_lane_changing = True
                else:
                    is_host_not_lane_changing = False

                # self.host_duration_to_maintain
                # self.gen6_obj

                event_is_host_vel_stabilised_during = df.loc[
                    event_start_end_groups_2[i, 1] - self.stabilisation_indices_length:
                    event_start_end_groups_2[i, 1],
                    'is_host_vel_stabilised']
                event_is_host_vel_stabilised_sum = np.sum(
                    event_is_host_vel_stabilised_during)

                if event_is_host_vel_stabilised_sum >= self._time_percent_duration_generic_condition*len(event_is_host_vel_stabilised_during):
                    is_host_vel_stabilised = True
                else:
                    is_host_vel_stabilised = False

                event_is_host_to_target_relative_vel_stabilised_during = df.loc[
                    event_start_end_groups_2[i, 1] - self.stabilisation_indices_length:
                    event_start_end_groups_2[i, 1],
                    'is_host_to_target_relative_vel_stabilised']
                event_is_host_to_target_relative_vel_stabilised_sum = np.sum(
                    event_is_host_to_target_relative_vel_stabilised_during)

                if event_is_host_to_target_relative_vel_stabilised_sum >= self._time_percent_duration_generic_condition*len(event_is_host_to_target_relative_vel_stabilised_during):
                    is_host_to_target_relative_vel_stabilised = True
                else:
                    is_host_to_target_relative_vel_stabilised = False

                # event_end_hard_stop event_start_hard_stop

                event_is_event_start_hard_stop = df.loc[event_start_end_groups_2[i, 0]:
                                                        event_start_end_groups_2[i, 2],
                                                        'event_start_hard_stop']
                event_is_event_start_hard_stop_sum = np.sum(
                    event_is_event_start_hard_stop)

                if event_is_event_start_hard_stop_sum >= self._time_percent_duration_generic_condition*len(event_is_event_start_hard_stop):
                    is_event_start_hard_stop = True
                else:
                    is_event_start_hard_stop = False

                if patch_asscalar(type_of_iteration) in 'polarion' and event_host_long_accel_flag:

                    event_host_long_accel_flag = True
                else:
                    event_host_long_accel_flag = False

                if patch_asscalar(type_of_iteration) in 'events_dma' and event_dem_only:

                    event_dem_only = True

                else:
                    event_dem_only = False

                event_is_event_end_hard_stop = df.loc[event_start_end_groups_2[i, 2]:
                                                      event_start_end_groups_2[i, 1],
                                                      'event_end_hard_stop']
                event_is_event_end_hard_stop_sum = np.sum(
                    event_is_event_end_hard_stop)

                if event_is_event_end_hard_stop_sum >= self._time_percent_duration_generic_condition*len(event_is_event_end_hard_stop):
                    is_event_end_hard_stop = True
                else:
                    is_event_end_hard_stop = False

                # target_time_to_cross = \
                #     df.loc[event_start_end_groups_2[i, 1], 'cTime'] \
                #     - df.loc[event_start_end_groups_2[i, 2], 'cTime']

                # index_min_tt_cross = df.loc[event_start_end_groups_2[i, 0]:
                #                                   event_start_end_groups_2[i, 1],
                #                                   'target_time_to_cross'].idxmin()

                # target_time_to_cross = df.loc[index_min_tt_cross,
                #                              'target_time_to_cross']

                target_time_to_cross = df.loc[event_start_end_groups_2[i, 0]:
                                              event_start_end_groups_2[i, 1],
                                              'target_time_to_cross'].min()

                # target_time_to_cross = df.loc[event_start_end_groups_2[i, 2],
                #                              'target_time_to_cross']

                # cTime_min_ttc = float(patch_asscalar(
                #     df.loc[index_min_ttc, 'cTime']))

                # event_host_long_accel_flag = df.loc[event_start_end_groups_2[i, 2],
                #                                  'host_long_accel_flag']

                # event_dem_only = df.loc[event_start_end_groups_2[i, 2],
                #                               'events_dma']
                ###############################################################

                # if event_dem_only and event_is_host_turning:

                #     continue

                if patch_asscalar(type_of_iteration) == 'dma':
                    dma_type_idx.append(i)
                elif patch_asscalar(type_of_iteration) == 'polarion':
                    host_long_accel_flag_type_idx.append(i)

                ###############
                target_id_frequency_dict = \
                    dict(Counter(df.loc[event_start_end_groups_2[i, 0]:
                                        event_start_end_groups_2[i, 1],
                                        'target_ID']).items())
                target_id_frequency_dict = {key: val
                                            for key, val in
                                            target_id_frequency_dict.items()
                                            if not key is np.nan
                                            and
                                            not key == 0
                                            }

                ###############

                # reqd_target_id = target_id_event_end - 1
                reqd_target_id = max(target_id_frequency_dict,
                                     key=target_id_frequency_dict.get) - 1
                if reqd_target_id >= 0:
                    target_rel_long_position_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                        'target_longitudinal_position_m_VCS_' + str(reqd_target_id)])
                    target_rel_long_vel_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                   'target_longitudinal_velocity_mps_VCS_' + str(reqd_target_id)])
                    target_rel_long_accel_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                     'target_longitudinal_acceleration_mps2_VCS_' + str(reqd_target_id)])
                    target_heading_angle_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                    'target_heading_rad_VCS_' + str(reqd_target_id)])
                    target_rel_lat_position_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                       'target_lateral_position_m_VCS_' + str(reqd_target_id)])
                    target_rel_lat_vel_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                  'target_lateral_velocity_mps_VCS_' + str(reqd_target_id)])
                    target_rel_lat_accel_event_start = float(df.loc[event_start_end_groups_2[i, 0],
                                                                    'target_lateral_acceleration_mps2_VCS_' + str(reqd_target_id)])
                    target_rel_long_position_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                      'target_longitudinal_position_m_VCS_' + str(reqd_target_id)])
                    target_rel_long_vel_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                 'target_longitudinal_velocity_mps_VCS_' + str(reqd_target_id)])
                    target_rel_long_accel_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                   'target_longitudinal_acceleration_mps2_VCS_' + str(reqd_target_id)])
                    target_heading_angle_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                  'target_heading_rad_VCS_' + str(reqd_target_id)])
                    target_rel_lat_position_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                     'target_lateral_position_m_VCS_' + str(reqd_target_id)])
                    target_rel_lat_vel_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                'target_lateral_velocity_mps_VCS_' + str(reqd_target_id)])
                    target_rel_lat_accel_event_end = float(df.loc[event_start_end_groups_2[i, 1],
                                                                  'target_lateral_acceleration_mps2_VCS_' + str(reqd_target_id)])
                    target_rel_lat_position_abpve_threshold_flag_event_start = True if abs(
                        target_rel_lat_position_event_start) > 1.5 else False
                    target_rel_lat_position_below_threshold_event_end = True if abs(
                        target_rel_lat_position_event_end) < 1.5 else False
                else:
                    target_rel_long_position_event_start = np.nan
                    target_rel_long_vel_event_start = np.nan
                    target_rel_long_accel_event_start = np.nan
                    target_heading_angle_event_start = np.nan
                    target_rel_lat_position_event_start = np.nan
                    target_rel_lat_vel_event_start = np.nan
                    target_rel_lat_accel_event_start = np.nan
                    target_rel_long_position_event_end = np.nan
                    target_rel_long_vel_event_end = np.nan
                    target_rel_long_accel_event_end = np.nan
                    target_heading_angle_event_end = np.nan
                    target_rel_lat_position_event_end = np.nan
                    target_rel_lat_vel_event_end = np.nan
                    target_rel_lat_accel_event_end = np.nan
                    target_rel_lat_position_abpve_threshold_flag_event_start = np.nan
                    target_rel_lat_position_below_threshold_event_end = np.nan

                # DA_status_vals_during_event = np.nan

                rec = list()
                rec = [log_path,
                       log_name,
                       event_host_long_accel_flag,
                       event_dem_only,
                       is_host_decelerating,
                       event_target_closing_in_to_host_lane_flag,
                       event_target_lat_vel_flag,
                       event_target_lat_accel_flag,
                       event_is_alc_disabled,
                       event_is_host_turning,
                       is_host_not_lane_changing,
                       is_host_vel_stabilised,
                       is_host_to_target_relative_vel_stabilised,
                       is_event_end_hard_stop,
                       is_event_start_hard_stop,
                       patch_asscalar(type_of_iteration),
                       target_time_to_cross,
                       base_name,
                       Vehicle,
                       start_cTime,
                       end_cTime,
                       # start_end_and_detection_indices[i, 0],
                       df.loc[start_end_and_detection_indices[i, 0],
                              'fusion_index'],
                       # start_end_and_detection_indices[i, 1],
                       df.loc[start_end_and_detection_indices[i, 1],
                              'fusion_index'],
                       df.loc[start_end_and_detection_indices[i, 2],
                              'host_longitudinal_velocity_mps'],
                       Time_to_collision_event_start,
                       Time_to_collision_event_duration_min,
                       Time_to_collision_event_end,
                       cTime_min_ttc,
                       target_headway_2_order,
                       Time_to_collision_event_start_interpolated,
                       Time_to_collision_event_duration_min_interpolated,
                       Time_to_collision_event_end_interpolated,
                       cTime_min_ttc_interp,
                       target_headway_interpolated,
                       host_long_accn,
                       max_host_long_deceleration,
                       max_host_long_accn_during_event,
                       host_longitudinal_jerk_mps3,
                       max_neg_long_jerk,
                       # minimum_ttc,
                       host_manoeuvre_latency,
                       PFS,
                       CFS,
                       missed_proximity_warning,
                       video_link,
                       image_link,
                       min_left_host_marker_with_conf,
                       max_left_host_marker_with_conf,
                       min_right_host_marker_with_conf,
                       max_right_host_marker_with_conf,
                       rt1_rel_long_position_event_start,
                       rt1_rel_long_vel_event_start,
                       rt1_rel_long_position_event_end,
                       rt1_rel_long_vel_event_end,
                       DA_status_event_start,
                       DA_status_event_end,
                       DA_status_vals_during_event,
                       target_id_event_start,
                       target_id_during_event,
                       target_id_event_end,
                       host_steering_angle_event_start,
                       host_steering_angle_event_end,
                       target_rel_long_position_event_start,
                       target_rel_long_vel_event_start,
                       target_rel_long_accel_event_start,
                       target_heading_angle_event_start,
                       target_rel_lat_position_event_start,
                       target_rel_lat_position_abpve_threshold_flag_event_start,
                       target_rel_lat_vel_event_start,
                       target_rel_lat_accel_event_start,
                       target_rel_long_position_event_end,
                       target_rel_long_vel_event_end,
                       target_rel_long_accel_event_end,
                       target_heading_angle_event_end,
                       target_rel_lat_position_event_end,
                       target_rel_lat_position_below_threshold_event_end,
                       target_rel_lat_vel_event_end,
                       target_rel_lat_accel_event_end,
                       _resim_path_to_orig_path(
                           log_name, log_path, is_path=True),
                       _resim_path_to_orig_path(
                           log_name, log_path, is_path=False)[:-4],
                       ]
                # vcs_long_pos[start_end_and_detection_indices[i][2]-1][acc_mov_track_ids[start_end_and_detection_indices[i][2]-1][int(rt_object_at_event[i])-1]]]
                complete_events += [rec]

        complete_events_trans = [list(x) for x in zip(*complete_events)]

        # calculate_overview = True

        if self.full_output:
            if bool(np.array(complete_events_trans, dtype=object).size):

                complete_events_dict = {key: val for key, val in
                                        zip(self._headers['output_data'],
                                            complete_events_trans)
                                        }

                # print('&&&&&&&&&&&&&&&&&& complete_events_dict')
                # print(complete_events_dict.keys())

                events_overview_vals = [log_path,
                                        log_name,
                                        detected_events_count,


                                        np.sum(np.array(complete_events_dict[
                                            'is_event_CSCSA_36417'])[host_long_accel_flag_type_idx]),
                                        # np.sum(np.array(complete_events_dict[
                                        #     'is_event_CSCSA_36417'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        # np.sum(np.array(complete_events_dict[
                                        #     'is_event_dma'])[host_long_accel_flag_type_idx]),
                                        np.sum(np.array(complete_events_dict[
                                            'is_event_dma'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_87591'])[host_long_accel_flag_type_idx]),

                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_36292'])[host_long_accel_flag_type_idx]),

                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_36295'])[host_long_accel_flag_type_idx]),

                                        np.sum(np.array(complete_events_dict[
                                            'is_event_end_hard_stop'])[host_long_accel_flag_type_idx]),

                                        np.sum(np.array(complete_events_dict[
                                            'is_event_start_hard_stop'])[host_long_accel_flag_type_idx]),

                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_87591'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_36292'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        np.sum(np.array(complete_events_dict[
                                            'event_f_CSCSA_36295'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        np.sum(np.array(complete_events_dict[
                                            'is_event_end_hard_stop'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        np.sum(np.array(complete_events_dict[
                                            'is_event_start_hard_stop'])[dma_type_idx]) if bool(dma_type_idx) else 'NA',
                                        ]

                events_overview_dict = {key: [val]
                                        for key, val in zip(self._headers['output_overview'],
                                                            events_overview_vals)}

            else:

                complete_events_dict = {}

            if bool(complete_events_dict):
                complete_events_df = pd.DataFrame(complete_events_dict)
                events_overview_df = pd.DataFrame(events_overview_dict)
            else:
                complete_events_df = pd.DataFrame(
                    columns=self._headers['output_data'])
                events_overview_df = pd.DataFrame(
                    columns=self._headers['output_overview'])

            final_data = dict()
            if self.full_output:
                if "lin" in sys.platform:
                    data = [np.array(complete_events_df),
                            np.array(events_overview_df)
                            ]
                elif "win" in sys.platform:
                    data = [complete_events_df,
                            events_overview_df
                            ]
            else:
                data = [np.array(complete_events_df),
                        np.array(events_overview_df)
                        ]

            keys = ['output_data', 'output_overview']
            final_data = {key: data_iter
                          for data_iter, key in zip(data, keys)}

        else:
            print(
                f'The output events shape is {np.array(complete_events_trans, dtype=object).shape}')
            final_data = (complete_events_trans,
                          host_long_accel_flag_type_idx,
                          dma_type_idx,
                          detected_events_count)

        return final_data

    def main(self, file_name, config_path, ):

        out_df, misc_out_dict = self.main_get_signals(config_path)

        return_val = self.event_extraction(out_df, file_name, **misc_out_dict)

        return return_val


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
    program = 'Thunder'  # 'Northstar'  #
    # 'config_northstar_v1_cut_in.yaml'  #
    config_file = 'config_thunder_cut_in.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'TNDR1_ASHN_20240403_141103_WDC5_rFLR240008243301_r4SRR240011243301_rM05_rVs05070011_rA24010124360402_dma_0016.mat'
        # 'TNDR1_KALU_20240822_045150_WDC5_rFLR240008243301_r4SRR240011243301_rM05_rVs05070011_rA24010124360402_dma_0067.mat'
        # 'TNDR1_ASHN_20240222_165037_WDC3_dma_0001.mat'
    )

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
    CUTIIN_core_logic_obj = coreEventExtractionCUTIN(mat_file_data)

    return_val_dict = CUTIIN_core_logic_obj.main(file_name, config_path)

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
