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
import scipy as sp

import pickle

import re
from pathlib import Path
from collections import Counter
import itertools
from itertools import groupby, accumulate, chain

from geopy.distance import geodesic, lonlat, distance
if __package__ is None:

    print('Here at none package 1')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, \n to change 1: {to_change_path}')
    from signal_mapping_tsi import signalMapping
    from get_signals_tsi import signalData
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
                                     merge_dicts,
                                     is_monotonic,
                                     find_ranges_in_iterable,
                                     ndix_unique,
                                     _time_duration_to_indices_length,
                                     )
    os.chdir(actual_package_path)


else:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, to change 1: {to_change_path}')

    from signal_mapping_tsi import signalMapping
    from get_signals_tsi import signalData
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
                                                     merge_dicts,
                                                     is_monotonic,
                                                     find_ranges_in_iterable,
                                                     ndix_unique,
                                                     _time_duration_to_indices_length,
                                                     )
    os.chdir(actual_package_path)
    # from . import signal_mapping_aeb

# class dummy(object):
#     pass


class coreEventExtractionTSI(signalData):

    def __init__(self, raw_data) -> None:

        # super().__init__(self, raw_data)
        signalData.__init__(self, raw_data)

        self._headers = dict()

        self._headers['output_overview'] = ['log_path',
                                            'log_name',

                                            ]

        self._headers['output_data'] = ['log_path',
                                        'log_name',

                                        ]

        self.performance_indicator_cols = [
            self.vision_avi_tsr_main_sign_col_name,
            self.vision_avi_tsr_main_sign_col_confidence,
            self.vision_avi_tsr_main_sign_long_dist,

        ]

        self.flickering_threshold_sec = 0.2  # [s]

        self.feature_distance_threshold = 0.5  # [m]
        # assume 5mph to travel 50 m
        self.feature_distance_time_threshold = 22.5  # [s]
        self.feature_min_time_sign_display = 1.75  # [s]
        self._time_to_look_forward = 5
        self._time_to_look_backward = 8

        self._no_stat_mode = True

        self._precise_look_back = False

        self.is_sign_ID_req = False

        self.is_feature_based = False
        self.is_non_speed_feature_based = False
        self.is_speed_source_considered = False

        self.misc_out_dict2 = {

            0: 'Sign_Not_Detected',
            1: 'Speed_Limit_5',
            2: 'Speed_Limit_10',
            3: 'Speed_Limit_15',
            4: 'Speed_Limit_20',
            5: 'Speed_Limit_25',
            6: 'Speed_Limit_30',
            7: 'Speed_Limit_35',
            8: 'Speed_Limit_40',
            9: 'Speed_Limit_45',
            10: 'Speed_Limit_50',
            11: 'Speed_Limit_55',
            12: 'Speed_Limit_60',
            13: 'Speed_Limit_65',
            14: 'Speed_Limit_70',
            15: 'Speed_Limit_75',
            16: 'Speed_Limit_80',
            17: 'Speed_Limit_85',
            18: 'Speed_Limit_90',
            19: 'Speed_Limit_95',
            20: 'Speed_Limit_100',
            21: 'Speed_Limit_105',
            22: 'Speed_Limit_110',
            23: 'Speed_Limit_115',
            24: 'Speed_Limit_120',
            25: 'Speed_Limit_125',
            26: 'Speed_Limit_130',
            27: 'Speed_Limit_135',
            28: 'Speed_Limit_140',
            29: 'Speed_Limit_145',
            30: 'Speed_Limit_150',
            31: 'Speed_Limit_155',
            32: 'Speed_Limit_160',
            62: 'Speed_Limit_Unlimited',
            63: 'End_Speed_Limit',

        }

        self._version_avi = '12_3'

        avi_mapping_pickle_path = os.path.join(
            Path(os.getcwd()).parent,
            'data', 'others',
            'version_'+str(self._version_avi)+'_avi_mapping.pickle')

        avi_supp_signs_mapping_pickle_path = os.path.join(
            Path(os.getcwd()).parent,
            'data', 'others',
            'version_'+str(self._version_avi)+'_avi_mapping_supp_signs.pickle')

        with open(avi_mapping_pickle_path, 'rb') as file:
            self._avi_mapping_df = pickle.load(file)

        self._avi_speed_limits_df = self._avi_mapping_df[
            self._avi_mapping_df['semantic_name'].str.contains("Speed")]

        self._avi_non_speed_limits_df = self._avi_mapping_df[
            self._avi_mapping_df['semantic_name'].apply(
                lambda x: ('Pass' in x and 'Start' in x)
                or ('LED' in x and 'Pass' in x) and 'End' not in x)]

        zone_df = self._avi_mapping_df[
            self._avi_mapping_df['semantic_name'].apply(
                lambda x: (('Road' in x and ('Work' in x or 'work' in x))
                           or ('School' in x and 'Zone' in x))
                and 'End' not in x)]

        with open(avi_supp_signs_mapping_pickle_path, 'rb') as file:
            self._supp_signs_mapping_df = pickle.load(file)

        self.supp_signs_dict = {key: val
                                for key, val in
                                zip(self._supp_signs_mapping_df['enum'],
                                    self._supp_signs_mapping_df['semantic_name'])
                                }

        self.zone_enum_dict = {key: val
                               for key, val in zip(zone_df['enum'],
                                                   zone_df['semantic_name'])
                               }

        self.speed_array = np.linspace(5, 150, 1000)
        max_possible_distance = 35
        self.time_array = max_possible_distance/(self.speed_array*0.44704)

    def _main_sign_extraction(self,
                              df,
                              main_sign_col_name,
                              supp_sign_1_col_name,
                              supp_sign_2_col_name,
                              **kwargs):

        return

    def _performance_indicators_monotonicity(self,
                                             series,
                                             cTime_series,
                                             ):
        (increasing, decreasing,
         non_increasing_count, non_decreasing_count) = is_monotonic(
             np.array(series))

        monotonically_decreasing_abberations_count_span_normalised = \
            non_decreasing_count/max((cTime_series.max()
                                      - cTime_series.min()),
                                     1E-15)

        monotonically_increasing_abberations_count_span_normalised = \
            non_decreasing_count/max((cTime_series.max()
                                      - cTime_series.min()),
                                     1E-15)

        return (monotonically_increasing_abberations_count_span_normalised,
                monotonically_decreasing_abberations_count_span_normalised)

    def _performance_indicators_signal_flickering(self,
                                                  series,
                                                  cTime_series,
                                                  no_nan_indices: bool = False
                                                  # df_req,
                                                  # event_start_end_groups,
                                                  ):
        if len(series) == 0:
            return (np.nan, np.nan)

        if no_nan_indices:

            nan_val_indices = list(set(range(min(series.index),
                                             max(series.index) + 1)
                                       )
                                   - set(series.index)
                                   )
        else:

            nan_val_indices = list(series[series.isnull()].index)

        nan_val_event_start_end_groups = \
            find_ranges_in_iterable(nan_val_indices)

        nan_val_events_count = len(list(nan_val_event_start_end_groups))

        # nan_val_event_max_len = np.max([end-start
        #                                 for start, end in
        #                                 nan_val_event_start_end_groups]
        #                                )
        # nan_val_event_duration_max = np.max([
        #     cTime_series[end]-cTime_series[start]
        #     for start, end in
        #     nan_val_event_start_end_groups]
        # )
        # nan_val_event_min_len = np.min([end-start
        #                                 for start, end in
        #                                 nan_val_event_start_end_groups]
        #                                )
        # nan_val_event_duration_min = np.min([
        #     cTime_series[end]-cTime_series[start]
        #     for start, end in
        #     nan_val_event_start_end_groups]
        # )
        nan_val_event_duration_mean = np.mean([
            cTime_series[end]-cTime_series[start]
            for start, end in
            nan_val_event_start_end_groups]
        )
        # nan_val_event_duration_mode = sp.stats.mode([
        #     cTime_series[end]-cTime_series[start]
        #     for start, end in
        #     nan_val_event_start_end_groups], keepdims=False, axis=None
        # )[0]

        nan_val_events_count_span_normalised = \
            nan_val_events_count/max((cTime_series.max()
                                      - cTime_series.min()),
                                     1E-15)
        # nan_val_events_count_std_normalised = \
        #     nan_val_events_count/(cTime_series.std())

        return (nan_val_events_count_span_normalised,
                nan_val_event_duration_mean,
                )

    def _performance_indicators(self,
                                df,
                                event_dict,

                                **kwargs):

        performance_indicators_list = [
            # 'speed_limit_sign_flickering_per_sec',
            # 'non_speed_limit_sign_flickering_per_sec',
            # 'sign_confidence_flickering_per_sec',
            # 'sign_distance_negative_monotone_breaks_per_sec',
            # 'sign_range_flickering_per_sec',
        ]

        indicators_dict = {key: np.array([])
                           for key in performance_indicators_list}

        event_dict = merge_dicts(event_dict, indicators_dict)

        return event_dict

    def _event_start_end_indices_extractor(self,
                                           signal_series: pd.Series):

        # series_for_events = signal_series.fillna(method='ffill')
        series_for_events = signal_series.ffill()

        nan_val_indices = signal_series.isnull()
        groups_of_nans = signal_series.groupby(
            nan_val_indices.ne(
                nan_val_indices.shift()).cumsum().values).transform('size').gt(
            self.req_indices_len)

        insert_nan_indices = groups_of_nans & nan_val_indices
        series_for_events[insert_nan_indices] = np.nan

        series_for_events = series_for_events.dropna()

        # idx_series = pd.Series(series_for_events.dropna().index)

        # event_start_end_groups = idx_series.groupby(
        #     idx_series.diff().ne(1)
        #     .cumsum()).apply(
        #     lambda x:
        #     [x.iloc[0], x.iloc[-1]]
        #     if len(x) >= 2
        #     else [x.iloc[0]]).tolist()

        # event_start_end_groups_with_counter = \
        #     {num: [i for i, x in
        #            enumerate(series_for_events.dropna())
        #            if x == num]
        #      for num, cnt in
        #      Counter(series_for_events.dropna()).items() if cnt > 1}

        # event_groups_count_dict = {num : cnt
        #                            for num, cnt in
        #                            Counter(series_for_events.dropna()).items()
        #                            if cnt > 1}

        # event_start_end_groups = [[item[0], item[-1]]
        #                           for item in
        #                           event_start_end_groups_with_counter.values()]

        # unique_vals = pd.unique(series_for_events.dropna())

        blocks = series_for_events.diff().ne(0).cumsum()

        out = (series_for_events.index.to_frame()
               .groupby(blocks)[0].agg(['min', 'max'])
               )
        out['max'] += 1

        event_start_end_groups = out[['min', 'max']].to_numpy()

        return event_start_end_groups

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
                                is_sign_ID: bool = False,
                                ):
        # print('##############################################')
        # req_cols_long_distance = [col
        #                           for col in df.columns
        #                           if self.vision_avi_tsr_main_sign_long_dist in col]
        # long_distance_test_array = df[req_cols_long_distance].values
        # long_distance_test_array_grouped = {
        #     ID:
        #         pd.Series(index=indices,
        #                   data=long_distance_test_array[item[:, 0],
        #                                                 item[:, 1]])
        #     for (ID, indices), item in zip(sign_indices_dict.items(),
        #                                    sign_unique_vals_idx)
        # }

        long_distance_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_long_dist,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        lateral_distance_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_lat_dist,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        host_bearing_test_array_grouped = self._grouped_data(
            self.host_bearing_deg,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        geodetic_latitude_deg_array_grouped = self._grouped_data(
            'geodetic_latitude_deg',
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        geodetic_longitude_deg_array_grouped = self._grouped_data(
            'geodetic_longitude_deg',
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        indices_length_feature_must_display = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self.feature_min_time_sign_display)

        indices_to_look_forward = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self._time_to_look_forward)

        indices_to_look_backward = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self._time_to_look_backward)

        # req_cols_confidence = [col
        #                        for col in df.columns
        #                        if self.vision_avi_tsr_main_sign_col_confidence
        #                        in col]
        # confidence_test_array = df[req_cols_confidence].values
        # confidence_test_array_grouped = {
        #     ID:
        #         pd.Series(index=indices,
        #                   data=confidence_test_array[item[:, 0],
        #                                              item[:, 1]])
        #     for (ID, indices), item in zip(sign_indices_dict.items(),
        #                                    sign_unique_vals_idx)
        # }

        if self.is_feature_based and not self.is_non_speed_feature_based:

            if self.is_speed_source_considered:
                print('In feature Speed Limit, source filtered related loop. ',
                      'If seen on HPC jobout, analyse')

            else:
                print('In feature Speed Limit related loop. ',
                      'If seen on HPC jobout, analyse')
        elif self.is_feature_based and self.is_non_speed_feature_based:

            print('In feature Non-Speed Limit related loop. ',
                  'If seen on HPC jobout, analyse')

        elif not self.is_feature_based and not is_sign_ID:

            print(
                'In Sign enum related loop. If seen on HPC jobout, observe and ignore')
        else:

            print('In Sign ID related loop. If seen on HPC jobout, can safely ignore')

        if is_sign_ID:

            # print('In Sign ID related loop. If seen on HPC jobout, can safely ignore')

            main_sign_test_array_grouped = self._grouped_data(
                self.vision_avi_tsr_main_sign_ID_col_name,
                sign_unique_vals_idx,
                sign_indices_dict,
                df)
        else:
            # if self.is_feature_based:
            #     print('In feature related loop. If seen on HPC jobout, analyse')
            # else:
            #     print(
            #         'In Sign enum related loop. If seen on HPC jobout, observe and ignore')
            main_sign_test_array_grouped = self._grouped_data(
                self.vision_avi_tsr_main_sign_col_name,
                sign_unique_vals_idx,
                sign_indices_dict,
                df)

        main_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp1_sign_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_1_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp1_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_1_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp2_sign_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_2_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)
        supp2_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_2_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        avi_mapping_df = self._avi_speed_limits_df
        # print('##############################################')

        event_start_cTime_list = []
        event_end_cTime_list = []
        sequence_start_cTime_list = []
        log_path_event_start_list = []
        sequence_name_event_start_list = []
        log_name_event_start_list = []
        log_name_event_end_list = []
        base_logname_event_start_list = []
        rTag_event_start_list = []
        main_sign_enum_list = []
        supp_1_sign_enum_list, supp_2_sign_enum_list = [], []

        host_gps_longitude_event_end_list = []
        host_gps_latitude_event_end_list = []

        host_gps_longitude_event_start_list = []
        host_gps_latitude_event_start_list = []

        target_gps_long_lat_coordinates_list = []
        to_trust_target_gps_list = []

        vision_frame_ID_event_start_list, readff_link_event_start_list = [], []
        vision_frame_ID_event_end_list = []
        readff_link_full_video_list = []

        main_sign_confidence_mean_list, main_sign_confidence_std_list = [], []
        supp_1_sign_confidence_mean_list, supp_1_sign_confidence_std_list = [], []
        supp_2_sign_confidence_mean_list, supp_2_sign_confidence_std_list = [], []

        main_sign_confidence_event_start_list = []
        supp_1_sign_confidence_event_start_list = []
        supp_2_sign_confidence_event_start_list = []

        main_sign_confidence_median_list = []
        main_sign_confidence_median_abs_deviation_list = []

        main_sign_ID_list = []

        host_long_velocity_median_list = []
        host_long_velocity_median_abs_deviation_list = []
        host_long_velocity_event_start_list = []

        main_sign_flickering_count_normalised_list = []
        main_sign_flickering_mean_duration_list = []
        main_sign_confidence_flickering_count_normalised_list = []
        main_sign_confidence_flickering_mean_duration_list = []
        main_sign_distance_monotonocity_abberations_count_normalised_list = []

        ff_20ms_TSR_status_list = []
        ff_20ms_fused_speed_limit_source_enum_list = []
        ff_20ms_fused_speed_limit_source_all_unique_enum_list = []
        ff_20ms_fused_speed_limit_sign_enum_list = []
        ff_20ms_fused_speed_limit_sign_enum_mapped_list = []

        main_sign_occurence_delta_cTime_list = []

        can_telematics_speed_limit_enum_list = []

        avi_SW_version_list = []

        ff_20ms_supp_speed_limit_sign_enum_list = []

        ff_20ms_overtake_sign_enum_list = []

        event_signal_iter_list = []
        ff_10ms_construction_area_enum_list = []
        ff_10ms_construction_area_enum_unique_list = []

        manual_label_comparison_list = []
        gt_confidence_list = []
        gt_type_list = []

        # event_start_end_groups_dict2 = copy.deepcopy(
        #     event_start_end_groups_dict)

        for enum, event_start_end_groups_array_list in \
                event_start_end_groups_dict.items():

            # print('*********************', enum, '^^^^^^^^^^^^^^^^^')

            # if isinstance(sign_indices_dict[enum], dict):

            #     series_main_list_idx = list(
            #         chain.from_iterable(
            #             sign_indices_dict[enum].values()))

            #     series_main_list = []

            # series_main_list = list(sign_indices_dict[enum].values())

            for enumerated_idx, event_start_end_groups in \
                    enumerate(event_start_end_groups_array_list):

                # event_start_end_groups = []
                # print('%%%%%%%% ', enumerated_idx, )

                if isinstance(sign_indices_dict[enum], dict):

                    req_indices_values = list(
                        chain.from_iterable(
                            sign_indices_dict[enum].values()
                        )
                    )[enumerated_idx][:, 0]

                    # series_main = series_main_list[enumerated_idx]

                    series_main = main_sign_test_array_grouped[
                        enum][enumerated_idx]
                    series_main_confidence = \
                        main_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]
                    long_distance_signal_iter = \
                        long_distance_test_array_grouped[enum][enumerated_idx]

                    lat_distance_signal_iter = \
                        lateral_distance_test_array_grouped[enum][enumerated_idx]

                    host_bearing_iter = \
                        host_bearing_test_array_grouped[enum][enumerated_idx]

                    geodetic_latitude_iter = \
                        geodetic_latitude_deg_array_grouped[enum][enumerated_idx]

                    geodetic_longitude_iter = \
                        geodetic_longitude_deg_array_grouped[enum][enumerated_idx]

                    series_supp_1 = supp1_sign_test_array_grouped[
                        enum][enumerated_idx]

                    series_supp_1_confidence = \
                        supp1_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]

                    series_supp_2 = supp2_sign_test_array_grouped[
                        enum][enumerated_idx]

                    series_supp_2_confidence = \
                        supp2_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]
                else:
                    req_indices_values = np.array(sign_indices_dict[enum],
                                                  dtype=int)

                    series_main = main_sign_test_array_grouped[
                        enum]
                    series_main_confidence = \
                        main_sign_confidence_test_array_grouped[
                            enum]
                    long_distance_signal_iter = \
                        long_distance_test_array_grouped[enum]

                    lat_distance_signal_iter = \
                        lateral_distance_test_array_grouped[enum]

                    host_bearing_iter = \
                        host_bearing_test_array_grouped[enum]

                    geodetic_latitude_iter = \
                        geodetic_latitude_deg_array_grouped[enum]

                    geodetic_longitude_iter = \
                        geodetic_longitude_deg_array_grouped[enum]

                    series_supp_1 = supp1_sign_test_array_grouped[
                        enum]

                    series_supp_1_confidence = \
                        supp1_sign_confidence_test_array_grouped[
                            enum]

                    series_supp_2 = supp2_sign_test_array_grouped[
                        enum]

                    series_supp_2_confidence = \
                        supp2_sign_confidence_test_array_grouped[
                            enum]

                # for event_start_end_groups in event_start_end_groups_array:
                traffic_sign_relative_angle_deg_iter = \
                    self._traffic_sign_to_host_angle2(
                        long_distance_signal_iter,
                        lat_distance_signal_iter)

                traffic_sign_bearing_deg_iter = (
                    host_bearing_iter
                    + traffic_sign_relative_angle_deg_iter) % 360

                traffic_sign_euclidean_distance_m_iter = pd.Series(
                    data=np.sqrt(np.sum(
                        np.square([long_distance_signal_iter,
                                   lat_distance_signal_iter]), axis=0)),
                    index=long_distance_signal_iter.index)

                to_trust_traffic_sign_lat_long_iter = pd.Series(
                    data=[
                        (np.isnan(euclidean_dist)
                         and np.isnan(geodetic_lat)
                         and np.isnan(geodetic_long)
                         and np.isnan(sign_bearing)
                         )
                        for (euclidean_dist,
                             geodetic_lat,
                             geodetic_long,
                             sign_bearing) in
                        zip(traffic_sign_euclidean_distance_m_iter,
                            geodetic_latitude_iter,
                            geodetic_longitude_iter,
                            traffic_sign_bearing_deg_iter)],
                    index=long_distance_signal_iter.index)

                if traffic_sign_bearing_deg_iter.isnull().values.any():

                    traffic_sign_bearing_deg_iter = \
                        traffic_sign_bearing_deg_iter.bfill().ffill()

                if traffic_sign_euclidean_distance_m_iter.isnull().values.any():
                    traffic_sign_euclidean_distance_m_iter = \
                        traffic_sign_euclidean_distance_m_iter.bfill().ffill()

                if (traffic_sign_euclidean_distance_m_iter.isnull(
                ).values.all()
                        or traffic_sign_bearing_deg_iter.isnull().values.all()
                        or geodetic_latitude_iter.isnull().values.all()
                        or geodetic_longitude_iter.isnull().values.all()
                ):

                    traffic_sign_latitude_iter = pd.Series(
                        data=[
                            np.nan for item in
                            traffic_sign_euclidean_distance_m_iter],
                        index=long_distance_signal_iter.index)

                    traffic_sign_longitude_iter = pd.Series(
                        data=[
                            np.nan for item in
                            traffic_sign_euclidean_distance_m_iter],
                        index=long_distance_signal_iter.index)
                else:

                    destination_lat_long_series = [
                        distance(kilometers=euclidean_dist/1E3).destination(
                            (geodetic_lat, geodetic_long), bearing=sign_bearing)
                        if (-90.0 <= geodetic_lat <= 90.0
                            and -180.0 <= geodetic_long <= 180.0)
                        else {'latitude': np.nan,
                              'longitude': np.nan}
                        for (euclidean_dist,
                             geodetic_lat,
                             geodetic_long,
                             sign_bearing) in
                        zip(traffic_sign_euclidean_distance_m_iter,
                            geodetic_latitude_iter,
                            geodetic_longitude_iter,
                            traffic_sign_bearing_deg_iter)
                    ]

                    traffic_sign_latitude_iter = pd.Series(
                        data=[
                            item['latitude'] if isinstance(item, dict)
                            else
                            item.latitude for item in destination_lat_long_series],
                        index=long_distance_signal_iter.index)

                    traffic_sign_longitude_iter = pd.Series(
                        data=[
                            item['longitude'] if isinstance(item, dict)
                            else
                            item.longitude for item in destination_lat_long_series],
                        index=long_distance_signal_iter.index)

                event_start_cTime = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'cTime'])
                event_end_cTime = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'cTime'])

                sequence_start_cTime = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'start_cTime_sequence'])

                log_path_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'log_path_flat'])

                sequence_name_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'seq_name'])

                log_name_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'log_name_flat'])
                log_name_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'log_name_flat'])

                vision_frame_ID_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'frame_ID'])

                vision_frame_ID_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'frame_ID'])

                readff_link_event_start_pre = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'readff_link'])

                readff_link_event_start = [item1 + '?iframe=' + str(
                    int(item2 + (item3-item2)*0.5)
                    if item3 > item2 else int(item2)
                )
                    for item1, item2, item3 in
                    zip(readff_link_event_start_pre,
                        vision_frame_ID_event_start,
                        vision_frame_ID_event_end)]

                readff_link_full_video = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'readff_link_full_video'])

                # if self.program_name_readff_map is not None:

                # req_prefix = self.program_name_readff_map

                base_logname_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'base_logname'])
                rTag_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'rTag'])

                host_gps_longitude_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'geodetic_longitude_deg'])
                host_gps_latitude_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'geodetic_latitude_deg'])

                host_gps_longitude_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'geodetic_longitude_deg'])
                host_gps_latitude_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'geodetic_latitude_deg'])

                avi_SW_version = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'avi_SW_version'])
                event_signal_iter = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'event_signal'])

                if self.is_feature_based:

                    # print('??????????????????????????????')

                    if self.is_non_speed_feature_based:
                        avi_mapping_df = self._avi_non_speed_limits_df

                    ff_20ms_TSR_status = np.array(
                        [df[self.feature_tsi_OW_TSR_status]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_source_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_source_enum]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_source_all_unique_enum = np.array(
                        [','.join(
                            map(str,
                                df[self.feature_tsi_OW_speed_limit_source_enum]
                                [req_indices_values[
                                    (req_indices_values >= start)
                                    & (req_indices_values <= end)]].unique()))
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_sign_enum_mapped = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         .map(self.misc_out_dict2)
                         .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])
                    ff_20ms_supp_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_supp_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_overtake_sign_enum = np.array(
                        [df[self.feature_tsi_OW_overtake_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_10ms_construction_area_enum = np.array(
                        [

                            (df[self.feature_tsi_IW_construction_enum]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]]
                             .value_counts(normalize=True)*100)
                            .reset_index().to_csv(header=None,
                                                  index=False,
                                                  float_format="%.2f").strip('\n')
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ])

                    ff_10ms_construction_area_enum_unique = np.array(
                        [','.join(
                            map(str,
                                df[self.feature_tsi_IW_construction_enum]
                                [req_indices_values[
                                    (req_indices_values >= start)
                                    & (req_indices_values <= end)]].unique()))
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                         ])

                    can_telematics_speed_limit_enum = [
                        np.nan
                        for item in ff_20ms_fused_speed_limit_sign_enum
                    ]

                    if self.can_telematics_speed_limit_enum in df.columns:
                        telematics_series = df[
                            self.can_telematics_speed_limit_enum]

                        can_telematics_speed_limit_enum_pre_01 = [

                            telematics_series[start:end].isin(
                                [self.feature_can_telematics_mapping.get(int(sign),
                                                                         np.nan)]
                            )
                            for sign, start, end in
                            zip(ff_20ms_fused_speed_limit_sign_enum,
                                event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ]

                        can_telematics_speed_limit_enum_pre = [
                            telematics_series[item[item == True].index].mode(
                                dropna=False)
                            for item in can_telematics_speed_limit_enum_pre_01]

                        can_telematics_speed_limit_enum = np.array(
                            [item[0] if len(item.values) > 0
                             else np.nan
                             for item in
                             can_telematics_speed_limit_enum_pre]
                        )

                    orig_event_start_end_groups = copy.deepcopy(
                        event_start_end_groups)

                    if self._precise_look_back:

                        host_long_velocity_event_end = np.array(
                            [df[self.vse_host_long_velocity_col][end]
                             for end in
                             event_start_end_groups[:, 1]
                             ])

                        look_back_indices_array = [
                            _time_duration_to_indices_length(
                                df['cTime'].reset_index(
                                    drop=True).to_frame(),
                                self.time_array[
                                    np.abs(self.speed_array
                                           - event_end_long_vel).argmin()])
                            for event_end_long_vel in host_long_velocity_event_end
                        ]

                        event_start_end_groups = np.array([
                            [max(start-look_back_index,
                                 df.index[0]),
                             min(start+indices_to_look_forward,
                                 df.index[-1])]
                            for start, look_back_index in
                            zip(event_start_end_groups[:, 0],
                                look_back_indices_array)
                        ])

                    else:

                        event_start_end_groups = np.array([
                            [max(start-indices_to_look_backward,
                                 df.index[0]),
                             min(start+indices_to_look_forward,
                                 df.index[-1])]
                            for start in event_start_end_groups[:, 0]
                        ])

                    # event_start_end_groups[:, 1] = event_start_end_groups[:, 0]

                    # event_start_end_groups[:, 0] = np.array(
                    #     [item if cond > item else cond for cond, item in zip(
                    #         host_long_distance_event_indices,
                    #         event_start_end_groups[:, 1]
                    #     )
                    #     ])
                    ############################################
                    req_indices_values = df.index[
                        np.min(event_start_end_groups, axis=None):
                        np.max(event_start_end_groups, axis=None)
                    ]
                    #######################################
                    # GT Check
                    unique_basenames = df.loc[req_indices_values,
                                              'base_logname'].unique()

                    avi_log_basename = [
                        df['base_logname'].loc[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]].unique()
                        # .dropna(axis=1, how='all')
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    if not self.df_GT.empty:

                        df_GT_iter = [
                            self.df_GT[
                                self.df_GT['base_logname'].isin(
                                    unique_avi_log_basename)]
                            for unique_avi_log_basename in avi_log_basename
                        ]

                        feature_20ms_SL_numeric = \
                            [int(re.findall(r'\d+', str(map_sign))[0])
                             if len(re.findall(r'\d+', str(map_sign))) > 0
                             else np.nan
                             for map_sign in
                                ff_20ms_fused_speed_limit_sign_enum_mapped]

                        manual_label_comparison = []
                        gt_confidence = []
                        gt_type = []

                        for sl, df_GT_iter_iter in zip(
                                feature_20ms_SL_numeric,
                                df_GT_iter
                        ):
                            if df_GT_iter_iter.empty:

                                manual_label_comparison.append('No GT')
                                gt_confidence.append(np.nan)
                                gt_type.append('No GT')
                                continue
                            manual_label_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['manual_label_main_sign'
                                                    ].astype(
                                        str)
                                 ]

                            signDB_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['signDB_main_sign'
                                                    ].astype(str)
                                 ]

                            yolo_v8_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['yolo_v8_main_sign'
                                                    ].astype(str)
                                 ]

                            ml_3_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['ml3_main_sign'].astype(
                                        str)
                                 ]

                            if (
                                sl in manual_label_main_sign_vals
                                or sl in yolo_v8_main_sign_vals
                                or sl in signDB_main_sign_vals
                                or sl in ml_3_main_sign_vals
                            ) and not np.isnan(sl):

                                if sl in manual_label_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(1)
                                    gt_type.append('Manual Label')
                                elif sl in signDB_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.9)
                                    gt_type.append('Sign Database')
                                elif sl in yolo_v8_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.6)
                                    gt_type.append('ML Yolo V8')
                                elif sl in ml_3_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.6)
                                    gt_type.append('ML assorted')

                            else:

                                if not sl in manual_label_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(1)
                                    gt_type.append('Manual Label')
                                elif not sl in signDB_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.9)
                                    gt_type.append('Sign Database')
                                elif not sl in yolo_v8_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.5)
                                    gt_type.append('ML Yolo V8')
                                elif not sl in ml_3_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.5)
                                    gt_type.append('ML assorted')
                                else:

                                    manual_label_comparison.append('No GT')
                                    gt_confidence.append(1)
                                    gt_type.append('No GT')

                    else:

                        manual_label_comparison = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                        gt_confidence = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                        gt_type = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                    ###############################

                    req_cols_main = [col
                                     for col in df.columns
                                     if self.vision_avi_tsr_main_sign_col_name in col]
                    series_main = df.loc[req_indices_values, req_cols_main]
                    series_main = series_main.where(series_main < pow(2, 15),
                                                    np.nan,
                                                    inplace=False).dropna(axis=1,
                                                                          how='all')

                    unique_enums = np.unique(series_main)
                    unique_enums = unique_enums[~np.isnan(unique_enums)]
                    df_enums = pd.DataFrame()
                    for enum_avi in unique_enums:
                        df_enums['col_'+str(int(enum_avi))] = \
                            series_main[
                                series_main == enum_avi].dropna(
                                    axis=1, how='all').bfill(
                                        axis=1).iloc[:, 0]

                    series_main = df_enums[
                        df_enums.isin(
                            avi_mapping_df['enum'].values)
                    ].dropna(axis=1, how='all')

                    if len(series_main) == 0 or series_main.empty:

                        if isinstance(sign_indices_dict[enum], dict):

                            req_indices_values = list(
                                chain.from_iterable(
                                    sign_indices_dict[enum].values()
                                )
                            )[enumerated_idx][:, 0]

                        else:
                            req_indices_values = np.array(sign_indices_dict[enum],
                                                          dtype=int)

                        event_start_end_groups = orig_event_start_end_groups

                        main_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        # can_telematics_speed_limit_enum = [
                        #     np.nan
                        #     for item in ff_20ms_fused_speed_limit_sign_enum
                        # ]

                        main_sign_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]
                        main_sign_confidence_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        #     main_sign_confidence_event_start

                        main_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median_abs_deviation = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        host_long_velocity_event_start = np.array(
                            [df[self.vse_host_long_velocity_col][start]
                             for start in
                             event_start_end_groups[:, 0]
                             ])
                        # host_long_distance_event_start = np.array(
                        #     [df[self.vse_host_long_velocity_col][start]
                        #      for start in
                        #      event_start_end_groups[:, 0]
                        #      ])

                        host_long_velocity_median = np.array(
                            [df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])
                        host_long_velocity_median_abs_deviation = np.array(
                            [sp.stats.median_abs_deviation(
                                df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]],
                                axis=None,
                                nan_policy='omit')
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])

                        # long_distance_signal_iter = \
                        #     self.vision_avi_tsr_main_sign_long_dist \
                        #     + '_' + col.split('_')[-1]
                        main_sign_distance_monotonocity_abberations_count_normalised = \
                            np.array(
                                [self._performance_indicators_monotonicity(
                                    long_distance_signal_iter[req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]],
                                    df['cTime'][req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]]
                                )[1]
                                    for start, end in
                                    zip(event_start_end_groups[:, 0],
                                        event_start_end_groups[:, 1])
                                ])

                        target_long_lat_median = \
                            [(traffic_sign_longitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True),
                             traffic_sign_latitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True))
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ]

                        to_trust_target_gps = np.array(
                            [
                                to_trust_traffic_sign_lat_long_iter
                                [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                                for start, end in
                                zip(event_start_end_groups[:, 0],
                                    event_start_end_groups[:, 1])
                            ])

                        main_sign_occurence_delta_cTime = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        to_trust_target_gps_list.append(to_trust_target_gps)

                        event_start_cTime_list.append(event_start_cTime)
                        event_end_cTime_list.append(event_end_cTime)
                        sequence_start_cTime_list.append(sequence_start_cTime)
                        log_path_event_start_list.append(log_path_event_start)
                        sequence_name_event_start_list.append(
                            sequence_name_event_start)
                        log_name_event_start_list.append(log_name_event_start)
                        log_name_event_end_list.append(log_name_event_end)
                        base_logname_event_start_list.append(
                            base_logname_event_start)
                        rTag_event_start_list.append(rTag_event_start)
                        main_sign_enum_list.append(main_sign_enum)

                        can_telematics_speed_limit_enum_list.append(
                            can_telematics_speed_limit_enum)

                        vision_frame_ID_event_start_list.append(
                            vision_frame_ID_event_start)
                        readff_link_event_start_list.append(
                            readff_link_event_start)

                        vision_frame_ID_event_end_list.append(
                            vision_frame_ID_event_end)

                        readff_link_full_video_list.append(
                            readff_link_full_video)

                        host_gps_longitude_event_end_list.append(
                            host_gps_longitude_event_end)
                        host_gps_latitude_event_end_list.append(
                            host_gps_latitude_event_end)

                        host_gps_longitude_event_start_list.append(
                            host_gps_longitude_event_start)
                        host_gps_latitude_event_start_list.append(
                            host_gps_latitude_event_start)

                        target_gps_long_lat_coordinates_list.append(
                            target_long_lat_median)

                        main_sign_confidence_mean_list.append(
                            main_sign_confidence_mean)
                        main_sign_confidence_std_list.append(
                            main_sign_confidence_std)

                        main_sign_confidence_event_start_list.append(
                            main_sign_confidence_event_start)

                        main_sign_confidence_median_list.append(
                            main_sign_confidence_median)
                        main_sign_confidence_median_abs_deviation_list.append(
                            main_sign_confidence_median_abs_deviation)

                        supp_1_sign_enum_list.append(supp_1_sign_enum)
                        supp_2_sign_enum_list.append(supp_2_sign_enum)

                        supp_1_sign_confidence_mean_list.append(
                            supp_1_sign_confidence_mean)
                        supp_1_sign_confidence_std_list.append(
                            supp_1_sign_confidence_std)

                        supp_1_sign_confidence_event_start_list.append(
                            supp_1_sign_confidence_event_start)

                        supp_2_sign_confidence_mean_list.append(
                            supp_2_sign_confidence_mean)
                        supp_2_sign_confidence_std_list.append(
                            supp_2_sign_confidence_std)

                        supp_2_sign_confidence_event_start_list.append(
                            supp_2_sign_confidence_event_start)

                        host_long_velocity_median_list.append(
                            host_long_velocity_median)
                        host_long_velocity_median_abs_deviation_list.append(
                            host_long_velocity_median_abs_deviation)
                        host_long_velocity_event_start_list.append(
                            host_long_velocity_event_start)

                        # main_sign_ID_list.append(main_sign_ID)

                        main_sign_flickering_count_normalised_list.append(
                            main_sign_flickering_count_normalised)
                        main_sign_flickering_mean_duration_list.append(
                            main_sign_flickering_mean_duration)
                        main_sign_confidence_flickering_count_normalised_list.append(
                            main_sign_confidence_flickering_count_normalised)
                        main_sign_confidence_flickering_mean_duration_list.append(
                            main_sign_confidence_flickering_mean_duration)
                        main_sign_distance_monotonocity_abberations_count_normalised_list\
                            .append(
                                main_sign_distance_monotonocity_abberations_count_normalised)

                        ff_20ms_fused_speed_limit_sign_enum_list.append(
                            ff_20ms_fused_speed_limit_sign_enum)
                        ff_20ms_fused_speed_limit_source_enum_list.append(
                            ff_20ms_fused_speed_limit_source_enum)
                        ff_20ms_fused_speed_limit_source_all_unique_enum_list.append(
                            ff_20ms_fused_speed_limit_source_all_unique_enum)
                        ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                        ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                            ff_20ms_fused_speed_limit_sign_enum_mapped)

                        manual_label_comparison_list.append(
                            manual_label_comparison)
                        gt_confidence_list.append(gt_confidence)
                        gt_type_list.append(gt_type)

                        ff_20ms_supp_speed_limit_sign_enum_list.append(
                            ff_20ms_supp_speed_limit_sign_enum)
                        ff_20ms_overtake_sign_enum_list.append(
                            ff_20ms_overtake_sign_enum)

                        main_sign_occurence_delta_cTime_list.append(
                            main_sign_occurence_delta_cTime)

                        avi_SW_version_list.append(avi_SW_version)
                        event_signal_iter_list.append(event_signal_iter)
                        ff_10ms_construction_area_enum_list.append(
                            ff_10ms_construction_area_enum)
                        ff_10ms_construction_area_enum_unique_list.append(
                            ff_10ms_construction_area_enum_unique)

                        continue

                    main_sign_enum_pre = [
                        series_main.loc[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)], :]
                        # .dropna(axis=1, how='all')
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    main_sign_enum_01 = [
                        item.isin(
                            self.feature_avi_mapping.get(int(sign),
                                                         [np.nan])
                        )
                        for sign, item in
                        zip(ff_20ms_fused_speed_limit_sign_enum,
                            main_sign_enum_pre)
                    ]

                    main_sign_enum_02 = [item2[item1 == True]
                                         .dropna(how='all', axis=1)
                                         .dropna(how='all', axis=0)
                                         for item1, item2 in zip(main_sign_enum_01,
                                                                 main_sign_enum_pre)
                                         ]

                    main_sign_occurence_index = []

                    for item, start in zip(main_sign_enum_02,
                                           orig_event_start_end_groups[:, 0]):

                        if not item.empty:

                            if item.index[-1] <= start:
                                main_sign_occurence_index.append(
                                    item.index[-1])
                            elif item.index[0] <= start <= item.index[-1]:
                                # if start - item.index[0] > item.index[-1] - start:
                                #     main_sign_occurence_index.append(
                                #         item.index[-1])
                                # else:
                                #     main_sign_occurence_index.append(
                                #         item.index[0])
                                main_sign_occurence_index.append(
                                    item.index[0])
                            elif item.index[0] >= start:
                                main_sign_occurence_index.append(
                                    item.index[0])
                            else:
                                main_sign_occurence_index.append(start+1)
                        else:
                            main_sign_occurence_index.append(np.nan)

                    main_sign_occurence_index = np.array(
                        main_sign_occurence_index)

                    # main_sign_occurence_index = np.array(
                    #     [item.index[-1] if not item.empty
                    #       else np.nan
                    #       for item in main_sign_enum_02
                    #       ]
                    # )

                    main_sign_occurence_delta_cTime = [
                        start_cTime - df.loc[index, 'cTime']
                        if not np.isnan(index)
                        else np.nan
                        for index, start_cTime in
                        zip(main_sign_occurence_index,
                            event_start_cTime)
                    ]

                    if self._no_stat_mode:

                        main_sign_enum = []

                        for idx, item in zip(main_sign_occurence_index,
                                             main_sign_enum_02):

                            if len(item) > 0:

                                if isinstance(item, pd.DataFrame):

                                    values_needed = item.loc[int(idx),
                                                             :].values.flatten()
                                elif isinstance(item, pd.Series):

                                    values_needed = item[int(
                                        idx)].values.flatten()

                                values_needed = values_needed[
                                    ~np.isnan(values_needed)]
                                main_sign_enum.append(int(values_needed[-1]))

                            else:
                                main_sign_enum.append('not available')

                        main_sign_enum = np.array(main_sign_enum)

                        # main_sign_enum = np.array(
                        #     [int(item.values.flatten()[-1])
                        #      if len(item) > 0
                        #      else 'not available'
                        #      for item in main_sign_enum_02
                        #      ]
                        # )

                        # if self.can_telematics_speed_limit_enum in df.columns:
                        #     telematics_series = df[
                        #         self.can_telematics_speed_limit_enum]
                        #     can_telematics_speed_limit_enum = np.array(
                        #         [telematics_series[int(idx)]
                        #          if not np.isnan(idx)
                        #          else np.nan
                        #          for idx in main_sign_occurence_index
                        #          ]
                        #     )

                        # can_telematics_speed_limit_enum =[
                        #     telematics_series[start:end].isin(
                        #         [self.feature_can_telematics_mapping[avi_sign]]
                        #     )
                        #     for avi_sign, start, end in
                        #     zip(ff_20ms_fused_speed_limit_sign_enum,
                        #         event_start_end_groups[:, 0],
                        #             event_start_end_groups[:, 1])
                        # ]

                    else:

                        main_sign_enum = np.array(
                            [
                                sp.stats.mode(item,
                                              nan_policy='omit',
                                              axis=None)[0]
                                if len(sp.stats.mode(item,
                                                     nan_policy='omit',
                                                     axis=None)) > 0
                                else 'not available'
                                for item in main_sign_enum_pre
                            ])

                        # if self.can_telematics_speed_limit_enum in df.columns:
                        #     telematics_series = df[
                        #         self.can_telematics_speed_limit_enum]

                        #     can_telematics_speed_limit_enum_pre = [
                        #         telematics_series[req_indices_values[
                        #             (req_indices_values >= start)
                        #             & (req_indices_values <= end)]].mode(
                        #                 dropna=True)
                        #         # .dropna(axis=1, how='all')
                        #         for start, end in
                        #         zip(event_start_end_groups[:, 0],
                        #             event_start_end_groups[:, 1])
                        #     ]

                        #     can_telematics_speed_limit_enum = np.array(
                        #         [
                        #             item[0]
                        #             if len(item) > 0
                        #             else 'not available'
                        #             for item in can_telematics_speed_limit_enum_pre
                        #         ])

                    (event_start_end_groups_dict_enum,
                     sign_data_dict_enum,
                     sign_indices_dict_enum,
                     sign_unique_vals_idx_enum) = self._events_from_col(
                        self.vision_avi_tsr_main_sign_col_name,
                        df)

                    main_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_main_sign_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_main = [main_sign_test_array_grouped_enum.get(
                        enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    # pre_series_main_series = [item
                    #                    for item in pre_series_main
                    #                    if isinstance(item, pd.Series) ]
                    # pre_series_main_list = [item
                    #                    for item in pre_series_main
                    #                    if isinstance(item, list) ]

                    series_main = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_main
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_main
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    if len(series_main) == 0 or series_main.empty:

                        if isinstance(sign_indices_dict[enum], dict):

                            req_indices_values = list(
                                chain.from_iterable(
                                    sign_indices_dict[enum].values()
                                )
                            )[enumerated_idx][:, 0]

                        else:
                            req_indices_values = np.array(sign_indices_dict[enum],
                                                          dtype=int)

                        event_start_end_groups = orig_event_start_end_groups

                        main_sign_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]
                        main_sign_confidence_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        #     main_sign_confidence_event_start

                        main_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median_abs_deviation = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        host_long_velocity_event_start = np.array(
                            [df[self.vse_host_long_velocity_col][start]
                             for start in
                             event_start_end_groups[:, 0]
                             ])
                        # host_long_distance_event_start = np.array(
                        #     [df[self.vse_host_long_velocity_col][start]
                        #      for start in
                        #      event_start_end_groups[:, 0]
                        #      ])

                        host_long_velocity_median = np.array(
                            [df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])
                        host_long_velocity_median_abs_deviation = np.array(
                            [sp.stats.median_abs_deviation(
                                df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]],
                                axis=None,
                                nan_policy='omit')
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])

                        # long_distance_signal_iter = \
                        #     self.vision_avi_tsr_main_sign_long_dist \
                        #     + '_' + col.split('_')[-1]
                        main_sign_distance_monotonocity_abberations_count_normalised = \
                            np.array(
                                [self._performance_indicators_monotonicity(
                                    long_distance_signal_iter[req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]],
                                    df['cTime'][req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]]
                                )[1]
                                    for start, end in
                                    zip(event_start_end_groups[:, 0],
                                        event_start_end_groups[:, 1])
                                ])

                        target_long_lat_median = \
                            [(traffic_sign_longitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True),
                             traffic_sign_latitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True))
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ]

                        to_trust_target_gps = np.array(
                            [
                                to_trust_traffic_sign_lat_long_iter
                                [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                                for start, end in
                                zip(event_start_end_groups[:, 0],
                                    event_start_end_groups[:, 1])
                            ])

                        to_trust_target_gps_list.append(to_trust_target_gps)

                        event_start_cTime_list.append(event_start_cTime)
                        event_end_cTime_list.append(event_end_cTime)
                        sequence_start_cTime_list.append(sequence_start_cTime)
                        log_path_event_start_list.append(log_path_event_start)
                        sequence_name_event_start_list.append(
                            sequence_name_event_start)
                        log_name_event_start_list.append(log_name_event_start)
                        log_name_event_end_list.append(log_name_event_end)
                        base_logname_event_start_list.append(
                            base_logname_event_start)
                        rTag_event_start_list.append(rTag_event_start)
                        main_sign_enum_list.append(main_sign_enum)

                        can_telematics_speed_limit_enum_list.append(
                            can_telematics_speed_limit_enum)

                        vision_frame_ID_event_start_list.append(
                            vision_frame_ID_event_start)
                        readff_link_event_start_list.append(
                            readff_link_event_start)

                        vision_frame_ID_event_end_list.append(
                            vision_frame_ID_event_end)

                        readff_link_full_video_list.append(
                            readff_link_full_video)

                        host_gps_longitude_event_end_list.append(
                            host_gps_longitude_event_end)
                        host_gps_latitude_event_end_list.append(
                            host_gps_latitude_event_end)

                        host_gps_longitude_event_start_list.append(
                            host_gps_longitude_event_start)
                        host_gps_latitude_event_start_list.append(
                            host_gps_latitude_event_start)

                        target_gps_long_lat_coordinates_list.append(
                            target_long_lat_median)

                        main_sign_confidence_mean_list.append(
                            main_sign_confidence_mean)
                        main_sign_confidence_std_list.append(
                            main_sign_confidence_std)

                        main_sign_confidence_event_start_list.append(
                            main_sign_confidence_event_start)

                        main_sign_confidence_median_list.append(
                            main_sign_confidence_median)
                        main_sign_confidence_median_abs_deviation_list.append(
                            main_sign_confidence_median_abs_deviation)

                        supp_1_sign_enum_list.append(supp_1_sign_enum)
                        supp_2_sign_enum_list.append(supp_2_sign_enum)

                        supp_1_sign_confidence_mean_list.append(
                            supp_1_sign_confidence_mean)
                        supp_1_sign_confidence_std_list.append(
                            supp_1_sign_confidence_std)

                        supp_1_sign_confidence_event_start_list.append(
                            supp_1_sign_confidence_event_start)

                        supp_2_sign_confidence_mean_list.append(
                            supp_2_sign_confidence_mean)
                        supp_2_sign_confidence_std_list.append(
                            supp_2_sign_confidence_std)

                        supp_2_sign_confidence_event_start_list.append(
                            supp_2_sign_confidence_event_start)

                        host_long_velocity_median_list.append(
                            host_long_velocity_median)
                        host_long_velocity_median_abs_deviation_list.append(
                            host_long_velocity_median_abs_deviation)
                        host_long_velocity_event_start_list.append(
                            host_long_velocity_event_start)

                        # main_sign_ID_list.append(main_sign_ID)

                        main_sign_flickering_count_normalised_list.append(
                            main_sign_flickering_count_normalised)
                        main_sign_flickering_mean_duration_list.append(
                            main_sign_flickering_mean_duration)
                        main_sign_confidence_flickering_count_normalised_list.append(
                            main_sign_confidence_flickering_count_normalised)
                        main_sign_confidence_flickering_mean_duration_list.append(
                            main_sign_confidence_flickering_mean_duration)
                        main_sign_distance_monotonocity_abberations_count_normalised_list\
                            .append(
                                main_sign_distance_monotonocity_abberations_count_normalised)

                        ff_20ms_fused_speed_limit_sign_enum_list.append(
                            ff_20ms_fused_speed_limit_sign_enum)
                        ff_20ms_fused_speed_limit_source_enum_list.append(
                            ff_20ms_fused_speed_limit_source_enum)
                        ff_20ms_fused_speed_limit_source_all_unique_enum_list.append(
                            ff_20ms_fused_speed_limit_source_all_unique_enum)
                        ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                        ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                            ff_20ms_fused_speed_limit_sign_enum_mapped)
                        manual_label_comparison_list.append(
                            manual_label_comparison)
                        gt_confidence_list.append(gt_confidence)
                        gt_type_list.append(gt_type)
                        ff_20ms_supp_speed_limit_sign_enum_list.append(
                            ff_20ms_supp_speed_limit_sign_enum)
                        ff_20ms_overtake_sign_enum_list.append(
                            ff_20ms_overtake_sign_enum)

                        main_sign_occurence_delta_cTime_list.append(
                            main_sign_occurence_delta_cTime)
                        avi_SW_version_list.append(avi_SW_version)
                        event_signal_iter_list.append(event_signal_iter)
                        ff_10ms_construction_area_enum_list.append(
                            ff_10ms_construction_area_enum)
                        ff_10ms_construction_area_enum_unique_list.append(
                            ff_10ms_construction_area_enum_unique)

                        continue

                    req_indices_values = series_main.index

                    main_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_main_sign_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_main_confidence = [
                        main_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_main_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_main_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_main_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp1_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_1_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_1 = [
                        supp1_sign_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_1 = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_1
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_1
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp1_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_1_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_1_confidence = [
                        supp1_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_1_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_1_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_1_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp2_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_2_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_2 = [
                        supp2_sign_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_2 = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_2
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_2
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp2_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_2_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_2_confidence = [
                        supp2_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_2_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_2_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_2_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    ################################

                    # print(event_start_end_groups)

                    # event_start_end_groups[:, 0] = \
                    #     host_long_distance_event_indices
                else:

                    # print('')

                    orig_event_start_end_groups = copy.deepcopy(
                        event_start_end_groups)

                    if self._precise_look_back:

                        host_long_velocity_event_end = np.array(
                            [df[self.vse_host_long_velocity_col][end]
                             for end in
                             event_start_end_groups[:, 1]
                             ])

                        look_back_indices_array = [
                            _time_duration_to_indices_length(
                                df['cTime'].reset_index(
                                    drop=True).to_frame(),
                                self.time_array[
                                    np.abs(self.speed_array
                                           - event_end_long_vel).argmin()])
                            for event_end_long_vel in host_long_velocity_event_end
                        ]

                        event_start_end_groups = np.array([
                            [max(start-look_back_index,
                                 df.index[0]),
                             min(start+indices_to_look_forward,
                                 df.index[-1])]
                            for start, look_back_index in
                            zip(event_start_end_groups[:, 0],
                                look_back_indices_array)
                        ])

                    else:

                        event_start_end_groups = np.array([
                            [max(start-indices_to_look_backward,
                                 df.index[0]),
                             min(start+indices_to_look_forward,
                                 df.index[-1])]
                            for start in event_start_end_groups[:, 0]
                        ])

                    ff_20ms_TSR_status = np.array(
                        [df[self.feature_tsi_OW_TSR_status]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_source_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_source_enum]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_source_all_unique_enum = np.array(
                        [','.join(
                            map(str,
                                df[self.feature_tsi_OW_speed_limit_source_enum]
                                [req_indices_values[
                                    (req_indices_values >= start)
                                    & (req_indices_values <= end)]].unique()))
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_sign_enum_mapped = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         .map(self.misc_out_dict2)
                         .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_supp_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_supp_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_overtake_sign_enum = np.array(
                        [df[self.feature_tsi_OW_overtake_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_10ms_construction_area_enum = np.array(
                        [

                            (df[self.feature_tsi_IW_construction_enum]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]]
                             .value_counts(normalize=True)*100)
                            .reset_index().to_csv(header=None,
                                                  index=False,
                                                  float_format="%.2f").strip('\n')
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ])

                    ff_10ms_construction_area_enum_unique = np.array(
                        [','.join(
                            map(str,
                                df[self.feature_tsi_IW_construction_enum]
                                [req_indices_values[
                                    (req_indices_values >= start)
                                    & (req_indices_values <= end)]].unique()))
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    can_telematics_speed_limit_enum = [
                        np.nan
                        for item in ff_20ms_fused_speed_limit_sign_enum
                    ]

                    if self.can_telematics_speed_limit_enum in df.columns:
                        telematics_series = df[
                            self.can_telematics_speed_limit_enum]

                        can_telematics_speed_limit_enum_pre_01 = [

                            telematics_series[start:end].isin(
                                [
                                    # self.feature_can_telematics_mapping[avi_sign]
                                    self.feature_can_telematics_mapping.get(int(sign),
                                                                            np.nan)
                                ]
                            )
                            for sign, start, end in
                            zip(ff_20ms_fused_speed_limit_sign_enum,
                                event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ]

                        can_telematics_speed_limit_enum_pre = [
                            telematics_series[item[item == True].index].mode(
                                dropna=False)
                            for item in can_telematics_speed_limit_enum_pre_01]

                        can_telematics_speed_limit_enum = np.array(
                            [item[0] if len(item.values) > 0
                             else np.nan
                             for item in
                             can_telematics_speed_limit_enum_pre]
                        )

                    event_start_end_groups = orig_event_start_end_groups

                    #######################################
                    # GT Check
                    unique_basenames = df.loc[req_indices_values,
                                              'base_logname'].unique()

                    avi_log_basename = [
                        df['base_logname'].loc[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]].unique()
                        # .dropna(axis=1, how='all')
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    if not self.df_GT.empty:

                        df_GT_iter = [
                            self.df_GT[
                                self.df_GT['base_logname'].isin(
                                    unique_avi_log_basename)]
                            for unique_avi_log_basename in avi_log_basename
                        ]

                        feature_20ms_SL_numeric = \
                            [int(re.findall(r'\d+', str(map_sign))[0])
                             if len(re.findall(r'\d+', str(map_sign))) > 0
                             else np.nan
                             for map_sign in
                                ff_20ms_fused_speed_limit_sign_enum_mapped]

                        manual_label_comparison = []
                        gt_confidence = []
                        gt_type = []

                        for idx_gt, (sl, df_GT_iter_iter) in enumerate(zip(
                                feature_20ms_SL_numeric,
                                df_GT_iter
                        )):
                            # print(f'Just after loop {idx_gt}, sl: {sl}')

                            if df_GT_iter_iter.empty:

                                manual_label_comparison.append('No GT')
                                gt_confidence.append(np.nan)
                                gt_type.append('No GT')
                                continue

                            manual_label_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['manual_label_main_sign'
                                                    ].astype(
                                        str)
                                 ]

                            signDB_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['signDB_main_sign'
                                                    ].astype(str)
                                 ]

                            yolo_v8_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['yolo_v8_main_sign'
                                                    ].astype(str)
                                 ]

                            ml_3_main_sign_vals = \
                                [int(re.findall(r'\d+', map_sign)[0])
                                 if len(re.findall(r'\d+', map_sign)) > 0
                                 else np.nan
                                 for map_sign in
                                    df_GT_iter_iter['ml3_main_sign'].astype(
                                        str)
                                 ]

                            if (
                                sl in manual_label_main_sign_vals
                                or sl in yolo_v8_main_sign_vals
                                or sl in signDB_main_sign_vals
                                or sl in ml_3_main_sign_vals
                            ) and not np.isnan(sl):
                                # print(f'inside conditional true {idx_gt}')

                                if sl in manual_label_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(1)
                                    gt_type.append('Manual Label')
                                elif sl in signDB_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.9)
                                    gt_type.append('Sign Database')
                                elif sl in yolo_v8_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.6)
                                    gt_type.append('ML Yolo V8')
                                elif sl in ml_3_main_sign_vals:

                                    manual_label_comparison.append('TP')
                                    gt_confidence.append(0.6)
                                    gt_type.append('ML assorted')

                            else:

                                # print(f'inside conditional false {idx_gt}, sl: {sl}')

                                if not sl in manual_label_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(1)
                                    gt_type.append('Manual Label')
                                elif not sl in signDB_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.9)
                                    gt_type.append('Sign Database')
                                elif not sl in yolo_v8_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.5)
                                    gt_type.append('ML Yolo V8')
                                elif not sl in ml_3_main_sign_vals:

                                    manual_label_comparison.append('FP')
                                    gt_confidence.append(0.5)
                                    gt_type.append('ML assorted')
                                else:

                                    manual_label_comparison.append('No GT')
                                    gt_confidence.append(1)
                                    gt_type.append('No GT')

                    else:

                        manual_label_comparison = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                        gt_confidence = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                        gt_type = [
                            np.nan for item in
                            ff_20ms_fused_speed_limit_sign_enum_mapped
                        ]

                    ###############################

                    main_sign_enum_pre = [
                        series_main[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]]
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    main_sign_enum_01 = [
                        item.isin(

                            # self.feature_avi_mapping[avi_sign]
                            self.feature_avi_mapping.get(int(sign),
                                                         [np.nan])
                        )
                        for sign, item in
                        zip(ff_20ms_fused_speed_limit_sign_enum,
                            main_sign_enum_pre)
                    ]

                    main_sign_enum_02 = [item2[item1 == True]
                                         .dropna(how='all', axis=0)
                                         for item1, item2 in zip(main_sign_enum_01,
                                                                 main_sign_enum_pre)
                                         ]

                    main_sign_occurence_index = []

                    for item, start in zip(main_sign_enum_02,
                                           orig_event_start_end_groups[:, 0]):

                        if not item.empty:

                            if item.index[-1] <= start:
                                main_sign_occurence_index.append(
                                    item.index[-1])
                            elif item.index[0] <= start <= item.index[-1]:
                                # if start - item.index[0] > item.index[-1] - start:
                                #     main_sign_occurence_index.append(
                                #         item.index[-1])
                                # else:
                                #     main_sign_occurence_index.append(
                                #         item.index[0])
                                main_sign_occurence_index.append(
                                    item.index[0])
                            elif item.index[0] >= start:
                                main_sign_occurence_index.append(
                                    item.index[0])
                            else:
                                main_sign_occurence_index.append(start+1)
                        else:
                            main_sign_occurence_index.append(np.nan)

                    main_sign_occurence_index = np.array(
                        main_sign_occurence_index)

                    # main_sign_occurence_index = np.array(
                    #     [item.index[-1] if not item.empty
                    #      else np.nan
                    #      for item in main_sign_enum_02
                    #      ]
                    # )

                    # main_sign_enum = []

                    # for idx, item in zip(main_sign_occurence_index,
                    #                      main_sign_enum_02):

                    #     if len(item) > 0:

                    #         if isinstance(item, pd.DataFrame):

                    #             values_needed = item.loc[int(idx),
                    #                                      :].values.flatten()
                    #         elif isinstance(item, pd.Series):

                    #             values_needed = item[int(
                    #                 idx)].values.flatten()

                    #         values_needed = values_needed[
                    #             ~np.isnan(values_needed)]
                    #         main_sign_enum.append(int(values_needed[-1]))

                    #     else:
                    #         main_sign_enum.append('not available')

                    main_sign_enum = np.array(
                        [
                            str(item.mode(dropna=True)[0])
                            if len(item.mode(dropna=True)) > 0
                            else 'not available'
                            for item in main_sign_enum_pre
                        ])

                    # if self.can_telematics_speed_limit_enum in df.columns:
                    #     telematics_series = df[
                    #         self.can_telematics_speed_limit_enum]

                    #     can_telematics_speed_limit_enum_pre = [
                    #         telematics_series[req_indices_values[
                    #             (req_indices_values >= start)
                    #             & (req_indices_values <= end)]].mode(
                    #                 dropna=True)
                    #         # .dropna(axis=1, how='all')
                    #         for start, end in
                    #         zip(event_start_end_groups[:, 0],
                    #             event_start_end_groups[:, 1])
                    #     ]

                    #     can_telematics_speed_limit_enum = np.array(
                    #         [
                    #             item[0]
                    #             if len(item) > 0
                    #             else 'not available'
                    #             for item in can_telematics_speed_limit_enum_pre
                    #         ])

                    main_sign_occurence_delta_cTime = [
                        start_cTime - df.loc[index, 'cTime']
                        if not np.isnan(index)
                        else np.nan
                        for index, start_cTime in
                        zip(main_sign_occurence_index,
                            event_start_cTime)
                    ]

                main_sign_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        df['cTime'][req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        no_nan_indices=True,
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_flickering_count_normalised = [
                    item[0] for item in main_sign_flickering
                ]
                main_sign_flickering_mean_duration = [
                    item[1] for item in main_sign_flickering
                ]

                main_sign_confidence_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main_confidence[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        df['cTime'][req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        no_nan_indices=True,
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_confidence_flickering_count_normalised = [
                    item[0] for item in main_sign_confidence_flickering
                ]
                main_sign_confidence_flickering_mean_duration = [
                    item[1] for item in main_sign_confidence_flickering
                ]

                main_sign_confidence_mean = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_std = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        series_main_confidence[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                main_sign_confidence_event_start = np.array(
                    [series_main_confidence[req_indices_values[
                        req_indices_values == start
                    ]].iloc[0]
                        if start in req_indices_values
                        else np.nan
                        for start in
                        event_start_end_groups[:, 0]
                    ]).flatten()

                supp_1_sign_enum_pre = [
                    series_supp_1[req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]].mode(
                        dropna=False
                    )
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                ]

                supp_1_sign_enum = np.array([item[0]
                                             if len(item) > 0 else np.nan
                                             for item in supp_1_sign_enum_pre])

                supp_1_sign_confidence_mean = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_1_sign_confidence_std = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_1_sign_confidence_event_start = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values == start)
                     ]].iloc[0]
                     if start in req_indices_values
                     else np.nan
                     for start in
                     event_start_end_groups[:, 0]
                     ]).flatten()

                supp_2_sign_enum_pre = [
                    series_supp_2
                    [req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]].mode(
                        dropna=False
                    )
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                supp_2_sign_enum = np.array([item[0]
                                             if len(item) > 0 else np.nan
                                             for item in supp_2_sign_enum_pre])

                supp_2_sign_confidence_mean = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_2_sign_confidence_std = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_2_sign_confidence_event_start = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values == start)
                     ]].iloc[0]
                     if start in req_indices_values
                     else np.nan
                     for start in
                     event_start_end_groups[:, 0]
                     ]).flatten()

                if self.is_feature_based:

                    if isinstance(sign_indices_dict[enum], dict):

                        req_indices_values = list(
                            chain.from_iterable(
                                sign_indices_dict[enum].values()
                            )
                        )[enumerated_idx][:, 0]

                    else:
                        req_indices_values = np.array(sign_indices_dict[enum],
                                                      dtype=int)

                    event_start_end_groups = orig_event_start_end_groups

                host_long_velocity_event_start = np.array(
                    [df[self.vse_host_long_velocity_col][start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])

                host_long_velocity_median = np.array(
                    [df[self.vse_host_long_velocity_col]
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                host_long_velocity_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        df[self.vse_host_long_velocity_col]
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                main_sign_distance_monotonocity_abberations_count_normalised = \
                    np.array(
                        [self._performance_indicators_monotonicity(
                            long_distance_signal_iter[req_indices_values[
                                (req_indices_values >= start)
                                & (req_indices_values <= end)]],
                            df['cTime'][req_indices_values[
                                (req_indices_values >= start)
                                & (req_indices_values <= end)]]
                        )[1]
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ])

                target_long_lat_median = \
                    [(traffic_sign_longitude_iter
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True),
                     traffic_sign_latitude_iter
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True))
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ]

                to_trust_target_gps = np.array(
                    [
                        to_trust_traffic_sign_lat_long_iter
                        [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                to_trust_target_gps_list.append(to_trust_target_gps)

                event_start_cTime_list.append(event_start_cTime)
                event_end_cTime_list.append(event_end_cTime)
                sequence_start_cTime_list.append(sequence_start_cTime)
                log_path_event_start_list.append(log_path_event_start)
                sequence_name_event_start_list.append(
                    sequence_name_event_start)
                log_name_event_start_list.append(log_name_event_start)
                log_name_event_end_list.append(log_name_event_end)
                base_logname_event_start_list.append(base_logname_event_start)
                rTag_event_start_list.append(rTag_event_start)
                main_sign_enum_list.append(main_sign_enum)

                can_telematics_speed_limit_enum_list.append(
                    can_telematics_speed_limit_enum)

                vision_frame_ID_event_start_list.append(
                    vision_frame_ID_event_start)
                readff_link_event_start_list.append(
                    readff_link_event_start)

                vision_frame_ID_event_end_list.append(
                    vision_frame_ID_event_end)

                readff_link_full_video_list.append(readff_link_full_video)

                host_gps_longitude_event_end_list.append(
                    host_gps_longitude_event_end)
                host_gps_latitude_event_end_list.append(
                    host_gps_latitude_event_end)

                host_gps_longitude_event_start_list.append(
                    host_gps_longitude_event_start)
                host_gps_latitude_event_start_list.append(
                    host_gps_latitude_event_start)

                target_gps_long_lat_coordinates_list.append(
                    target_long_lat_median)

                main_sign_confidence_mean_list.append(
                    main_sign_confidence_mean)
                main_sign_confidence_std_list.append(
                    main_sign_confidence_std)

                main_sign_confidence_event_start_list.append(
                    main_sign_confidence_event_start)

                main_sign_confidence_median_list.append(
                    main_sign_confidence_median)
                main_sign_confidence_median_abs_deviation_list.append(
                    main_sign_confidence_median_abs_deviation)

                supp_1_sign_enum_list.append(supp_1_sign_enum)
                supp_2_sign_enum_list.append(supp_2_sign_enum)

                supp_1_sign_confidence_mean_list.append(
                    supp_1_sign_confidence_mean)
                supp_1_sign_confidence_std_list.append(
                    supp_1_sign_confidence_std)

                supp_1_sign_confidence_event_start_list.append(
                    supp_1_sign_confidence_event_start)

                supp_2_sign_confidence_mean_list.append(
                    supp_2_sign_confidence_mean)
                supp_2_sign_confidence_std_list.append(
                    supp_2_sign_confidence_std)

                supp_2_sign_confidence_event_start_list.append(
                    supp_2_sign_confidence_event_start)

                host_long_velocity_median_list.append(
                    host_long_velocity_median)
                host_long_velocity_median_abs_deviation_list.append(
                    host_long_velocity_median_abs_deviation)
                host_long_velocity_event_start_list.append(
                    host_long_velocity_event_start)

                # main_sign_ID_list.append(main_sign_ID)

                main_sign_flickering_count_normalised_list.append(
                    main_sign_flickering_count_normalised)
                main_sign_flickering_mean_duration_list.append(
                    main_sign_flickering_mean_duration)
                main_sign_confidence_flickering_count_normalised_list.append(
                    main_sign_confidence_flickering_count_normalised)
                main_sign_confidence_flickering_mean_duration_list.append(
                    main_sign_confidence_flickering_mean_duration)
                main_sign_distance_monotonocity_abberations_count_normalised_list\
                    .append(
                        main_sign_distance_monotonocity_abberations_count_normalised)

                ff_20ms_fused_speed_limit_sign_enum_list.append(
                    ff_20ms_fused_speed_limit_sign_enum)
                ff_20ms_fused_speed_limit_source_enum_list.append(
                    ff_20ms_fused_speed_limit_source_enum)
                ff_20ms_fused_speed_limit_source_all_unique_enum_list.append(
                    ff_20ms_fused_speed_limit_source_all_unique_enum)
                ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                    ff_20ms_fused_speed_limit_sign_enum_mapped)
                manual_label_comparison_list.append(
                    manual_label_comparison)
                gt_confidence_list.append(gt_confidence)
                gt_type_list.append(gt_type)
                ff_20ms_supp_speed_limit_sign_enum_list.append(
                    ff_20ms_supp_speed_limit_sign_enum)
                ff_20ms_overtake_sign_enum_list.append(
                    ff_20ms_overtake_sign_enum)

                main_sign_occurence_delta_cTime_list.append(
                    main_sign_occurence_delta_cTime)

                avi_SW_version_list.append(avi_SW_version)
                event_signal_iter_list.append(event_signal_iter)
                ff_10ms_construction_area_enum_list.append(
                    ff_10ms_construction_area_enum)
                ff_10ms_construction_area_enum_unique_list.append(
                    ff_10ms_construction_area_enum_unique)

        main_sign_enum_list_stacked = (np.hstack(main_sign_enum_list)
                                       if bool(main_sign_enum_list)
                                       else np.array([]))

        can_telematics_speed_limit_enum_list_stacked = (
            np.hstack(can_telematics_speed_limit_enum_list)
            if bool(can_telematics_speed_limit_enum_list)
            else np.array([]))

        ff_20ms_fused_speed_limit_sign_enum_list_stacked = (np.hstack(
            ff_20ms_fused_speed_limit_sign_enum_list)
            if bool(ff_20ms_fused_speed_limit_sign_enum_list)
            else np.array([]))

        is_main_sign_feature_match = [
            'Matched'
            if (
                ((isinstance(item, str) and item.isdigit())
                 or (isinstance(item, (int, np.integer))
                     or isinstance(item, (float, np.floating))
                     and not np.isnan(item))
                 ) and

                int(item) in
                self.avi_feature_mapping.keys() and
                self.avi_feature_mapping.get(int(item), None) == int(f_item))
            else 'Not Matched'
            for f_item, item in
            zip(
                ff_20ms_fused_speed_limit_sign_enum_list_stacked,
                main_sign_enum_list_stacked)
        ]

        if self.is_feature_based and self.is_non_speed_feature_based:

            ff_20ms_overtake_sign_enum_list_stacked = (np.hstack(
                ff_20ms_overtake_sign_enum_list)
                if bool(ff_20ms_overtake_sign_enum_list)
                else np.array([]))

            is_main_sign_feature_match = [
                'Matched'
                if (
                    ((isinstance(item, str) and item.isdigit())
                     or (isinstance(item, (int, np.integer))
                         or isinstance(item, (float, np.floating))
                         and not np.isnan(item))
                     ) and

                    int(item) in avi_mapping_df['enum'].values
                    and not np.isnan(f_item)
                    and int(f_item) != self._default_val_overtake_ff_20ms)
                else 'Not Matched'
                for f_item, item in
                zip(
                    ff_20ms_overtake_sign_enum_list_stacked,
                    main_sign_enum_list_stacked)
            ]

        is_can_telematics_feature_match = [
            'Matched'
            if (
                ((isinstance(item, str) and item.isdigit())
                 or ((isinstance(item, (int, np.integer))
                     or isinstance(item, (float, np.floating)))
                     and not np.isnan(item))
                 ) and

                int(item) in
                self.feature_can_telematics_mapping.keys() and
                self.feature_can_telematics_mapping.get(int(f_item),
                                                        None) == int(item))
            else 'Not Matched'
            for f_item, item in
            zip(
                ff_20ms_fused_speed_limit_sign_enum_list_stacked,
                can_telematics_speed_limit_enum_list_stacked)
        ]

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
            'start_cTime_sequence':
            np.hstack(sequence_start_cTime_list)
            if bool(sequence_start_cTime_list)
            else np.array([]),

            'feature_20ms_OW_fused_speed_limit_sign_enum_mapped':
            np.hstack(ff_20ms_fused_speed_limit_sign_enum_mapped_list)
            if bool(ff_20ms_fused_speed_limit_sign_enum_mapped_list)
            else np.array([]),

            'feature_20ms_OW_fused_speed_limit_sign_enum':
            ff_20ms_fused_speed_limit_sign_enum_list_stacked,

            'feature_20ms_OW_fused_speed_limit_source_enum':
            np.hstack(ff_20ms_fused_speed_limit_source_enum_list)
            if bool(ff_20ms_fused_speed_limit_source_enum_list)
            else np.array([]),

            'feature_20ms_OW_fused_speed_limit_source_all_unique_enum':
            np.hstack(ff_20ms_fused_speed_limit_source_all_unique_enum_list)
            if bool(ff_20ms_fused_speed_limit_source_all_unique_enum_list)
            else np.array([]),



            'feature_20ms_OW_TSR_status':
            np.hstack(ff_20ms_TSR_status_list)
            if bool(ff_20ms_TSR_status_list)
            else np.array([]),

            'feature_20ms_OW_supp_speed_limit_sign_enum':
            np.hstack(ff_20ms_supp_speed_limit_sign_enum_list)
            if bool(ff_20ms_supp_speed_limit_sign_enum_list)
            else np.array([]),

            'feature_20ms_OW_overtake_sign_enum':
            np.hstack(ff_20ms_overtake_sign_enum_list)
            if bool(ff_20ms_overtake_sign_enum_list)
            else np.array([]),

            'feature_10ms_OW_construction_area_enum':
            np.hstack(ff_10ms_construction_area_enum_list)
            if bool(ff_10ms_construction_area_enum_list)
            else np.array([]),

            'feature_10ms_OW_construction_area_enum_unique':
            np.hstack(ff_10ms_construction_area_enum_unique_list)
            if bool(ff_10ms_construction_area_enum_unique_list)
            else np.array([]),



            'target_gps_long_lat_coordinates':
            list(chain.from_iterable(target_gps_long_lat_coordinates_list))
            if bool(target_gps_long_lat_coordinates_list)
            else np.array([]),

            'to_trust_target_gps':
            np.hstack(to_trust_target_gps_list)
            if bool(to_trust_target_gps_list)
            else np.array([]),

            'readff_link_event_start':
                np.hstack(readff_link_event_start_list)
                if bool(readff_link_event_start_list)
                else np.array([]),
            'readff_link_full_video':
                np.hstack(readff_link_full_video_list)
                if bool(readff_link_full_video_list)
                else np.array([]),

            'avi_SW_version':
            np.hstack(avi_SW_version_list)
            if bool(avi_SW_version_list)
            else np.array([]),

            'vision_frame_ID_event_start':
                np.hstack(vision_frame_ID_event_start_list)
                if bool(vision_frame_ID_event_start_list)
                else np.array([]),
            'vision_frame_ID_event_end':
                np.hstack(vision_frame_ID_event_end_list)
                if bool(vision_frame_ID_event_end_list)
                else np.array([]),

            'host_gps_latitude_event_end':
            np.hstack(host_gps_latitude_event_end_list)
            if bool(host_gps_latitude_event_end_list)
            else np.array([]),

            'host_gps_longitude_event_end':
            np.hstack(host_gps_longitude_event_end_list)
            if bool(host_gps_longitude_event_end_list)
            else np.array([]),

            'host_gps_latitude_event_start':
            np.hstack(host_gps_latitude_event_start_list)
            if bool(host_gps_latitude_event_start_list)
            else np.array([]),

            'host_gps_longitude_event_start':
            np.hstack(host_gps_longitude_event_start_list)
            if bool(host_gps_longitude_event_start_list)
            else np.array([]),

            # 'main_sign_ID':
            # np.hstack(main_sign_ID_list)
            # if bool(main_sign_ID_list)
            # else np.array([]),
            'main_sign_enum':
            main_sign_enum_list_stacked,
            'is_main_sign_feature_match': is_main_sign_feature_match,
            'can_telematics_sign_enum':
            can_telematics_speed_limit_enum_list_stacked,
            'is_can_telematics_feature_match': is_can_telematics_feature_match,

            # manual_label_comparison_list
            # gt_confidence_list gt_type_list.append(gt_type)
            'ground_truth':
            np.hstack(manual_label_comparison_list)
            if bool(manual_label_comparison_list)
            else np.array([]),

            'gt_confidence':
            np.hstack(gt_confidence_list)
            if bool(gt_confidence_list)
            else np.array([]),

            'gt_type':
            np.hstack(gt_type_list)
            if bool(gt_type_list)
            else np.array([]),

            'feature_avi_match_delta_cTime':
            np.hstack(main_sign_occurence_delta_cTime_list)
            if bool(main_sign_occurence_delta_cTime_list)
            else np.array([]),
            'main_sign_confidence_mean':
            np.hstack(main_sign_confidence_mean_list)
            if bool(main_sign_confidence_mean_list)
            else np.array([]),
            'main_sign_confidence_std':
            np.hstack(main_sign_confidence_std_list)
            if bool(main_sign_confidence_std_list)
            else np.array([]),
            'main_sign_confidence_event_start':
            np.hstack(main_sign_confidence_event_start_list)
            if bool(main_sign_confidence_event_start_list)
            else np.array([]),
            'main_sign_confidence_median':
            np.hstack(main_sign_confidence_median_list)
            if bool(main_sign_confidence_median_list)
            else np.array([]),
            'main_sign_confidence_median_abs_deviation':
            np.hstack(
                main_sign_confidence_median_abs_deviation_list)
            if bool(
                main_sign_confidence_median_abs_deviation_list)
            else np.array([]),
            'supp_1_sign_enum':
            np.hstack(supp_1_sign_enum_list)
            if bool(supp_1_sign_enum_list)
            else np.array([]),

            'supp_1_sign_confidence_mean':
            np.hstack(supp_1_sign_confidence_mean_list)
            if bool(supp_1_sign_confidence_mean_list)
            else np.array([]),
            'supp_1_sign_confidence_std':
            np.hstack(supp_1_sign_confidence_std_list)
            if bool(supp_1_sign_confidence_std_list)
            else np.array([]),
            'supp_1_sign_confidence_event_start':
            np.hstack(supp_1_sign_confidence_event_start_list)
            if bool(supp_1_sign_confidence_event_start_list)
            else np.array([]),
            'supp_2_sign_enum':
            np.hstack(supp_2_sign_enum_list)
            if bool(supp_2_sign_enum_list)
            else np.array([]),
            'supp_2_sign_confidence_mean':
            np.hstack(supp_2_sign_confidence_mean_list)
            if bool(supp_2_sign_confidence_mean_list)
            else np.array([]),
            'supp_2_sign_confidence_std':
            np.hstack(supp_2_sign_confidence_std_list)
            if bool(supp_2_sign_confidence_std_list)
            else np.array([]),
            'supp_2_sign_confidence_event_start':
            np.hstack(supp_2_sign_confidence_event_start_list)
            if bool(supp_2_sign_confidence_event_start_list)
            else np.array([]),

            'host_long_velocity_median_mps':
            np.hstack(host_long_velocity_median_list)
            if bool(host_long_velocity_median_list)
            else np.array([]),
            'host_long_velocity_median_abs_deviation_mps':
            np.hstack(
                host_long_velocity_median_abs_deviation_list)
            if bool(host_long_velocity_median_abs_deviation_list)
            else np.array([]),
            'host_long_velocity_event_start_mps':
            np.hstack(host_long_velocity_event_start_list)
            if bool(host_long_velocity_event_start_list)
            else np.array([]),
            'main_sign_flickering_count_per_sec':
            np.hstack(main_sign_flickering_count_normalised_list)
            if bool(main_sign_flickering_count_normalised_list)
            else np.array([]),
            'main_sign_flickering_mean_duration_sec':
            np.hstack(main_sign_flickering_mean_duration_list)
            if bool(main_sign_flickering_mean_duration_list)
            else np.array([]),
            'main_sign_confidence_flickering_count_per_sec':
            np.hstack(
                main_sign_confidence_flickering_count_normalised_list)
            if bool(
                main_sign_confidence_flickering_count_normalised_list)
            else np.array([]),
            'main_sign_confidence_flickering_mean_duration_sec':
            np.hstack(
                main_sign_confidence_flickering_mean_duration_list)
            if bool(main_sign_confidence_flickering_mean_duration_list)
            else np.array([]),
            'main_sign_distance_monotonicity_abberations_count_per_sec':
            np.hstack(
                main_sign_distance_monotonocity_abberations_count_normalised_list)
            if bool(main_sign_distance_monotonocity_abberations_count_normalised_list)
            else np.array([]),

            'event_signal':
            np.hstack(
                event_signal_iter_list)
            if bool(event_signal_iter_list)
            else np.array([]),

            'log_name_event_start':
            np.hstack(log_name_event_start_list)
            if bool(log_name_event_start_list)
            else np.array([]),
            'log_name_event_end':
            np.hstack(log_name_event_end_list)
            if bool(log_name_event_end_list)
            else np.array([]),
        }

        event_dict['supp_1_sign_enum_mapped'] = [
            # self.zone_enum_dict.get(item, np.nan)
            self.supp_signs_dict.get(item, np.nan)
            for item in event_dict['supp_1_sign_enum']
        ]
        event_dict['supp_2_sign_enum_mapped'] = [
            # self.zone_enum_dict.get(item, np.nan)
            self.supp_signs_dict.get(item, np.nan)
            for item in event_dict['supp_2_sign_enum']
        ]

        if self.is_feature_based and not self.is_non_speed_feature_based:

            req_keys = ['feature_20ms_OW_overtake_sign_enum',]

            event_dict.update({key: [np.nan]*len(val)
                               for key, val in event_dict.items()
                               if key in req_keys})

            if self.is_speed_source_considered:
                event_dict['event_type'] = [
                    # self.zone_enum_dict.get(item, np.nan)
                    'Feature Speed Limit, source considered'
                    for item in event_dict['log_name']
                ]
            else:
                event_dict['event_type'] = [
                    # self.zone_enum_dict.get(item, np.nan)
                    'Feature Speed Limit'
                    for item in event_dict['log_name']
                ]
        elif self.is_feature_based and self.is_non_speed_feature_based:

            req_keys = [
                # 'log_path',
                # 'sequence_name',
                # 'log_name',
                # 'base_logname',
                # 'rTag',
                # 'event_start_cTime',
                # 'event_end_cTime',
                # 'start_cTime_sequence',
                'feature_20ms_OW_fused_speed_limit_sign_enum_mapped',
                'feature_20ms_OW_fused_speed_limit_sign_enum',
                'feature_20ms_OW_fused_speed_limit_source_enum',
                'feature_20ms_OW_fused_speed_limit_source_all_unique_enum',
                # 'feature_20ms_OW_TSR_status',
                'feature_20ms_OW_supp_speed_limit_sign_enum',
                # 'feature_20ms_OW_overtake_sign_enum',
                'feature_10ms_OW_construction_area_enum',
                'feature_10ms_OW_construction_area_enum_unique',
                # 'target_gps_long_lat_coordinates',
                # 'to_trust_target_gps',
                # 'readff_link_event_start',
                # 'readff_link_full_video',
                # 'avi_SW_version',
                # 'vision_frame_ID_event_start',
                # 'vision_frame_ID_event_end',
                # 'host_gps_latitude_event_end',
                # 'host_gps_longitude_event_end',
                # 'host_gps_latitude_event_start',
                # 'host_gps_longitude_event_start',
                # 'main_sign_enum',
                # 'is_main_sign_feature_match',
                'can_telematics_sign_enum',
                'is_can_telematics_feature_match',
                # 'feature_avi_match_delta_cTime',
                # 'main_sign_confidence_mean',
                # 'main_sign_confidence_std',
                # 'main_sign_confidence_event_start',
                # 'main_sign_confidence_median',
                # 'main_sign_confidence_median_abs_deviation',
                'supp_1_sign_enum',
                'supp_1_sign_enum_mapped',
                'supp_1_sign_confidence_mean',
                'supp_1_sign_confidence_std',

                'supp_2_sign_enum',
                'supp_2_sign_enum_mapped',
                'supp_2_sign_confidence_mean',
                'supp_2_sign_confidence_std',

                # 'host_long_velocity_median_mps',
                # 'host_long_velocity_median_abs_deviation_mps',
                # 'host_long_velocity_event_start_mps',
                # 'main_sign_flickering_count_per_sec',
                # 'main_sign_flickering_mean_duration_sec',
                # 'main_sign_confidence_flickering_count_per_sec',
                # 'main_sign_confidence_flickering_mean_duration_sec',
                # 'main_sign_distance_monotonicity_abberations_count_per_sec',
                # 'event_signal',

            ]

            event_dict.update({key: [np.nan]*len(val)
                               for key, val in event_dict.items()
                               if key in req_keys})

            event_dict['event_type'] = [
                # self.zone_enum_dict.get(item, np.nan)
                'Feature Non-Speed Limit'
                for item in event_dict['log_name']
            ]
        elif not self.is_feature_based and not is_sign_ID:

            event_dict['event_type'] = [
                # self.zone_enum_dict.get(item, np.nan)
                'AVI Speed Limit'
                for item in event_dict['log_name']
            ]
        else:

            event_dict['event_type'] = [
                # self.zone_enum_dict.get(item, np.nan)
                'AVI Sign ID'
                for item in event_dict['log_name']
            ]

        # if not is_sign_ID

        # event_dict = {key + '_sign_ID_based'
        #               if is_sign_ID
        #               else key: val
        #               for key, val in event_dict.items()
        #               }

        return event_dict

    def _sign_extraction2(self,
                          df,
                          main_sign_col_name,
                          supp_sign_1_col_name,
                          supp_sign_2_col_name,
                          # performance_indicator_cols,
                          **kwargs):

        # Assumption : default_val for main and supp signs is same

        if 'default_val' in kwargs:
            default_val = kwargs['default_val']
        else:
            default_val = 0

        # req_cols = sort_list([col for col in df.columns
        #                       if main_sign_col_name in col
        #                       ])

        # req_cols_confidence = [col for col in df.columns
        #                        if self.vision_avi_tsr_main_sign_col_confidence
        #                        in col
        #                        or
        #                        self.vision_avi_tsr_supp_sign_1_col_confidence
        #                        in col
        #                        or
        #                        self.vision_avi_tsr_supp_sign_2_col_confidence
        #                        in col]

        # req_cols_supp_1 = sort_list([col for col in df.columns
        #                              if supp_sign_1_col_name in col
        #                              ])

        # req_cols_supp_2 = sort_list([col for col in df.columns
        #                              if supp_sign_2_col_name in col
        #                              ])

        # req_cols_all = (req_cols
        #                 # + req_cols_confidence
        #                 + req_cols_supp_1
        #                 + req_cols_supp_2)

        # df_req = df.copy(deep=True)[
        #     ['cTime']
        #     + req_cols
        #     # + req_cols_confidence
        #     + req_cols_supp_1
        #     + req_cols_supp_2
        # ]

        # df_confidence = df.copy(deep=True)[
        #     ['cTime']
        #     + req_cols_confidence
        #     # + req_cols_supp_1
        #     # + req_cols_supp_2
        # ]

        # df = None

        # df_req_test = df_req[df_req != default_val[0]].dropna(
        #     axis=1, how='all')

        # df = df[
        #     df != default_val[0]].dropna(
        #     axis=1, how='all')

        # ASSUMPTION : there shall not be flickering duration beyond
        # what is threshold
        self.req_indices_len = _time_duration_to_indices_length(
            df['cTime'].to_frame(),
            self.flickering_threshold_sec, )

        # bool_df_req = df_req != default_val

        print('************** Event Extraction Start *******************')

        # sign_ID_cols = [col
        #                 for col in df.columns
        #                 if self.vision_avi_tsr_main_sign_ID_col_name in col]

        # sign_ID_test_array = df[sign_ID_cols].values

        # unique_vals_1, unique_vals_idx_1 = ndix_unique(sign_ID_test_array)

        # sign_ID_indices = {ID: df.index[df[sign_ID_cols]
        #                                 [df[sign_ID_cols] == ID]
        #                                 .any(axis=1)].tolist()
        #                    for ID in unique_vals_1}

        # sign_ID_indices_processed = {}
        # for ID, series in sign_ID_indices.items():
        #     sign_ID_indices_processed[ID] = series.groupby(
        #         series.diff().ne(1)
        #         .cumsum()).apply(
        #         lambda x:
        #         [x.iloc[0], x.iloc[-1]]
        #         if len(x) >= 2
        #         else [x.iloc[0]]).tolist()

        event_dict_enum_based = {}

        (event_start_end_groups_dict,
         sign_data_dict,
         sign_indices_dict,
         sign_unique_vals_idx) = self._events_from_col(
            main_sign_col_name,
            df)

        event_dict_enum_based = self._sign_extraction_helper(
            df,
            sign_indices_dict,
            sign_unique_vals_idx,
            event_start_end_groups_dict,
            is_sign_ID=False,
        )

        event_dict_ID_based = {}

        if self.is_sign_ID_req:

            (event_start_end_groups_ID_dict,
             sign_ID_data_dict,
             sign_ID_indices_dict,
             sign_ID_unique_vals_idx) = self._events_from_col(
                self.vision_avi_tsr_main_sign_ID_col_name,
                df)

            event_dict_ID_based = self._sign_extraction_helper(
                df,
                sign_ID_indices_dict,
                sign_ID_unique_vals_idx,
                event_start_end_groups_ID_dict,
                is_sign_ID=True,
            )
        f_req_indices_len = self.req_indices_len

        self.req_indices_len = 1

        (event_start_end_groups_dict_feature_stakeholder,
         sign_data_dict_feature_stakeholder,
         sign_indices_dict_feature_stakeholder,
         sign_unique_vals_idx_feature_stakeholder) = self._events_from_col(
            self.feature_tsi_OW_speed_limit_sign_enum,
            df,
            is_feature=True)

        self.is_feature_based = True
        event_dict_feature_based_present = self._sign_extraction_helper(
            df,
            sign_indices_dict_feature_stakeholder,
            sign_unique_vals_idx_feature_stakeholder,
            event_start_end_groups_dict_feature_stakeholder,
            is_sign_ID=False,
        )

        self.feature_tsi_OW_speed_limit_sign_enum_edited = \
            'feature_tsi_OW_vehicle_speed_limit_sign_enum_edited'
        req_edited_vals_avi_sign_enum = \
            np.array(df[self.feature_tsi_OW_speed_limit_sign_enum])
        rolled_indices = \
            np.where(
                np.roll(
                    df[self.feature_tsi_OW_speed_limit_source_enum], 1)
                != df[self.feature_tsi_OW_speed_limit_source_enum])[0][1:]
        req_edited_vals_avi_sign_enum[rolled_indices] = \
            self.default_vals_ff_20ms

        df[self.feature_tsi_OW_speed_limit_sign_enum_edited] = \
            req_edited_vals_avi_sign_enum

        (event_start_end_groups_dict_feature,
         sign_data_dict_feature,
         sign_indices_dict_feature,
         sign_unique_vals_idx_feature) = self._events_from_col(
            self.feature_tsi_OW_speed_limit_sign_enum_edited,
            df,
            is_feature=True)

        self.is_feature_based = True
        self.is_speed_source_considered = True
        event_dict_feature_based = self._sign_extraction_helper(
            df,
            sign_indices_dict_feature,
            sign_unique_vals_idx_feature,
            event_start_end_groups_dict_feature,
            is_sign_ID=False,
        )

        self.req_indices_len = f_req_indices_len

        (event_start_end_groups_dict_feature_non_speed,
         sign_data_dict_feature_non_speed,
         sign_indices_dict_feature_non_speed,
         sign_unique_vals_idx_feature_non_speed) = self._events_from_col(
            self.feature_tsi_OW_overtake_sign_enum,
            df,
            is_feature=True)

        self.is_feature_based = True
        self.is_non_speed_feature_based = True
        event_dict_feature_non_speed_based = self._sign_extraction_helper(
            df,
            sign_indices_dict_feature_non_speed,
            sign_unique_vals_idx_feature_non_speed,
            event_start_end_groups_dict_feature_non_speed,
            is_sign_ID=False,
        )

        # event_dict = {**event_dict_enum_based,
        #               **event_dict_ID_based}

        event_dict = {'enum_based': event_dict_enum_based,
                      'ID_based': event_dict_ID_based,
                      'feature_based': event_dict_feature_based,
                      'feature_based_present': event_dict_feature_based_present,
                      'feature_non_speed_based': event_dict_feature_non_speed_based}

        print('************** Event Extraction Done *******************')

        event_dict = self._performance_indicators(df, event_dict, **kwargs)

        return event_dict

    def _events_from_col(self, column_name, df, is_feature: bool = False):

        # self.req_indices_len

        req_cols = [col
                    for col in df.columns
                    if column_name in col]

        req_df = df[req_cols]

        if is_feature:

            req_cols_test_array = req_df.where((req_df < pow(2, 15)) &
                                               (req_df > self.default_vals_ff_20ms),
                                               np.nan,
                                               inplace=False).values
        else:
            req_cols_test_array = req_df.where(req_df < pow(2, 15),
                                               np.nan,
                                               inplace=False).values

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

            non_repeating_idx = (indices[:, 0] ==
                                 non_repeated_groups[:, None]).any(0).nonzero()[0]

            non_repeating_all = indices[non_repeating_idx]

            repeating_indices_only = list(set(range(len(indices)))
                                          - set(non_repeating_idx))

            repeating_all = indices[repeating_indices_only]

            # yy = {key : val
            #       for key, val in Counter(indices[:, 0]).items()}

            # yy3 = {key :
            #        indices[:, 1]
            #        [indices[:, 0] == key]
            #        for key, val in Counter(indices[:, 0]).items()}

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

                # for repeated_group in repeated_groups:

                # repeated_idx = np.argwhere(np.all(
                #     indices[:, 0] == repeated_group, axis=1)
                #     )
                # unique_cols = np.unique(indices[:, 1])
                # unique_col_dict = {col: [] for col in unique_cols}

                # for idx, col in indices:
                #     unique_col_dict[col].append(idx)

                # req_cols_indices_dict[col_identifier] = unique_col_dict

                # indices_collection = []

                # for repeat_count in unique_repeat_counts:

                #     req_indices_01 = unique_indices[
                #         repeated_indices_count == repeat_count]

                #     indices_idx_req_1 = (indices[:, 0] ==
                #                          req_indices_01[:, None]
                #                          ).any(0).nonzero()[0]

                #     for repeat_ in range(repeat_count):

                #         (unique_indices_02,
                #          repeated_indices_idx_02,
                #          repeated_indices_count_02) = np.unique(
                #              indices[indices_idx_req_1],
                #              return_counts=True,
                #              return_index=True)

                #         req_indices_02 = unique_indices[
                #             repeated_indices_count == repeat_count]

                #         indices_idx_req_2 = (indices[indices_idx_req_1] ==
                #                              req_indices_02[:, None]
                #                              ).any(0).nonzero()[0]

                #         indices_collection.append(
                #             indices[indices_idx_req_2])

                req_cols_indices_dict[col_identifier] = indices_collection

        # req_cols_indices_dict = {col_identifier: df.index[df[req_cols]
        #                                                   [df[req_cols] ==
        #                                                       col_identifier]
        #                                                   .any(axis=1)].tolist()
        #                          for col_identifier in unique_vals_1}

        # req_col_data_dict = {col_identifier: df[req_cols][df[req_cols] ==
        #                                                   col_identifier]
        #                      .any(axis=1)
        #                      for col_identifier in unique_vals_1}

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        # req_col_data_dict = {col_identifier: df[req_cols]
        #                      .iloc[idx_array[:, 0], idx_array[:, 1]]
        #                      .all(axis=1)
        #                      for idx_array, col_identifier in
        #                      zip(unique_vals_idx_2, unique_vals_2)
        #                      if not np.isnan(col_identifier)
        #                      }
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

    def _sign_extraction(self,
                         df,
                         main_sign_col_name,
                         supp_sign_1_col_name,
                         supp_sign_2_col_name,
                         # performance_indicator_cols,
                         **kwargs):

        # Assumption : default_val for main and supp signs is same

        if 'default_val' in kwargs:
            default_val = kwargs['default_val']
        else:
            default_val = 0

        req_cols = sort_list([col for col in df.columns
                              if main_sign_col_name in col
                              ])

        req_cols_confidence = [col for col in df.columns
                               if self.vision_avi_tsr_main_sign_col_confidence
                               in col
                               or
                               self.vision_avi_tsr_supp_sign_1_col_confidence
                               in col
                               or
                               self.vision_avi_tsr_supp_sign_2_col_confidence
                               in col]

        req_cols_supp_1 = sort_list([col for col in df.columns
                                     if supp_sign_1_col_name in col
                                     ])

        req_cols_supp_2 = sort_list([col for col in df.columns
                                     if supp_sign_2_col_name in col
                                     ])

        df_req = df.copy(deep=True)[
            ['cTime']
            + req_cols
            # + req_cols_confidence
            + req_cols_supp_1
            + req_cols_supp_2
        ]

        df_confidence = df.copy(deep=True)[
            ['cTime']
            + req_cols_confidence
            # + req_cols_supp_1
            # + req_cols_supp_2
        ]

        # df = None

        df_req_test = df_req[df_req != default_val[0]].dropna(
            axis=1, how='all')

        # ASSUMPTION : there shall not be flickering duration beyond
        # what is threshold
        self.req_indices_len = self._time_duration_to_indices_len(
            df_req['cTime'].to_frame(),
            self.flickering_threshold_sec, )

        # bool_df_req = df_req != default_val

        event_start_cTime_list = []
        event_end_cTime_list = []
        main_sign_enum_list = []
        supp_1_sign_enum_list, supp_2_sign_enum_list = [], []

        main_sign_confidence_mean_list, main_sign_confidence_std_list = [], []
        supp_1_sign_confidence_mean_list, supp_1_sign_confidence_std_list = [], []
        supp_2_sign_confidence_mean_list, supp_2_sign_confidence_std_list = [], []

        main_sign_confidence_event_start_list = []
        supp_1_sign_confidence_event_start_list = []
        supp_2_sign_confidence_event_start_list = []

        main_sign_confidence_median_list = []
        main_sign_confidence_median_abs_deviation_list = []

        main_sign_ID_list = []

        host_long_velocity_median_list = []
        host_long_velocity_median_abs_deviation_list = []
        host_long_velocity_event_start_list = []

        main_sign_flickering_count_normalised_list = []
        main_sign_flickering_mean_duration_list = []
        main_sign_confidence_flickering_count_normalised_list = []
        main_sign_confidence_flickering_mean_duration_list = []
        main_sign_distance_monotonocity_abberations_count_normalised_list = []

        print('************** Event Extraction Start *******************')
        for enum, col in enumerate(req_cols):

            if col in df_req_test.columns:
                # print('&&&&&&&&&&&&&&\n', col, '&&&&&&&&&&&&&&\n',)

                ID_col = \
                    self.vision_avi_tsr_main_sign_ID_col_name \
                    + '_' \
                    + col.split('_')[-1]

                performance_indicator_cols_iter = [
                    perf_col + '_' + col.split('_')[-1]
                    for perf_col in self.performance_indicator_cols
                ]

                series_ID = df[ID_col]

                series_ID = series_ID.replace(
                    to_replace=default_val,
                    value=np.nan)
                series_ID = series_ID.where(series_ID < pow(2, 15), np.nan,
                                            inplace=False)

                if series_ID is None:
                    continue

                # series = pd.Series(series_ID.dropna().index)

                event_start_end_groups_ID = \
                    self._event_start_end_indices_extractor(series_ID)

                event_start_end_groups_ID = np.array(event_start_end_groups_ID)

                # event_start_end_groups_ID = series.groupby(
                #     series.diff().ne(1)
                #     .cumsum()).apply(
                #     lambda x:
                #     [x.iloc[0], x.iloc[-1]]
                #     if len(x) >= 2
                #     else [x.iloc[0]]).tolist()

                series_main = df_req[col].replace(
                    to_replace=default_val,
                    value=np.nan)
                series_main = series_main.where(series_main < pow(2, 15), np.nan,
                                                inplace=False)
                # if series_main is None:
                #     continue
                confidence_col = \
                    self.vision_avi_tsr_main_sign_col_confidence \
                    + '_' \
                    + col.split('_')[-1]
                series_main_confidence = df_confidence[confidence_col]

                # series = pd.Series(series_main.dropna().index)

                event_start_end_groups = \
                    self._event_start_end_indices_extractor(series_main)

                # event_start_end_groups = series.groupby(
                #     series.diff().ne(1)
                #     .cumsum()).apply(
                #     lambda x:
                #     [x.iloc[0], x.iloc[-1]]
                #     if len(x) >= 2
                #     else [x.iloc[0]]).tolist()

                event_start_end_groups = np.array(event_start_end_groups)
                event_start_cTime = np.array(
                    df_req.loc[event_start_end_groups[:, 0],
                               'cTime'])
                event_end_cTime = np.array(
                    df_req.loc[event_start_end_groups[:, 1],
                               'cTime'])

                main_sign_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main[start: end],
                        df_req['cTime'][start: end]
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_flickering_count_normalised = [
                    item[0] for item in main_sign_flickering
                ]
                main_sign_flickering_mean_duration = [
                    item[1] for item in main_sign_flickering
                ]

                main_sign_confidence_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main_confidence[start: end],
                        df_req['cTime'][start: end]
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_confidence_flickering_count_normalised = [
                    item[0] for item in main_sign_confidence_flickering
                ]
                main_sign_confidence_flickering_mean_duration = [
                    item[1] for item in main_sign_confidence_flickering
                ]

                long_distance_signal_iter = \
                    self.vision_avi_tsr_main_sign_long_dist \
                    + '_' + col.split('_')[-1]
                main_sign_distance_monotonocity_abberations_count_normalised = \
                    np.array(
                        [self._performance_indicators_monotonicity(
                            df[long_distance_signal_iter][start: end],
                            df_req['cTime'][start: end]
                        )[1]
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ])

                main_sign_enum = np.array(
                    [series_main[start: end].mode(dropna=True)[0]
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_ID = np.array(
                    [series_ID[start: end].mode(dropna=True)[0]
                     for start, end in
                     zip(event_start_end_groups_ID[:, 0],
                         event_start_end_groups_ID[:, 1])
                     ])
                main_sign_confidence_mean = np.array(
                    [series_main_confidence[start: end].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_std = np.array(
                    [series_main_confidence[start: end].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median = np.array(
                    [series_main_confidence[start: end].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        series_main_confidence[start: end],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                main_sign_confidence_event_start = np.array(
                    [series_main_confidence[start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])

                series_supp_1 = df_req[req_cols_supp_1[enum]].replace(
                    to_replace=default_val,
                    value=np.nan)

                confidence_supp_1_col = \
                    self.vision_avi_tsr_supp_sign_1_col_confidence \
                    + '_' \
                    + col.split('_')[-1]

                series_supp_1_confidence = df_confidence[confidence_supp_1_col]

                series_supp_2 = df_req[req_cols_supp_2[enum]].replace(
                    to_replace=default_val,
                    value=np.nan)

                confidence_supp_2_col = \
                    self.vision_avi_tsr_supp_sign_2_col_confidence \
                    + '_' \
                    + col.split('_')[-1]

                series_supp_2_confidence = df_confidence[confidence_supp_2_col]

                supp_1_sign_enum = np.array(
                    [series_supp_1[start: end].mode(
                        dropna=False
                    )[0]
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                supp_1_sign_confidence_mean = np.array(
                    [series_supp_1_confidence[start: end].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_1_sign_confidence_std = np.array(
                    [series_supp_1_confidence[start: end].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_1_sign_confidence_event_start = np.array(
                    [series_supp_1_confidence[start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])

                supp_2_sign_enum = np.array(
                    [series_supp_2[start: end].mode(
                        dropna=False
                    )[0]
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                supp_2_sign_confidence_mean = np.array(
                    [series_supp_2_confidence[start: end].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_2_sign_confidence_std = np.array(
                    [series_supp_2_confidence[start: end].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_2_sign_confidence_event_start = np.array(
                    [series_supp_2_confidence[start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])

                host_long_velocity_event_start = np.array(
                    [df[self.vse_host_long_velocity_col][start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])

                host_long_velocity_median = np.array(
                    [df[self.vse_host_long_velocity_col]
                     [start: end].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                host_long_velocity_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        df[self.vse_host_long_velocity_col]
                     [start: end],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                event_start_cTime_list.append(event_start_cTime)
                event_end_cTime_list.append(event_end_cTime)
                main_sign_enum_list.append(main_sign_enum)

                main_sign_confidence_mean_list.append(
                    main_sign_confidence_mean)
                main_sign_confidence_std_list.append(main_sign_confidence_std)

                main_sign_confidence_event_start_list.append(
                    main_sign_confidence_event_start)

                main_sign_confidence_median_list.append(
                    main_sign_confidence_median)
                main_sign_confidence_median_abs_deviation_list.append(
                    main_sign_confidence_median_abs_deviation)

                supp_1_sign_enum_list.append(supp_1_sign_enum)
                supp_2_sign_enum_list.append(supp_2_sign_enum)

                supp_1_sign_confidence_mean_list.append(
                    supp_1_sign_confidence_mean)
                supp_1_sign_confidence_std_list.append(
                    supp_1_sign_confidence_std)

                supp_1_sign_confidence_event_start_list.append(
                    supp_1_sign_confidence_event_start)

                supp_2_sign_confidence_mean_list.append(
                    supp_2_sign_confidence_mean)
                supp_2_sign_confidence_std_list.append(
                    supp_2_sign_confidence_std)

                supp_2_sign_confidence_event_start_list.append(
                    supp_2_sign_confidence_event_start)

                host_long_velocity_median_list.append(
                    host_long_velocity_median)
                host_long_velocity_median_abs_deviation_list.append(
                    host_long_velocity_median_abs_deviation)
                host_long_velocity_event_start_list.append(
                    host_long_velocity_event_start)

                main_sign_ID_list.append(main_sign_ID)

                main_sign_flickering_count_normalised_list.append(
                    main_sign_flickering_count_normalised)
                main_sign_flickering_mean_duration_list.append(
                    main_sign_flickering_mean_duration)
                main_sign_confidence_flickering_count_normalised_list.append(
                    main_sign_confidence_flickering_count_normalised)
                main_sign_confidence_flickering_mean_duration_list.append(
                    main_sign_confidence_flickering_mean_duration)
                main_sign_distance_monotonocity_abberations_count_normalised_list.append(
                    main_sign_distance_monotonocity_abberations_count_normalised)
        print('************** Event Extraction Done *******************')
        event_dict = {'event_start_cTime':
                      np.hstack(event_start_cTime_list)
                      if bool(event_start_cTime_list)
                      else np.array([]),
                      'event_end_cTime':
                          np.hstack(event_end_cTime_list)
                          if bool(event_end_cTime_list)
                          else np.array([]),
                      'main_sign_ID':
                      np.hstack(main_sign_ID_list)
                      if bool(main_sign_ID_list)
                      else np.array([]),
                      'main_sign_enum':
                          np.hstack(main_sign_enum_list)
                          if bool(main_sign_enum_list)
                          else np.array([]),
                      'main_sign_confidence_mean':
                          np.hstack(main_sign_confidence_mean_list)
                          if bool(main_sign_confidence_mean_list)
                          else np.array([]),
                      'main_sign_confidence_std':
                          np.hstack(main_sign_confidence_std_list)
                          if bool(main_sign_confidence_std_list)
                          else np.array([]),
                      'main_sign_confidence_event_start':
                          np.hstack(main_sign_confidence_event_start_list)
                          if bool(main_sign_confidence_event_start_list)
                          else np.array([]),
                      'main_sign_confidence_median':
                          np.hstack(main_sign_confidence_median_list)
                          if bool(main_sign_confidence_median_list)
                          else np.array([]),
                      'main_sign_confidence_median_abs_deviation':
                          np.hstack(
                              main_sign_confidence_median_abs_deviation_list)
                          if bool(
                          main_sign_confidence_median_abs_deviation_list)
                          else np.array([]),
                      'supp_1_sign_enum':
                          np.hstack(supp_1_sign_enum_list)
                          if bool(supp_1_sign_enum_list)
                          else np.array([]),

                      'supp_1_sign_confidence_mean':
                          np.hstack(supp_1_sign_confidence_mean_list)
                          if bool(supp_1_sign_confidence_mean_list)
                          else np.array([]),
                      'supp_1_sign_confidence_std':
                          np.hstack(supp_1_sign_confidence_std_list)
                          if bool(supp_1_sign_confidence_std_list)
                          else np.array([]),
                      'supp_1_sign_confidence_event_start':
                          np.hstack(supp_1_sign_confidence_event_start_list)
                          if bool(supp_1_sign_confidence_event_start_list)
                          else np.array([]),
                      'supp_2_sign_enum':
                          np.hstack(supp_2_sign_enum_list)
                          if bool(supp_2_sign_enum_list)
                          else np.array([]),
                      'supp_2_sign_confidence_mean':
                          np.hstack(supp_2_sign_confidence_mean_list)
                          if bool(supp_2_sign_confidence_mean_list)
                          else np.array([]),
                      'supp_2_sign_confidence_std':
                          np.hstack(supp_2_sign_confidence_std_list)
                          if bool(supp_2_sign_confidence_std_list)
                          else np.array([]),
                      'supp_2_sign_confidence_event_start':
                          np.hstack(supp_2_sign_confidence_event_start_list)
                          if bool(supp_2_sign_confidence_event_start_list)
                          else np.array([]),

                      'host_long_velocity_median_mps':
                          np.hstack(host_long_velocity_median_list)
                          if bool(host_long_velocity_median_list)
                          else np.array([]),
                      'host_long_velocity_median_abs_deviation_mps':
                          np.hstack(
                              host_long_velocity_median_abs_deviation_list)
                          if bool(host_long_velocity_median_abs_deviation_list)
                          else np.array([]),
                      'host_long_velocity_event_start_mps':
                          np.hstack(host_long_velocity_event_start_list)
                          if bool(host_long_velocity_event_start_list)
                          else np.array([]),
                      'main_sign_flickering_count_per_sec':
                          np.hstack(main_sign_flickering_count_normalised_list)
                          if bool(main_sign_flickering_count_normalised_list)
                          else np.array([]),
                      'main_sign_flickering_mean_duration_sec':
                          np.hstack(main_sign_flickering_mean_duration_list)
                          if bool(main_sign_flickering_mean_duration_list)
                          else np.array([]),
                      'main_sign_confidence_flickering_count_per_sec':
                          np.hstack(
                              main_sign_confidence_flickering_count_normalised_list)
                          if bool(
                          main_sign_confidence_flickering_count_normalised_list)
                          else np.array([]),
                      'main_sign_confidence_flickering_mean_duration_sec':
                          np.hstack(
                              main_sign_confidence_flickering_mean_duration_list)
                          if bool(main_sign_confidence_flickering_mean_duration_list)
                          else np.array([]),
                      'main_sign_distance_monotonicity_abberations_count_per_sec':
                          np.hstack(
                          main_sign_distance_monotonocity_abberations_count_normalised_list)
                          if bool(main_sign_distance_monotonocity_abberations_count_normalised_list)
                          else np.array([]),
                      }

        event_dict = self._performance_indicators(df, event_dict, **kwargs)

        return event_dict

    def _sign_extraction_helper_old(self,
                                    df,
                                    sign_indices_dict,
                                    sign_unique_vals_idx,
                                    event_start_end_groups_dict,
                                    is_sign_ID: bool = False,
                                    ):
        # print('##############################################')
        # req_cols_long_distance = [col
        #                           for col in df.columns
        #                           if self.vision_avi_tsr_main_sign_long_dist in col]
        # long_distance_test_array = df[req_cols_long_distance].values
        # long_distance_test_array_grouped = {
        #     ID:
        #         pd.Series(index=indices,
        #                   data=long_distance_test_array[item[:, 0],
        #                                                 item[:, 1]])
        #     for (ID, indices), item in zip(sign_indices_dict.items(),
        #                                    sign_unique_vals_idx)
        # }

        long_distance_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_long_dist,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        lateral_distance_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_lat_dist,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        host_bearing_test_array_grouped = self._grouped_data(
            self.host_bearing_deg,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        geodetic_latitude_deg_array_grouped = self._grouped_data(
            'geodetic_latitude_deg',
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        geodetic_longitude_deg_array_grouped = self._grouped_data(
            'geodetic_longitude_deg',
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        indices_length_feature_must_display = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self.feature_min_time_sign_display)

        indices_to_look_forward = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self._time_to_look_forward)

        indices_to_look_backward = \
            _time_duration_to_indices_length(df['cTime'].reset_index(
                drop=True).to_frame(),
                self._time_to_look_backward)

        # req_cols_confidence = [col
        #                        for col in df.columns
        #                        if self.vision_avi_tsr_main_sign_col_confidence
        #                        in col]
        # confidence_test_array = df[req_cols_confidence].values
        # confidence_test_array_grouped = {
        #     ID:
        #         pd.Series(index=indices,
        #                   data=confidence_test_array[item[:, 0],
        #                                              item[:, 1]])
        #     for (ID, indices), item in zip(sign_indices_dict.items(),
        #                                    sign_unique_vals_idx)
        # }
        if is_sign_ID:

            print('In Sign ID related loop. If seen on HPC jobout, can safely ignore')

            main_sign_test_array_grouped = self._grouped_data(
                self.vision_avi_tsr_main_sign_ID_col_name,
                sign_unique_vals_idx,
                sign_indices_dict,
                df)
        else:
            if self.is_feature_based:
                print('In feature related loop. If seen on HPC jobout, analyse')
            else:
                print(
                    'In Sign enum related loop. If seen on HPC jobout, observe and ignore')
            main_sign_test_array_grouped = self._grouped_data(
                self.vision_avi_tsr_main_sign_col_name,
                sign_unique_vals_idx,
                sign_indices_dict,
                df)

        main_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_main_sign_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp1_sign_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_1_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp1_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_1_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)

        supp2_sign_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_2_col_name,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)
        supp2_sign_confidence_test_array_grouped = self._grouped_data(
            self.vision_avi_tsr_supp_sign_2_col_confidence,
            sign_unique_vals_idx,
            sign_indices_dict,
            df)
        # print('##############################################')

        event_start_cTime_list = []
        event_end_cTime_list = []
        sequence_start_cTime_list = []
        log_name_event_start_list = []
        base_logname_event_start_list = []
        main_sign_enum_list = []
        supp_1_sign_enum_list, supp_2_sign_enum_list = [], []

        host_gps_longitude_event_end_list = []
        host_gps_latitude_event_end_list = []

        host_gps_longitude_event_start_list = []
        host_gps_latitude_event_start_list = []

        target_gps_long_lat_coordinates_list = []
        to_trust_target_gps_list = []

        vision_frame_ID_event_start_list, readff_link_event_start_list = [], []
        vision_frame_ID_event_end_list = []
        readff_link_full_video_list = []

        main_sign_confidence_mean_list, main_sign_confidence_std_list = [], []
        supp_1_sign_confidence_mean_list, supp_1_sign_confidence_std_list = [], []
        supp_2_sign_confidence_mean_list, supp_2_sign_confidence_std_list = [], []

        main_sign_confidence_event_start_list = []
        supp_1_sign_confidence_event_start_list = []
        supp_2_sign_confidence_event_start_list = []

        main_sign_confidence_median_list = []
        main_sign_confidence_median_abs_deviation_list = []

        main_sign_ID_list = []

        host_long_velocity_median_list = []
        host_long_velocity_median_abs_deviation_list = []
        host_long_velocity_event_start_list = []

        main_sign_flickering_count_normalised_list = []
        main_sign_flickering_mean_duration_list = []
        main_sign_confidence_flickering_count_normalised_list = []
        main_sign_confidence_flickering_mean_duration_list = []
        main_sign_distance_monotonocity_abberations_count_normalised_list = []

        ff_20ms_TSR_status_list = []
        ff_20ms_fused_speed_limit_source_enum_list = []
        ff_20ms_fused_speed_limit_sign_enum_list = []
        ff_20ms_fused_speed_limit_sign_enum_mapped_list = []

        avi_SW_version_list = []

        # event_start_end_groups_dict2 = copy.deepcopy(
        #     event_start_end_groups_dict)

        for enum, event_start_end_groups_array_list in \
                event_start_end_groups_dict.items():

            # print('*********************', enum, '^^^^^^^^^^^^^^^^^')

            # if isinstance(sign_indices_dict[enum], dict):

            #     series_main_list_idx = list(
            #         chain.from_iterable(
            #             sign_indices_dict[enum].values()))

            #     series_main_list = []

            # series_main_list = list(sign_indices_dict[enum].values())

            for enumerated_idx, event_start_end_groups in \
                    enumerate(event_start_end_groups_array_list):

                # event_start_end_groups = []
                # print('%%%%%%%% ',
                #       enumerated_idx, )

                if isinstance(sign_indices_dict[enum], dict):

                    req_indices_values = list(
                        chain.from_iterable(
                            sign_indices_dict[enum].values()
                        )
                    )[enumerated_idx][:, 0]

                    # series_main = series_main_list[enumerated_idx]

                    series_main = main_sign_test_array_grouped[
                        enum][enumerated_idx]
                    series_main_confidence = \
                        main_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]
                    long_distance_signal_iter = \
                        long_distance_test_array_grouped[enum][enumerated_idx]

                    lat_distance_signal_iter = \
                        lateral_distance_test_array_grouped[enum][enumerated_idx]

                    host_bearing_iter = \
                        host_bearing_test_array_grouped[enum][enumerated_idx]

                    geodetic_latitude_iter = \
                        geodetic_latitude_deg_array_grouped[enum][enumerated_idx]

                    geodetic_longitude_iter = \
                        geodetic_longitude_deg_array_grouped[enum][enumerated_idx]

                    series_supp_1 = supp1_sign_test_array_grouped[
                        enum][enumerated_idx]

                    series_supp_1_confidence = \
                        supp1_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]

                    series_supp_2 = supp2_sign_test_array_grouped[
                        enum][enumerated_idx]

                    series_supp_2_confidence = \
                        supp2_sign_confidence_test_array_grouped[
                            enum][enumerated_idx]
                else:
                    req_indices_values = np.array(sign_indices_dict[enum],
                                                  dtype=int)

                    series_main = main_sign_test_array_grouped[
                        enum]
                    series_main_confidence = \
                        main_sign_confidence_test_array_grouped[
                            enum]
                    long_distance_signal_iter = \
                        long_distance_test_array_grouped[enum]

                    lat_distance_signal_iter = \
                        lateral_distance_test_array_grouped[enum]

                    host_bearing_iter = \
                        host_bearing_test_array_grouped[enum]

                    geodetic_latitude_iter = \
                        geodetic_latitude_deg_array_grouped[enum]

                    geodetic_longitude_iter = \
                        geodetic_longitude_deg_array_grouped[enum]

                    series_supp_1 = supp1_sign_test_array_grouped[
                        enum]

                    series_supp_1_confidence = \
                        supp1_sign_confidence_test_array_grouped[
                            enum]

                    series_supp_2 = supp2_sign_test_array_grouped[
                        enum]

                    series_supp_2_confidence = \
                        supp2_sign_confidence_test_array_grouped[
                            enum]

                # for event_start_end_groups in event_start_end_groups_array:
                traffic_sign_relative_angle_deg_iter = \
                    self._traffic_sign_to_host_angle2(
                        long_distance_signal_iter,
                        lat_distance_signal_iter)

                traffic_sign_bearing_deg_iter = (
                    host_bearing_iter
                    + traffic_sign_relative_angle_deg_iter) % 360

                traffic_sign_euclidean_distance_m_iter = pd.Series(
                    data=np.sqrt(np.sum(
                        np.square([long_distance_signal_iter,
                                   lat_distance_signal_iter]), axis=0)),
                    index=long_distance_signal_iter.index)

                to_trust_traffic_sign_lat_long_iter = pd.Series(
                    data=[
                        (np.isnan(euclidean_dist)
                         and np.isnan(geodetic_lat)
                         and np.isnan(geodetic_long)
                         and np.isnan(sign_bearing)
                         )
                        for (euclidean_dist,
                             geodetic_lat,
                             geodetic_long,
                             sign_bearing) in
                        zip(traffic_sign_euclidean_distance_m_iter,
                            geodetic_latitude_iter,
                            geodetic_longitude_iter,
                            traffic_sign_bearing_deg_iter)],
                    index=long_distance_signal_iter.index)

                if traffic_sign_bearing_deg_iter.isnull().values.any():

                    traffic_sign_bearing_deg_iter = \
                        traffic_sign_bearing_deg_iter.bfill().ffill()

                if traffic_sign_euclidean_distance_m_iter.isnull().values.any():
                    traffic_sign_euclidean_distance_m_iter = \
                        traffic_sign_euclidean_distance_m_iter.bfill().ffill()

                if (traffic_sign_euclidean_distance_m_iter.isnull(
                ).values.all()
                        or traffic_sign_bearing_deg_iter.isnull().values.all()
                        or geodetic_latitude_iter.isnull().values.all()
                        or geodetic_longitude_iter.isnull().values.all()
                ):

                    traffic_sign_latitude_iter = pd.Series(
                        data=[
                            np.nan for item in
                            traffic_sign_euclidean_distance_m_iter],
                        index=long_distance_signal_iter.index)

                    traffic_sign_longitude_iter = pd.Series(
                        data=[
                            np.nan for item in
                            traffic_sign_euclidean_distance_m_iter],
                        index=long_distance_signal_iter.index)
                else:

                    destination_lat_long_series = [
                        distance(kilometers=euclidean_dist/1E3).destination(
                            (geodetic_lat, geodetic_long), bearing=sign_bearing)
                        for (euclidean_dist,
                             geodetic_lat,
                             geodetic_long,
                             sign_bearing) in
                        zip(traffic_sign_euclidean_distance_m_iter,
                            geodetic_latitude_iter,
                            geodetic_longitude_iter,
                            traffic_sign_bearing_deg_iter)
                    ]

                    traffic_sign_latitude_iter = pd.Series(
                        data=[
                            item.latitude for item in destination_lat_long_series],
                        index=long_distance_signal_iter.index)

                    traffic_sign_longitude_iter = pd.Series(
                        data=[
                            item.longitude for item in destination_lat_long_series],
                        index=long_distance_signal_iter.index)

                event_start_cTime = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'cTime'])
                event_end_cTime = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'cTime'])

                sequence_start_cTime = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'start_cTime_sequence'])

                log_name_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'log_name_flat'])

                vision_frame_ID_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'frame_ID'])

                vision_frame_ID_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'frame_ID'])

                readff_link_event_start_pre = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'readff_link'])

                readff_link_event_start = [item1 + '?iframe=' + str(
                    int(item2 + (item3-item2)*0.5)
                    if item3 > item2 else int(item2)
                )
                    for item1, item2, item3 in
                    zip(readff_link_event_start_pre,
                        vision_frame_ID_event_start,
                        vision_frame_ID_event_end)]

                readff_link_full_video = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'readff_link_full_video'])

                # if self.program_name_readff_map is not None:

                # req_prefix = self.program_name_readff_map

                base_logname_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'base_logname'])

                host_gps_longitude_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'geodetic_longitude_deg'])
                host_gps_latitude_event_end = np.array(
                    df.loc[event_start_end_groups[:, 1],
                           'geodetic_latitude_deg'])

                host_gps_longitude_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'geodetic_longitude_deg'])
                host_gps_latitude_event_start = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'geodetic_latitude_deg'])

                avi_SW_version = np.array(
                    df.loc[event_start_end_groups[:, 0],
                           'avi_SW_version'])

                if self.is_feature_based:

                    # print('??????????????????????????????')

                    host_long_distance_event = [-pd.Series(
                        sp.integrate.cumulative_trapezoid(
                            y=df[self.vse_host_long_velocity_col]
                            [:start-indices_length_feature_must_display][::-1],
                            x=df['cTime']
                            [:start-indices_length_feature_must_display][::-1],
                            initial=0), index=df['cTime']
                        [:start-indices_length_feature_must_display][::-1].index)
                        if start > indices_length_feature_must_display
                        else pd.Series([self.feature_distance_threshold - 0.01,
                                        self.feature_distance_threshold - 0.02],
                                       index=[max(start-1, 0),
                                              max(start, 1)])
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]
                    host_long_distance_event_indices = [
                        series[series < self.feature_distance_threshold].index[-1]
                        if len(series[
                            series > self.feature_distance_threshold].index) > 0
                        else series.index[-1]
                        for series in host_long_distance_event
                    ]

                    host_long_distance_event_indices = [
                        [max(start-indices_to_look_backward,
                             series_main.index[0]),
                         min(start+indices_to_look_forward,
                             series_main.index[-1])]
                        for start in event_start_end_groups[:, 0]
                    ]

                    # ff_20ms_TSR_status = np.array(
                    #     [df[self.feature_tsi_OW_TSR_status][::-1]
                    #      [threshold:
                    #       threshold-indices_length_feature_must_display].mode(
                    #           dropna=False)[0]
                    #      if threshold < end else np.nan
                    #      for end, threshold in
                    #      zip(event_start_end_groups[:, 1],
                    #          host_long_distance_event_indices)
                    #      ])

                    ff_20ms_TSR_status = np.array(
                        [df[self.feature_tsi_OW_TSR_status]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    # ff_20ms_fused_speed_limit_source_enum = np.array(
                    #     [df[self.feature_tsi_OW_speed_limit_source_enum][::-1]
                    #      [threshold:
                    #       threshold-indices_length_feature_must_display].mode(
                    #           dropna=False)[0]
                    #      for end, threshold in
                    #      zip(event_start_end_groups[:, 1],
                    #          host_long_distance_event_indices)
                    #      ])

                    ff_20ms_fused_speed_limit_source_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_source_enum]
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    # ff_20ms_fused_speed_limit_sign_enum = np.array(
                    #     [df[self.feature_tsi_OW_speed_limit_sign_enum][::-1]
                    #      .map(self.misc_out_dict['main_signs_enums_dict'])
                    #      .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                    #      [threshold:
                    #       threshold-indices_length_feature_must_display].mode(
                    #           dropna=False)[0]
                    #      for end, threshold in
                    #      zip(event_start_end_groups[:, 1],
                    #          host_long_distance_event_indices)
                    #      ])

                    ff_20ms_fused_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])

                    ff_20ms_fused_speed_limit_sign_enum_mapped = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         .map(self.misc_out_dict2)
                         .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [req_indices_values[
                             (req_indices_values >= start)
                             & (req_indices_values <= end)]].mode(dropna=False)[0]
                         for start, end in
                         zip(event_start_end_groups[:, 0],
                             event_start_end_groups[:, 1])
                         ])
                    # event_start_end_groups[:, 1] = np.array([
                    #     item if item > 0 else item+1 for item in
                    #     event_start_end_groups[:, 0]
                    # ])

                    orig_event_start_end_groups = copy.deepcopy(
                        event_start_end_groups)

                    event_start_end_groups[:, 1] = event_start_end_groups[:, 0]

                    event_start_end_groups[:, 0] = np.array(
                        [item if cond > item else cond for cond, item in zip(
                            host_long_distance_event_indices,
                            event_start_end_groups[:, 1]
                        )
                        ])
                    ############################################
                    req_indices_values = df.index[
                        np.min(event_start_end_groups, axis=None):
                        np.max(event_start_end_groups, axis=None)
                    ]
                    req_cols_main = [col
                                     for col in df.columns
                                     if self.vision_avi_tsr_main_sign_col_name in col]
                    series_main = df.loc[req_indices_values, req_cols_main]
                    series_main = series_main.where(series_main < pow(2, 15),
                                                    np.nan,
                                                    inplace=False).dropna(axis=1,
                                                                          how='all')
                    unique_enums = np.unique(series_main)
                    unique_enums = unique_enums[~np.isnan(unique_enums)]
                    df_enums = pd.DataFrame()
                    for enum_avi in unique_enums:
                        df_enums['col_'+str(int(enum_avi))] = \
                            series_main[
                                series_main == enum_avi].dropna(
                                    axis=1, how='all').bfill(
                                        axis=1).iloc[:, 0]

                    series_main = df_enums[
                        df_enums.isin(
                            self._avi_speed_limits_df['enum'].values)
                    ].dropna(axis=1, how='all')

                    if len(series_main) == 0 or series_main.empty:

                        if isinstance(sign_indices_dict[enum], dict):

                            req_indices_values = list(
                                chain.from_iterable(
                                    sign_indices_dict[enum].values()
                                )
                            )[enumerated_idx][:, 0]

                        else:
                            req_indices_values = np.array(sign_indices_dict[enum],
                                                          dtype=int)

                        event_start_end_groups = orig_event_start_end_groups

                        main_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]
                        main_sign_confidence_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        #     main_sign_confidence_event_start

                        main_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median_abs_deviation = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        host_long_velocity_event_start = np.array(
                            [df[self.vse_host_long_velocity_col][start]
                             for start in
                             event_start_end_groups[:, 0]
                             ])
                        # host_long_distance_event_start = np.array(
                        #     [df[self.vse_host_long_velocity_col][start]
                        #      for start in
                        #      event_start_end_groups[:, 0]
                        #      ])

                        host_long_velocity_median = np.array(
                            [df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])
                        host_long_velocity_median_abs_deviation = np.array(
                            [sp.stats.median_abs_deviation(
                                df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]],
                                axis=None,
                                nan_policy='omit')
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])

                        # long_distance_signal_iter = \
                        #     self.vision_avi_tsr_main_sign_long_dist \
                        #     + '_' + col.split('_')[-1]
                        main_sign_distance_monotonocity_abberations_count_normalised = \
                            np.array(
                                [self._performance_indicators_monotonicity(
                                    long_distance_signal_iter[req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]],
                                    df['cTime'][req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]]
                                )[1]
                                    for start, end in
                                    zip(event_start_end_groups[:, 0],
                                        event_start_end_groups[:, 1])
                                ])

                        target_long_lat_median = \
                            [(traffic_sign_longitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True),
                             traffic_sign_latitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True))
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ]

                        to_trust_target_gps = np.array(
                            [
                                to_trust_traffic_sign_lat_long_iter
                                [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                                for start, end in
                                zip(event_start_end_groups[:, 0],
                                    event_start_end_groups[:, 1])
                            ])

                        to_trust_target_gps_list.append(to_trust_target_gps)

                        event_start_cTime_list.append(event_start_cTime)
                        event_end_cTime_list.append(event_end_cTime)
                        sequence_start_cTime_list.append(sequence_start_cTime)
                        log_name_event_start_list.append(log_name_event_start)

                        base_logname_event_start_list.append(
                            base_logname_event_start)
                        main_sign_enum_list.append(main_sign_enum)

                        vision_frame_ID_event_start_list.append(
                            vision_frame_ID_event_start)
                        readff_link_event_start_list.append(
                            readff_link_event_start)

                        vision_frame_ID_event_end_list.append(
                            vision_frame_ID_event_end)

                        readff_link_full_video_list.append(
                            readff_link_full_video)

                        host_gps_longitude_event_end_list.append(
                            host_gps_longitude_event_end)
                        host_gps_latitude_event_end_list.append(
                            host_gps_latitude_event_end)

                        host_gps_longitude_event_start_list.append(
                            host_gps_longitude_event_start)
                        host_gps_latitude_event_start_list.append(
                            host_gps_latitude_event_start)

                        target_gps_long_lat_coordinates_list.append(
                            target_long_lat_median)

                        main_sign_confidence_mean_list.append(
                            main_sign_confidence_mean)
                        main_sign_confidence_std_list.append(
                            main_sign_confidence_std)

                        main_sign_confidence_event_start_list.append(
                            main_sign_confidence_event_start)

                        main_sign_confidence_median_list.append(
                            main_sign_confidence_median)
                        main_sign_confidence_median_abs_deviation_list.append(
                            main_sign_confidence_median_abs_deviation)

                        supp_1_sign_enum_list.append(supp_1_sign_enum)
                        supp_2_sign_enum_list.append(supp_2_sign_enum)

                        supp_1_sign_confidence_mean_list.append(
                            supp_1_sign_confidence_mean)
                        supp_1_sign_confidence_std_list.append(
                            supp_1_sign_confidence_std)

                        supp_1_sign_confidence_event_start_list.append(
                            supp_1_sign_confidence_event_start)

                        supp_2_sign_confidence_mean_list.append(
                            supp_2_sign_confidence_mean)
                        supp_2_sign_confidence_std_list.append(
                            supp_2_sign_confidence_std)

                        supp_2_sign_confidence_event_start_list.append(
                            supp_2_sign_confidence_event_start)

                        host_long_velocity_median_list.append(
                            host_long_velocity_median)
                        host_long_velocity_median_abs_deviation_list.append(
                            host_long_velocity_median_abs_deviation)
                        host_long_velocity_event_start_list.append(
                            host_long_velocity_event_start)

                        # main_sign_ID_list.append(main_sign_ID)

                        main_sign_flickering_count_normalised_list.append(
                            main_sign_flickering_count_normalised)
                        main_sign_flickering_mean_duration_list.append(
                            main_sign_flickering_mean_duration)
                        main_sign_confidence_flickering_count_normalised_list.append(
                            main_sign_confidence_flickering_count_normalised)
                        main_sign_confidence_flickering_mean_duration_list.append(
                            main_sign_confidence_flickering_mean_duration)
                        main_sign_distance_monotonocity_abberations_count_normalised_list\
                            .append(
                                main_sign_distance_monotonocity_abberations_count_normalised)

                        ff_20ms_fused_speed_limit_sign_enum_list.append(
                            ff_20ms_fused_speed_limit_sign_enum)
                        ff_20ms_fused_speed_limit_source_enum_list.append(
                            ff_20ms_fused_speed_limit_source_enum)
                        ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                        ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                            ff_20ms_fused_speed_limit_sign_enum_mapped)

                        continue

                    main_sign_enum_pre = [
                        df_enums.loc[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)], :]
                        # .dropna(axis=1, how='all')
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    main_sign_enum = np.array(
                        [
                            sp.stats.mode(item,
                                          nan_policy='omit',
                                          axis=None)[0]
                            if len(sp.stats.mode(item, nan_policy='omit', axis=None)) > 0
                            else 'not available'
                            for item in main_sign_enum_pre
                        ])

                    # main_sign_enum = unique_enums[~np.isnan(main_sign_enum)]

                    # series_main = pd.concat(
                    #     [series_main[
                    #         series_main == enum].dropna(
                    #             axis=1, how='all').bfill(
                    #                 axis=1).iloc[:, 0]
                    #      for enum in main_sign_enum
                    #      ],
                    #     axis=0,
                    #     ignore_index=False)

                    (event_start_end_groups_dict_enum,
                     sign_data_dict_enum,
                     sign_indices_dict_enum,
                     sign_unique_vals_idx_enum) = self._events_from_col(
                        self.vision_avi_tsr_main_sign_col_name,
                        df)

                    main_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_main_sign_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_main = [main_sign_test_array_grouped_enum.get(
                        enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    # pre_series_main_series = [item
                    #                    for item in pre_series_main
                    #                    if isinstance(item, pd.Series) ]
                    # pre_series_main_list = [item
                    #                    for item in pre_series_main
                    #                    if isinstance(item, list) ]

                    series_main = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_main
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_main
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    if len(series_main) == 0 or series_main.empty:

                        if isinstance(sign_indices_dict[enum], dict):

                            req_indices_values = list(
                                chain.from_iterable(
                                    sign_indices_dict[enum].values()
                                )
                            )[enumerated_idx][:, 0]

                        else:
                            req_indices_values = np.array(sign_indices_dict[enum],
                                                          dtype=int)

                        event_start_end_groups = orig_event_start_end_groups

                        main_sign_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_flickering_count_normalised = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]
                        main_sign_confidence_flickering_mean_duration = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        #     main_sign_confidence_event_start

                        main_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_median_abs_deviation = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        main_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_1_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_enum = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_mean = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_std = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        supp_2_sign_confidence_event_start = [
                            np.nan
                            for item in ff_20ms_fused_speed_limit_sign_enum
                        ]

                        host_long_velocity_event_start = np.array(
                            [df[self.vse_host_long_velocity_col][start]
                             for start in
                             event_start_end_groups[:, 0]
                             ])
                        # host_long_distance_event_start = np.array(
                        #     [df[self.vse_host_long_velocity_col][start]
                        #      for start in
                        #      event_start_end_groups[:, 0]
                        #      ])

                        host_long_velocity_median = np.array(
                            [df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])
                        host_long_velocity_median_abs_deviation = np.array(
                            [sp.stats.median_abs_deviation(
                                df[self.vse_host_long_velocity_col]
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]],
                                axis=None,
                                nan_policy='omit')
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ])

                        # long_distance_signal_iter = \
                        #     self.vision_avi_tsr_main_sign_long_dist \
                        #     + '_' + col.split('_')[-1]
                        main_sign_distance_monotonocity_abberations_count_normalised = \
                            np.array(
                                [self._performance_indicators_monotonicity(
                                    long_distance_signal_iter[req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]],
                                    df['cTime'][req_indices_values[
                                        (req_indices_values >= start)
                                        & (req_indices_values <= end)]]
                                )[1]
                                    for start, end in
                                    zip(event_start_end_groups[:, 0],
                                        event_start_end_groups[:, 1])
                                ])

                        target_long_lat_median = \
                            [(traffic_sign_longitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True),
                             traffic_sign_latitude_iter
                             [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True))
                             for start, end in
                             zip(event_start_end_groups[:, 0],
                                 event_start_end_groups[:, 1])
                             ]

                        to_trust_target_gps = np.array(
                            [
                                to_trust_traffic_sign_lat_long_iter
                                [req_indices_values[
                                 (req_indices_values >= start)
                                 & (req_indices_values <= end)]].median(skipna=True)
                                for start, end in
                                zip(event_start_end_groups[:, 0],
                                    event_start_end_groups[:, 1])
                            ])

                        to_trust_target_gps_list.append(to_trust_target_gps)

                        event_start_cTime_list.append(event_start_cTime)
                        event_end_cTime_list.append(event_end_cTime)
                        sequence_start_cTime_list.append(sequence_start_cTime)
                        log_name_event_start_list.append(log_name_event_start)
                        base_logname_event_start_list.append(
                            base_logname_event_start)
                        main_sign_enum_list.append(main_sign_enum)

                        vision_frame_ID_event_start_list.append(
                            vision_frame_ID_event_start)
                        readff_link_event_start_list.append(
                            readff_link_event_start)

                        vision_frame_ID_event_end_list.append(
                            vision_frame_ID_event_end)

                        readff_link_full_video_list.append(
                            readff_link_full_video)

                        host_gps_longitude_event_end_list.append(
                            host_gps_longitude_event_end)
                        host_gps_latitude_event_end_list.append(
                            host_gps_latitude_event_end)

                        host_gps_longitude_event_start_list.append(
                            host_gps_longitude_event_start)
                        host_gps_latitude_event_start_list.append(
                            host_gps_latitude_event_start)

                        target_gps_long_lat_coordinates_list.append(
                            target_long_lat_median)

                        main_sign_confidence_mean_list.append(
                            main_sign_confidence_mean)
                        main_sign_confidence_std_list.append(
                            main_sign_confidence_std)

                        main_sign_confidence_event_start_list.append(
                            main_sign_confidence_event_start)

                        main_sign_confidence_median_list.append(
                            main_sign_confidence_median)
                        main_sign_confidence_median_abs_deviation_list.append(
                            main_sign_confidence_median_abs_deviation)

                        supp_1_sign_enum_list.append(supp_1_sign_enum)
                        supp_2_sign_enum_list.append(supp_2_sign_enum)

                        supp_1_sign_confidence_mean_list.append(
                            supp_1_sign_confidence_mean)
                        supp_1_sign_confidence_std_list.append(
                            supp_1_sign_confidence_std)

                        supp_1_sign_confidence_event_start_list.append(
                            supp_1_sign_confidence_event_start)

                        supp_2_sign_confidence_mean_list.append(
                            supp_2_sign_confidence_mean)
                        supp_2_sign_confidence_std_list.append(
                            supp_2_sign_confidence_std)

                        supp_2_sign_confidence_event_start_list.append(
                            supp_2_sign_confidence_event_start)

                        host_long_velocity_median_list.append(
                            host_long_velocity_median)
                        host_long_velocity_median_abs_deviation_list.append(
                            host_long_velocity_median_abs_deviation)
                        host_long_velocity_event_start_list.append(
                            host_long_velocity_event_start)

                        # main_sign_ID_list.append(main_sign_ID)

                        main_sign_flickering_count_normalised_list.append(
                            main_sign_flickering_count_normalised)
                        main_sign_flickering_mean_duration_list.append(
                            main_sign_flickering_mean_duration)
                        main_sign_confidence_flickering_count_normalised_list.append(
                            main_sign_confidence_flickering_count_normalised)
                        main_sign_confidence_flickering_mean_duration_list.append(
                            main_sign_confidence_flickering_mean_duration)
                        main_sign_distance_monotonocity_abberations_count_normalised_list\
                            .append(
                                main_sign_distance_monotonocity_abberations_count_normalised)

                        ff_20ms_fused_speed_limit_sign_enum_list.append(
                            ff_20ms_fused_speed_limit_sign_enum)
                        ff_20ms_fused_speed_limit_source_enum_list.append(
                            ff_20ms_fused_speed_limit_source_enum)
                        ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                        ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                            ff_20ms_fused_speed_limit_sign_enum_mapped)

                        continue

                    req_indices_values = series_main.index

                    main_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_main_sign_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_main_confidence = [
                        main_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_main_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_main_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_main_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp1_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_1_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_1 = [
                        supp1_sign_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_1 = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_1
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_1
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp1_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_1_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_1_confidence = [
                        supp1_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_1_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_1_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_1_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp2_sign_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_2_col_name,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_2 = [
                        supp2_sign_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_2 = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_2
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_2
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    supp2_sign_confidence_test_array_grouped_enum = self._grouped_data(
                        self.vision_avi_tsr_supp_sign_2_col_confidence,
                        sign_unique_vals_idx_enum,
                        sign_indices_dict_enum,
                        df)

                    pre_series_supp_2_confidence = [
                        supp2_sign_confidence_test_array_grouped_enum.get(
                            enum, pd.Series())
                        for enum in main_sign_enum
                    ]

                    series_supp_2_confidence = pd.concat(
                        list(itertools.chain.from_iterable(
                            [item for item in pre_series_supp_2_confidence
                             if isinstance(item, list)])
                             ) + [item for item in pre_series_supp_2_confidence
                                  if isinstance(item, pd.Series)],
                        axis=0,
                        ignore_index=False)

                    ################################

                    # print(event_start_end_groups)

                    # event_start_end_groups[:, 0] = \
                    #     host_long_distance_event_indices
                else:

                    # print('')

                    host_long_distance_event = [pd.Series(
                        sp.integrate.cumulative_trapezoid(
                            y=df[self.vse_host_long_velocity_col][end:],
                            x=df['cTime'][end:],
                            initial=0), index=df['cTime'].loc[end:].index)
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]
                    host_long_distance_event_indices = [
                        series[series > self.feature_distance_threshold].index[0]
                        if len(series[
                            series > self.feature_distance_threshold].index) > 0
                        else series.index[0]
                        for series in host_long_distance_event
                    ]
                    # feature_20ms_related =
                    ff_20ms_TSR_status = np.array(
                        [df[self.feature_tsi_OW_TSR_status]
                         [threshold:
                          threshold+indices_length_feature_must_display].mode(
                              dropna=False)[0]
                         if threshold > end else np.nan
                         for end, threshold in
                         zip(event_start_end_groups[:, 1],
                             host_long_distance_event_indices)
                         ])
                    ff_20ms_fused_speed_limit_source_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_source_enum]
                         [threshold:
                          threshold+indices_length_feature_must_display].mode(
                              dropna=False)[0]
                         if threshold > end else np.nan
                         for end, threshold in
                         zip(event_start_end_groups[:, 1],
                             host_long_distance_event_indices)
                         ])

                    ff_20ms_fused_speed_limit_sign_enum = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         # .map(self.misc_out_dict2)
                         # .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         # [req_indices_values[
                         #     (req_indices_values >= start)
                         #     & (req_indices_values <= end)]].mode(dropna=False)[0]
                         [threshold:
                          threshold+indices_length_feature_must_display].mode(
                              dropna=False)[0]
                         if threshold > end else np.nan
                         for end, threshold in
                         zip(event_start_end_groups[:, 1],
                             host_long_distance_event_indices)
                         ])

                    ff_20ms_fused_speed_limit_sign_enum_mapped = np.array(
                        [df[self.feature_tsi_OW_speed_limit_sign_enum]
                         .map(self.misc_out_dict2)
                         .fillna(df[self.feature_tsi_OW_speed_limit_sign_enum])
                         [threshold:
                          threshold+indices_length_feature_must_display].mode(
                              dropna=False)[0]
                         if threshold > end else np.nan
                         for end, threshold in
                         zip(event_start_end_groups[:, 1],
                             host_long_distance_event_indices)
                         ])

                    main_sign_enum_pre = [
                        series_main[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]]
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ]

                    main_sign_enum = np.array(
                        [
                            item.mode(dropna=True)[0]
                            if len(item.mode(dropna=True)) > 0
                            else 'not available'
                            for item in main_sign_enum_pre
                        ])

                main_sign_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        df['cTime'][req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        no_nan_indices=True,
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_flickering_count_normalised = [
                    item[0] for item in main_sign_flickering
                ]
                main_sign_flickering_mean_duration = [
                    item[1] for item in main_sign_flickering
                ]

                main_sign_confidence_flickering = np.array(
                    [self._performance_indicators_signal_flickering(
                        series_main_confidence[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        df['cTime'][req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        no_nan_indices=True,
                    )
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                main_sign_confidence_flickering_count_normalised = [
                    item[0] for item in main_sign_confidence_flickering
                ]
                main_sign_confidence_flickering_mean_duration = [
                    item[1] for item in main_sign_confidence_flickering
                ]

                # main_sign_enum = np.array(
                #     [series_main[req_indices_values[
                #         (req_indices_values >= start)
                #         & (req_indices_values <= end)]].mode(dropna=True)[0]
                #      for start, end in
                #      zip(event_start_end_groups[:, 0],
                #          event_start_end_groups[:, 1])
                #      ])

                # main_sign_ID = np.array(
                #     [series_ID[start: end].mode(dropna=True)[0]
                #      for start, end in
                #      zip(event_start_end_groups_ID[:, 0],
                #          event_start_end_groups_ID[:, 1])
                #      ])
                main_sign_confidence_mean = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_std = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median = np.array(
                    [series_main_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                main_sign_confidence_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        series_main_confidence[req_indices_values[
                            (req_indices_values >= start)
                            & (req_indices_values <= end)]],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                main_sign_confidence_event_start = np.array(
                    [series_main_confidence[req_indices_values[
                        req_indices_values == start
                    ]].iloc[0]
                        if start in req_indices_values
                        else np.nan
                        for start in
                        event_start_end_groups[:, 0]
                    ]).flatten()

                # series_supp_1 = df_req[req_cols_supp_1[enum]].replace(
                #     to_replace=default_val,
                #     value=np.nan)

                # if not is_sign_ID

                # confidence_supp_1_col = \
                #     self.vision_avi_tsr_supp_sign_1_col_confidence \
                #     + '_' \
                #     + col.split('_')[-1]

                # series_supp_1_confidence = df_confidence[confidence_supp_1_col]

                # series_supp_2 = df_req[req_cols_supp_2[enum]].replace(
                #     to_replace=default_val,
                #     value=np.nan)

                # confidence_supp_2_col = \
                #     self.vision_avi_tsr_supp_sign_2_col_confidence \
                #     + '_' \
                #     + col.split('_')[-1]

                # series_supp_2_confidence = df_confidence[confidence_supp_2_col]

                supp_1_sign_enum_pre = [
                    series_supp_1[req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]].mode(
                        dropna=False
                    )
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                ]

                supp_1_sign_enum = np.array([item[0]
                                             if len(item) > 0 else np.nan
                                             for item in supp_1_sign_enum_pre])

                supp_1_sign_confidence_mean = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_1_sign_confidence_std = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_1_sign_confidence_event_start = np.array(
                    [series_supp_1_confidence
                     [req_indices_values[
                         (req_indices_values == start)
                     ]].iloc[0]
                     if start in req_indices_values
                     else np.nan
                     for start in
                     event_start_end_groups[:, 0]
                     ]).flatten()

                supp_2_sign_enum_pre = [
                    series_supp_2
                    [req_indices_values[
                        (req_indices_values >= start)
                        & (req_indices_values <= end)]].mode(
                        dropna=False
                    )
                    for start, end in
                    zip(event_start_end_groups[:, 0],
                        event_start_end_groups[:, 1])
                ]

                supp_2_sign_enum = np.array([item[0]
                                             if len(item) > 0 else np.nan
                                             for item in supp_2_sign_enum_pre])

                supp_2_sign_confidence_mean = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].mean(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                supp_2_sign_confidence_std = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].std(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                supp_2_sign_confidence_event_start = np.array(
                    [series_supp_2_confidence
                     [req_indices_values[
                         (req_indices_values == start)
                     ]].iloc[0]
                     if start in req_indices_values
                     else np.nan
                     for start in
                     event_start_end_groups[:, 0]
                     ]).flatten()

                if self.is_feature_based:

                    if isinstance(sign_indices_dict[enum], dict):

                        req_indices_values = list(
                            chain.from_iterable(
                                sign_indices_dict[enum].values()
                            )
                        )[enumerated_idx][:, 0]

                    else:
                        req_indices_values = np.array(sign_indices_dict[enum],
                                                      dtype=int)

                    event_start_end_groups = orig_event_start_end_groups

                host_long_velocity_event_start = np.array(
                    [df[self.vse_host_long_velocity_col][start]
                     for start in
                     event_start_end_groups[:, 0]
                     ])
                # host_long_distance_event_start = np.array(
                #     [df[self.vse_host_long_velocity_col][start]
                #      for start in
                #      event_start_end_groups[:, 0]
                #      ])

                host_long_velocity_median = np.array(
                    [df[self.vse_host_long_velocity_col]
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])
                host_long_velocity_median_abs_deviation = np.array(
                    [sp.stats.median_abs_deviation(
                        df[self.vse_host_long_velocity_col]
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]],
                        axis=None,
                        nan_policy='omit')
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ])

                # long_distance_signal_iter = \
                #     self.vision_avi_tsr_main_sign_long_dist \
                #     + '_' + col.split('_')[-1]
                main_sign_distance_monotonocity_abberations_count_normalised = \
                    np.array(
                        [self._performance_indicators_monotonicity(
                            long_distance_signal_iter[req_indices_values[
                                (req_indices_values >= start)
                                & (req_indices_values <= end)]],
                            df['cTime'][req_indices_values[
                                (req_indices_values >= start)
                                & (req_indices_values <= end)]]
                        )[1]
                            for start, end in
                            zip(event_start_end_groups[:, 0],
                                event_start_end_groups[:, 1])
                        ])

                target_long_lat_median = \
                    [(traffic_sign_longitude_iter
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True),
                     traffic_sign_latitude_iter
                     [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True))
                     for start, end in
                     zip(event_start_end_groups[:, 0],
                         event_start_end_groups[:, 1])
                     ]

                to_trust_target_gps = np.array(
                    [
                        to_trust_traffic_sign_lat_long_iter
                        [req_indices_values[
                         (req_indices_values >= start)
                         & (req_indices_values <= end)]].median(skipna=True)
                        for start, end in
                        zip(event_start_end_groups[:, 0],
                            event_start_end_groups[:, 1])
                    ])

                to_trust_target_gps_list.append(to_trust_target_gps)

                event_start_cTime_list.append(event_start_cTime)
                event_end_cTime_list.append(event_end_cTime)
                sequence_start_cTime_list.append(sequence_start_cTime)
                log_name_event_start_list.append(log_name_event_start)
                base_logname_event_start_list.append(base_logname_event_start)
                main_sign_enum_list.append(main_sign_enum)

                vision_frame_ID_event_start_list.append(
                    vision_frame_ID_event_start)
                readff_link_event_start_list.append(
                    readff_link_event_start)

                vision_frame_ID_event_end_list.append(
                    vision_frame_ID_event_end)

                readff_link_full_video_list.append(readff_link_full_video)

                host_gps_longitude_event_end_list.append(
                    host_gps_longitude_event_end)
                host_gps_latitude_event_end_list.append(
                    host_gps_latitude_event_end)

                host_gps_longitude_event_start_list.append(
                    host_gps_longitude_event_start)
                host_gps_latitude_event_start_list.append(
                    host_gps_latitude_event_start)

                target_gps_long_lat_coordinates_list.append(
                    target_long_lat_median)

                main_sign_confidence_mean_list.append(
                    main_sign_confidence_mean)
                main_sign_confidence_std_list.append(
                    main_sign_confidence_std)

                main_sign_confidence_event_start_list.append(
                    main_sign_confidence_event_start)

                main_sign_confidence_median_list.append(
                    main_sign_confidence_median)
                main_sign_confidence_median_abs_deviation_list.append(
                    main_sign_confidence_median_abs_deviation)

                supp_1_sign_enum_list.append(supp_1_sign_enum)
                supp_2_sign_enum_list.append(supp_2_sign_enum)

                supp_1_sign_confidence_mean_list.append(
                    supp_1_sign_confidence_mean)
                supp_1_sign_confidence_std_list.append(
                    supp_1_sign_confidence_std)

                supp_1_sign_confidence_event_start_list.append(
                    supp_1_sign_confidence_event_start)

                supp_2_sign_confidence_mean_list.append(
                    supp_2_sign_confidence_mean)
                supp_2_sign_confidence_std_list.append(
                    supp_2_sign_confidence_std)

                supp_2_sign_confidence_event_start_list.append(
                    supp_2_sign_confidence_event_start)

                host_long_velocity_median_list.append(
                    host_long_velocity_median)
                host_long_velocity_median_abs_deviation_list.append(
                    host_long_velocity_median_abs_deviation)
                host_long_velocity_event_start_list.append(
                    host_long_velocity_event_start)

                # main_sign_ID_list.append(main_sign_ID)

                main_sign_flickering_count_normalised_list.append(
                    main_sign_flickering_count_normalised)
                main_sign_flickering_mean_duration_list.append(
                    main_sign_flickering_mean_duration)
                main_sign_confidence_flickering_count_normalised_list.append(
                    main_sign_confidence_flickering_count_normalised)
                main_sign_confidence_flickering_mean_duration_list.append(
                    main_sign_confidence_flickering_mean_duration)
                main_sign_distance_monotonocity_abberations_count_normalised_list\
                    .append(
                        main_sign_distance_monotonocity_abberations_count_normalised)

                ff_20ms_fused_speed_limit_sign_enum_list.append(
                    ff_20ms_fused_speed_limit_sign_enum)
                ff_20ms_fused_speed_limit_source_enum_list.append(
                    ff_20ms_fused_speed_limit_source_enum)
                ff_20ms_TSR_status_list.append(ff_20ms_TSR_status)

                ff_20ms_fused_speed_limit_sign_enum_mapped_list.append(
                    ff_20ms_fused_speed_limit_sign_enum_mapped)

        main_sign_enum_list_stacked = (np.hstack(main_sign_enum_list)
                                       if bool(main_sign_enum_list)
                                       else np.array([]))

        ff_20ms_fused_speed_limit_sign_enum_list_stacked = (np.hstack(
            ff_20ms_fused_speed_limit_sign_enum_list)
            if bool(ff_20ms_fused_speed_limit_sign_enum_list)
            else np.array([]))

        is_main_sign_feature_match = [
            'Matched'
            if (item in
                self.avi_feature_mapping.keys() and
                self.avi_feature_mapping[item] == f_item)
            else 'Not Matched'
            for f_item, item in
            zip(
                ff_20ms_fused_speed_limit_sign_enum_list_stacked,
                main_sign_enum_list_stacked)
        ]

        event_dict = {
            'log_name':
            np.hstack(log_name_event_start_list)
            if bool(log_name_event_start_list)
            else np.array([]),

            'base_logname':
            np.hstack(base_logname_event_start_list)
            if bool(base_logname_event_start_list)
            else np.array([]),

            'event_start_cTime':
            np.hstack(event_start_cTime_list)
            if bool(event_start_cTime_list)
            else np.array([]),
            'event_end_cTime':
            np.hstack(event_end_cTime_list)
            if bool(event_end_cTime_list)
            else np.array([]),
            'start_cTime_sequence':
            np.hstack(sequence_start_cTime_list)
            if bool(sequence_start_cTime_list)
            else np.array([]),



            'feature_20ms_OW_fused_speed_limit_sign_enum_mapped':
            np.hstack(ff_20ms_fused_speed_limit_sign_enum_mapped_list)
            if bool(ff_20ms_fused_speed_limit_sign_enum_mapped_list)
            else np.array([]),

            'feature_20ms_OW_fused_speed_limit_sign_enum':
            ff_20ms_fused_speed_limit_sign_enum_list_stacked,

            'feature_20ms_OW_fused_speed_limit_source_enum':
            np.hstack(ff_20ms_fused_speed_limit_source_enum_list)
            if bool(ff_20ms_fused_speed_limit_source_enum_list)
            else np.array([]),

            'feature_20ms_OW_TSR_status':
            np.hstack(ff_20ms_TSR_status_list)
            if bool(ff_20ms_TSR_status_list)
            else np.array([]),

            'target_gps_long_lat_coordinates':
            list(chain.from_iterable(target_gps_long_lat_coordinates_list))
            if bool(target_gps_long_lat_coordinates_list)
            else np.array([]),

            'to_trust_target_gps':
            np.hstack(to_trust_target_gps_list)
            if bool(to_trust_target_gps_list)
            else np.array([]),

            'readff_link_event_start':
                np.hstack(readff_link_event_start_list)
                if bool(readff_link_event_start_list)
                else np.array([]),
            'readff_link_full_video':
                np.hstack(readff_link_full_video_list)
                if bool(readff_link_full_video_list)
                else np.array([]),

            'vision_frame_ID_event_start':
                np.hstack(vision_frame_ID_event_start_list)
                if bool(vision_frame_ID_event_start_list)
                else np.array([]),
            'vision_frame_ID_event_end':
                np.hstack(vision_frame_ID_event_end_list)
                if bool(vision_frame_ID_event_end_list)
                else np.array([]),

            'host_gps_latitude_event_end':
            np.hstack(host_gps_latitude_event_end_list)
            if bool(host_gps_latitude_event_end_list)
            else np.array([]),

            'host_gps_longitude_event_end':
            np.hstack(host_gps_longitude_event_end_list)
            if bool(host_gps_longitude_event_end_list)
            else np.array([]),

            'host_gps_latitude_event_start':
            np.hstack(host_gps_latitude_event_start_list)
            if bool(host_gps_latitude_event_start_list)
            else np.array([]),

            'host_gps_longitude_event_start':
            np.hstack(host_gps_longitude_event_start_list)
            if bool(host_gps_longitude_event_start_list)
            else np.array([]),

            # 'main_sign_ID':
            # np.hstack(main_sign_ID_list)
            # if bool(main_sign_ID_list)
            # else np.array([]),
            'main_sign_enum':
            main_sign_enum_list_stacked,
            'is_main_sign_feature_match': is_main_sign_feature_match,
            'main_sign_confidence_mean':
            np.hstack(main_sign_confidence_mean_list)
            if bool(main_sign_confidence_mean_list)
            else np.array([]),
            'main_sign_confidence_std':
            np.hstack(main_sign_confidence_std_list)
            if bool(main_sign_confidence_std_list)
            else np.array([]),
            'main_sign_confidence_event_start':
            np.hstack(main_sign_confidence_event_start_list)
            if bool(main_sign_confidence_event_start_list)
            else np.array([]),
            'main_sign_confidence_median':
            np.hstack(main_sign_confidence_median_list)
            if bool(main_sign_confidence_median_list)
            else np.array([]),
            'main_sign_confidence_median_abs_deviation':
            np.hstack(
                main_sign_confidence_median_abs_deviation_list)
            if bool(
                main_sign_confidence_median_abs_deviation_list)
            else np.array([]),
            'supp_1_sign_enum':
            np.hstack(supp_1_sign_enum_list)
            if bool(supp_1_sign_enum_list)
            else np.array([]),

            'supp_1_sign_confidence_mean':
            np.hstack(supp_1_sign_confidence_mean_list)
            if bool(supp_1_sign_confidence_mean_list)
            else np.array([]),
            'supp_1_sign_confidence_std':
            np.hstack(supp_1_sign_confidence_std_list)
            if bool(supp_1_sign_confidence_std_list)
            else np.array([]),
            'supp_1_sign_confidence_event_start':
            np.hstack(supp_1_sign_confidence_event_start_list)
            if bool(supp_1_sign_confidence_event_start_list)
            else np.array([]),
            'supp_2_sign_enum':
            np.hstack(supp_2_sign_enum_list)
            if bool(supp_2_sign_enum_list)
            else np.array([]),
            'supp_2_sign_confidence_mean':
            np.hstack(supp_2_sign_confidence_mean_list)
            if bool(supp_2_sign_confidence_mean_list)
            else np.array([]),
            'supp_2_sign_confidence_std':
            np.hstack(supp_2_sign_confidence_std_list)
            if bool(supp_2_sign_confidence_std_list)
            else np.array([]),
            'supp_2_sign_confidence_event_start':
            np.hstack(supp_2_sign_confidence_event_start_list)
            if bool(supp_2_sign_confidence_event_start_list)
            else np.array([]),

            'host_long_velocity_median_mps':
            np.hstack(host_long_velocity_median_list)
            if bool(host_long_velocity_median_list)
            else np.array([]),
            'host_long_velocity_median_abs_deviation_mps':
            np.hstack(
                host_long_velocity_median_abs_deviation_list)
            if bool(host_long_velocity_median_abs_deviation_list)
            else np.array([]),
            'host_long_velocity_event_start_mps':
            np.hstack(host_long_velocity_event_start_list)
            if bool(host_long_velocity_event_start_list)
            else np.array([]),
            'main_sign_flickering_count_per_sec':
            np.hstack(main_sign_flickering_count_normalised_list)
            if bool(main_sign_flickering_count_normalised_list)
            else np.array([]),
            'main_sign_flickering_mean_duration_sec':
            np.hstack(main_sign_flickering_mean_duration_list)
            if bool(main_sign_flickering_mean_duration_list)
            else np.array([]),
            'main_sign_confidence_flickering_count_per_sec':
            np.hstack(
                main_sign_confidence_flickering_count_normalised_list)
            if bool(
                main_sign_confidence_flickering_count_normalised_list)
            else np.array([]),
            'main_sign_confidence_flickering_mean_duration_sec':
            np.hstack(
                main_sign_confidence_flickering_mean_duration_list)
            if bool(main_sign_confidence_flickering_mean_duration_list)
            else np.array([]),
            'main_sign_distance_monotonicity_abberations_count_per_sec':
            np.hstack(
                main_sign_distance_monotonocity_abberations_count_normalised_list)
            if bool(main_sign_distance_monotonocity_abberations_count_normalised_list)
            else np.array([]),
        }

        # if not is_sign_ID

        # event_dict = {key + '_sign_ID_based'
        #               if is_sign_ID
        #               else key: val
        #               for key, val in event_dict.items()
        #               }

        return event_dict

    def event_extraction(self, df, **kwargs):

        event_dict = self._sign_extraction2(
            df,
            self.vision_avi_tsr_main_sign_col_name,
            self.vision_avi_tsr_supp_sign_1_col_name,
            self.vision_avi_tsr_supp_sign_2_col_name,
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
    program = 'MCIP'  # 'Thunder'  # 'Northstar'  #
    # 'config_northstar_v1_cut_in.yaml'  #
    config_file = 'config_mcip_v1_tsi.yaml'  # 'config_thunder_v1_tsi.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'ThunderMCIP_WS11656_20250804_125911_0010_p01.mat'
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
    TSI_core_logic_obj = coreEventExtractionTSI(mat_file_data)

    return_val_dict = TSI_core_logic_obj.main(config_path)

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
