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


if __package__ is None:

    print('Here at none package 1')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, \n to change 1: {to_change_path}')
    from signal_mapping_tsi import signalMapping
    print('Here at none package 2')
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(to_change_path)
    print(f'Current dir 2: {os.getcwd()}, to change 2: {to_change_path}')

    from utils.utils_generic import (
        loadmat,
        _calc_derivative,
        _get_bearing_from_lat_long,

    )
    os.chdir(actual_package_path)


else:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, to change 1: {to_change_path}')

    from signal_mapping_tsi import signalMapping

    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

    os.chdir(to_change_path)
    print(f'Current dir 2: {os.getcwd()}, to change 2: {to_change_path}')

    # from .. import utils
    try:
        from eventExtraction.utils.utils_generic import (loadmat,
                                                         _calc_derivative,
                                                         _get_bearing_from_lat_long

                                                         )
    except:
        from utils.utils_generic import (loadmat,
                                         _calc_derivative,
                                         _get_bearing_from_lat_long

                                         )


class signalData(signalMapping):

    def __init__(self, raw_data) -> None:

        # super().__init__(self, raw_data)
        signalMapping.__init__(self, raw_data)

        # self.raw_data = raw_data

        self.vision_avi_tsr_main_sign_col_name = \
            'vision_avi_tsr_sign_name'
        self.vision_avi_tsr_supp_sign_1_col_name = \
            'vision_avi_tsr_supp_sign_1_name'
        self.vision_avi_tsr_supp_sign_2_col_name = \
            'vision_avi_tsr_supp_sign_2_name'

        self.vision_avi_tsr_main_sign_col_confidence = \
            'vision_avi_tsr_sign_confidence'
        self.vision_avi_tsr_supp_sign_1_col_confidence = \
            'vision_avi_tsr_supp_sign_1_confidence'
        self.vision_avi_tsr_supp_sign_2_col_confidence = \
            'vision_avi_tsr_supp_sign_2_confidence'

        self.vision_avi_tsr_main_sign_long_dist = \
            'vision_avi_tsr_sign_longitudinal_distance'
        self.vision_avi_tsr_main_sign_lat_dist = \
            'vision_avi_tsr_sign_lateral_distance'

        self.vse_host_long_velocity_col = 'host_longitudinal_velocity_mps'
        # self.vse_host_long_distance_col =

        self.vision_avi_tsr_main_sign_ID_col_name = \
            'vision_avi_tsr_sign_ID_name'

        self.feature_tsi_OW_speed_limit_sign_enum = \
            'feature_tsi_OW_vehicle_speed_limit_sign_enum'
        self.feature_tsi_OW_speed_limit_source_enum = \
            'feature_tsi_OW_source_fused_speed_limit'
        self.feature_tsi_OW_TSR_status = \
            'feature_tsi_OW_traffic_sign_recognition_status'

        # self.can_telematics_speed_limit_enum = \
        #     'can_FD3_telematic_FD11_effective_speed_limit'
        self.can_telematics_speed_limit_enum = \
            'feature_tsi_vnet_IW_effective_speed_limit'

        self.kwargs_processing = {'default_val': [0, pow(2, 16)-1, ]

                                  }
        self.default_vals_ff_20ms = 0

        self.vision_avi_tsr_sign_confidence_limit = 0.4  # [-], percentage

        # [-] periods to shift to calculate host bearing
        self._bearing_period_shift = 5

        self.host_bearing_deg = 'host_bearing_deg'  # [deg]

        self.feature_tsi_OW_speed_limit_supp_sign_enum = \
            'feature_tsi_OW_vehicle_speed_limit_supp_sign_enum'

        self.feature_tsi_OW_overtake_sign_enum = \
            'feature_tsi_OW_overtaking_sign_enum'
        self.feature_tsi_IW_construction_enum = \
            'feature_avi_tsi_construction_area_enum'

    def _main_non_speed_signals_processing(self, ):

        return

    def _avi_SW_version_processing(self, df):

        try:

            df['avi_SW_version'] = \
                str(
                    df['vision_avi_SW_major_version']
                    .mode(dropna=False)[0]) + '.' + str(
                        df['vision_avi_SW_minor_version']
                        .mode(dropna=False)[0]) + '.' + str(
                            math.floor(df['vision_avi_SW_mapping_version']
                                       .mode(dropna=False)[0]/2**8))
        except:

            print('Issue with SW version extraction for AVI')

            df['avi_SW_version'] = 'UNKNOWN'

        return df

    def _traffic_sign_to_host_angle2(self,
                                     longitudinal_distance_array,
                                     lateral_distance_array):

        relative_angle_to_host_array = np.rad2deg(
            np.arctan2(longitudinal_distance_array,
                       lateral_distance_array)
        )

        return relative_angle_to_host_array

    def _traffic_sign_to_host_angle(self, df):

        relative_angle_to_host_series = np.rad2deg(
            np.arctan2(df[self.vision_avi_tsr_main_sign_long_dist],
                       df[self.vision_avi_tsr_main_sign_lat_dist])
        )

        return relative_angle_to_host_series

    def _host_bearing_helper(self, df, ):

        df['geodetic_latitude_deg_lag'] = df[
            'geodetic_latitude_deg'].shift(
                periods=self._bearing_period_shift).bfill()
        df['geodetic_longitude_deg_lag'] = df[
            'geodetic_longitude_deg'].shift(
                periods=self._bearing_period_shift).bfill()

        return_series = df.apply(lambda x:
                                 _get_bearing_from_lat_long(
                                     x['geodetic_latitude_deg_lag'],
                                     x['geodetic_latitude_deg'],
                                     x['geodetic_longitude_deg_lag'],
                                     x['geodetic_longitude_deg']), axis=1)

        return return_series

    def _target_gps_finder(self, df, ):

        df['host_bearing_deg'] = self._host_bearing_helper(
            df[['geodetic_latitude_deg',
                'geodetic_longitude_deg']])

        # df['traffic_sign_relative_angle_deg'] = \
        #     self._traffic_sign_to_host_angle(df[[
        #         self.vision_avi_tsr_main_sign_long_dist,
        #         self.vision_avi_tsr_main_sign_lat_dist]])

        # df['traffic_sign_bearing_deg'] = (
        #     df['host_bearing_deg']
        #     + df['traffic_sign_relative_angle_deg']
        # ) % 360

        # df['traffic_sign_euclidean_distance_m'] = np.sum(
        #     np.square(
        #         [df[self.vision_avi_tsr_main_sign_long_dist],
        #          df[self.vision_avi_tsr_main_sign_lat_dist]]),
        #     axis=0)

        # destination_lat_long_series = \
        #     df.apply(lambda x: distance(
        #         kilometers=x['traffic_sign_euclidean_distance_m'])
        #         .destination((x['geodetic_latitude_deg'],
        #                       x['geodetic_longitude_deg']),
        #                      bearing=x['traffic_sign_bearing_deg'])
        #     )
        # df['traffic_sign_geodetic_latitude_deg'] = [
        #     item.latitude for item in destination_lat_long_series]
        # df['traffic_sign_geodetic_longitude_deg'] = [
        #     item.longitude for item in destination_lat_long_series]

        return df

    def _main_speed_signals_processing(self, ):

        return

    def _time_duration_to_indices_len(self, df, time_duration_sec, ):

        start_index = 0
        start_cTime = float(df.loc[start_index, 'cTime'])
        delta_cTime = time_duration_sec
        end_cTime = start_cTime + delta_cTime
        end_index = int(df[df['cTime']
                           <= end_cTime]['cTime'].idxmax())

        indices_len = end_index - start_index

        return indices_len

    def _signals_processing_confidence_filter(self,
                                              df,
                                              sign_col_name,
                                              confidence_col_name,
                                              **kwargs):

        if 'default_val' in kwargs:
            default_val = kwargs['default_val']
        else:
            default_val = 0

        req_cols = [col for col in df.columns
                    if sign_col_name in col
                    ]

        req_confidence_cols = [col for col in df.columns
                               if confidence_col_name in col
                               ]

        df_req = df.copy(
            # deep=True
        )[
            req_cols +
            req_confidence_cols
        ]

        df_req[df_req[req_confidence_cols] <=
               self.vision_avi_tsr_sign_confidence_limit][req_cols] = default_val[0]

        df[req_cols] = df_req[req_cols].values

        return df

    def _supplementary_signals_processing(self, ):

        return

    def _process_signals(self, out_df, enums_dict, ):

        out_df = self._signals_processing_confidence_filter(
            out_df,
            self.vision_avi_tsr_main_sign_col_name,
            self.vision_avi_tsr_main_sign_col_confidence,
            **self.kwargs_processing
        )

        out_df = self._signals_processing_confidence_filter(
            out_df,
            self.vision_avi_tsr_supp_sign_1_col_name,
            self.vision_avi_tsr_supp_sign_1_col_confidence,
            **self.kwargs_processing
        )

        out_df = self._signals_processing_confidence_filter(
            out_df,
            self.vision_avi_tsr_supp_sign_2_col_name,
            self.vision_avi_tsr_supp_sign_2_col_confidence,
            **self.kwargs_processing
        )

        try:

            out_df = self._target_gps_finder(out_df)

        except:
            print('*************************************\n',
                  'GPS data is absent in log or absent in config file')
            print('creating dummy GPS values')
            out_df['host_bearing_deg'] = 0
            # out_df['geodetic_latitude_deg'] = -999
            # out_df['geodetic_longitude_deg'] = -999

        out_df = self._avi_SW_version_processing(out_df)

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
    program = 'Thunder'  # 'Northstar'
    config_file = 'config_thunder_v1_tsi.yaml'  # 'config_northstar_v1_cut_in.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,  # .parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'TNDR1_MENI_20240808_042406_WDC5_dma_0004.mat')

    config_path = os.path.join(
        Path(os.getcwd()).parent,  # .parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        config_file,

    )

    mat_file_data = loadmat(file_name)
    TSI_signal_data_obj = signalData(mat_file_data)

    df, misc_out_dict = TSI_signal_data_obj.main_get_signals(config_path)

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
