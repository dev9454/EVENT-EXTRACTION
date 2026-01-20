# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:42:58 2024

@author: mfixlz
"""
import sys
from os import path
from pathlib import Path
import os
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
from collections.abc import Iterable
import itertools
if __package__ is None:

    print('Here at none package 1')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    actual_package_path = to_change_path
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, \n to change 1: {to_change_path}')
    from signal_mapping_cut_in import signalMapping
    print('Here at none package 2')
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(to_change_path)
    print(f'Current dir 2: {os.getcwd()}, to change 2: {to_change_path}')

    from utils.utils_generic import (
        loadmat,
        _calc_derivative

    )
    os.chdir(actual_package_path)


else:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(to_change_path)
    print(f'Current dir 1: {os.getcwd()}, to change 1: {to_change_path}')

    from signal_mapping_cut_in import signalMapping

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

                                                         )
    except:
        from utils.utils_generic import (loadmat,
                                         _calc_derivative,

                                         )


class signalData(signalMapping):

    def __init__(self, raw_data) -> None:

        # super().__init__(self, raw_data)
        signalMapping.__init__(self, raw_data)

        # self.raw_data = raw_data
        self.smoothen_y_data = True
        self.smoothen_window_len_sec = 0.5

        self.target_mapping = {'target_lateral_position_m':
                               'target_lateral_position_m_VCS',
                               'target_lateral_velocity_mps':
                               'target_lateral_velocity_mps_VCS',
                               'target_lateral_acceleration_mps2':
                               'target_lateral_acceleration_mps2_VCS',
                               'target_longitudinal_position_m':
                               'target_longitudinal_position_m_VCS',
                               'target_longitudinal_velocity_mps':
                               'target_longitudinal_velocity_mps_VCS',
                               'target_longitudinal_acceleration_mps2':
                               'target_longitudinal_acceleration_mps2_VCS',
                               'target_width_m':
                               'target_width_m',
                               'target_length_m':
                               'target_length_m',
                               'target_heading_rad':
                               'target_heading_rad_VCS',
                               }
        self.target_signals_mapping = {
            'LA_Line_Side_eyeQ': {6: "LEFT_NEXT_NEXT",
                                  5: "RIGHT_NEXT_NEXT",
                                  4: "RIGHT_RIGHT_LANEMARK",
                                  3: "RIGHT_LEFT_LANEMARK",
                                  2: "LEFT_RIGHT_LANEMARK",
                                  1: "LEFT_LEFT_LANEMARK",
                                  0: "NONE",
                                  },
            'LA_Line_Side_AVI': {
                255: "INVALID",
                10: "NEXT_RIGHT_LEFT_LANEMARK",
                9: "NEXT_RIGHT_RIGHT_LANEMARK",
                8: "RIGHT_LEFT_LANEMARK",
                7: "RIGHT_RIGHT_LANEMARK",
                6: "NEXT_LEFT_LEFT_LANEMARK",
                5: "NEXT_LEFT_RIGHT_LANEMARK",
                4: "LEFT_LEFT_LANEMARK",
                3: "LEFT_RIGHT_LANEMARK",
                2: "RIGHT",
                1: "LEFT",
                0: "NONE",
            }

        }

        required_roles = [
            # "LEFT_RIGHT_LANEMARK",
            #   "LEFT_LEFT_LANEMARK",
            #   "RIGHT_RIGHT_LANEMARK",
            #   "RIGHT_LEFT_LANEMARK",
            "LEFT",
            "RIGHT",
            #   "NONE",
            #   "INVALID",
        ]

        self.roles_mapping_avi = {val: key
                                  for key, val in
                                  self.target_signals_mapping['LA_Line_Side_AVI'].items(
                                  )
                                  if val in required_roles
                                  }

        self.host_signals_mapping = {
            'LH_Side': {2: "RIGHT",
                        1: "LEFT",
                        0: "UNKNOWN",
                        }
        }

        self.vision_host_confidence_limit = 0.2  # [-], percentage

        self.target_time_to_approach_lane_limit = 0.1  # [s], seconds
        self.target_to_lane_marker_distance_threshold = 0.5  # [m]
        self.target_condition_target_lat_vel_flag = 0.5
        self.target_condition_target_lat_accel_flag = 0

        self.host_condition_host_long_accel_flag = 0
        self.duration_check_for_host_lane_change_sec = 0.25
        self.min_host_to_lane_marker_gap = 0.3
        self.max_host_to_lane_marker_gap = 3.0
        self.host_ALC_type_feature_disabled_enum = 0
        self.vision_host_lane_not_changing_enum = 0

        self.host_duration_to_maintain_sec = 0.25  # [s]

        self.host_long_position_linearised = True
        self.ttc_order = 2

    def _time_duration_to_indices_len(self, df, time_duration_sec, ):

        start_index = 0
        start_cTime = float(df.loc[start_index, 'cTime'])
        delta_cTime = time_duration_sec
        end_cTime = start_cTime + delta_cTime
        end_index = int(df[df['cTime']
                           <= end_cTime]['cTime'].idxmax())

        indices_len = end_index - start_index

        return indices_len

    def _helper_target_id_change_from(self, row,

                                      ):

        index = row.name

        # left_id = row['target_ID_2']
        # right_id = row['target_ID_3']

        if index == 0:

            return np.nan

        elif row['RT3_to_RT1_left_check']:  # :??? check here

            return_val = 'from_left'

        elif row['RT4_to_RT1_right_check']:

            return_val = 'from_right'

        else:

            return np.nan

        return return_val

    def _helper_target_properties(self, row, property_col: str, ID_col: str):

        index = row.name

        req_col_relobj = row[ID_col]-1

        if np.isnan(req_col_relobj) or req_col_relobj < 0:
            return np.nan

        property_val = row[property_col + '_' + str(int(req_col_relobj))]

        return property_val

    def _helper_host_to_lane_distance(self, row, left_right_enum,
                                      is_host_to_lane: bool = False):

        index = row.name

        host_width = row['host_width_m']

        if left_right_enum == 1:

            req_col = row['lanes_req_col_LEFT']

        elif left_right_enum == 2:

            req_col = row['lanes_req_col_RIGHT']
            host_width = -host_width

        if np.isnan(req_col):

            return np.nan
        else:
            req_col = str(int(req_col))

        host_C0 = row['vision_host_lane_parameters_C0_' + req_col]
        signal_confidence = row['vision_host_signal_confidence_' + req_col]

        if signal_confidence >= self.vision_host_confidence_limit:

            if is_host_to_lane:

                return host_C0

            host_to_lane_distance = 0.5*host_width + host_C0
        else:
            if is_host_to_lane:

                return np.nan
            host_to_lane_distance = np.nan  # 0.5*host_width + host_C0

        return host_to_lane_distance

    def _aptiv_vision_interface_signals(self, df, ):

        # req roles: left left : 4, left right : 3, right right : 7, right left : 8

        role_cols = [col for col in df.columns
                     if (
                         #      'vision_lane_side_from_host' in col
                         #  or
                         'vision_host_lane_side' in col
                     )]

        roles_array = df[role_cols].to_numpy()

        lane_side_from_host_indices_dict = {}

        for key, val in self.roles_mapping_avi.items():

            df_key = 'lanes_req_col_' + key

            df[df_key] = np.nan

            req_indices = np.where(roles_array == val)

            df.loc[req_indices[0], df_key] = req_indices[1]

            lane_side_from_host_indices_dict[key] = np.where(
                roles_array == val)

        return df

    def _host_lane_change(self, df, ):

        indices_host_closing_left = \
            df.query('host_lateral_distance_left_wheel_to_left_lane_marker '
                     + f'<= {self.min_host_to_lane_marker_gap} '
                     + 'or host_lateral_distance_right_wheel_to_right_lane_marker '
                     + '>= 0.5*vision_host_lane_width'
                     ).index
        indices_host_closing_right = \
            df.query('host_lateral_distance_right_wheel_to_right_lane_marker '
                     + f'<= {self.min_host_to_lane_marker_gap} '
                     + 'or host_lateral_distance_left_wheel_to_left_lane_marker '
                     + '>= 0.5*vision_host_lane_width'
                     ).index

        self.duration_check_for_host_lane_change = \
            self._time_duration_to_indices_length(df,
                                                  self.duration_check_for_host_lane_change_sec)

        # self.duration_check_for_host_lane_change, use lines 1175-1255 in find_cutin_generic
        indices_gap_left = pd.Series(indices_host_closing_left)
        start_end_indices_left = indices_gap_left.groupby(indices_gap_left.diff().ne(1)
                                                          .cumsum()).apply(lambda x:
                                                                           [x.iloc[0],
                                                                               x.iloc[-1]]
                                                                           if len(x) >= 2
                                                                           else [x.iloc[0]]).tolist()
        start_end_indices_left = [indices
                                  for indices in start_end_indices_left
                                  if len(indices) > 1]

        indices_length_left = [abs(indices[1] - indices[0])
                               for indices in start_end_indices_left]
        req_indices_left = [left_indices for left_len, left_indices in
                            zip(indices_length_left,
                                start_end_indices_left)
                            if left_len >= self.duration_check_for_host_lane_change
                            ]

        req_indices_left = [np.sort(item).tolist()
                            for item in req_indices_left]

        indices_left_event_start = [indices[0] for indices in req_indices_left]

        req_indices_left = [list(range(indices[0], indices[1]+1))
                            for indices in req_indices_left]
        req_indices_left = list(
            itertools.chain.from_iterable(req_indices_left))

        # RIGHT
        indices_gap_right = pd.Series(indices_host_closing_right)
        start_end_indices_right = indices_gap_right.groupby(indices_gap_right.diff().ne(1)
                                                            .cumsum()).apply(lambda x:
                                                                             [x.iloc[0],
                                                                              x.iloc[-1]]
                                                                             if len(x) >= 2
                                                                             else [x.iloc[0]]).tolist()
        start_end_indices_right = [indices
                                   for indices in start_end_indices_right
                                   if len(indices) > 1]

        indices_length_right = [abs(indices[1] - indices[0])
                                for indices in start_end_indices_right]
        req_indices_right = [right_indices for right_len, right_indices in
                             zip(indices_length_right,
                                 start_end_indices_right)
                             if right_len >= self.duration_check_for_host_lane_change
                             ]

        req_indices_right = [np.sort(item).tolist()
                             for item in req_indices_right]

        indices_right_event_start = [indices[0]
                                     for indices in req_indices_right]

        req_indices_right = [list(range(indices[0], indices[1]+1))
                             for indices in req_indices_right]
        req_indices_right = list(
            itertools.chain.from_iterable(req_indices_right))

        indices_to_exclude = req_indices_left + req_indices_right

        return (indices_to_exclude,
                indices_left_event_start,
                indices_right_event_start,
                )

    def _helper_target_to_host_lane_distance(self,
                                             row,
                                             left_right_enum: int = None,
                                             for_confidence: bool = False,
                                             for_min_host_target_distance: bool = False):

        index = row.name

        host_width = row['host_width_m']

        # left = True if row['target_ID_change_from'] == 'from_left' else False
        # right = True if row['target_ID_change_from'] == 'from_right' else False

        if left_right_enum == 1:

            req_col = row['lanes_req_col_LEFT']

            long_dist_target = row['target_longitudinal_position_m_RT3']
            lat_dist_target = row['target_lateral_position_m_RT3']
            target_width = row['target_width_m_RT3']
            target_length = row['target_length_m_RT3']

            if for_min_host_target_distance:

                target_to_host_min_distance = (lat_dist_target
                                               - 0.5*target_width
                                               - 0.5*host_width)
                return target_to_host_min_distance

            if np.isnan(req_col):

                return np.nan
            else:
                req_col = str(int(req_col))

            host_C0 = row['vision_host_lane_parameters_C0_' + req_col]
            host_C1 = row['vision_host_lane_parameters_C1_' + req_col]
            host_C2 = row['vision_host_lane_parameters_C2_' + req_col]
            host_C3 = row['vision_host_lane_parameters_C3_' + req_col]
            target_width = -1*target_width
            lat_dist_target = -1*lat_dist_target
            signal_confidence = row['vision_host_signal_confidence_' + req_col]
        elif left_right_enum == 2:

            req_col = row['lanes_req_col_RIGHT']

            long_dist_target = row['target_longitudinal_position_m_RT4']
            lat_dist_target = row['target_lateral_position_m_RT4']
            target_width = row['target_width_m_RT4']
            target_length = row['target_length_m_RT4']

            if for_min_host_target_distance:

                target_to_host_min_distance = (lat_dist_target
                                               - 0.5*target_width
                                               - 0.5*host_width)
                return target_to_host_min_distance

            if np.isnan(req_col):

                return np.nan
            else:
                req_col = str(int(req_col))

            host_C0 = row['vision_host_lane_parameters_C0_' + req_col]
            host_C1 = row['vision_host_lane_parameters_C1_' + req_col]
            host_C2 = row['vision_host_lane_parameters_C2_' + req_col]
            host_C3 = row['vision_host_lane_parameters_C3_' + req_col]
            signal_confidence = row['vision_host_signal_confidence_' + req_col]
        else:

            return np.nan

        if for_confidence:

            return signal_confidence

        lat_dist_host_to_lane_at_target = (host_C0
                                           + host_C1 *
                                           (long_dist_target + 0.5*target_length)
                                           + host_C2 *
                                           (long_dist_target +
                                            0.5*target_length)**2
                                           + host_C3 *
                                           (long_dist_target +
                                            0.5*target_length)**3
                                           )

        # if signal_confidence >= self.vision_host_confidence_limit:

        #     lat_dist_host_to_lane_at_target = (host_C0
        #                                     + host_C1*long_dist_target
        #                                     + host_C2*long_dist_target**2
        #                                     + host_C3*long_dist_target**3
        #                                     )
        # else:

        #     lat_dist_host_to_lane_at_target = (host_C0
        #                                     + host_C1*long_dist_target
        #                                     + host_C2*long_dist_target**2
        #                                     + host_C3*long_dist_target**3
        #                                     ) - (
        #                                         self.target_time_to_approach_lane_limit*
        #                                         row['target_lateral_velocity_mps']
        #                                     )

        lat_dist_target_to_lane = (lat_dist_target
                                   - 0.5*target_width
                                   - lat_dist_host_to_lane_at_target
                                   )

        return lat_dist_target_to_lane

    def _helper_host_lane_change(self, row, left_right_enum):

        index = row.name

        if left_right_enum == 1:

            req_col = row['lanes_req_col_LEFT']

        elif left_right_enum == 2:

            req_col = row['lanes_req_col_RIGHT']

        if np.isnan(req_col):

            return np.nan
        else:
            req_col = str(int(req_col))

        signal_confidence = row['vision_host_signal_confidence_' + req_col]

        if signal_confidence >= self.vision_host_confidence_limit:

            host_lane_change_status_vision = row['vision_host_lane_change_' + req_col]
        else:
            host_lane_change_status_vision = np.nan  # 0.5*host_width + host_C0

        return host_lane_change_status_vision

    def _time_duration_to_indices_length(self,
                                         out_df,
                                         time_duration):

        start_index = 0
        start_cTime = float(out_df.loc[start_index, 'cTime'])
        delta_cTime = time_duration
        end_cTime = start_cTime + delta_cTime
        end_index = int(out_df[out_df['cTime']
                               <= end_cTime]['cTime'].idxmax())

        indices_length = end_index - start_index

        return indices_length

    def _helper_time_to_collison2(self, row,
                                  is_headway: bool = False,

                                  ):

        index = row.name
        long_pos = row['target_longitudinal_position_m']

        long_vel_host = row['host_longitudinal_velocity_mps']
        long_accel_host = row['host_longitudinal_acceleration_mps2']

        if not self.host_long_position_linearised:

            # print('\ncurve length based distance not implemented',
            #       ' in currently\n'
            #       )
            total_distance = long_pos
        else:
            total_distance = long_pos

        if is_headway:
            rt1_long_vel = 0.0
            rt1_long_accel = 0.0

        else:

            rt1_long_vel = row['target_longitudinal_velocity_mps']
            rt1_long_accel = row['target_longitudinal_acceleration_mps2']

        relative_vel = long_vel_host - rt1_long_vel
        relative_accel = long_accel_host - rt1_long_accel

        if self.ttc_order == 1:
            if relative_vel < 0:
                return_val = np.inf
            else:

                return_val = total_distance / (relative_vel + 1E-16)
        elif self.ttc_order == 2:
            try:

                if (np.power(relative_vel, 2)
                        - 4*(0.5 * relative_accel)*(-total_distance)) < 0:
                    return_val = np.inf
                else:

                    return_val = np.roots([0.5 * relative_accel,
                                           relative_vel,
                                           -total_distance])

                    return_val = return_val[return_val > 0]

                    if len(return_val) > 1:
                        return_val = np.min(return_val)
            except:
                # print('&&&&&&&&&&&&&&&&&&&&&&&&&&')
                # print('TTC with 2nd order failed. going with 1st order value')
                if relative_vel < 0:
                    return_val = np.inf
                else:

                    return_val = total_distance / (relative_vel + 1E-16)

        return return_val  # np.abs(return_val)

    def _process_signals_host(self,
                              out_df,
                              smoothen_window_len):

        out_df['host_comp_steering_angle_deg'] = -999  # FIXME

        misc_out_dict = {}

        # start_index = 0
        # start_cTime = float(out_df.loc[start_index, 'cTime'])
        # delta_cTime = self.host_duration_to_maintain_sec
        # end_cTime = start_cTime + delta_cTime
        # end_index = int(out_df[out_df['cTime']
        #                        <= end_cTime]['cTime'].idxmax())

        # misc_out_dict['host_duration_to_maintain'] = end_index - start_index

        misc_out_dict['host_duration_to_maintain'] = \
            self._time_duration_to_indices_length(out_df,
                                                  self.host_duration_to_maintain_sec)

        out_df['host_longitudinal_jerk_mps3'] = _calc_derivative(
            out_df['host_longitudinal_acceleration_mps2'].values,
            out_df['cTime'].values,
            self.smoothen_y_data,
            smoothen_window_len
        )
        out_df['host_lateral_jerk_mps3'] = _calc_derivative(
            out_df['host_lateral_acceleration_mps2'].values,
            out_df['cTime'].values,
            self.smoothen_y_data,
            smoothen_window_len
        )

        # ??? assumption is camera mounted at Front Center
        # ??? asmption left side negative, right side positive
        # out_df['host_lateral_distance_left_wheel_to_left_lane_marker'] = \
        #     (0.5*out_df['host_width_m']
        #      + out_df['vision_host_lane_parameters_C0_0']).mask(
        #          (out_df['vision_host_signal_confidence_0'] <
        #           self.vision_host_confidence_limit)
        #         # | (out_df['vision_host_lane_parameters_C0_0'] == 0)
        # )\
        #     .interpolate(method='pchip', fill_value='extrapolate')\
        #     .bfill(downcast='infer').abs()

        out_df['host_lateral_distance_left_wheel_to_left_lane_marker'] = out_df.apply(
            self._helper_host_to_lane_distance,
            axis=1,
            left_right_enum=1
        )

        out_df['host_lateral_distance_to_left_lane'] = out_df.apply(
            self._helper_host_to_lane_distance,
            axis=1,
            left_right_enum=1,
            is_host_to_lane=True,
        )
        # out_df['host_lateral_distance_right_wheel_to_right_lane_marker'] = \
        #     (-0.5*out_df['host_width_m']
        #      + out_df['vision_host_lane_parameters_C0_1']).mask(
        #          (out_df['vision_host_signal_confidence_1'] <
        #           self.vision_host_confidence_limit)
        #         # | (out_df['vision_host_lane_parameters_C0_1'] == 0)
        # )\
        #     .interpolate(method='pchip', fill_value='extrapolate')\
        #     .bfill(downcast='infer').abs()

        out_df['host_lateral_distance_right_wheel_to_right_lane_marker'] = out_df.apply(
            self._helper_host_to_lane_distance,
            axis=1,
            left_right_enum=2
        )

        out_df['host_lateral_distance_to_right_lane'] = out_df.apply(
            self._helper_host_to_lane_distance,
            axis=1,
            left_right_enum=2,
            is_host_to_lane=True,
        )

        out_df['vision_host_lane_change_left'] = out_df.apply(
            self._helper_host_lane_change,
            axis=1,
            left_right_enum=1
        )

        out_df['vision_host_lane_change_right'] = out_df.apply(
            self._helper_host_lane_change,
            axis=1,
            left_right_enum=2
        )

        vision_host_lane_change_left_bool = np.where(
            (out_df['vision_host_lane_change_left'] == 0) |
            (out_df['vision_host_lane_change_left'] == 1),
            out_df['vision_host_lane_change_left'],
            0
        )
        vision_host_lane_change_right_bool = np.where(
            (out_df['vision_host_lane_change_right'] == 0) |
            (out_df['vision_host_lane_change_right'] == 1),
            out_df['vision_host_lane_change_right'],
            0
        )
        out_df['vision_host_lane_change'] = np.logical_or(vision_host_lane_change_left_bool,
                                                          vision_host_lane_change_right_bool)

        indices_to_exclude, \
            indices_left_event_start, \
            indices_right_event_start = self._host_lane_change(out_df, )

        out_df['is_host_turning'] = False

        host_turning_indices = out_df.query('road_curvature.abs() <  ' +
                                            f'{self.curvature_limit_straight_path} ' +
                                            'and ' +
                                            'road_curvature_signal_confidence > ' +
                                            f'{self.vision_host_confidence_limit}' +
                                            'and '
                                            'host_yaw_rate_rps.abs() > ' +
                                            f'{self.abs_yaw_rate_limit_straight_path}').index

        out_df.loc[host_turning_indices, 'is_host_turning'] = True

        misc_out_dict['host_initiated_cutout_start_list'] = [indices_left_event_start,
                                                             indices_right_event_start
                                                             ]

        out_df['host_veh_cut_out_initiated'] = False

        out_df.loc[indices_to_exclude, 'host_veh_cut_out_initiated'] = True

        # ALC_type_feature_enum
        out_df['is_alc_disabled'] = False

        indices_alc_disabled = out_df.query('ALC_type_feature_enum == '
                                            + f'{self.host_ALC_type_feature_disabled_enum}'
                                            ).index

        out_df.loc[indices_alc_disabled,
                   'is_alc_disabled'] = True

        out_df['is_host_not_lane_changing'] = False

        indices_host_not_changing_lanes = out_df.query('vision_host_lane_change_left ==  '
                                                       + f'{self.vision_host_lane_not_changing_enum}'
                                                       + 'or '
                                                       + 'vision_host_lane_change_right == '
                                                       + f'{self.vision_host_lane_not_changing_enum}'
                                                       ).index

        out_df.loc[indices_host_not_changing_lanes,
                   'is_host_not_lane_changing'] = True

        out_df['is_host_decelerating'] = False

        indices_host_decelerating = out_df.query('host_longitudinal_acceleration_mps2 < 0'
                                                 ).index

        out_df.loc[indices_host_decelerating,
                   'is_host_decelerating'] = True

        out_df['host_long_accel_flag'] = False

        indices_CSCSA_36417 = out_df.query('host_longitudinal_acceleration_mps2 > '
                                           + f'{self.host_condition_host_long_accel_flag}'
                                           ).index

        out_df.loc[indices_CSCSA_36417, 'host_long_accel_flag'] = True

        return out_df, misc_out_dict

    def _process_signals_target(self,
                                out_df,
                                smoothen_window_len):

        misc_out_dict = {}

        to_del_cols = [col for col in out_df.columns
                       if 'target_ID' in col]

        out_df['target_ID_RT3'] = pd.to_numeric(out_df['moving_target_ID_SATA_2'],
                                                downcast='integer')

        for target_df_col, target_property_col in self.target_mapping.items():

            out_df[target_df_col + '_RT3'] = out_df.apply(self._helper_target_properties,
                                                          axis=1,
                                                          property_col=target_property_col,
                                                          ID_col='target_ID_RT3'
                                                          )

        out_df['target_ID_RT4'] = pd.to_numeric(out_df['moving_target_ID_SATA_3'],
                                                downcast='integer')

        for target_df_col, target_property_col in self.target_mapping.items():

            out_df[target_df_col + '_RT4'] = out_df.apply(self._helper_target_properties,
                                                          axis=1,
                                                          property_col=target_property_col,
                                                          ID_col='target_ID_RT4'
                                                          )

        misc_out_dict['rt3_target_unique_ids'] = list(
            out_df['target_ID_RT3'].unique())
        misc_out_dict['rt4_target_unique_ids'] = list(
            out_df['target_ID_RT4'].unique())

        unique_target_ids = \
            misc_out_dict['rt3_target_unique_ids'] \
            + misc_out_dict['rt4_target_unique_ids']

        left_shift = out_df['moving_target_ID_SATA_2'].shift(periods=1)
        right_shift = out_df['moving_target_ID_SATA_3'].shift(periods=1)

        rt1_id_series = out_df['moving_target_ID_SATA_0']

        # check_vals = left_shift == rt1_id_series
        # out_df['RT3_to_RT1_left_check'] = check_vals != 0

        check_vals = (left_shift == rt1_id_series) & (
            left_shift != 0) & (rt1_id_series != 0)
        out_df['RT3_to_RT1_left_check'] = check_vals

        # check_vals = right_shift == rt1_id_series
        # out_df['RT4_to_RT1_right_check'] = check_vals != 0

        check_vals = (right_shift == rt1_id_series) & (
            right_shift != 0) & (rt1_id_series != 0)
        out_df['RT4_to_RT1_right_check'] = check_vals

        out_df['target_ID_change_from'] = out_df.apply(
            self._helper_target_id_change_from,
            axis=1, )

        out_df['right_target_to_lane_marker_m'] = out_df.apply(
            self._helper_target_to_host_lane_distance,
            axis=1,
            left_right_enum=2
        )

        out_df['left_target_to_lane_marker_m'] = out_df.apply(
            self._helper_target_to_host_lane_distance,
            axis=1,
            left_right_enum=1
        )

        out_df['target_to_left_lane_marker_conf'] = out_df.apply(
            self._helper_target_to_host_lane_distance,
            axis=1,
            left_right_enum=1,
            for_confidence=True
        )

        out_df['target_to_right_lane_marker_conf'] = out_df.apply(
            self._helper_target_to_host_lane_distance,
            axis=1,
            left_right_enum=2,
            for_confidence=True
        )

        out_df['left_target_to_host_min_distance'] = out_df.apply(self._helper_target_to_host_lane_distance,
                                                                  axis=1,
                                                                  left_right_enum=1,
                                                                  for_min_host_target_distance=True)
        out_df['right_target_to_host_min_distance'] = out_df.apply(self._helper_target_to_host_lane_distance,
                                                                   axis=1,
                                                                   left_right_enum=2,
                                                                   for_min_host_target_distance=True)

        out_df['target_to_host_min_distance'] = out_df[['left_target_to_host_min_distance',
                                                        'right_target_to_host_min_distance']].min(axis=1)

        out_df['target_ID'] = pd.to_numeric(out_df['moving_target_ID_SATA_0'],
                                            downcast='integer')
        out_df = out_df.drop(columns=to_del_cols, )

        misc_out_dict['rt1_target_unique_ids'] = list(
            out_df['target_ID'].unique())

        unique_target_ids = unique_target_ids + \
            misc_out_dict['rt1_target_unique_ids']

        self.exception_to_delete_cols = ['target_heading_rad',
                                         ]

        for target_df_col, target_property_col in self.target_mapping.items():

            to_del_cols = [col for col in out_df.columns
                           if target_property_col in col]

            if any(target_df_col in mystring
                    for mystring in self.exception_to_delete_cols):

                save_exception_cols = [target_df_col + '_VCS_' + str(int(id_)-1)
                                       for id_ in unique_target_ids
                                       if (id_ != 0 and not np.isnan(id_))
                                       ]

                to_del_cols = list(set(to_del_cols) - set(save_exception_cols))

            out_df[target_df_col] = out_df.apply(self._helper_target_properties,
                                                 axis=1,
                                                 property_col=target_property_col,
                                                 ID_col='target_ID'
                                                 )

            # out_df = out_df.drop(columns=to_del_cols, )  # ??? check if needed
        # ??? assumption is target coordinates at Body Center

        out_df['target_time_to_collison_2_order'] = out_df.apply(self._helper_time_to_collison2,
                                                                 axis=1,

                                                                 )

        out_df['target_headway_2_order'] = out_df.apply(self._helper_time_to_collison2,
                                                        axis=1,
                                                        is_headway=True,

                                                        )
        self.host_long_position_linearised = False
        out_df['target_time_to_collison_interpolated'] = out_df.apply(self._helper_time_to_collison2,
                                                                      axis=1,

                                                                      )
        out_df['target_headway_interpolated'] = out_df.apply(self._helper_time_to_collison2,
                                                             axis=1,
                                                             is_headway=True,

                                                             )

        out_df['target_time_to_cross'] = out_df['target_to_host_min_distance'].abs() / \
            (out_df['target_lateral_velocity_mps'] -
             out_df['host_lateral_velocity_mps']).abs()

        out_df['target_closing_in_to_host_lane_flag'] = False

        indices_CSCSA_87591 = out_df.query('(left_target_to_lane_marker_m < '
                                           + f'{self.target_to_lane_marker_distance_threshold} '
                                           + 'and not left_target_to_lane_marker_m.isnull())'
                                           + 'or '
                                           + '(right_target_to_lane_marker_m < '
                                           + f'{self.target_to_lane_marker_distance_threshold} '
                                           + 'and not right_target_to_lane_marker_m.isnull())'
                                           ).index
        out_df.loc[indices_CSCSA_87591,
                   'target_closing_in_to_host_lane_flag'] = True

        out_df['target_lat_vel_flag'] = False

        indices_CSCSA_36292 = out_df.query('target_lateral_velocity_mps > '
                                           + f'{self.target_condition_target_lat_vel_flag}'
                                           ).index

        out_df.loc[indices_CSCSA_36292, 'target_lat_vel_flag'] = True

        out_df['target_lat_accel_flag'] = False

        indices_CSCSA_36295 = out_df.query('target_lateral_acceleration_mps2 > '
                                           + f'{self.target_condition_target_lat_accel_flag}'
                                           ).index

        out_df.loc[indices_CSCSA_36295, 'target_lat_accel_flag'] = True

        return out_df, misc_out_dict

    def _process_signals(self, out_df):

        smoothen_window_len = self._time_duration_to_indices_len(
            out_df[['cTime', ]],
            self.smoothen_window_len_sec,
        )

        out_df = self._aptiv_vision_interface_signals(out_df, )

        out_df, misc_out_dict_host = self._process_signals_host(out_df,
                                                                smoothen_window_len)
        out_df, misc_out_dict_target = self._process_signals_target(out_df,
                                                                    smoothen_window_len)

        misc_out_dict = {**misc_out_dict_host,
                         **misc_out_dict_target
                         }

        # out_df = out_df.apply(pd.to_numeric, axis=1, errors='ignore')
        return out_df, misc_out_dict

    def get_signal_mapping(self, config_path, ):

        out_df = self._signal_mapping(config_path)

        return out_df

    def main_get_signals(self, config_path):

        # signal_data_obj = signalData(mat_file_data)
        out_df = self.get_signal_mapping(config_path)

        out_df, misc_out_dict = self._process_signals(out_df)

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
    config_file = 'config_thunder_cut_in.yaml'  # 'config_northstar_v1_cut_in.yaml'

    file_name = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data',
        program,
        'extracted_data',
        'TNDR1_MENI_20240808_042406_WDC5_dma_0004.mat')

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
    CUTIIN_signal_data_obj = signalData(mat_file_data)

    df, misc_out_dict = CUTIIN_signal_data_obj.main_get_signals(config_path)

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
