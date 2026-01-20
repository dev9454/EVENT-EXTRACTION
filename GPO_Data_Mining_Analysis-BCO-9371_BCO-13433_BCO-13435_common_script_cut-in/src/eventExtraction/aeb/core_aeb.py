# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:08:59 2024

@author: mfixlz
"""
import sys
from os import path
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
import copy
if __package__ is None:
    print('Here at none package')
    sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.insert(1, path.dirname(path.abspath(__file__)))
    from utils.utils_generic import (read_platform,
                                     loadmat,
                                     stream_check,
                                     transform_df,
                                     merge_pandas_df,
                                     sort_list,
                                     get_all_sw_ver_no_mid_map,
                                     )
    from signal_mapping_aeb import signalMapping
    from get_signals_aeb import signalData

else:

    # from .. import utils
    from utils.utils_generic import (read_platform, loadmat,
                                     stream_check, transform_df,
                                     merge_pandas_df, sort_list,
                                     get_all_sw_ver_no_mid_map, )
    # from . import signal_mapping_aeb
    from signal_mapping_aeb import signalMapping
    from get_signals_aeb import signalData


class coreEventExtractionAEB:

    # def __init__(self) -> None:

    #     pass

    def _helper_enum(self, row,
                     col_name,
                     col_enum_dict,
                     is_aeb_type: bool = True):

        if is_aeb_type:

            return_val = col_enum_dict[row[col_name]]['aeb_type']
        else:
            return_val = col_enum_dict[row[col_name]]['name']

        return return_val

    def extract_ttc(self, data_df,
                    aeb_target_type_arr,
                    event_type_arr,
                    event_properties_df,
                    start_event_indices,
                    df_column):

        value_map_aeb_target_type = \
            {start_index: aeb_target
             for start_index, aeb_target in
             zip(start_event_indices,
                 aeb_target_type_arr)
             }

        value_map_event_type = \
            {start_index: event_type
             for start_index, event_type in
             zip(start_event_indices,
                 event_type_arr)
             }

        event_properties_df['start_indices_enum'] = \
            np.arange(len(event_properties_df), dtype=int)

        event_properties_df['aeb_target_type'] = \
            event_properties_df['start_indices'].apply(
                lambda x: value_map_aeb_target_type.get(x))

        event_properties_df['event_type'] = \
            event_properties_df['start_indices'].apply(
                lambda x: value_map_event_type.get(x))

        event_properties_df['ttc'] = ''

        mapping_dict = {'FCW': 'FCW_TTC',
                        'AEB': 'AEB_TTC',
                        'PEB': 'PEB_Target_TTC', }

        for aeb_target, start_index, start_index_enum in \
            np.array(event_properties_df[['aeb_target_type',
                                          'start_indices',
                                          'start_indices_enum']]):

            if aeb_target == 'OCTAP':
                req_col = 'ICA_OCTAP_TTC_Masked'
            elif aeb_target == 'SCP':
                req_col = 'ICA_SCP_TTC_Masked'
            else:
                req_col = mapping_dict[aeb_target]

            event_properties_df.loc[start_index_enum, 'ttc'] = \
                data_df.loc[start_index,
                            req_col]

        event_properties_df['signal_value'] = \
            data_df.loc[event_properties_df['start_indices'],
                        df_column].values

        event_properties_df['CADM_MID_SW_Version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'cadm_mid_sw_ver'].values
        event_properties_df['CADM_MID_LRR_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'lrr_ver'].values
        event_properties_df['CADM_MID_MRR_FR_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'mrr_fr_ver'].values
        event_properties_df['CADM_MID_MRR_RR_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'mrr_rr_ver'].values
        event_properties_df['CADM_MAP_SW_Version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'cadm_map_sw_ver'].values
        event_properties_df['CADM_MAP_LRR_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'lrr_ver'].values
        event_properties_df['CADM_MAP_MRR_FL_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'mrr_fl_ver'].values
        event_properties_df['CADM_MAP_MRR_RL_version'] = \
            data_df.loc[event_properties_df['start_indices'],
                        'mrr_rl_ver'].values

        # TODO: ADD the prefill cols, other cols in enums dict

        return event_properties_df

    def extract_properties(self, data_df,
                           target_type_map,
                           target_type_dict,
                           event_target_id_arr_,
                           start_event_indices,
                           end_event_indices,
                           event_start_ctime_arr,
                           event_end_ctime_arr,
                           host_long_vel_event_start_arr,
                           host_lat_vel_event_start_arr,
                           host_veh_index_event_start_arr,
                           host_veh_index_event_end_arr,
                           object_class_arr,
                           aeb_target_type_arr,
                           event_type_arr,
                           df_column):
        df_list = []

        target_columns = list(target_type_map.keys())

        req_target_cols = [col for col in target_columns
                           if 'track_id' in col]

        target_types = list(target_type_dict.keys())

        for (event_target_id_arr, start_index, end_index,
             start_ctime, end_ctime, host_long_vel,
             host_lat_vel, host_idx_start, host_idx_end, object_cls) in \
                zip(event_target_id_arr_, start_event_indices, end_event_indices,
                    event_start_ctime_arr, event_end_ctime_arr,
                    host_long_vel_event_start_arr, host_lat_vel_event_start_arr,
                    host_veh_index_event_start_arr, host_veh_index_event_end_arr,
                    object_class_arr,):

            req_df = data_df.loc[start_index:end_index, target_columns]

            out_val = {}

            for col in req_target_cols:
                out_val[col] = req_df.index[np.in1d(req_df[col],
                                                    event_target_id_arr)].tolist()

            out_dict = {key: val
                        for key, val in out_val.items()
                        if len(val) > 0
                        }

            out_dict = {key: max(val)
                        for key, val in out_dict.items()
                        }

            req_cols_root_list = ['vcs_long_posn',
                                  'vcs_lat_posn',
                                  'vcs_long_vel',
                                  'vcs_lat_vel']

            dummy = {target_type: {val
                                   + '_' + target_type: list()
                                   for val in req_cols_root_list
                                   }
                     for target_type in target_types
                     }

            dummy_2 = {target_type: {val: list()
                                     for val in req_cols_root_list
                                     }
                       for target_type in target_types
                       }

            for (target_type_, inner_dict), \
                (target_type_2, inner_dict_2) in zip(dummy.items(),
                                                     dummy_2.items()):

                new_inner_dict = {}

                out_dict_req = {key: val
                                for key, val in out_dict.items()
                                if target_type_ == key.split('_')[-2]}

                # print('Boolen of out_dict_req', bool(
                #     out_dict_req), list(out_dict_req.keys()))

                if not bool(out_dict_req):
                    continue

                out_dict_req = [max(out_dict_req.items(),
                                    key=itemgetter(1))]

                out_dict_req = {k: v for k, v in out_dict_req}

                new_col_list = list(out_dict_req.keys())

                col_enum_list = [col.split('_')[-1]
                                 for col in new_col_list]

                object_type_list = np.array([target_type_ + '_' +
                                            str(int(col_enum) + 1)
                                            for col_enum in col_enum_list])

                indices_list = [out_dict_req['track_id_'
                                             + target_type_ + '_' + col_enum]
                                for col_enum in col_enum_list]

                new_inner_dict = copy.deepcopy(inner_dict_2)

                for (target_col_type, col_list), \
                    (target_col_type_2, _) in zip(inner_dict.items(),
                                                  inner_dict_2.items()):

                    req_col_list_2 = [target_col_type + '_' +
                                      str(int(col_enum) + 1)
                                      for col_enum in col_enum_list]

                    for col_enum, index in zip(col_enum_list, indices_list):

                        new_inner_dict[target_col_type_2]\
                            .extend(req_df.loc[indices_list,
                                    target_col_type + '_'
                                    + col_enum].values)

                new_inner_dict['object_type'] = object_type_list
                new_inner_dict['target_type'] = np.array([
                    target_type_]*len(object_type_list))
                new_inner_dict['start_indices'] = np.array([
                    start_index]*len(object_type_list)).astype(int)
                new_inner_dict['end_indices'] = np.array([
                    end_index]*len(object_type_list)).astype(int)
                new_inner_dict['start_ctimes'] = np.array([
                    start_ctime]*len(object_type_list)).astype(float)
                new_inner_dict['end_ctimes'] = np.array([
                    end_ctime]*len(object_type_list)).astype(float)
                new_inner_dict['host_long_vel'] = np.array([
                    host_long_vel]*len(object_type_list)).astype(float)
                new_inner_dict['host_lat_vel'] = np.array([
                    host_lat_vel]*len(object_type_list)).astype(float)

                new_inner_dict['host_indices_start'] = np.array([
                    host_idx_start]*len(object_type_list)).astype(int)
                new_inner_dict['host_indices_end'] = np.array([
                    host_idx_end]*len(object_type_list)).astype(int)

                new_inner_dict['object_class'] = np.array([
                    object_cls]*len(object_type_list)).astype(int)
                new_inner_dict['object_id'] = np.array([
                    event_target_id_arr]*len(object_type_list)).astype(int)

                df = pd.DataFrame(new_inner_dict)

                df_list.append(df)

        req_out_df = pd.concat(df_list, axis=0, ignore_index=True)

        req_out_df = self.extract_ttc(data_df,
                                      aeb_target_type_arr,
                                      event_type_arr,
                                      req_out_df,
                                      start_event_indices,
                                      df_column)
        # req_out = req_out_df.to_dict(orient='list')

        req_out_df['AEB_Type'] = np.array(
            [event_type
             if '_AEB_Type' in df_column else 'None'
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['Brake_Jerk_Req'] = np.array(
            [event_type
             if '_Jerk' in df_column else 'None'
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['Prefill_Req'] = np.array(
            [event_type
             if 'Prefill' in df_column else 'None'
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['AEB_DispPopupSts'] = np.array(
            [event_type
             if 'AEB_DispPopupSts' in df_column else 'None'
             for event_type in
             req_out_df['event_type'].values]
        )

        # event_type_arr

        req_out_df['Brake Prefill-FCW'] = np.array(
            [event_type
             if 'Brake_Prefill_FCW' in event_type else 0
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['Brake Prefill-PEB'] = np.array(
            [event_type
             if 'Brake_Prefill_PEB' in event_type else 0
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['Brake Prefill-ICA'] = np.array(
            [event_type
             if 'Brake_Prefill_ICA' in event_type else 0
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['ABA-FCW'] = np.array(
            [event_type
             if 'ABA-FCW' in event_type else 0
             for event_type in
             req_out_df['event_type'].values]
        )

        req_out_df['ABA-ICA'] = np.array(
            [event_type
             if 'ABA-ICA' in event_type else 0
             for event_type in
             req_out_df['event_type'].values]
        )

        ###############
        req_out_df['Brake Jerk'] = np.array(
            [signal_value
             if '_Jerk' in df_column else 0
             for signal_value in
             req_out_df['signal_value'].values]
        )

        req_out_df['AEB-Type value'] = np.array(
            [signal_value
             if '_AEB_Type' in df_column else 0
             for signal_value in
             req_out_df['signal_value'].values]
        )

        req_out_df['Warnings'] = np.array(
            [signal_value
             if 'AEB_DispPopupSts' in df_column else 0
             for signal_value in
             req_out_df['signal_value'].values]
        )

        return req_out_df

    def _extract_events(self, data_df, enums_dict, target_type_list):

        data_df = data_df.apply(pd.to_numeric, errors='ignore')

        df_columns = list(data_df.columns)

        req_signals = list(enums_dict.keys())
        jVal = "|".join(req_signals)

        req_df_columns = list(enums_dict.keys())

        target_cols = '|'.join(target_type_list)

        df_cols_split = [col.split("_") for col in df_columns]

        target_type_map = {df_columns[i]: col
                           for col in target_type_list
                           for i, split_col in enumerate(df_cols_split)
                           if col in split_col
                           }

        target_type_dict = {col: list()
                            for col in target_type_list}

        for col in target_type_list:
            for i, split_col in enumerate(df_cols_split):
                if col in split_col:
                    target_type_dict[col].append(df_columns[i])

        all_track_ID_cols = [col for col in df_columns
                             if 'Track_ID' in col]

        event_properties_list = []

        for i, column in enumerate(req_df_columns):

            # print(f'column is : {column}')

            aeb_type_col_name = column + '_aeb_type'
            data_df[aeb_type_col_name] = data_df.apply(
                self._helper_enum, axis=1,
                col_name=column,
                col_enum_dict=enums_dict[column],
                is_aeb_type=True)

            event_type_col_name = column + '_event_type'
            data_df[event_type_col_name] = data_df.apply(
                self._helper_enum, axis=1,
                col_name=column,
                col_enum_dict=enums_dict[column],
                is_aeb_type=False)

            start_event_indices = self.event_start_end_extractor(
                data_df[column].values)
            if start_event_indices.size == 0:
                print(
                    f'Event name : {column}, No events')
                continue
            else:
                print(
                    f'Event name : {column}, \n{enums_dict[column]}\n\n')
            event_exist = True
            end_event_indices = self.event_start_end_extractor(data_df[column].values,
                                                               start_index=False)

            aeb_target_type_arr = \
                data_df.loc[start_event_indices, aeb_type_col_name].values
            event_type_arr = \
                data_df.loc[start_event_indices, event_type_col_name].values

            req_change_idx = [idx
                              for idx, val in
                              enumerate(aeb_target_type_arr)
                              if (val != 'None')]

            start_event_indices = start_event_indices[req_change_idx]
            end_event_indices = end_event_indices[req_change_idx]
            aeb_target_type_arr = aeb_target_type_arr[req_change_idx]

            event_start_ctime_arr = \
                data_df.loc[start_event_indices, 'cTime'].values
            event_end_ctime_arr = \
                data_df.loc[end_event_indices, 'cTime'].values

            host_long_vel_event_start_arr = \
                data_df.loc[start_event_indices, 'host_vcs_long_vel'].values
            host_lat_vel_event_start_arr = \
                data_df.loc[start_event_indices, 'host_vcs_lat_vel'].values
            host_veh_index_event_start_arr = \
                data_df.loc[start_event_indices, 'vse_index'].values    # ???
            host_veh_index_event_end_arr = \
                data_df.loc[end_event_indices, 'vse_index'].values    # ???

            aeb_target_type_ID_cols = [aeb_target + '_Track_ID'
                                       for aeb_target in aeb_target_type_arr
                                       ]

            event_target_id_arr = np.array([data_df.loc[event_start, aeb_target_ID_col]
                                            for event_start, aeb_target_ID_col in
                                            zip(start_event_indices,
                                                aeb_target_type_ID_cols)
                                            ])

            object_class_cols = [aeb_target + '_Fusion_Source'
                                 for aeb_target in aeb_target_type_arr]
            object_class_arr = np.array([data_df.loc[event_start, aeb_object_class]
                                        for event_start, aeb_object_class in
                                        zip(start_event_indices,
                                            object_class_cols)
                                         ])

            event_properties = self.extract_properties(data_df,
                                                       target_type_map,
                                                       target_type_dict,
                                                       event_target_id_arr,
                                                       start_event_indices,
                                                       end_event_indices,
                                                       event_start_ctime_arr,
                                                       event_end_ctime_arr,
                                                       host_long_vel_event_start_arr,
                                                       host_lat_vel_event_start_arr,
                                                       host_veh_index_event_start_arr,
                                                       host_veh_index_event_end_arr,
                                                       object_class_arr,
                                                       aeb_target_type_arr,
                                                       event_type_arr,
                                                       column)
            # object_type_arr = event_properties['object_type']
            # vcs_long_posn_arr = event_properties['vcs_long_posn']
            # vcs_lat_posn_arr = event_properties['vcs_lat_posn']
            # vcs_long_vel_arr = event_properties['vcs_long_vel']
            # vcs_lat_vel_arr = event_properties['vcs_lat_vel']

            # target_type_arr = event_properties['target_type']

            # ttc_arr = event_properties['ttc']

            print('reached_end')

            event_properties_list.append(event_properties)

        event_properties_df = pd.concat(event_properties_list,
                                        axis=0, ignore_index=True)

        return event_properties_df

    @staticmethod
    def event_start_end_extractor(array: np.array,
                                  start_index: bool = True) -> np.array:
        """
        Function to extract indexes where value is different than zero or change to different than zero value
        @param array: Given array
        @param start_index: indicate whether extract start index or end index (start is default)
        @return: Array with event indices
        """
        array = array.reshape((-1, 1))
        if start_index:
            # array with additional row at the beginning
            tmp_arr = np.concatenate((np.zeros((1, 1)), array), axis=0)
        else:
            # array with additional row at the end
            tmp_arr = np.concatenate((array, np.zeros((1, 1))), axis=0)
        tmp_arr_diff = np.diff(tmp_arr[:, 0])
        # array with indices where signal starts change (raising or dropping)
        edge_idx = np.where(np.logical_or(
            tmp_arr_diff < 0, tmp_arr_diff > 0) == True)[0]
        # array with indices where signal starts change (raising or dropping) and different than zero
        edge_idx = edge_idx[(array[edge_idx] != 0).flatten()]

        return edge_idx


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
        'FCAWL_20220110_WL75DMX_RWUP_CD_DC_DC_MF_IC_100734_073.mat')
    mat_file_data = loadmat(file_name_resim)

    signal_data_obj = signalData(mat_file_data)

    # signal_mapping_dict, enum_mapping_dict, target_type_list = \
    #     signal_data_obj.get_signal_mapping()
    # df_dict = signal_data_obj.get_signals_dict(signal_mapping_dict)

    # df_merged = merge_pandas_df(list(df_dict.values()))

    (df_merged,
     enum_mapping_dict,
     target_type_list) = signal_data_obj.main()

    log_path, log_name = os.path.split(file_name_resim)

    df_merged.insert(0, 'log_path', log_path)
    df_merged.insert(1, 'log_name', log_name)

    core_aeb_obj = coreEventExtractionAEB()

    out_df_list = core_aeb_obj._extract_events(df_merged,
                                               enum_mapping_dict,
                                               target_type_list)
