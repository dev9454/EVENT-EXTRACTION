# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:09:10 2023

@author: ioiny8
"""

import os
import sys
import numpy as np
import time
import pandas as pd
from datetime import timedelta

from datetime import datetime
from datetime import timedelta
from openpyxl import load_workbook, Workbook
from functools import reduce


def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
               [(t*1000,), 1000, 60, 60])


if "win" in sys.platform:
    print('platform : ', sys.platform)
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import contextlib
    import joblib
    sys.path.insert(
        0, r'C:\Users\mfixlz\OneDrive - Aptiv\Documents\DM_A\PO_Chaitanya_K\Projects\GPO Data Mining Analysis\GPO_Data_Mining_Analysis\src')

else:
    package_path_NA = \
        r'/mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/CES_related/GPO_event_extraction/GPO_Data_Mining_Analysis/src'
    package_path_EU = \
        r"/net/8k3/e0fs01/irods/PLKRA-PROJECTS/STRADVISION/8-Users/DMA_Team/mfixlz/GPO_data_mining_project_GITLAB/GPO_Data_Mining_Analysis/src"

    package_path = package_path_NA
    if (os.path.isdir(package_path_NA,)
            and os.access(package_path_NA, os.R_OK)
            and os.access(package_path_NA, os.W_OK)
            and os.access(package_path_NA, os.X_OK)
        ):

        package_path = package_path_NA

    elif (os.path.isdir(package_path_EU,)
          and os.access(package_path_EU, os.R_OK)
          and os.access(package_path_EU, os.W_OK)
          and os.access(package_path_EU, os.X_OK)
          ):

        package_path = package_path_EU
    sys.path.insert(0, package_path)


class MissingMatfilePath(Exception):
    """Exception raised for errors in the input mat file path.

    Attributes:
        mat_path -- input mat path which caused the error
        message -- explanation of the error
    """

    def __init__(self, mat_path, message="This path cannot find in mat file"):
        self.mat_path = mat_path
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.mat_path} : {self.message}'


class MatgenCeerMcip():
    """
    This class contains the headers, function name, and event extraction process related to the Close Cutin Scenario for L2+ Production Data.
    """

    def __init__(self):
        self._func_name = os.path.basename(__file__).replace(".py", "")
        self._headers = dict()
        self._version = '1.2'
        venv_path_NA = '/mnt/usmidet/projects/GPO-IFV7XX/7-Tools/DMA_Venv/data_analytics_mdf_modified_venv/dma_mcip_3_11/bin/activate'
        venv_path_EU = '/net/8k3/e0fs01/irods/PLKRA-PROJECTS/STRADVISION/7-Tools/DMA_Venv/data_analytics_mdf_modified_venv/python_3_11/bin/activate'

        venv_path = venv_path_NA
        if (os.access(venv_path_NA, os.F_OK)
                and os.access(venv_path_NA, os.R_OK)
                and os.access(venv_path_NA, os.W_OK)
                and os.access(venv_path_NA, os.X_OK)
            ):

            venv_path = venv_path_NA

        elif (os.access(venv_path_NA, os.F_OK)
              and os.access(venv_path_NA, os.R_OK)
              and os.access(venv_path_NA, os.W_OK)
              and os.access(venv_path_NA, os.X_OK)
              ):

            venv_path = venv_path_EU

        self._venv = venv_path
        self._red_venv = venv_path

        self._headers['output_sheet'] = ['log_path',
                                         'log_name',
                                         'Information',
                                         'MATGEN_run_time',
                                         'stream_check_dict',
                                         'stream_mapping_pickle_path',]

        self._yaml_input_path = '/mnt/usmidet/projects/GPO-IFV7XX/7-Tools/DMA_Venv/MCIP_Related/config_inputs_MCIP.yaml'
        self._yaml_output_path = '/mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/Thunder_related/config_outputs.yaml'

        self.run_mode_mapping_dict = {'0000': -999,
                                      '0001': 1,
                                      '0010': 2,
                                      '0100': 3,
                                      '1000': 4,
                                      '0011': -1,
                                      '0110': -2,
                                      '1100': -3,
                                      '1001': -4,
                                      '0101': -5,
                                      '1010': -6,
                                      '1111': -999,
                                      'g03': 312,
                                      }

        # run_mode == -999 -> runs all bxx
        # run_mode == 1 -> runs b03 only
        # run_mode == 2 -> runs b04 only
        # run_mode == 3 -> runs b05 only
        # run_mode == -1 -> runs b03 and b04
        # run_mode == -2 -> runs b04 and b05
        # run_mode == -3 -> runs b05 and b03

        if 'lin' in sys.platform:

            _save_mcip_stream_map_path = \
                os.path.join(r'/mnt/usmidet/projects/GPO-IFV7XX',
                             r'7-Tools/DMA_Venv/MCIP_Related',
                             'mcip_stream_map.pickle')
        elif 'win' in sys.platform:
            _save_mcip_stream_map_path = \
                _save_mcip_stream_map_path = \
                os.path.join(r'C:\Users\mfixlz\Downloads\IFV720_MCIP',
                             'misc',
                             'mcip_stream_map.pickle')

        self._save_mcip_stream_map_path = _save_mcip_stream_map_path

    def get_version(self):
        """
        :return: Return script version
        """
        return self._version

    def get_func_name(self):
        """
        :return: Returns function name
        """
        return self._func_name

    def get_headers(self):
        """
        :return: returns headers
        """
        return self._headers

    def get_venv(self):
        """
        Return virtual env from constructor
        """
        return self._venv

    def _process_stream_check(self, mcip_obj_after_run, ):

        from eventExtraction.utils\
            .utils_generic import get_from_dict

        req_map_dict = {
            'key': [key for key in
                    mcip_obj_after_run.stream_check_dict.keys()],

            'source_name': [mcip_obj_after_run.colloquial_source_map.get(
                int(key.split('_')[-3]), "unknown_source")
                for key in mcip_obj_after_run.stream_check_dict.keys()],

            'channel_name': [mcip_obj_after_run.colloquial_channel_map.get(
                int(key.split('_')[1]), "unknown_channel")
                for key in mcip_obj_after_run.stream_check_dict.keys()],

            'stream_name': [get_from_dict(
                mcip_obj_after_run.channel_source_stream_map_dict,
                [int(key.split('_')[1]),
                 int(key.split('_')[-3]),
                 int(key.split('_')[-1])])

                for key in
                mcip_obj_after_run.stream_check_dict.keys()],
        }

        can_map_dict = {
            2102: 'front_corners',
            2103: 'rear_corners',
            2120: 'FDCAN_LRRF',
            2101: 'ALL_CORNERS_SRR',
            2111: 'FDCAN3_MCIP',
            2112: 'FDCAN14_MCIP',
            2211: 'FDCAN3_MCIP_IFV',
            2212: 'FDCAN14_MCIP_IFV',
        }

        swift_nav_dict = {
            258: 'gnss_inertial_gps_time',
            259: 'utc_time',
            521: 'earth_centered_earth_fixed_coord_position',
            522: 'geodetic_position',
            525: 'earth_centered_earth_fixed_coord_velocity',
            526: 'north_east_down_coord_velocity',
            529: 'geodetic_position_with_covariance',
            530: 'north_east_down_coord_velocity_with_covariance',
            532: 'earth_centered_earth_fixed_coord_position_with_covariance',
            533: 'earth_centered_earth_fixed_coord_velocity_with_covariance',
            536: 'geodetic_position_and_accuracy',
            540: 'velocity_as_course_over_ground',
            545: 'orientation_euler_angles',
            65283: 'inertial_nav_system_status',
            65286: 'inertial_nav_system_status_update',
            65294: 'solution_sensors_metadata',
        }

        req_can_dict = {
            'key': [key for key in
                    mcip_obj_after_run.can_check_dict.keys()],
            'name': [can_map_dict.get(int(key), 'Unknown') for key in
                     mcip_obj_after_run.can_check_dict.keys()],

        }

        req_swift_nav_dict = {
            'key': [key for key in
                    mcip_obj_after_run.check_dict_swift_nav.keys()],
            'name': [swift_nav_dict.get(int(key), 'msg_' + str(key)) for key in
                     mcip_obj_after_run.check_dict_swift_nav.keys()],

        }

        req_map_df = pd.DataFrame(req_map_dict)
        req_can_map_df = pd.DataFrame(req_can_dict)
        req_swift_nav_map_df = pd.DataFrame(req_swift_nav_dict)

        with open(self._save_mcip_stream_map_path, 'wb') as handle:
            pickle.dump([req_map_df,
                         req_can_map_df,
                         req_swift_nav_map_df], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        return req_map_df, req_can_map_df

    def kpi_sheet_generation(self, output_excel_sheet):

        def _df_from_list_dicts(excel_out_path, sheet_name, col_name):

            req_streams_eth = ['busID_1001060_source_23_stream_81',
                               'busID_1001060_source_23_stream_85',
                               'busID_1001060_source_23_stream_80',
                               'busID_1001060_source_23_stream_117',
                               'busID_1001060_source_23_stream_18',
                               'busID_1001060_source_23_stream_19',
                               'busID_1001060_source_104_stream_37', ]

            req_channels_can = [2102, 2103, 2120, 2111, 2112, 2211, 2212,]

            req_msg_swift_nav = [522, 535, 529]

            req_cols_data_all = pd.read_excel(excel_out_path,
                                              sheet_name=sheet_name)

            req_cols_data = req_cols_data_all[col_name].to_list()

            req_cols_data_va_tool = req_cols_data_all[['log_path',
                                                       'log_name']]

            req_cols_data = [eval(item) for item in req_cols_data]

            req_df_lists_eth = [pd.DataFrame({key.strip(): val
                                              if isinstance(val, list)
                                              else [val]
                                              for key, val in list_item[0].items()
                                              if key.strip() in req_streams_eth})
                                for list_item in req_cols_data]
            req_df_eth = pd.concat(req_df_lists_eth,
                                   axis=0, ignore_index=True,).fillna('not_found')

            req_df_lists_can = [pd.DataFrame({'CAN_busID_'+str(key): val
                                              if isinstance(val, list)
                                              else [val]
                                              for key, val in list_item[1].items()
                                              if int(key) in req_channels_can})
                                for list_item in req_cols_data]
            req_df_can = pd.concat(req_df_lists_can,
                                   axis=0, ignore_index=True,).fillna('not_found')

            req_df_lists_swift_nav = [pd.DataFrame({'swiftNav_msgID_'+str(key): val
                                                    if isinstance(val, list)
                                                    else [val]
                                                    for key, val in list_item[2].items()
                                                    if int(key) in req_msg_swift_nav})
                                      for list_item in req_cols_data]
            req_df_swift_nav = pd.concat(req_df_lists_swift_nav,
                                         axis=0, ignore_index=True,).fillna('not_found')

            out_df = pd.concat([req_cols_data_va_tool,
                                req_df_can,
                                req_df_eth,
                                req_df_swift_nav,], axis=1)

            return out_df

        excel_out_path = output_excel_sheet
        sheet_name = 'output_sheet'
        col_name = 'stream_check_dict'

        out_df = _df_from_list_dicts(excel_out_path, sheet_name, col_name)

        out_df = out_df[self._headers['overview_stream_check']]

        with pd.ExcelWriter(output_excel_sheet,
                            engine='openpyxl',
                            mode='a') as writer:
            out_df.to_excel(writer,
                            sheet_name='overview_stream_check',
                            index=False)

        if "win" in sys.platform:

            return out_df

    def run(self, file_name, **kwargs):
        '''


        Parameters
        ----------
        file_name : str
            complete log path.
        **kwargs : TYPE
            As and how they are defined in VA tool

        Returns
        -------
        output : pd.DataFrame or np.array
            depending on whether it is run on VAtool or on local system.

        '''
        from joblib import Parallel, delayed, parallel_backend, parallel_config
        from tqdm import tqdm
        import contextlib
        import joblib
        import yaml
        import scipy as sp
        import pickle
        from scipy.io import loadmat as load_mat_scipy
        import re
        import psutil
        import time

        def process_memory():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss, mem_info.vms

        print(
            f'\n \n ********  Log with path is : {file_name}   \n***********')

        log_path, log_name = os.path.split(file_name)

        if "lin" in sys.platform:
            from eventExtraction.utils\
                .signal_extraction import mcipSignalExtraction

            from eventExtraction.utils\
                .utils_generic import merge_dicts

            if not 'out_df' in kwargs.keys():

                kwargs['out_df'] = False

            with open(self._yaml_input_path) as stream:

                yaml_input_dict = yaml.load(stream, Loader=yaml.Loader)

            with open(self._yaml_output_path) as stream2:

                yaml_output_dict = yaml.load(stream2, Loader=yaml.Loader)

        else:

            from eventExtraction.utils\
                .signal_extraction import mcipSignalExtraction

            from eventExtraction.utils\
                .utils_generic import merge_dicts

            with open(kwargs['yaml_input_path']) as stream:

                yaml_input_dict = yaml.load(stream, Loader=yaml.Loader)

            with open(kwargs['yaml_output_path']) as stream2:

                yaml_output_dict = yaml.load(stream2, Loader=yaml.Loader)

        if ('stream_def_path' in kwargs.keys()
                    and (kwargs['stream_def_path'] != ''
                         or kwargs['stream_def_path'] is not None)
                and os.path.isdir(kwargs['stream_def_path'])
                ):

            stream_def_dir_path = os.path.join(kwargs['stream_def_path'])

        else:

            stream_def_dir_path = yaml_input_dict['stream_def_dir_path']
        # trimble_config_path = yaml_input_dict['trimble_config_path']

        # ############################################################
        # # Vinay 03062025 hard coding

        # is_exist = np.sum(self.df2['req_file_names'].str.contains(
        #     os.path.split(file_name)[-1].split('.')[0]))

        # if is_exist != 0:

        #     stream_def_dir_path = self.alternate_stream_def_path

        # ############################################################

        b03_dict = yaml_input_dict['b03_dict']
        b04_dict = yaml_input_dict['b04_dict']
        b05_dict = yaml_input_dict['b05_dict']
        p01_dict = yaml_input_dict['p01_dict']
        r03_dict = yaml_input_dict['r03_dict']

        p01_dict['mudp_input_dict']['bus_channel'] = int(1001)

        mcip_obj = mcipSignalExtraction(
            #  stream_def_dir_path,
            # trimble_config_path
        )
        # mcip_obj.is_decoding = False

        kwargs_main = {}

        if 'run_mode' in kwargs.keys():
            run_mode = self.run_mode_mapping_dict[kwargs['run_mode']]
            # run_mode = int(kwargs['run_mode'])
        else:
            run_mode = 1

        debug_str = 'MAT file already exists'

        # out_dict = thun_obj.main_bus(bus_dict, log_path)
        if ('output_path' in kwargs.keys()
                    and os.path.isdir(kwargs['output_path'])
                ):
            mat_out_path = kwargs['output_path']

        else:
            # mat_out_path = yaml_output_dict['mat_out_path']
            mat_out_path = log_path

        if '1-Raw' in mat_out_path:

            mat_out_path = mat_out_path.replace('1-Raw', '2-Sim', 1)

        sub_string_list = [
            '_b01', '_b02',
            '_b03', '_b04', '_b05',
            '_p01',
            '_r03',
        ]
        # boolean_substring = list(map(log_name.__contains__,
        #                              sub_string_list))
        longest_first = sorted(sub_string_list, key=len, reverse=True)
        compiled_strings = re.compile(r'(?:{})'
                                      .format('|'.join(
                                          map(re.escape, longest_first))))

        out_mat_file_name = re.sub(compiled_strings, '_p01',
                                   log_name.split('.')[0]+'.mat')

        # out_mat_file_name = log_name.split('.')[0]+'.mat'

        out_mat_existing_file_name = out_mat_file_name
        existing_mat_full_path = os.path.join(mat_out_path,
                                              out_mat_existing_file_name)

        # if run_mode == 312:

        #     out_mat_file_name = re.sub(compiled_strings, '_dma',
        #                                out_mat_file_name)
        if not run_mode in [312, ]:

            out_mat_file_name = re.sub(compiled_strings, '_dma',
                                       out_mat_file_name)
        save_mat_full_path = os.path.join(mat_out_path,
                                          out_mat_file_name)

        # save_mat_full_path = save_mat_full_path.replace('_bus_','_dma_',1)

        if 'partial_overwrite' in kwargs.keys():

            if (kwargs['partial_overwrite'].lower() in
                ['true', '1', 't', 'y',
                 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']):

                partial_overwrite = True

            elif (kwargs['partial_overwrite'].lower() in
                  ['false', '0', 'f', 'n',
                   'no', 'nah', 'nope', ]):

                partial_overwrite = False

            else:

                partial_overwrite = False

        else:
            partial_overwrite = True

        if 'overwrite_existing_mat' in kwargs.keys():

            if (kwargs['overwrite_existing_mat'].lower() in
                ['true', '1', 't', 'y',
                 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']):

                overwrite_existing_mat = True

            elif (kwargs['overwrite_existing_mat'].lower() in
                  ['false', '0', 'f', 'n',
                   'no', 'nah', 'nope', ]):

                overwrite_existing_mat = False

            else:

                overwrite_existing_mat = False

        else:
            overwrite_existing_mat = False

        if (os.path.isfile(save_mat_full_path)
            or
                os.path.isfile(existing_mat_full_path)):  # or overwrite_existing_mat:

            kwargs_load_mat_scipy = {  # 'struct_as_record' : False,
                # 'squeeze_me' : True,
                'simplify_cells': True,
                'verify_compressed_data_integrity': True
            }

            if os.path.isfile(existing_mat_full_path):

                data = load_mat_scipy(existing_mat_full_path,
                                      **kwargs_load_mat_scipy
                                      )
            elif os.path.isfile(save_mat_full_path):

                data = load_mat_scipy(save_mat_full_path,
                                      **kwargs_load_mat_scipy
                                      )

            if ('DMA_MATGEN_VERSION' not in data.keys()
                or data['DMA_MATGEN_VERSION'] < self._version
                or overwrite_existing_mat
                ):

                start_time = time.time()
                mem_before_phy, mem_before_virtual = process_memory()

                out_dict, debug_str = mcip_obj.main(b03_dict,
                                                    b04_dict,
                                                    b05_dict,
                                                    p01_dict,
                                                    r03_dict,
                                                    file_name,
                                                    run_mode=run_mode,
                                                    **kwargs_main)

                if partial_overwrite:
                    # from utils_generic import merge_dicts
                    out_dict = merge_dicts(data, out_dict)
                    # out_dict = {**data, **out_dict}

                mem_after_phy, mem_after_virtual = process_memory()

                end_time = time.time()

                # missing_channels = thun_obj._existence_channels(log_path, [51, 917504, 17, 18])

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

                out_dict['DMA_MATGEN_VERSION'] = self._version

                if os.path.isfile(save_mat_full_path):
                    os.remove(save_mat_full_path)

            else:
                start_time = end_time = time.time()

                # missing_channels = thun_obj._existence_channels(log_path, [51, 917504, 17, 18])

                elapsed_time = secondsToStr(end_time-start_time)

                # stream_check_dict = {key: [val]
                #                      for key, val in mcip_obj.stream_check_dict.items()
                #                      }

                # if not os.path.isfile(self._save_mcip_stream_map_path):

                #     req_map_df = self._process_stream_check(mcip_obj)

                stream_check_dict = {}
                can_check_dict = {}
                swift_nav_check_dict = {}

                vatool_out_dict = {'log_path': [log_path],
                                   'log_name': [log_name],
                                   'debug_str': [debug_str],
                                   'MATGEN_run_time [hh:mm:ss]': [elapsed_time],
                                   'stream_check_dict': [[stream_check_dict,
                                                          can_check_dict,
                                                          swift_nav_check_dict]],
                                   'stream_mapping_pickle_path':
                                       [self._save_mcip_stream_map_path]
                                   }
                # vatool_out_dict = {'log_path': [log_path],
                #                    'log_name': [log_name],
                #                    'debug_str': [debug_str],
                #                    'MATGEN_run_time [hh:mm:ss]': [elapsed_time],
                #                    }

                vatool_out_df = pd.DataFrame(vatool_out_dict)

                output = {}

                if kwargs['out_df']:
                    output['output_sheet'] = vatool_out_df

                else:
                    output['output_sheet'] = np.array(vatool_out_df)
                return output

        else:
            start_time = time.time()
            mem_before_phy, mem_before_virtual = process_memory()

            # if run_mode == 312:

            print('\n**********************************************\n',
                  'No existing mat file available. Check the path',
                  '\n For now, proceeding with DMA MATGEN',
                  '**********************************************\n')

            # if run_mode == 312:

            #     print('**********************************************',
            #           'No existing mat file available. Check the path',
            #           'For now, proceeding with DMA MATGEN',
            #           '**********************************************')

            #     run_mode = 1

            out_dict, debug_str = mcip_obj.main(b03_dict,
                                                b04_dict,
                                                b05_dict,
                                                p01_dict,
                                                r03_dict,
                                                file_name,
                                                run_mode=run_mode,
                                                **kwargs_main)

            out_dict['DMA_MATGEN_VERSION'] = self._version

            mem_after_phy, mem_after_virtual = process_memory()

            end_time = time.time()

            # missing_channels = thun_obj._existence_channels(log_path, [51, 917504, 17, 18])

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

            print(
                f'\n \n ********  Mat out path is : {mat_out_path}   \n***********')
        if 'Error message' in out_dict:
            del out_dict['Error message']
        sp.io.savemat(save_mat_full_path,
                      out_dict,
                      **{'long_field_names': True,
                         'oned_as': 'column'},
                      do_compression=True,
                      )
        if "lin" in sys.platform:

            os.system("chmod 777 " + save_mat_full_path)

        # if yaml_output_dict['is_mat_out'] or os.path.isdir(kwargs['output_path']) or os.path.isdir(mat_out_path):

        #     sp.io.savemat(save_mat_full_path,
        #                   out_dict,
        #                   **{'long_field_names': True,
        #                      'oned_as': 'column'},
        #                   do_compression=True,
        #                   )

        # if yaml_output_dict['is_pickle_out']:

        #     out_pickle_file_name = log_name.split('.')[0]+'.pickle'

        #     out_pickle_file_name = re.sub(compiled_strings, '_dma',
        #                                   out_pickle_file_name)
        #     save_pickle_full_path = os.path.join(yaml_output_dict['pickle_out_path'],
        #                                          out_pickle_file_name)

        #     with open(save_pickle_full_path,
        #               'wb') as handle:
        #         pickle.dump(out_dict, handle,
        #                     protocol=5)
        stream_check_dict = {key: [val]
                             for key, val in mcip_obj.stream_check_dict.items()
                             }

        can_check_dict = {key: [val]
                          for key, val in mcip_obj.can_check_dict.items()
                          }
        swift_nav_check_dict = {key: [val]
                                for key, val in mcip_obj.check_dict_swift_nav.items()
                                }

        if not os.path.isfile(self._save_mcip_stream_map_path):

            req_map_df = self._process_stream_check(mcip_obj)

        vatool_out_dict = {'log_path': [log_path],
                           'log_name': [log_name],
                           'debug_str': [debug_str],
                           'MATGEN_run_time [hh:mm:ss]': [elapsed_time],
                           'stream_check_dict': [[stream_check_dict,
                                                  can_check_dict,
                                                  swift_nav_check_dict]],
                           'stream_mapping_pickle_path':
                               [self._save_mcip_stream_map_path]
                           }

        vatool_out_df = pd.DataFrame(vatool_out_dict)

        output = {}

        if 'is_stream_check_dict' in kwargs and kwargs['is_stream_check_dict']:

            stream_check_df = pd.DataFrame(stream_check_dict, )

            return stream_check_df

        if kwargs['out_df']:
            output['output_sheet'] = vatool_out_df

        else:
            output['output_sheet'] = np.array(vatool_out_df)
        return output

    def parallel_wrapper_run(self, args):

        file_name, kwargs = args

        return self.run(file_name,
                        **kwargs
                        )


def read_pickle_parallel(pickle_path):

    df = pd.read_pickle(pickle_path)

    return df


if __name__ == "__main__":

    import warnings
    import os
    import more_itertools as mit
    from ray.util.joblib import register_ray
    from functools import reduce
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import contextlib
    import joblib
    import yaml
    import scipy as sp
    import pickle
    warnings.filterwarnings("ignore")

    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm 
        progress bar given as argument"""
        os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):

                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()

    class_obj = MatgenCeerMcip()

    is_parallel_run = False  # make it true for HPCC / remote windows
    is_remote_windows = False  # make it true for remote windows desktop

    hpcc_cmd_run = True

    dir_name = datetime.now().strftime('%d_%b_%Y__%H%M%S')

    kwargs = {}

    # kwargs['stream_def_dir_path'] = os.path.join(
    #     r'C:\Users\mfixlz\Downloads\DEXT_Aswin',
    #     r'DEXT v1.1.0_Patched\stream_definitions\stream_defs')
    # kwargs['can_db_path'] = os.path.join(
    #     r'C:\Users\mfixlz\Downloads\Databases\Databases', )

    if "win" in sys.platform:

        kwargs['yaml_input_path'] = os.path.join(os.getcwd(),
                                                 'config_inputs_MCIP_win.yaml')
        kwargs['yaml_output_path'] = os.path.join(os.getcwd(),
                                                  'config_outputs_MCIP_win.yaml')

        kwargs['output_path'] = os.path.join(os.getcwd(),)

        kwargs['partial_overwrite'] = 'Y'
        kwargs['overwrite_existing_mat'] = 'Y'
        kwargs['run_mode'] = 'g03'
        # kwargs['is_stream_check_dict'] = True
        is_pickle = False

        sys.path.insert(
            0, os.path.join(r'C:\Users\mfixlz',
                            r'OneDrive - Aptiv\Documents\DM_A\PO_Chaitanya_K',
                            r'Projects\GPO Data Mining Analysis',
                            'GPO_Data_Mining_Analysis',
                            'src',)
        )

        req_dir = os.path.join(os.getcwd())
        file_list_path = os.path.join(req_dir, 'file_list_mcip.txt')
        file_names = list(pd.read_csv(file_list_path, header=None)[0].values)

        if is_remote_windows:

            back_end = "ray"
            register_ray()

        else:
            back_end = "loky"
            # register_ray()

            if not is_parallel_run:

                kwargs['out_df'] = False
                final_out = []

                for file_name in file_names:

                    final_out.append(class_obj.run(file_name, **kwargs))

    elif "lin" in sys.platform:

        kwargs['yaml_input_path'] = os.path.join(os.getcwd(),
                                                 'config_inputs_win.yaml')
        kwargs['yaml_output_path'] = os.path.join(os.getcwd(),
                                                  'config_outputs_win.yaml')

        back_end = "loky"  # 'ray'  #
        # register_ray()

        is_pickle = True
        try:
            file_list_path = sys.argv[1]

        except:

            file_list_path = r'/mnt/usmidet/projects/STLA-THUNDER/11-Development/Filelist/Test/FLR4_Test'

        try:
            kwargs['output_path'] = sys.argv[2]
        except:
            kwargs['output_path'] = os.path.split(file_list_path)[0]

        file_names = list(pd.read_csv(file_list_path, header=None)[0].values)

    if is_parallel_run:

        bool_val = True if hpcc_cmd_run else False

        kwargs['out_df'] = True

        kwargs_list = [kwargs
                       for _ in file_names]

        args = list(zip(file_names, kwargs_list))

        slice_length = 800
        args_sliced = list(mit.chunked(args, slice_length))
        num_jobs = -1

        # -1 :  all cores used intelligently
        # 1  : single core, i.e., no parallel
        # -n : all but n cores are used for parallel
        # n  : n cores used parallely

        def secondsToStr(t):
            return "%d:%02d:%02d.%03d" % \
                reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                       [(t*1000,), 1000, 60, 60])

        start_time = time.time()
        if is_pickle:
            cwd = os.getcwd()
            req_path = os.path.join(cwd, dir_name)

            if not os.path.isdir(req_path):

                # if the demo_folder2 directory is
                # not present then create it.
                os.makedirs(req_path)

        results = []

        with parallel_config(backend=back_end):

            with tqdm_joblib(tqdm(desc="My calculation",
                                  total=len(args)
                                  )) as progress_bar:

                # os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
                for i, args_ in enumerate(args_sliced):
                    results_ = Parallel(n_jobs=num_jobs,
                                        prefer='processes',
                                        # return_as="generator",
                                        )(delayed(
                                            class_obj.parallel_wrapper_run)(a)
                                          for a in args_)
                    if is_pickle:
                        results_ = [result for result in results_
                                    if isinstance(result, pd.DataFrame)]

                        if len(results_) == 0:
                            results_iter_df = pd.DataFrame(
                                columns=class_obj._headers['output_sheet'])
                        else:
                            results_iter_df = pd.concat(
                                results_, ignore_index=True)

                        req_file_path = os.path.join(req_path, str(i)+'.pkl')
                        results_iter_df.to_pickle(req_file_path,
                                                  protocol=5)
                    else:
                        results.append(results_)

            # results = results_

        end_time = time.time()

        if is_pickle:
            path_list = file_names = [os.path.join(req_path, file)
                                      for file in os.listdir(req_path)
                                      if file.endswith(".pkl")]
            # read_pickle_parallel
            results = Parallel(n_jobs=num_jobs,
                               prefer='processes',
                               # return_as="generator",
                               )(delayed(
                                   read_pickle_parallel)(a)
                                 for a in path_list)

        results_refined = [result for result in results
                           if isinstance(result, pd.DataFrame)]
        results_df = pd.concat(results_refined, ignore_index=True)

        results_df.to_csv(
            f"results_production_{dir_name}.csv")

        elapsed_time = secondsToStr(end_time-start_time)

        print(f'&&&&&&&&&&&& Elapsed time is {elapsed_time} %%%%%%%%%%%%%%%%')
