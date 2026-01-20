###########################################################################
# The value of this variable should match the project folder name on HPC.
# If you are not sure, check the projects list inside - /mnt/usmidet/projects/ - for Southfield HPC
# This path will be different for other HPC clusters
from random import randint
import sys
import os
import numpy as np
import contextlib
import re
from pathlib import Path

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


###########################################################################

###########################################################################
# DO NOT CHANGE THIS PART


###########################################################################


class FindTsiGeneric:
    """
    User should add a general description of what this script does in here
    **DESCRIPTION**
    """

    def __init__(self):
        ###########################################################################
        # DO NOT CHANGE THIS PART
        if "lin" in sys.platform:
            self._func_name = os.path.basename(__file__).replace(".py", "")
        ###########################################################################

        ###########################################################################
        # Update the header dictionary. The final output excel sheet can include several sub-sheets.
        # The key of self._headers should be sheet name and value is the header name of that sub-sheet.
        # In this sample, it only have one sub-sheet.
        # function that user can uncomment : generate_plot.

        # THE FIRST ENTRIES SHOULD ALWAYS BE "log_path", "log_name",

        self._version = '1.0'

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

        self._headers = dict()
        # keys = ['output', 'overview', ]
        # for key in keys:
        #     self._headers[key] = []

        self._headers['output_data_feature_based'] = [
            'log_path',
            'sequence_name',
            'log_name',
            'base_logname',
            'rTag',
            'event_start_cTime',
            'event_end_cTime',
            'start_cTime_sequence',
            'feature_20ms_OW_fused_speed_limit_sign_enum_mapped',
            'feature_20ms_OW_fused_speed_limit_sign_enum',
            'feature_20ms_OW_fused_speed_limit_source_enum',
            'feature_20ms_OW_fused_speed_limit_source_all_unique_enum',
            'feature_20ms_OW_TSR_status',
            'feature_20ms_OW_supp_speed_limit_sign_enum',
            'feature_20ms_OW_overtake_sign_enum',
            'feature_10ms_OW_construction_area_enum',
            'feature_10ms_OW_construction_area_enum_unique',
            'target_gps_long_lat_coordinates',
            'to_trust_target_gps',
            'readff_link_event_start',
            'readff_link_full_video',
            'avi_SW_version',
            'vision_frame_ID_event_start',
            'vision_frame_ID_event_end',
            'host_gps_latitude_event_end',
            'host_gps_longitude_event_end',
            'host_gps_latitude_event_start',
            'host_gps_longitude_event_start',
            # 'main_sign_ID',
            'main_sign_enum',
            'is_main_sign_feature_match',
            'can_telematics_sign_enum',
            'is_can_telematics_feature_match',
            'ground_truth',
            'gt_confidence',
            'gt_type',
            'feature_avi_match_delta_cTime',
            'main_sign_confidence_mean',
            'main_sign_confidence_std',
            'main_sign_confidence_event_start',
            'main_sign_confidence_median',
            'main_sign_confidence_median_abs_deviation',
            'supp_1_sign_enum',
            'supp_1_sign_enum_mapped',
            'supp_1_sign_confidence_mean',
            'supp_1_sign_confidence_std',
            'supp_1_sign_confidence_event_start',
            'supp_2_sign_enum',
            'supp_2_sign_enum_mapped',
            'supp_2_sign_confidence_mean',
            'supp_2_sign_confidence_std',
            'supp_2_sign_confidence_event_start',
            'host_long_velocity_median_mps',
            'host_long_velocity_median_abs_deviation_mps',
            'host_long_velocity_event_start_mps',
            'main_sign_flickering_count_per_sec',
            'main_sign_flickering_mean_duration_sec',
            'main_sign_confidence_flickering_count_per_sec',
            'main_sign_confidence_flickering_mean_duration_sec',
            'main_sign_distance_monotonicity_abberations_count_per_sec',
            'event_signal',
            'event_type',
            'log_name_event_start',
            'log_name_event_end',
        ]

        self._headers['output_data_feature_based_all'] = [
            item
            for item in self._headers['output_data_feature_based']]

        self._headers['output_data_feature_based_p'] = [
            item
            for item in self._headers['output_data_feature_based']]

        self._headers['output_feature_non_speed_based'] = [
            item
            for item in self._headers['output_data_feature_based']]

        self._headers['output_data_sign_ID_based'] = [
            item
            # + '_sign_ID_based'
            for item in self._headers['output_data_feature_based']]

        self._headers['output_data_sign_enum_based'] = [
            item
            # + '_feature_based'
            for item in self._headers['output_data_feature_based']]

        self._headers['output_overview'] = [
        ]

        self.program_config_name_map = {
            'thunder': 'config_thunder_v1_tsi.yaml',
            'mcip': 'config_mcip_v1_tsi.yaml',
        }

        self.program_name_readff_map = {
            'thunder': 'Thunder',
            'mcip': 'GPO-IFV7XX',

        }

        self.readff_map_dict = {'Thunder': ['_dma', '_v01'],
                                'GPO-IFV7XX': ['_p01', '_v01']
                                }

        if 'win' in sys.platform:

            self.debug_win = True
            self.gif_generation = True
            self.SIMULATE_LINUX = False

        else:
            self.debug_win = False
            self.gif_generation = False
            self.SIMULATE_LINUX = True

        self.table_name_GT = 'DMA_MCIP_TSI_GT'

        ###########################################################################

    ###########################################################################
    # DO NOT CHANGE THIS PART

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

    def get_red_venv(self):
        """
        Return virtual env from constructor
        """
        return self._red_venv

    ###########################################################################

    ###########################################################################

    def kpi_sheet_generation(self, output_excel_sheet):

        from statsmodels.stats.stattools import medcouple
        import pandas as pd
        import numpy as np
        import sys
        import re

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

        # if "lin" in sys.platform:
        #     sys.path.insert(
        #         0, r'/mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/CES_related/GPO_event_extraction/GPO_Data_Mining_Analysis/src')

        #     from eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
        #                                                      _write_gifs_to_excel,
        #                                                      _resim_path_to_orig_path,
        #                                                      )
        # elif "win" in sys.platform:
        #     root_path = os.path.join(r'C:\Users\mfixlz',
        #                              r'OneDrive - Aptiv\Documents\DM_A\PO_Chaitanya_K',
        #                              r'Projects\GPO Data Mining Analysis',
        #                              'GPO_Data_Mining_Analysis',
        #                              'src',)
        #     sys.path.insert(0, root_path)
        #     from eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
        #                                                      _write_gifs_to_excel,
        #                                                      _resim_path_to_orig_path,
        #                                                      )

        from eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
                                                         _write_gifs_to_excel,
                                                         _resim_path_to_orig_path,
                                                         )

        if self.gif_generation:

            job_out_excel_path = output_excel_sheet
            events_sheet_name = 'output_data_feature_based_p'

            excel_output_path = os.path.join(os.path.split(
                job_out_excel_path)[0], 'gif_excel_out.xlsx')
            end_cTime_col = 'event_end_cTime'
            log_path_col = 'log_path'
            log_name_col = 'log_name'
            start_cTime_col = 'event_start_cTime'

            log_name_replace_list: list = []  # ['dma', 'v01']
            video_extension: str = ''  # '.webm'
            prepend_to_start_time: float = 3
            append_to_end_time: float = 0.0
            req_fps: int = 5
            is_bw_required: bool = True
            gif_resize_factor: float = 0.25
            gif_output_path: os.path.join = None

            total_df = pd.read_excel(job_out_excel_path,
                                     sheet_name=events_sheet_name)

            total_df[log_path_col] = total_df.apply(lambda x:
                                                    os.path.join(x[log_path_col],
                                                                 x['base_logname']),
                                                    axis=1)

            # total_df[log_path_col] = total_df.apply(
            #     self._resim_path_to_orig_path,
            #     axis=1,
            #     is_path=True,
            # )
            # total_df[log_name_col] = total_df.apply(
            #     self._resim_path_to_orig_path,
            #     axis=1,
            #     is_path=False,
            # )
            # RESIM
            # total_df_2 = total_df[[log_path_col,
            #                        log_name_col]].copy(deep=True)

            # total_df[log_path_col] = total_df_2.apply(lambda row:
            #                                           _resim_path_to_orig_path(
            #                                               row['log_name'],
            #                                               row['log_path'],
            #                                               is_path=True),
            #                                           axis=1)
            # total_df[log_name_col] = total_df_2.apply(lambda row:
            #                                           _resim_path_to_orig_path(
            #                                               row['log_name'],
            #                                               row['log_path'],
            #                                               is_path=False),
            #                                           axis=1)

            ####################################################

            # config_path = os.path.join(
            #     r"C:\Users\mfixlz\Downloads\TNDR1_DRUK_20240808_221741_WDC5_v01_0047.webm.json")
            # config_file = open(config_path, 'r')

            # config_json = json.load(config_file)
            # LD = config_json['frames']
            # DA = _list_of_dicts_to_dict_of_arrays(LD)
            # video_cTimes_array = DA['Time']

            video_cTimes_array_signal_and_multiplier_list = ['', 1E6]

            ##################################################

            _write_gifs_to_excel(job_out_excel_path,
                                 events_sheet_name,
                                 excel_output_path,
                                 start_cTime_col,
                                 end_cTime_col,
                                 log_path_col,
                                 log_name_col,
                                 video_cTimes_array_signal_and_multiplier_list,
                                 log_name_replace_list,
                                 video_extension,
                                 prepend_to_start_time,
                                 append_to_end_time,
                                 req_fps,
                                 is_bw_required,
                                 gif_resize_factor,
                                 gif_output_path,
                                 total_df=total_df
                                 )

        excel_out_path = output_excel_sheet
        sheet_name = 'output_data_feature_based_p'
        col_name = 'feature_20ms_OW_fused_speed_limit_sign_enum_mapped'
        renamed_col = 'feature_20ms_OW_fused_speed_limit_sign'
        group_by_column = 'feature_20ms_OW_fused_speed_limit_source_enum'

        df = (pd.read_excel(excel_out_path, sheet_name=sheet_name)
              .rename(columns={col_name: renamed_col, },))

        try:

            req_output = df.groupby(group_by_column,
                                    group_keys=True)[renamed_col].apply(
                                        pd.Series.value_counts,
                                        dropna=True).reset_index()

            other_col = list(set(req_output.columns) - set([renamed_col,
                                                            group_by_column]))[0]
            req_output = req_output.rename(columns={renamed_col: 'count',
                                                    group_by_column: 'source',
                                                    other_col: 'Speed Limit Sign'},)

            req_output = req_output.sort_values(
                [group_by_column, other_col], ascending=[True, True])
            req_index = False

        except:
            print('&&&&&&& Group by enum source threw errors',
                  '\nProceeding with combined aggregation')
            req_output = (df[[renamed_col]]
                          .apply(pd.Series.value_counts, dropna=True)
                          .rename(columns={renamed_col: 'count', },))
            req_index = True

        with pd.ExcelWriter(output_excel_sheet,
                            engine='openpyxl',
                            mode='a') as writer:
            req_output.to_excel(writer,
                                sheet_name='kpi_overview',
                                index=req_index
                                )

        if "win" in sys.platform:

            return req_output

    def _run_helper(self, file_name, is_continuous=False,  **kwargs):
        ###########################################################################

        ###########################################################################
        # ADD A BRIEF DESCRIPTION OF THE LOGIC USED IN THIS SCRIPT HERE
        # ALONG WITH INPUT PARAMS IF ANY OTHER THAN THE MAT FILE
        ## Mandatory details for th script ##
        """
        Description: Write briefly about your script
        Input to the script: filelist with file names
        Input file type: .mat or .mudp
        Output of the script: Excel sheet with events

        """
        import pandas as pd
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")
        ###########################################################################

        ###########################################################################
        # DO NOT REMOVE THE "TRY" STATEMENT
        if "lin" in sys.platform:

            root_path = os.path.join(r'/mnt/usmidet/projects/STLA-THUNDER',
                                     r'7-Tools/DMA_Venv/CES_related',
                                     'GPO_event_extraction',
                                     'GPO_Data_Mining_Analysis',
                                     'src',
                                     )

            # sys.path.insert(0, root_path)
            from eventExtraction.tsi.core_tsi import coreEventExtractionTSI
            from eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
                                                             _write_gifs_to_excel,
                                                             _resim_path_to_orig_path,
                                                             loadmat,
                                                             _create_flat_file_list,
                                                             _create_base_name_list_from_file_list,
                                                             )
            kwargs['config_path'] = os.path.join(root_path,
                                                 'eventExtraction',
                                                 'data',
                                                 kwargs['program'],
                                                 self.program_config_name_map[
                                                     kwargs['program'].lower()])

        elif "win" in sys.platform:
            print('****************', os.getcwd())
            # root_path = os.path.join(r'C:\Users\mfixlz',
            #                          r'OneDrive - Aptiv\Documents\DM_A\PO_Chaitanya_K',
            #                          r'Projects\GPO Data Mining Analysis',
            #                          'GPO_Data_Mining_Analysis',
            #                          'src',)
            # sys.path.insert(0, root_path)
            from eventExtraction.tsi.core_tsi import coreEventExtractionTSI
            # from eventExtraction.tsi.signal_mapping_tsi import signalMapping
            from eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
                                                             _write_gifs_to_excel,
                                                             _resim_path_to_orig_path,
                                                             loadmat,
                                                             _create_flat_file_list,
                                                             _create_base_name_list_from_file_list,
                                                             )
        import pickle

        from eventExtraction.utils.utils_generic import create_mysql_engine_fn

        try:

            if is_continuous:

                if isinstance(file_name, list):

                    file_name_list = file_name
                    (log_path_list,
                     base_name_list,
                     rTag_list,
                     seq_path_list,
                     original_log_name_list) = _create_base_name_list_from_file_list(
                        file_name_list)

                else:

                    (file_name_list, base_name_list, rTag_list,
                     seq_path_list, original_log_name_list) = \
                        _create_flat_file_list(
                            file_name,
                            SIMULATE_LINUX=self.SIMULATE_LINUX)

                if self.debug_win:

                    file_name_list = file_name_list[:1]

                df_list = []
                collect_cTime = True

                for idx, (file_name_flat, basename,
                          seq_path, orig_log_name) in enumerate(
                        zip(file_name_list, base_name_list,
                            seq_path_list, original_log_name_list)):

                    log_path, log_name = os.path.split(file_name_flat)
                    rTag = rTag_list[idx]

                    try:

                        mat_file_data = loadmat(file_name_flat)
                        # TSI_signal_map_obj = signalMapping(mat_file_data)
                        TSI_core_logic_obj = coreEventExtractionTSI(
                            mat_file_data)

                        out_df, enums_dict = TSI_core_logic_obj._signal_mapping(
                            kwargs['config_path'], log_name=log_name)

                        out_df['log_path_flat'] = log_path

                        out_df['log_name_flat'] = log_name

                        out_df['orig_log_name_flat'] = orig_log_name

                        # basename = '_'.join(log_name.split('.')[
                        #                     0].split('_')[:-1])

                        out_df['base_logname'] = basename
                        out_df['rTag'] = rTag

                        out_df['frame_ID'] = (
                            out_df['vision_avi_tsr_camera_stream_ref_index'] -
                            out_df['vision_avi_tsr_camera_stream_ref_index'].min(
                                skipna=True)
                        )

                        # out_df['frame_ID'] = (
                        #     out_df['vision_avi_tsr_camera_frame_ID'] -
                        #     out_df['vision_avi_tsr_camera_frame_ID'].min(
                        #         skipna=True)
                        # )
                        # out_df['frame_ID'] = np.array([
                        #     f_id
                        #     for f_id in range(
                        #         len(out_df[
                        #             'vision_avi_tsr_camera_frame_ID']))],
                        #     dtype=int)

                        if collect_cTime:
                            start_cTime_sequence = np.array(out_df['cTime'])[0]
                            collect_cTime = False

                        df_list.append(out_df)
                    except:
                        print('***************************',
                              f'Log at with name "{log_name}" is corrupt ',
                              f"or missing required streams as per {kwargs['config_path']}"
                              '\n skipping the log')
                        continue

                if not self.debug_win:
                    req_df = pd.concat(df_list, axis=0, ignore_index=True)

                    req_df['start_cTime_sequence'] = start_cTime_sequence

                    req_df, misc_out_dict = \
                        TSI_core_logic_obj._process_signals(req_df,
                                                            enums_dict)

                    TSI_core_logic_obj.misc_out_dict = misc_out_dict

                    if 'win' in sys.platform:

                        req_path_pickle = os.path.join(r'C:\Users\mfixlz',
                                                       r'OneDrive - Aptiv\Documents\DM_A\Aravind',
                                                       r'Projects\2025\KW_31\BCO-14777',
                                                       'req_df_6.pkl')

                        print('**********************************',
                              req_path_pickle)

                        with open(req_path_pickle, 'wb') as file:

                            pickle.dump(req_df, file)
                else:

                    # req_path_pickle = os.path.join(os.getcwd(), 'req_df_3.pkl')

                    # print('**********************************',
                    #       req_path_pickle)

                    # with open(req_path_pickle, 'wb') as file:

                    #     pickle.dump(req_df, file)

                    df_path = os.path.join(r'C:\Users\mfixlz',
                                           r'OneDrive - Aptiv\Documents\DM_A\Aravind',
                                           r'Projects\2025\KW_31\BCO-14777',
                                           'req_df_6.pkl')
                    with open(df_path, 'rb') as file:
                        req_df = pickle.load(file)

                # seq_name = '_'.join(log_name.split('.')[0].split('_')[:-2])
                seq_name = '_'.join(basename.split('_')[:-1])

                req_df['seq_name'] = seq_name
                # FIXME: make it compatible for THUNDER
                # seq_path = os.path.dirname(log_path)

                # log_name = seq_name
                # log_path = seq_path

                # req_df['sequence_path'] = seq_path
                # req_df['sequence_name'] = seq_name

                # req_df['sequence_path'] = req_df['base_logname'].apply(
                #     lambda x: os.path.join(seq_path, x))

                if kwargs['program'].lower() == 'mcip':

                    req_df['sequence_path'] = req_df['base_logname'].apply(
                        lambda x: os.path.join(seq_path, x))
                elif kwargs['program'].lower() == 'thunder':

                    req_df['sequence_path'] = log_path

                req_df['readff_link'] = req_df[['sequence_path',
                                                'orig_log_name_flat']].apply(
                    lambda x:
                        'https://readff-na.aptiv.com/vidFile/' +
                        f'{os.path.join(*Path(x["sequence_path"]).parts[3:])}/' +
                        '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
                            *self.readff_map_dict[
                                self.program_name_readff_map[
                                    kwargs["program"].lower()]])), axis=1
                )

                req_df['readff_link_full_video'] = req_df[['sequence_path',
                                                           'orig_log_name_flat']].apply(
                    lambda x:
                        'https://readff-na.aptiv.com/files/' +
                        f'{os.path.join(*Path(x["sequence_path"]).parts[3:])}/' +
                        '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
                            *self.readff_map_dict[
                                self.program_name_readff_map[
                                    kwargs["program"].lower()]])), axis=1
                )

                # req_df['frame_ID'] = (
                #     req_df['vision_avi_tsr_camera_frame_ID'] -
                #     req_df['vision_avi_tsr_camera_frame_ID'].min(
                #         skipna=True)
                # )
                req_df['event_signal'] = 'TSI'
                TSI_core_logic_obj.program_name_readff_map = \
                    self.program_name_readff_map[kwargs['program'].lower()]
                try:
                    TSI_core_logic_obj.df_GT = \
                        pd.read_sql(f'SELECT * FROM {self.table_name_GT}',
                                    con=create_mysql_engine_fn())
                except:
                    print('Error with MySQL connection. GT values shall be nan')
                    TSI_core_logic_obj.df_GT = pd.DataFrame()

                return_val_dict = TSI_core_logic_obj.event_extraction(
                    req_df,
                    **TSI_core_logic_obj.kwargs_processing)

            else:

                log_path, log_name = os.path.split(file_name)

                (log_path_list, base_name_list,
                 rTag_list,
                 seq_path_list,
                 original_log_name_list) = _create_base_name_list_from_file_list(
                    [file_name])
                basename = base_name_list[0]
                rTag = rTag_list[0]
                seq_path = seq_path_list[0]
                orig_log_name = original_log_name_list[0]

                mat_file_data = loadmat(file_name)
                TSI_core_logic_obj = coreEventExtractionTSI(mat_file_data)

                out_df, enums_dict = TSI_core_logic_obj._signal_mapping(
                    kwargs['config_path'], log_name=log_name)

                out_df['log_path_flat'] = log_path

                out_df['log_name_flat'] = log_name

                out_df['orig_log_name_flat'] = orig_log_name

                # basename = '_'.join(log_name.split('.')[
                #                     0].split('_')[:-1])

                out_df['base_logname'] = basename
                out_df['rTag'] = rTag

                start_cTime_sequence = np.array(out_df['cTime'])[0]

                out_df['start_cTime_sequence'] = start_cTime_sequence

                out_df, misc_out_dict = \
                    TSI_core_logic_obj._process_signals(out_df,
                                                        enums_dict)

                TSI_core_logic_obj.misc_out_dict = misc_out_dict

                # seq_name = '_'.join(log_name.split('.')[0].split('_')[:-2])
                seq_name = '_'.join(basename.split('_')[:-1])

                out_df['seq_name'] = seq_name
                # FIXME: make it compatible for THUNDER
                # seq_path = os.path.dirname(log_path)

                # log_name = seq_name
                # log_path = seq_path

                if kwargs['program'].lower() == 'mcip':

                    out_df['sequence_path'] = out_df['base_logname'].apply(
                        lambda x: os.path.join(seq_path, x))
                elif kwargs['program'].lower() == 'thunder':

                    out_df['sequence_path'] = log_path

                out_df['readff_link'] = out_df[['sequence_path',
                                                'orig_log_name_flat']].apply(
                    lambda x:
                        'https://readff-na.aptiv.com/vidFile/' +
                        f'{os.path.join(*Path(x["sequence_path"]).parts[4:])}/' +
                        '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
                            *self.readff_map_dict[
                                self.program_name_readff_map[
                                    kwargs["program"].lower()]])), axis=1
                )

                out_df['readff_link_full_video'] = out_df[['sequence_path',
                                                           'orig_log_name_flat']].apply(
                    lambda x:
                        'https://readff-na.aptiv.com/files/' +
                        f'{os.path.join(*Path(x["sequence_path"]).parts[4:])}/' +
                        '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
                            *self.readff_map_dict[
                                self.program_name_readff_map[
                                    kwargs["program"].lower()]])), axis=1
                )

                out_df['frame_ID'] = (
                    out_df['vision_avi_tsr_camera_stream_ref_index'] -
                    out_df['vision_avi_tsr_camera_stream_ref_index'].min(
                        skipna=True)
                )
                # out_df['frame_ID'] = (
                #     out_df['vision_avi_tsr_camera_frame_ID'] -
                #     out_df['vision_avi_tsr_camera_frame_ID'].min(
                #         skipna=True)
                # )
                # out_df['frame_ID'] = np.array([
                #     f_id
                #     for f_id in range(
                #         len(out_df[
                #             'vision_avi_tsr_camera_frame_ID']))],
                #     dtype=int)
                out_df['event_signal'] = 'TSI'
                TSI_core_logic_obj.program_name_readff_map = \
                    self.program_name_readff_map[kwargs['program'].lower()]

                return_val_dict = TSI_core_logic_obj.event_extraction(
                    out_df,
                    **TSI_core_logic_obj.kwargs_processing)

            try:
                return_val_df_enum_based = pd.DataFrame(
                    return_val_dict.get('enum_based', {}))
                return_val_df_ID_based = pd.DataFrame(
                    return_val_dict.get('ID_based', {}))
                return_val_df_feature_based = pd.DataFrame(
                    return_val_dict.get('feature_based', {}))
                return_val_df_feature_based_present = pd.DataFrame(
                    return_val_dict.get('feature_based_present', {}))
                return_val_df_feature_non_speed_based = pd.DataFrame(
                    return_val_dict.get('feature_non_speed_based', {}))
            except:
                print('***********************************',
                      'uneven lengths in the output. Debug',
                      '***********************************',)
                return_val_df_enum_based = pd.DataFrame.from_dict(
                    return_val_dict.get('enum_based', {}),
                    orient='index')

                return_val_df_enum_based = return_val_df_enum_based.transpose()

                return_val_df_ID_based = pd.DataFrame.from_dict(
                    return_val_dict.get('ID_based', {}),
                    orient='index')
                return_val_df_ID_based = return_val_df_ID_based.transpose()

                return_val_df_feature_based = pd.DataFrame.from_dict(
                    return_val_dict.get('feature_based', {}),
                    orient='index')
                return_val_df_feature_based = return_val_df_feature_based.transpose()

                return_val_df_feature_based_present = pd.DataFrame.from_dict(
                    return_val_dict.get('feature_based_present', {}),
                    orient='index')
                return_val_df_feature_based_present = \
                    return_val_df_feature_based_present.transpose()

                return_val_df_feature_non_speed_based = pd.DataFrame.from_dict(
                    return_val_dict.get('feature_non_speed_based', {}),
                    orient='index')
                return_val_df_feature_non_speed_based = \
                    return_val_df_feature_non_speed_based.transpose()

            if not return_val_df_enum_based.empty:
                # log_path_out = return_val_df_enum_based.pop('log_path')
                # return_val_df_enum_based.insert(0, 'log_path', log_path_out)
                # return_val_df_enum_based.insert(1, 'sequence_name', seq_name)
                return_val_df_enum_based = \
                    return_val_df_enum_based[
                        self._headers['output_data_sign_enum_based']]

            if not return_val_df_ID_based.empty:
                # log_path_out = return_val_df_ID_based.pop('log_path')
                # return_val_df_ID_based.insert(0, 'log_path', log_path_out)
                # return_val_df_ID_based.insert(1, 'sequence_name', seq_name)
                return_val_df_ID_based = \
                    return_val_df_ID_based[
                        self._headers['output_data_sign_ID_based']]

            if not return_val_df_feature_based.empty:
                # log_path_out = return_val_df_feature_based.pop('log_path')
                # return_val_df_feature_based.insert(0, 'log_path', log_path_out)
                # return_val_df_feature_based.insert(
                #     1, 'sequence_name', seq_name)
                return_val_df_feature_based = return_val_df_feature_based[
                    self._headers['output_data_feature_based']]

            if not return_val_df_feature_based_present.empty:
                # log_path_out = return_val_df_feature_based_present.pop(
                #     'log_path')
                # return_val_df_feature_based_present.insert(
                #     0, 'log_path', log_path_out)
                # return_val_df_feature_based_present.insert(
                #     1, 'sequence_name', seq_name)
                return_val_df_feature_based_present = \
                    return_val_df_feature_based_present[
                        self._headers['output_data_feature_based']]

            if not return_val_df_feature_non_speed_based.empty:
                # log_path_out = return_val_df_feature_non_speed_based.pop(
                #     'log_path')
                # return_val_df_feature_non_speed_based.insert(
                #     0, 'log_path', log_path_out)
                # return_val_df_feature_non_speed_based.insert(
                #     1, 'sequence_name', seq_name)
                return_val_df_feature_non_speed_based = \
                    return_val_df_feature_non_speed_based[
                        self._headers['output_feature_non_speed_based']]

            out_dict = dict()
            if 'local' in kwargs and kwargs['local']:

                out_dict['output_data_sign_enum_based'] = return_val_df_enum_based
                out_dict['output_data_sign_ID_based'] = return_val_df_ID_based
                out_dict['output_data_feature_based'] = return_val_df_feature_based
                out_dict['output_data_feature_based_p'] = \
                    return_val_df_feature_based_present
                out_dict['output_feature_non_speed_based'] = \
                    return_val_df_feature_non_speed_based

                out_dict['output_data_feature_based_all'] = \
                    pd.concat([return_val_df_feature_based,
                               return_val_df_feature_based_present,
                               return_val_df_feature_non_speed_based,],
                              axis=0)

            else:

                print('********** Writing values to dict')

                # out_dict['output_data_sign_enum_based'] = np.array(
                #     return_val_df_enum_based)
                # out_dict['output_data_sign_ID_based'] = np.array(
                #     return_val_df_ID_based)
                out_dict['output_data_feature_based'] = np.array(
                    return_val_df_feature_based)

                out_dict['output_data_feature_based_p'] = np.array(
                    return_val_df_feature_based_present)

                out_dict['output_feature_non_speed_based'] = np.array(
                    return_val_df_feature_non_speed_based)

                out_dict['output_data_feature_based_all'] = np.array(
                    pd.concat([
                        return_val_df_feature_based,
                        # return_val_df_feature_based_present,
                        return_val_df_feature_non_speed_based,],
                        axis=0)
                )

            return out_dict

        #######################################################################
        # DO NOT CHANGE THIS PART
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return str(exc_obj.args[0]) + " FOUND IN LINE: " + str(exc_tb.tb_lineno)

    def run(self, file_name, **kwargs):

        #######################################################################

        compiled_regex = re.compile(r"\[(.*?)\]")

        is_continuous = False

        if isinstance(file_name,
                      list) or bool(compiled_regex.search(file_name)):
            is_continuous = True

        return self._run_helper(file_name,
                                is_continuous=is_continuous, **kwargs)

        # if isinstance(file_name, list):

        #     return self._run_helper(file_name,
        #                             is_continuous=is_continuous, **kwargs)
        # else:

        #     return self._run_helper(file_name,
        #                             is_continuous=is_continuous, **kwargs)

    def parallel_wrapper_run(self, args):

        file_name, kwargs = args

        return self.run(file_name,
                        **kwargs
                        )


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

###########################################################################
# THIS PART IF NOT REQUIRED BY VA TOOL
# BUT IS USEFUL FOR LOCALLY TESTING THE SCRIPT ON YOUR PC


if __name__ == '__main__':

    import warnings
    import os
    import sys
    import more_itertools as mit
    from openpyxl import load_workbook, Workbook
    import pandas as pd
    from scipy.io import loadmat as load_mat_scipy
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import contextlib
    import joblib
    from ray.util.joblib import register_ray

    warnings.filterwarnings("ignore")

    file_names = [

        'ThunderMCIP_WS11656_20250801_090508_0001_p01.mat',
        # 'ThunderMCIP_WS11656_20250801_090508_0002_p01.mat',
        # 'ThunderMCIP_WS11656_20250801_090508_0003_p01.mat',
        # 'ThunderMCIP_WS11656_20250801_090508_0004_p01.mat',
        # 'ThunderMCIP_WS11656_20250801_090508_0005_p01.mat',
        # 'ThunderMCIP_WS11656_20250801_090508_0006_p01.mat',
        # 'ThunderMCIP_WS11656_20250727_083502_0008_p01.mat',

    ]

    file_name_paths = [os.path.join(os.getcwd(),
                                    'data',
                                    file_name)
                       for file_name in file_names
                       ][0]

    # cont_file_list_name = \
    #     'ThunderMCIP_WS11656_20250805_084635_[0054]_p01.mat,3'

    cont_file_list_name = \
        'ThunderMCIP_WS11656_20250729_215140_[0030]_p01.mat, 25'
    # 'ThunderMCIP_WS11656_20250809_181542_[0525]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250730_121739_[0022]_p01.mat, 3'
    # 'ThunderMCIP_WS11656_20250730_121739_0022_p01.mat'
    #
    # 'ThunderMCIP_WS11656_20250902_201941_[0220]_p01.mat, 1'
    # 'ThunderMCIP_WS11656_20250902_093343_[0038]_p01.mat, 1'
    # 'ThunderMCIP_WS11656_20250827_084725_[1328]_p01.mat,16'
    # 'ThunderMCIP_WS11656_20250807_080343_[0000]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250801_112829_[0001]_p01_rMC58003.mat,10'
    # 'ThunderMCIP_gps_removed_WS11656_20250729_215140_[0118]_p01.mat, 1'
    # 'ThunderMCIP_WS11656_20250729_215140_[0118]_p01.mat, 1'

    #
    #
    # 'ThunderMCIP_WS11656_20250809_181542_[0525]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250809_181542_[0350]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250911_180704_[0150]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250826_171555_[0475]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250904_015917_[0000]_p01.mat,16'
    # 'ThunderMCIP_WS11656_20250729_215140_[0000]_p01.mat,19'
    # 'ThunderMCIP_WS11656_20250729_215140_[0125]_p01.mat, 1'
    # 'ThunderMCIP_WS11656_20250729_215140_[0038]_p01.mat, 1'
    #
    #
    # 'ThunderMCIP_WS11656_20250829_095931_[1023]_p01.mat, 1'
    #
    #
    #
    # 'ThunderMCIP_WS11656_20250828_170317_[0500]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250802_091005_[0089]_p01.mat,25'
    # 'ThunderMCIP_WS11656_20250827_193618_[0899]_p01.mat,25'
    #

    # file_name_paths = os.path.join(os.getcwd(),
    #                                'data',
    #                                cont_file_list_name)

    program = 'MCIP'  # 'Thunder'

    config_name = 'config_mcip_v1_tsi.yaml'

    kwargs = {'local': False,
              'config_path': os.path.join(r'C:\Users\mfixlz',
                                          r'OneDrive - Aptiv\Documents',
                                          'DM_A',
                                          'PO_Chaitanya_K',
                                          r'Projects\GPO Data Mining Analysis',
                                          'GPO_Data_Mining_Analysis',
                                          'src',
                                          'eventExtraction',
                                          'data',
                                          program,
                                          config_name),
              'program': program,

              }

    class_obj = FindTsiGeneric()

    final_out = class_obj.run(file_name_paths, **kwargs)

    # output_excel_sheet = os.path.join(
    #     r"C:\Users\mfixlz\Downloads\find_tsi_generic_2025-09-11_04-56-00_0_test.xlsx")

    # xx = class_obj.kpi_sheet_generation(output_excel_sheet)


##########################      END      ##################################


# test the script and take extra argument
