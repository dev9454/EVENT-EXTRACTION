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

import sys
import os
from pathlib import Path

# 1. Dynamically find the path to the 'src' directory
# Assumes script is in: GPO_Data_Mining_Analysis/scripts/E2E/
current_script_path = Path(__file__).resolve()
# .parents[1] goes from E2E -> scripts -> GPO_Data_Mining_Analysis
project_root = current_script_path.parents[1] 
package_path = str(project_root / "src")

# 2. Add to sys.path
if package_path not in sys.path:
    sys.path.insert(0, package_path)

# 2. Apply platform-specific logic and update sys.path
if "win" in sys.platform:
    print('platform : ', sys.platform)
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import contextlib
    import joblib

    if package_path not in sys.path:
        sys.path.insert(0, package_path)

else:
    # Linux/HPC validation logic
    if (os.path.isdir(package_path)
            and os.access(package_path, os.R_OK)
            and os.access(package_path, os.W_OK)
            and os.access(package_path, os.X_OK)
        ):
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
    else:
        # Fallback to insert path even if strict permissions check fails
        if os.path.isdir(package_path) and package_path not in sys.path:
            sys.path.insert(0, package_path)


###########################################################################

###########################################################################
# DO NOT CHANGE THIS PART


###########################################################################


class FindDaGeneric:
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

        self._headers['output_data_da'] = [
            'log_path',
            'sequence_name',
            'log_name',
            'base_logname',
            'rTag',
            'event_start_cTime',
            'event_end_cTime',
            'event_duration',
            'da_status',
            'drive_mode_pre_event',
            'drive_mode_event',
            'drive_mode_post_event',
            'alc_status',
            'turn_indicator_presence',
            'vehicle_turn_status',
            'event_type',
            'event_comment',
            'vision_frame_ID_event_start',
            'vision_frame_ID_event_end',
            'log_name_event_start',
            'log_name_event_end',
            'remarks',
            'nexus_event_type',
        ]

        self._headers['output_overview'] = [
        ]

        self.program_config_name_map = {
            'thunder': '',
            'mcip': '',
            'e2e': 'config_e2e_v1_da_basic.yaml',
        }

        self.program_name_readff_map = {
            'thunder': 'Thunder',
            'mcip': 'GPO-IFV7XX',
            'e2e': 'GPO-E2E',

        }

        self.readff_map_dict = {'Thunder': ['_dma', '_v01'],
                                'GPO-IFV7XX': ['_p01', '_v01']
                                }

        if 'win' in sys.platform:

            self.debug_win = True
            self.gif_generation = False
            self.SIMULATE_LINUX = False

        else:
            self.debug_win = False
            self.gif_generation = False
            self.SIMULATE_LINUX = True

        self.table_name_GT = 'DMA_MCIP_TSI_GT'

        self.error_skip = True
        self.multi_col_kpi = False

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
        import os
        import re
        from pathlib import Path

        # 1. Dynamically find the src directory relative to this script
        # Assuming script is in GPO_Data_Mining_Analysis/scripts/E2E/
        current_script_path = Path(__file__).resolve()
        package_path = str(current_script_path.parents[2] / "src")

        # 2. Add to sys.path if not already present
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

        if "win" in sys.platform:
            print('platform : ', sys.platform)
            from joblib import Parallel, delayed, parallel_backend, parallel_config
            from tqdm import tqdm
            import contextlib
            import joblib
            # Hardcoded sys.path.insert removed as package_path is handled above
        else:
            # 3. Linux/HPC permission validation using the dynamic path
            if (os.path.isdir(package_path)
                    and os.access(package_path, os.R_OK)
                    and os.access(package_path, os.W_OK)
                    and os.access(package_path, os.X_OK)
                ):
                # Path is valid and has correct permissions
                pass 
            else:
                print(f"Warning: Permissions check failed or directory missing at {package_path}")
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
            events_sheet_name = 'output_data_da'

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
        sheet_name = 'output_data_da'
        turn_status_col = 'vehicle_turn_status'
        lane_change_col = 'alc_status'
        event_type_col = 'event_type'
        turn_indicator_presence_col = 'turn_indicator_presence'
        turn_indicator_presence_col_rename = 'Turn indicator status'

        req_categories = ['left turn',
                          'right turn',
                          'Lane change',
                          'Disengagement during event',
                          'Disengagement after event',
                          ]

        df = pd.read_excel(excel_out_path, sheet_name=sheet_name)

        df['disengagement_in_event'] = 'No Disengagement'
        disengagement_in_event_indices = df.query(
            f'{event_type_col} == "Turn_status" ' +
            f'and {turn_status_col} != "No Turn" ' +
            'and (drive_mode_pre_event == "Longitudinal + Lateral" ' +
            'and drive_mode_event != "Longitudinal + Lateral" )'
        ).index

        df.loc[disengagement_in_event_indices,
               'disengagement_in_event'] = 'Disengagement during event'

        df['disengagement_after_event'] = 'No Disengagement'
        disengagement_after_event_indices = df.query(
            f'{event_type_col} == "Turn_status" ' +
            f'and {turn_status_col} != "No Turn" ' +
            'and (drive_mode_pre_event == "Longitudinal + Lateral" ' +
            'and drive_mode_event == "Longitudinal + Lateral" ' +
            'and drive_mode_post_event != "Longitudinal + Lateral" )'
        ).index

        df.loc[disengagement_after_event_indices,
               'disengagement_in_event'] = 'Disengagement after event'

        if self.multi_col_kpi:

            turn_counts = df.query(
                f'{event_type_col} == "Turn_status" ' +
                f'and {turn_status_col} != "No Turn" ').groupby(
                [turn_status_col,
                 turn_indicator_presence_col])[turn_status_col].count()

            lane_change_counts = df.query(
                f'{event_type_col} == "DA_status" ' +
                f'and {turn_indicator_presence_col} != "No Turn"').groupby(
                    [lane_change_col,
                     turn_indicator_presence_col])[lane_change_col].count()

            disengagement_in_event_counts = df.groupby(
                ['disengagement_in_event',
                 turn_indicator_presence_col])[turn_status_col].count()

            disengagement_after_event_counts = df.groupby(
                ['disengagement_after_event',
                 turn_indicator_presence_col])[turn_status_col].count()

            req_output = pd.concat([lane_change_counts,
                                    turn_counts,
                                    disengagement_in_event_counts,
                                    disengagement_after_event_counts,
                                    ],
                                   axis=0)

            req_output = req_output.reset_index(level=1)

            req_output.index.name = 'Vehicle status / Description'

            req_output = req_output.query('index in @req_categories').rename(
                columns={0: 'Count',
                         'turn_indicator_presence': 'Turn indicator status'}
            )

            if not 'Disengagement during event' in req_output.index:
                req_output.loc['Disengagement during event'] = [
                    'Unknown', 0]

            if not 'Disengagement after event' in req_output.index:
                req_output.loc['Disengagement after event'] = ['Unknown', 0]

        else:

            turn_counts = df.query(
                f'{event_type_col} == "Turn_status" ' +
                f'and {turn_status_col} != "No Turn" ')[
                    turn_status_col].value_counts(
                normalize=False,
                sort=True,
                ascending=False,
                dropna=True
            )

            lane_change_counts = df.query(
                f'{event_type_col} == "DA_status" ' +
                f'and {turn_indicator_presence_col} != "No Turn"')[
                    lane_change_col].value_counts(
                normalize=False,
                sort=True,
                ascending=False,
                dropna=True
            )

            disengagement_in_event_counts = \
                df['disengagement_in_event'].value_counts(
                    normalize=False,
                    sort=True,
                    ascending=False,
                    dropna=True
                )

            disengagement_after_event_counts = \
                df['disengagement_after_event'].value_counts(
                    normalize=False,
                    sort=True,
                    ascending=False,
                    dropna=True
                )

            req_output = pd.concat([lane_change_counts,
                                    turn_counts,
                                    disengagement_in_event_counts,
                                    disengagement_after_event_counts,
                                    ],
                                   axis=0)

            req_output.index.name = 'Vehicle status / Description'

            req_output = req_output.to_frame().query('index in @req_categories')

            if not 'Disengagement during event' in req_output.index:
                req_output.loc['Disengagement during event'] = 0

            if not 'Disengagement after event' in req_output.index:
                req_output.loc['Disengagement after event'] = 0

        with pd.ExcelWriter(output_excel_sheet,
                            engine='openpyxl',
                            mode='a') as writer:
            req_output.to_excel(writer,
                                sheet_name='kpi_overview',

                                )

        if "win" in sys.platform:

            return req_output
    
    def _run_helper(self, file_name, is_continuous=False,  **kwargs):
        """
        Description: Main execution helper for DA event extraction.
        Input: file_name (str or list), is_continuous (bool), program (str in kwargs)
        Output: Dictionary containing 'output_data_da' DataFrame or Array.
        """
        import pandas as pd
        import numpy as np
        import warnings
        import sys
        import os
        import pickle
        from pathlib import Path
        warnings.filterwarnings("ignore")

        # 1. DYNAMIC PATH SETUP
        # Automatically find the 'src' directory relative to this script
        # Assumes structure: GPO_Data_Mining_Analysis/scripts/E2E/find_da_generic.py
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parents[2] 
        src_path = str(project_root / "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Determine the root path for data/config lookups
        if "lin" in sys.platform:
            # Fallback for HPC if the dynamic path doesn't exist, otherwise use dynamic
            root_path = src_path if os.path.exists(src_path) else r'/mnt/usmidet/projects/STLA-THUNDER/7-Tools/DMA_Venv/CES_related/GPO_event_extraction/GPO_Data_Mining_Analysis/src'
        else:
            root_path = src_path

        # 2. DYNAMIC CONFIG PATH
        program_key = kwargs.get('program', 'e2e').lower()
        config_file = self.program_config_name_map.get(program_key, 'config_e2e_v1_da_basic.yaml')
        
        if 'config_path' not in kwargs or not kwargs['config_path']:
            kwargs['config_path'] = os.path.join(root_path, 'eventExtraction', 'data', kwargs.get('program', 'E2E'), config_file)

        # 3. RELATIVE IMPORTS (Using the search path defined above)
        from eventExtraction.da.core_da import coreEventExtractionDA
        from eventExtraction.utils.utils_generic import (
            _list_of_dicts_to_dict_of_arrays,
            _write_gifs_to_excel,
            _resim_path_to_orig_path,
            loadmat,
            _create_flat_file_list,
            _create_base_name_list_from_file_list,
            create_mysql_engine_fn
        )

        try:
            # CASE A: CONTINUOUS PROCESSING (File Lists)
            if is_continuous:
                if isinstance(file_name, list):
                    file_name_list = file_name
                    (log_path_list, base_name_list, rTag_list, seq_path_list, original_log_name_list) = \
                        _create_base_name_list_from_file_list(file_name_list)
                else:
                    (file_name_list, base_name_list, rTag_list, seq_path_list, original_log_name_list) = \
                        _create_flat_file_list(file_name, SIMULATE_LINUX=self.SIMULATE_LINUX)

                if self.debug_win:
                    file_name_list = file_name_list[:1]

                df_list = []
                collect_cTime = True

                for idx, (file_name_flat, basename, seq_path, orig_log_name) in enumerate(
                        zip(file_name_list, base_name_list, seq_path_list, original_log_name_list)):

                    log_path, log_name = os.path.split(file_name_flat)
                    
                    try:
                        print(f'\n########## Processing Log: {log_name}')
                        mat_file_data = loadmat(file_name_flat)
                        DA_core_logic_obj = coreEventExtractionDA(mat_file_data, file_name_flat)

                        out_df, enums_dict = DA_core_logic_obj._signal_mapping(kwargs['config_path'], log_name=log_name)

                        if collect_cTime:
                            start_cTime_sequence = np.array(out_df['cTime'])[0]
                            collect_cTime = False

                        df_list.append(out_df)
                    except Exception as e:
                        if not self.error_skip: raise e
                        print(f'*** Skipping corrupt log: {log_name}')
                        continue

                if not self.debug_win:
                    req_df = pd.concat(df_list, axis=0, ignore_index=True)
                    req_df['start_cTime_sequence'] = start_cTime_sequence
                    req_df, misc_out_dict = DA_core_logic_obj._process_signals(req_df, enums_dict)
                    DA_core_logic_obj.misc_out_dict = misc_out_dict

                    # Save pickle to local working directory instead of OneDrive
                    req_path_pickle = os.path.join(os.getcwd(), 'req_df_extraction_cache.pkl')
                    with open(req_path_pickle, 'wb') as file:
                        pickle.dump(req_df, file)
                else:
                    # In debug mode, attempt to load from local cache
                    df_path = os.path.join(os.getcwd(), 'req_df_extraction_cache.pkl')
                    if os.path.exists(df_path):
                        with open(df_path, 'rb') as file:
                            req_df = pickle.load(file)
                    else:
                        print("Debug cache not found, processing from scratch.")
                        req_df = pd.concat(df_list, axis=0, ignore_index=True)

                req_df['event_signal'] = 'DA'
                DA_core_logic_obj.program_name_readff_map = self.program_name_readff_map[kwargs['program'].lower()]
                return_val_dict = DA_core_logic_obj.event_extraction(req_df, **DA_core_logic_obj.kwargs_processing)

            # CASE B: SINGLE FILE PROCESSING
            else:
                log_path, log_name = os.path.split(file_name)
                mat_file_data = loadmat(file_name)
                DA_core_logic_obj = coreEventExtractionDA(mat_file_data, file_name)
                mat_file_data = None # Free memory

                out_df, enums_dict = DA_core_logic_obj._signal_mapping(kwargs['config_path'], log_name=log_name)
                start_cTime_sequence = np.array(out_df['cTime'])[0]
                out_df['start_cTime_sequence'] = start_cTime_sequence

                out_df, misc_out_dict = DA_core_logic_obj._process_signals(out_df, enums_dict)
                DA_core_logic_obj.misc_out_dict = misc_out_dict

                out_df['event_signal'] = 'DA'
                DA_core_logic_obj.program_name_readff_map = self.program_name_readff_map[kwargs['program'].lower()]

                return_val_dict = DA_core_logic_obj.event_extraction(out_df, **DA_core_logic_obj.kwargs_processing)

            # 4. PREPARE OUTPUT
            try:
                return_val_df_udp_based = pd.DataFrame(return_val_dict.get('overall_UDP', {}))
            except:
                print('Handling uneven length output...')
                return_val_df_udp_based = pd.DataFrame.from_dict(return_val_dict.get('overall_UDP', {}), orient='index').transpose()

            if not return_val_df_udp_based.empty:
                # Filter by headers defined in __init__
                valid_headers = [h for h in self._headers['output_data_da'] if h in return_val_df_udp_based.columns]
                return_val_df_udp_based = return_val_df_udp_based[valid_headers]

            out_dict = dict()
            if kwargs.get('local', False):
                out_dict['output_data_da'] = return_val_df_udp_based
            else:
                out_dict['output_data_da'] = np.array(return_val_df_udp_based)

            return out_dict

        # Replace the return string at the end of _run_helper (around line 1050)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_msg = f"{str(e)} FOUND IN LINE: {str(exc_tb.tb_lineno)}"
            print(f"CRITICAL ERROR: {error_msg}")
            raise # This will stop the script and show you exactly what failed
    # def _run_helper(self, file_name, is_continuous=False,  **kwargs):
    #     ###########################################################################

    #     ###########################################################################
    #     # ADD A BRIEF DESCRIPTION OF THE LOGIC USED IN THIS SCRIPT HERE
    #     # ALONG WITH INPUT PARAMS IF ANY OTHER THAN THE MAT FILE
    #     ## Mandatory details for th script ##
    #     """
    #     Description: Write briefly about your script
    #     Input to the script: filelist with file names
    #     Input file type: .mat or .mudp
    #     Output of the script: Excel sheet with events

    #     """
    #     import pandas as pd
    #     import numpy as np
    #     import warnings
    #     import sys
    #     import os
    #     from pathlib import Path
    #     import pickle
    #     warnings.filterwarnings("ignore")
    #     ###########################################################################

    #     ###########################################################################
    #     # DO NOT REMOVE THE "TRY" STATEMENT
    #     if "lin" in sys.platform:

    #         root_path = os.path.join(r'/mnt/usmidet/projects/STLA-THUNDER',
    #                                  r'7-Tools/DMA_Venv/CES_related',
    #                                  'GPO_event_extraction',
    #                                  'GPO_Data_Mining_Analysis',
    #                                  'src',
    #                                  )

    #         # sys.path.insert(0, root_path)
    #         from GPO_Data_Mining_Analysis.src.eventExtraction.da.core_da import coreEventExtractionDA
    #         from GPO_Data_Mining_Analysis.src.eventExtraction.utils.utils_generic import (_list_of_dicts_to_dict_of_arrays,
    #                                                          _write_gifs_to_excel,
    #                                                          _resim_path_to_orig_path,
    #                                                          loadmat,
    #                                                          _create_flat_file_list,
    #                                                          _create_base_name_list_from_file_list,
    #                                                          )
    #         kwargs['config_path'] = os.path.join(root_path,
    #                                              'eventExtraction',
    #                                              'data',
    #                                              kwargs['program'],
    #                                              self.program_config_name_map[
    #                                                  kwargs['program'].lower()])

    #     elif "win" in sys.platform:
    #         print('****************', os.getcwd())
    #         # root_path = os.path.join(r'C:\Users\mfixlz',
    #         #                          r'OneDrive - Aptiv\Documents\DM_A\PO_Chaitanya_K',
    #         #                          r'Projects\GPO Data Mining Analysis',
    #         #                          'GPO_Data_Mining_Analysis',
    #         #                          'src',)
    #         # sys.path.insert(0, root_path)
    #         from GPO_Data_Mining_Analysis.src.eventExtraction.da.core_da import coreEventExtractionDA
    #         # from eventExtraction.tsi.signal_mapping_tsi import signalMapping
    #         from GPO_Data_Mining_Analysis.src.eventExtraction.utils.utils_generic  import (_list_of_dicts_to_dict_of_arrays,
    #                                                          _write_gifs_to_excel,
    #                                                          _resim_path_to_orig_path,
    #                                                          loadmat,
    #                                                          _create_flat_file_list,
    #                                                          _create_base_name_list_from_file_list,
    #                                                          )
    #     import pickle

    #     from GPO_Data_Mining_Analysis.src.eventExtraction.utils.utils_generic  import create_mysql_engine_fn

    #     try:

    #         if is_continuous:

    #             if isinstance(file_name, list):

    #                 file_name_list = file_name
    #                 (log_path_list,
    #                  base_name_list,
    #                  rTag_list,
    #                  seq_path_list,
    #                  original_log_name_list) = _create_base_name_list_from_file_list(
    #                     file_name_list)

    #             else:

    #                 (file_name_list, base_name_list, rTag_list,
    #                  seq_path_list, original_log_name_list) = \
    #                     _create_flat_file_list(
    #                         file_name,
    #                         SIMULATE_LINUX=self.SIMULATE_LINUX)

    #             if self.debug_win:

    #                 file_name_list = file_name_list[:1]

    #             df_list = []
    #             collect_cTime = True

    #             for idx, (file_name_flat, basename,
    #                       seq_path, orig_log_name) in enumerate(
    #                     zip(file_name_list, base_name_list,
    #                         seq_path_list, original_log_name_list)):

    #                 log_path, log_name = os.path.split(file_name_flat)
    #                 rTag = rTag_list[idx]

    #                 if self.error_skip:

    #                     try:

    #                         mat_file_data = loadmat(file_name_flat)
    #                         # TSI_signal_map_obj = signalMapping(mat_file_data)
    #                         print('\n########## Object initialisation\n',
    #                               f'{log_name}')
    #                         DA_core_logic_obj = coreEventExtractionDA(
    #                             mat_file_data, file_name_flat)

    #                         print('\n########## Signal interface mapping\n')
    #                         out_df, enums_dict = DA_core_logic_obj._signal_mapping(
    #                             kwargs['config_path'], log_name=log_name)

    #                         if collect_cTime:
    #                             start_cTime_sequence = np.array(
    #                                 out_df['cTime'])[0]
    #                             collect_cTime = False

    #                         df_list.append(out_df)
    #                     except Exception:
    #                         exc_type, exc_obj, exc_tb = sys.exc_info()
    #                         print('***************************',
    #                               f'Log at with name "{log_name}" is corrupt ',
    #                               f"or missing required streams as per {kwargs['config_path']}"
    #                               '\n skipping the log' +
    #                               f'\n logpath is : {file_name_flat}')
    #                         print(str(exc_obj.args[0])
    #                               + " FOUND IN LINE: " + str(exc_tb.tb_lineno))
    #                         continue
    #                 else:
    #                     print('\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n',
    #                           f'COMPLETE FILE PATH : {file_name_flat}',
    #                           '\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n')
    #                     mat_file_data = loadmat(file_name_flat)
    #                     # TSI_signal_map_obj = signalMapping(mat_file_data)
    #                     print('\n########## Object initialisation\n',
    #                           f'{log_name}')
    #                     DA_core_logic_obj = coreEventExtractionDA(
    #                         mat_file_data, file_name_flat)

    #                     print('\n########## Signal interface mapping\n')
    #                     out_df, enums_dict = DA_core_logic_obj._signal_mapping(
    #                         kwargs['config_path'], log_name=log_name)

    #                     if collect_cTime:
    #                         start_cTime_sequence = np.array(out_df['cTime'])[0]
    #                         collect_cTime = False

    #                     df_list.append(out_df)

    #             if not self.debug_win:
    #                 req_df = pd.concat(df_list, axis=0, ignore_index=True)

    #                 req_df['start_cTime_sequence'] = start_cTime_sequence

    #                 req_df, misc_out_dict = \
    #                     DA_core_logic_obj._process_signals(req_df,
    #                                                        enums_dict)

    #                 DA_core_logic_obj.misc_out_dict = misc_out_dict

    #                 if 'win' in sys.platform:

    #                     req_path_pickle = os.path.join(r'C:\Users\mfixlz',
    #                                                    r'OneDrive - Aptiv\Documents\DM_A\PO_Vinay',
    #                                                    r'Projects\2025\KW52',
    #                                                    'req_df_8.pkl')

    #                     print('**********************************',
    #                           req_path_pickle)

    #                     with open(req_path_pickle, 'wb') as file:

    #                         pickle.dump(req_df, file)
    #             else:

    #                 # req_path_pickle = os.path.join(os.getcwd(), 'req_df_3.pkl')

    #                 # print('**********************************',
    #                 #       req_path_pickle)

    #                 # with open(req_path_pickle, 'wb') as file:

    #                 #     pickle.dump(req_df, file)

    #                 df_path = os.path.join(r'C:\Users\mfixlz',
    #                                        r'OneDrive - Aptiv\Documents\DM_A\PO_Vinay',
    #                                        r'Projects\2025\KW52',
    #                                        'req_df_8.pkl')
    #                 with open(df_path, 'rb') as file:
    #                     req_df = pickle.load(file)

    #             # seq_name = '_'.join(log_name.split('.')[0].split('_')[:-2])
    #             # seq_name = '_'.join(basename.split('_')[:-1])

    #             # req_df['seq_name'] = seq_name
    #             # FIXME: make it compatible for THUNDER
    #             # seq_path = os.path.dirname(log_path)

    #             # log_name = seq_name
    #             # log_path = seq_path

    #             # req_df['sequence_path'] = seq_path
    #             # req_df['sequence_name'] = seq_name

    #             # req_df['sequence_path'] = req_df['base_logname'].apply(
    #             #     lambda x: os.path.join(seq_path, x))
    #             # FIXME:
    #             # if kwargs['program'].lower() == 'mcip':

    #             #     req_df['sequence_path'] = req_df['base_logname'].apply(
    #             #         lambda x: os.path.join(seq_path, x))
    #             # elif kwargs['program'].lower() == 'thunder':

    #             #     req_df['sequence_path'] = log_path

    #             # if kwargs['program'].lower() == 'e2e':

    #             #     req_df['sequence_path'] = req_df['base_logname'].apply(
    #             #         lambda x: os.path.join(seq_path, x))

    #             # req_df['readff_link'] = req_df[['sequence_path',
    #             #                                 'orig_log_name_flat']].apply(
    #             #     lambda x:
    #             #         'https://readff-na.aptiv.com/vidFile/' +
    #             #         f'{os.path.join(*Path(x["sequence_path"]).parts[3:])}/' +
    #             #         '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
    #             #             *self.readff_map_dict[
    #             #                 self.program_name_readff_map[
    #             #                     kwargs["program"].lower()]])), axis=1
    #             # )

    #             # req_df['readff_link_full_video'] = req_df[['sequence_path',
    #             #                                            'orig_log_name_flat']].apply(
    #             #     lambda x:
    #             #         'https://readff-na.aptiv.com/files/' +
    #             #         f'{os.path.join(*Path(x["sequence_path"]).parts[3:])}/' +
    #             #         '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
    #             #             *self.readff_map_dict[
    #             #                 self.program_name_readff_map[
    #             #                     kwargs["program"].lower()]])), axis=1
    #             # )

    #             # req_df['frame_ID'] = (
    #             #     req_df['vision_avi_tsr_camera_frame_ID'] -
    #             #     req_df['vision_avi_tsr_camera_frame_ID'].min(
    #             #         skipna=True)
    #             # )
    #             req_df['event_signal'] = 'DA'
    #             DA_core_logic_obj.program_name_readff_map = \
    #                 self.program_name_readff_map[kwargs['program'].lower()]
    #             # try:
    #             #     DA_core_logic_obj.df_GT = \
    #             #         pd.read_sql(f'SELECT * FROM {self.table_name_GT}',
    #             #                     con=create_mysql_engine_fn())
    #             # except:
    #             #     print('Error with MySQL connection. GT values shall be nan')
    #             #     DA_core_logic_obj.df_GT = pd.DataFrame()

    #             return_val_dict = DA_core_logic_obj.event_extraction(
    #                 req_df,
    #                 **DA_core_logic_obj.kwargs_processing)

    #         else:

    #             log_path, log_name = os.path.split(file_name)

    #             mat_file_data = loadmat(file_name)
    #             DA_core_logic_obj = coreEventExtractionDA(mat_file_data,
    #                                                       file_name)
    #             mat_file_data = None

    #             out_df, enums_dict = DA_core_logic_obj._signal_mapping(
    #                 kwargs['config_path'], log_name=log_name)

    #             start_cTime_sequence = np.array(out_df['cTime'])[0]

    #             out_df['start_cTime_sequence'] = start_cTime_sequence

    #             out_df, misc_out_dict = \
    #                 DA_core_logic_obj._process_signals(out_df,
    #                                                    enums_dict)

    #             DA_core_logic_obj.misc_out_dict = misc_out_dict

    #             # seq_name = '_'.join(log_name.split('.')[0].split('_')[:-2])
    #             # seq_name = '_'.join(basename.split('_')[:-1])

    #             # out_df['seq_name'] = seq_name
    #             # FIXME: make it compatible for THUNDER
    #             # seq_path = os.path.dirname(log_path)

    #             # log_name = seq_name
    #             # log_path = seq_path

    #             # if kwargs['program'].lower() == 'mcip':

    #             #     out_df['sequence_path'] = out_df['base_logname'].apply(
    #             #         lambda x: os.path.join(seq_path, x))
    #             # elif kwargs['program'].lower() == 'thunder':

    #             #     out_df['sequence_path'] = log_path

    #             # out_df['readff_link'] = out_df[['sequence_path',
    #             #                                 'orig_log_name_flat']].apply(
    #             #     lambda x:
    #             #         'https://readff-na.aptiv.com/vidFile/' +
    #             #         f'{os.path.join(*Path(x["sequence_path"]).parts[4:])}/' +
    #             #         '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
    #             #             *self.readff_map_dict[
    #             #                 self.program_name_readff_map[
    #             #                     kwargs["program"].lower()]])), axis=1
    #             # )

    #             # out_df['readff_link_full_video'] = out_df[['sequence_path',
    #             #                                            'orig_log_name_flat']].apply(
    #             #     lambda x:
    #             #         'https://readff-na.aptiv.com/files/' +
    #             #         f'{os.path.join(*Path(x["sequence_path"]).parts[4:])}/' +
    #             #         '{}.MF4'.format(x["orig_log_name_flat"].split(".")[0].replace(
    #             #             *self.readff_map_dict[
    #             #                 self.program_name_readff_map[
    #             #                     kwargs["program"].lower()]])), axis=1
    #             # )

    #             # out_df['frame_ID'] = (
    #             #     out_df['vision_avi_tsr_camera_stream_ref_index'] -
    #             #     out_df['vision_avi_tsr_camera_stream_ref_index'].min(
    #             #         skipna=True)
    #             # )
    #             # out_df['frame_ID'] = (
    #             #     out_df['vision_avi_tsr_camera_frame_ID'] -
    #             #     out_df['vision_avi_tsr_camera_frame_ID'].min(
    #             #         skipna=True)
    #             # )
    #             # out_df['frame_ID'] = np.array([
    #             #     f_id
    #             #     for f_id in range(
    #             #         len(out_df[
    #             #             'vision_avi_tsr_camera_frame_ID']))],
    #             #     dtype=int)
    #             out_df['event_signal'] = 'DA'
    #             DA_core_logic_obj.program_name_readff_map = \
    #                 self.program_name_readff_map[kwargs['program'].lower()]

    #             return_val_dict = DA_core_logic_obj.event_extraction(
    #                 out_df,
    #                 **DA_core_logic_obj.kwargs_processing)

    #         try:
    #             return_val_df_udp_based = pd.DataFrame(
    #                 return_val_dict.get('overall_UDP', {}))

    #         except:
    #             print('***********************************',
    #                   'uneven lengths in the output. Debug',
    #                   '***********************************',)
    #             return_val_df_udp_based = pd.DataFrame.from_dict(
    #                 return_val_dict.get('overall_UDP', {}),
    #                 orient='index')

    #             return_val_df_udp_based = return_val_df_udp_based.transpose()

    #         if not return_val_df_udp_based.empty:
    #             return_val_df_udp_based = \
    #                 return_val_df_udp_based[
    #                     self._headers['output_data_da']]

    #         out_dict = dict()
    #         if 'local' in kwargs and kwargs['local']:

    #             out_dict['output_data_da'] = return_val_df_udp_based

    #         else:

    #             print('********** Writing values to dict')

    #             out_dict['output_data_da'] = np.array(
    #                 return_val_df_udp_based)

    #         return out_dict

    #     #######################################################################
    #     # DO NOT CHANGE THIS PART
    #     except Exception:
    #         exc_type, exc_obj, exc_tb = sys.exc_info()
    #         return str(exc_obj.args[0]) + " FOUND IN LINE: " + str(exc_tb.tb_lineno)

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
    # from ray.util.joblib import register_ray # Comment out if ray is not installed
    from functools import reduce
    from datetime import datetime
    from pathlib import Path
    import psutil
    import time

    warnings.filterwarnings("ignore")

    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss, mem_info.vms

    def secondsToStr(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(t*1000,), 1000, 60, 60])

    start_time = time.time()
    mem_before_phy, mem_before_virtual = process_memory()

    program = 'E2E'  
    config_name = 'config_e2e_v1_da_basic.yaml'

    # 1. DYNAMIC PATH RESOLUTION
    current_script_path = Path(__file__).resolve()
    PROJECT_ROOT = current_script_path.parents[2] # Points to GPO_Data_Mining_Analysis/

    if "win" in sys.platform:
        print(f"Project Root Detected: {PROJECT_ROOT}")
        
        # 2. LOCAL DATA PATH (IGNITION CYCLE SETUP)
        data_dir = PROJECT_ROOT / 'data' / program / 'extracted_data'
        
        # List all files belonging to the same ignition cycle here in order
        file_names = [
            'SDV_E2EML_M16_20251122_180824_0000_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0001_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0002_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0003_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0004_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0005_merged.mat',
            'SDV_E2EML_M16_20251122_180824_0006_merged.mat',
        ]
        
        # Convert the list of filenames into absolute path strings
        file_name_paths = [str(data_dir / f) for f in file_names]

        # 3. DYNAMIC CONFIG PATH
        local_config = str(PROJECT_ROOT / 'src' / 'eventExtraction' / 'data' / program / config_name)

        kwargs = {
            'local': True,
            'program': program,
            'config_path': local_config
        }
        
        print(f"Starting Ignition Cycle with {len(file_name_paths)} files.")
        print(f"Using Config: {local_config}")

    elif "lin" in sys.platform:
        # Dynamic Linux path resolution (Example)
        cont_file_list_name = 'SDV_E2EML_M16_20251229_111748_[0000]_p01.mat, 19'
        file_name_paths = os.path.join('/mnt/usmidet/projects/GPO-E2E', '9-Upload/KPI_Drives_KPI_Data/12292025',
                                       'M16/AIRT_111748_route_3_v0332', 'SDV_E2EML_M16_20251229_111748_0000', cont_file_list_name)
        kwargs = {'program': program}

    # 4. EXECUTION
    class_obj = FindDaGeneric()
    # If file_name_paths is a list, the run method internally triggers continuous processing
    final_out = class_obj.run(file_name_paths, **kwargs)

    # 5. ERROR HANDLING
    if isinstance(final_out, str):
        print("\n" + "!"*30)
        print(f"CRITICAL ERROR IN CYCLE EXTRACTION:\n{final_out}")
        print("!"*30 + "\n")
        sys.exit(1)

    # --- 6. OUTPUT DIRECTORY SETUP ---
    output_dir = PROJECT_ROOT / 'output'
    os.makedirs(output_dir, exist_ok=True)

    # --- 7. OUTPUT SAVING ---
    # Define a base name for the cycle (using the first file name + suffix)
    if isinstance(file_name_paths, list):
        mat_base_name = Path(file_names[0]).stem + "_cycle"
    else:
        mat_base_name = Path(file_name_paths).stem 
    
    json_save_path = output_dir / f"{mat_base_name}.json"
    excel_save_path = output_dir / f"{mat_base_name}.xlsx"

    if 'output_data_da' in final_out:
        # Save combined cycle results to JSON
        final_out['output_data_da'].to_json(
            json_save_path, 
            orient='records', 
            indent=4
        )
        print(f"Cycle JSON generated: {json_save_path}")
        
        # Save combined cycle results to Excel
        final_out['output_data_da'].to_excel(
            excel_save_path, 
            sheet_name='output_data_da', 
            index=False
        )
        print(f"Cycle Excel generated: {excel_save_path}")
        
        # 8. KPI GENERATION
        try:
            kpi_output = class_obj.kpi_sheet_generation(str(excel_save_path))
        except Exception as e:
            print(f"Warning: KPI sheet generation failed: {e}")
    else:
        print("Error: 'output_data_da' not found in cycle extraction output.")

    # Performance Monitoring
    mem_after_phy, mem_after_virtual = process_memory()
    end_time = time.time()
    elapsed_time = secondsToStr(end_time-start_time)
    consumed_memory_phy = (mem_after_phy - mem_before_phy)*1E-6
    
    print(f'Done! Total Cycle Elapsed time: {elapsed_time}')
    print(f'Consumed physical memory: {consumed_memory_phy:.2f} MB')
