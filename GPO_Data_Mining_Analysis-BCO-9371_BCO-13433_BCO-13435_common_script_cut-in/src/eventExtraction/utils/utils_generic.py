# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:39:47 2024

@author: mfixlz
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from functools import reduce
from scipy.io import loadmat as load_mat_scipy
import re
import pickle
from fxpmath import Fxp
from collections import defaultdict
from collections.abc import Mapping
import copy
from itertools import combinations
import sys
import hashlib

import xlsxwriter
import math
import json
from pathlib import Path

import more_itertools as mit
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.BlackAndWhite import BlackAndWhite
import logging

from sqlalchemy import create_engine, text
# from sqlalchemy.dialects.mysql import insert
# import mysql.connector

from geographiclib.geodesic import Geodesic


if __package__ is None:
    from os import path
    print('Here at none package')
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    to_change_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(to_change_path)
    print(f'Current dir: {os.getcwd()}, to change : {to_change_path}')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(1, str(Path(__file__).resolve().parent))

    from utils.mat73 import loadmat as loadmat_v7_3
else:

    # from .. import utils
    sys.path.insert(0,
                    os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))

    os.chdir(to_change_path)
    print(f'Current dir *: {os.getcwd()}, \n to change *: {to_change_path}')
    try:

        from utils.mat73 import loadmat as loadmat_v7_3
    except:

        from mat73 import loadmat as loadmat_v7_3


class Fxp_2(Fxp, ):

    def __init__(self, dtype_fxp: str, **kwargs):

        super().__init__()

        self.fxp_dtype = Fxp(dtype=dtype_fxp)

    def __call__(self, value: np.ndarray):

        value = '0b'+''.join(map(str, np.unpackbits(value)))
        return_val = self.fxp_dtype(value).get_val()

        return return_val


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


def deep_get(log_data, keys: str) -> Any:
    """
    Get value of given stream from data parameter
    :param keys: name of stream (eg. dvlExtDBC.FDCAN14.ADAS_FD_INFO.Prefill_Req)
    :return: Value of given key or None if no key in dictionary
    """
    return reduce(lambda d, key: d.get(key)
                  if isinstance(d, dict)
                  else None, keys.split("."), log_data)


def stream_check(raw_data, stream: str) -> Any:
    """
    Function to extract and check if stream is valid (not none).
    If stream is invalid error parameter is updated with that stream
    :param stream: name of string (eg. dvlExtDBC.FDCAN14.ADAS_FD_INFO.Prefill_Req)
    :return: None if stream is valid or data from that stream
    """
    stream_data = deep_get(raw_data, stream)
    error = ''
    if stream_data is None:
        error += "signal {} not available".format(stream)
        return error
    elif type(stream_data) == str:
        error += "{}: {}".format(stream, stream_data)
        return error
    else:
        return stream_data


def get_from_dict(dataDict,
                  mapList,
                  default_error: str = 'unknown'):
    """Iterate nested dictionary"""
    try:
        return reduce(dict.get, mapList, dataDict)
    except TypeError:
        return default_error


def array_at_mat_dict(data, mat_path):
    """
    A function which constructs from mat_path like nested dictionaries and return whole array.

    :param data, mat_path
    :ret array at specific path
    """
    mat_path = "['" + mat_path.replace('.', "']['") + "']"
    try:
        return eval('data' + mat_path)
    except:
        raise MissingMatfilePath(mat_path)


def read_platform(raw_data) -> str:
    """
    Function to check vehicle platform for given log
    :return: Name of platform
    """
    try:
        proxi_data = raw_data['mudp']['inst']['Proxi']
    except (KeyError, TypeError):
        raise Exception("No stream: mudp.inst.Proxi")
    if type(proxi_data) == str:
        raise Exception('Wrong mudp.inst.Proxi data: {}'.format(proxi_data))
    if 'Wheelbase' not in proxi_data.keys() or 'Vehicle_Line_Configuration' not in proxi_data.keys():
        raise Exception(
            'Wheelbase or Vehicle_Line_Configuration stream not in mudp.inst.Proxi data')

    def extract_proxi_data(proxi_stream: str) -> int:
        stream_data = proxi_data[proxi_stream]
        if type(stream_data).__module__ == np.__name__:
            stream_val = stream_data[-1]
        elif type(stream_data) == int:
            stream_val = stream_data
        else:
            raise Exception('Wrong {} data: {}'.format(
                proxi_stream, stream_data))
        return stream_val

    wheel_base = extract_proxi_data('Wheelbase')
    vehicle_line = extract_proxi_data('Vehicle_Line_Configuration')

    if vehicle_line == 101 and wheel_base == 2:  # WL75
        return 'WL'
    elif vehicle_line == 101 and wheel_base == 1:  # WL74
        return 'WL'
    elif vehicle_line == 104 and (wheel_base == 1 or wheel_base == 2):  # WS
        return 'WL'
    elif vehicle_line == 108:  # MAS182
        return 'WL'
    elif vehicle_line == 109:  # MAS189
        return 'WL'
    elif vehicle_line == 124:  # DT
        return 'WL'
    elif vehicle_line == 49:  # JLBUX
        return 'JL_BUX'
    else:
        raise Exception(
            'Unknown platform (vehicle line: {})'.format(vehicle_line))


def loadmat(file_name: os.path.join):

    try:
        kwargs_load_mat_scipy = {
            # 'struct_as_record' : False,
            # 'squeeze_me' : True,
            'simplify_cells': True,
            'verify_compressed_data_integrity': True
        }
        data = load_mat_scipy(file_name,
                              **kwargs_load_mat_scipy
                              )
    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('***************************',
              f'failed to load "{file_name}" with scipy ',
              'trying with mat73 ')
        print(str(exc_obj.args[0])
              + " FOUND IN LINE: " + str(exc_tb.tb_lineno),
              'After failing at scipy mat file load')

        data = loadmat_v7_3(file_name,)
    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('***************************',
              f'failed to load "{file_name}" with scipy and mat73 ',
              'going to raise the error ')
        print(str(exc_obj.args[0])
              + " FOUND IN LINE: " + str(exc_tb.tb_lineno),
              'After failing at scipy mat file load and mat73')

        raise Exception(f'Error in loading matfile at {file_name},')

    return data


def transform_df(data_dict, cTime_check: bool = True):
    """
    Creates dataFrame from nested eyeq data

    Args:
        data_dict: dictionary with eyeq data

    Returns:
        Pandas dataFrame with extracted data
    """
    df_list = []

    if cTime_check:

        if ('cTime' in data_dict.keys()
                and isinstance(data_dict['cTime'], str)):

            return list(data_dict.keys())

        data_dict = {key: val/1E6
                     if 'cTime' in key
                     else val
                     for key, val in data_dict.items()
                     }
    max_len = max([len(item)
                   for item in data_dict.values()])
    for key, value in data_dict.items():

        if isinstance(value, str) and 'not available' in value:

            df = pd.DataFrame({key: [value]*max_len})

        elif isinstance(value[0], str) and len(value) == 1:

            df = pd.DataFrame({key: value*max_len})
        else:
            if value.size != len(value):
                df = pd.DataFrame(
                    value, columns=[key + '_' + str(i)
                                    for i in range(value.shape[1])
                                    ])
            else:
                df = pd.DataFrame({key: value})

        cTime_cols = [col for
                      col in list(df.columns) if 'cTime' in col]
        if len(cTime_cols) > 1:

            sorted_cTime_cols = sort_list(cTime_cols)
            col_to_remove = sorted_cTime_cols[1:]
            col_to_keep = sorted_cTime_cols[0]
            df = df.drop(columns=col_to_remove)
            df = df.rename(columns={col_to_keep: 'cTime'})
        df_list.append(df)

    return pd.concat(df_list, axis=1).apply(pd.to_numeric, errors='ignore')


def merge_pandas_df(list_of_dfs,
                    merge_key_list: List[str] = 'cTime'):

    df_merged = reduce(lambda left, right: pd.merge_asof(left, right,
                                                         on=merge_key_list,
                                                         direction='nearest',
                                                         ),
                       list_of_dfs
                       )

    df_merged = df_merged.apply(pd.to_numeric, errors='ignore')

    return df_merged


def sort_list(list_):
    def convert(text): return float(text) \
        if text.isdigit() else text
    def alphanum(key): return [convert(c)
                               for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    list_.sort(key=alphanum)
    return list_


def get_all_sw_ver_no_mid_map(mat_data):
    """
    This function returns the list all possible software version number as per CGQ-3740
    """

    # CADM_MID_SW_Version
    try:
        vehcal_data = mat_data['mudp']['vehCal']
        cadm_mid_sw_ver = (str(vehcal_data['variant_version'][0]) + '.' +
                           str(vehcal_data['major_version'][0]) + '.' +
                           str(vehcal_data['minor_version'][0]) + '.' +
                           str(vehcal_data['patch_version'][0]))
    except:
        cadm_mid_sw_ver = 'N/A'

    # CADM_MAP_SW_Version
    try:
        vehcal_data = mat_data['mudp']['MAP']['vehCal']
        cadm_map_sw_ver = (str(vehcal_data['variant_version'][0]) + '.' +
                           str(vehcal_data['major_version'][0]) + '.' +
                           str(vehcal_data['minor_version'][0]) + '.' +
                           str(vehcal_data['patch_version'][0]))
    except:
        cadm_map_sw_ver = 'N/A'

    # LRR_version
    try:
        try:
            host_sw_ver_data = mat_data['brr']['mrr']['z4']['Host_Sw_Version']
            cadm_mid_lrr_ver = (str(host_sw_ver_data[0, 0]) + '.' +
                                str(host_sw_ver_data[0, 1]) + '.' +
                                str(host_sw_ver_data[0, 2]) + '.' +
                                str(host_sw_ver_data[0, 3]))
        except:
            host_sw_ver_data = mat_data['dvlExtDBC']['LRR']['LRRF1_Status_SwVersion']
            cadm_mid_lrr_ver = (str(host_sw_ver_data['LRRF1_CAN_SW_Release_Revision'][0]) + '.' +
                                str(host_sw_ver_data['LRRF1_CAN_SW_Promote_Revision'][0]) + '.' +
                                str(host_sw_ver_data['LRRF1_CAN_SW_Field_Revision'][0]) + '.' +
                                str(host_sw_ver_data['LRRF1_CAN_SW_Field'][0]))

    except:
        cadm_mid_lrr_ver = 'N/A'

    # MRR_FR_version
    try:
        try:
            host_sw_ver_data = mat_data['brr']['srr_fr']['z4']['Host_Sw_Version']
            cadm_mid_mrr_fr_ver = (str(host_sw_ver_data[0, 0]) + '.' +
                                   str(host_sw_ver_data[0, 1]) + '.' +
                                   str(host_sw_ver_data[0, 2]) + '.' +
                                   str(host_sw_ver_data[0, 3]))
        except:
            host_sw_ver_data = mat_data['dvlExtDBC']['MRR_Right']['MRR_FR_SW_Version_Status']
            cadm_mid_mrr_fr_ver = (str(host_sw_ver_data['MRR_FR_CAN_SW_Release_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FR_CAN_SW_Promote_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FR_CAN_SW_Field_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FR_CAN_SW_Field'][0]))

    except:
        cadm_mid_mrr_fr_ver = 'N/A'

    # MRR_RR_version
    try:
        try:
            host_sw_ver_data = mat_data['brr']['srr_rr']['z4']['Host_Sw_Version']
            cadm_mid_mrr_rr_ver = (str(host_sw_ver_data[0, 0]) + '.' +
                                   str(host_sw_ver_data[0, 1]) + '.' +
                                   str(host_sw_ver_data[0, 2]) + '.' +
                                   str(host_sw_ver_data[0, 3]))
        except:
            host_sw_ver_data = mat_data['dvlExtDBC']['MRR_Right']['MRR_RR_SW_Version_Status']
            cadm_mid_mrr_rr_ver = (str(host_sw_ver_data['MRR_RR_CAN_SW_Release_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RR_CAN_SW_Promote_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RR_CAN_SW_Field_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RR_CAN_SW_Field'][0]))

    except:
        cadm_mid_mrr_rr_ver = 'N/A'

    # MRR_FL_version
    try:
        try:
            host_sw_ver_data = mat_data['brr']['srr_fl']['z4']['Host_Sw_Version']
            cadm_mid_mrr_fl_ver = (str(host_sw_ver_data[0, 0]) + '.' +
                                   str(host_sw_ver_data[0, 1]) + '.' +
                                   str(host_sw_ver_data[0, 2]) + '.' +
                                   str(host_sw_ver_data[0, 3]))
        except:
            host_sw_ver_data = mat_data['dvlExtDBC']['MRR_Left']['MRR_FL_SW_Version_Status']
            cadm_mid_mrr_fl_ver = (str(host_sw_ver_data['MRR_FL_CAN_SW_Release_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FL_CAN_SW_Promote_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FL_CAN_SW_Field_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_FL_CAN_SW_Field'][0]))

    except:
        cadm_mid_mrr_fl_ver = 'N/A'

    # MRR_RL_version
    try:
        try:
            host_sw_ver_data = mat_data['brr']['srr_rl']['z4']['Host_Sw_Version']
            cadm_mid_mrr_rl_ver = (str(host_sw_ver_data[0, 0]) + '.' +
                                   str(host_sw_ver_data[0, 1]) + '.' +
                                   str(host_sw_ver_data[0, 2]) + '.' +
                                   str(host_sw_ver_data[0, 3]))
        except:
            host_sw_ver_data = mat_data['dvlExtDBC']['MRR_Left']['MRR_RL_SW_Version_Status']
            cadm_mid_mrr_rl_ver = (str(host_sw_ver_data['MRR_RL_CAN_SW_Release_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RL_CAN_SW_Promote_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RL_CAN_SW_Field_Revision'][0]) + '.' +
                                   str(host_sw_ver_data['MRR_RL_CAN_SW_Field'][0]))

    except:
        cadm_mid_mrr_rl_ver = 'N/A'

    # MID TR version
    try:
        host_sw_ver_data = mat_data['dvlExtDBC']['INSTCAN_Mid']['CADM_SW_Version_Info1']
        mid_tr_version = (str(host_sw_ver_data['tr_software_version_aa'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_bb'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_cc'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_dd'][0]))
    except:
        mid_tr_version = 'N/A'

    # MAP TR version
    try:
        host_sw_ver_data = mat_data['dvlExtDBC']['INSTCAN_Map']['CADM_SW_Version_Info1']
        map_tr_version = (str(host_sw_ver_data['tr_software_version_aa'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_bb'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_cc'][0]) + '.' +
                          str(host_sw_ver_data['tr_software_version_dd'][0]))
    except:
        map_tr_version = 'N/A'

    # EYEQ version info
    try:
        host_sw_ver_data = mat_data['dvlExtDBC']['INSTCAN_Mid']['CADM_SW_Version_Info1']
        eyeq_version = (str(host_sw_ver_data['eyeqSwVersionMajor'][0]) + '.' +
                        str(host_sw_ver_data['eyeqSwVersionMinor'][0]) + '.' +
                        str(host_sw_ver_data['eyeqSwVersionSubMajor'][0]) + '.' +
                        str(host_sw_ver_data['eyeqSwVersionSubMinor'][0]))
    except:
        eyeq_version = 'N/A'

    # Tracker version
    try:
        tracker_ver = mat_data['mudp']['fus']['log_data_fusion_tracker']['trackerLibVersion']
        tracker_version = (str(tracker_ver[0][0]) + '.' +
                           str(tracker_ver[0][1]) + '.' +
                           str(tracker_ver[0][2]) + '.' +
                           str(tracker_ver[0][3]))
    except:
        tracker_version = 'N/A'

    # SATA version
    try:
        sata_version_stream = mat_data['mudp']['tsel']['versionInfo']
        sata_version = (str(sata_version_stream['SATA_Version_Major'][0]) + '.' +
                        str(sata_version_stream['SATA_Version_Minor'][0]) + '.' +
                        str(sata_version_stream['SATA_Version_Wrapper'][0]) + '.' +
                        str(sata_version_stream['SATA_Version_Revision'][0]))
    except:
        sata_version = 'N/A'

    # SPP version
    try:
        spp_stream = mat_data['mudp']['MAP']['SPP']
        spp_version = (str(spp_stream['sppLibVersion'][0][0]) + '.' +
                       str(spp_stream['sppLibVersion'][0][1]) + '.' +
                       str(spp_stream['sppLibVersion'][0][2]) + '.' +
                       str(spp_stream['sppLibVersion'][0][3]) + '.' +
                       str(spp_stream['sppLibVersion'][0][4]))
    except:
        spp_version = 'N/A'

    # VSE version
    try:
        vse_stream = mat_data['mudp']['MAP']['VSE']['versions']
        vse_version = (str(vse_stream['major'][0]) + '.' +
                       str(vse_stream['minor'][0]) + '.' +
                       str(vse_stream['field'][0]) + '.' +
                       str(vse_stream['calibration'][0]))
    except:
        vse_version = 'N/A'

    # MAP version
    try:
        map_version = str(mat_data['mudp']['MAP']
                          ['rwm']['rwm209']['part1.mapVersion'])
    except:
        map_version = 'N/A'

    # Update these when available in data.
    # Localization
    localization_version = 'N/A'
    # Motion Model
    Motion_Model_version = 'N/A'
    # IW2
    lw2_version = 'N/A'
    # VS
    vs_version = 'N/A'
    # CA
    ca_version = 'N/A'
    # HMI2
    hmi_version = 'N/A'
    # OW2
    ow_version = 'N/A'
    # MRM
    mrm_version = 'N/A'
    # LC2
    try:
        LC2_version_stream = mat_data['mudp']['MAP']['inst']['LC2']['LC2_Logging_Msg']
        lc_version = (str(LC2_version_stream['LC2_MajorVersion'][0]) + '.' +
                      str(LC2_version_stream['LC2_MinorVersion'][0]) + '.' +
                      str(LC2_version_stream['LC2_PatchVersion'][0]))
    except:
        lc_version = 'N/A'
    # MAPH
    try:
        maph_stream = mat_data['mudp']['MAP']['inst']['MAPH']['MAPH_Logging2_Msg']
        maph_version = (str(maph_stream['MAPH_MajorVersion'][0]) + '.' +
                        str(maph_stream['MAPH_MinorVersion'][0]) + '.' +
                        str(maph_stream['MAPH_PatchVersion'][0]))
    except:
        maph_version = 'N/A'
    # TPC
    try:
        tpc_stream = mat_data['mudp']['MAP']['inst']['TPC2']['TPC1_Logging_Msg']
        tpc_version = (str(tpc_stream['TPC1_MajorVersion'][0]) + '.' +
                       str(tpc_stream['TPC1_MinorVersion'][0]) + '.' +
                       str(tpc_stream['TPC1_PatchVersion'][0]))
    except:
        tpc_version = 'N/A'
    # MMC
    try:
        mmc_stream = mat_data['mudp']['MAP']['inst']['MMC2']['MMC2_Logging_Msg']
        mmc_version = (str(mmc_stream['VsMMC2_MajorVersion'][0]) + '.' +
                       str(mmc_stream['VsMMC2_MinorVersion'][0]) + '.' +
                       str(mmc_stream['VsMMC2_PatchVersion'][0]))
    except:
        mmc_version = 'N/A'
    # SDM
    try:
        sdm_stream = mat_data['mudp']['MAP']['inst']['SDM2']['SDM2_Logging_Msg']
        sdm_version = (str(sdm_stream['VsSDM2_MajorVersion'][0]) + '.' +
                       str(sdm_stream['VsSDM2_MinorVersion'][0]) + '.' +
                       str(sdm_stream['VsSDM2_PatchVersion'][0]))
    except:
        sdm_version = 'N/A'

    # Return SW versions as a dict.
    sw_ver_no_dict = dict
    sw_ver_no_dict = {'cadm_mid_sw_ver': cadm_mid_sw_ver,
                      'cadm_map_sw_ver': cadm_map_sw_ver,
                      'mrr_fr_ver': cadm_mid_mrr_fr_ver,
                      'mrr_rr_ver': cadm_mid_mrr_rr_ver,
                      'mrr_fl_ver': cadm_mid_mrr_fl_ver,
                      'mrr_rl_ver': cadm_mid_mrr_rl_ver,
                      'lrr_ver': cadm_mid_lrr_ver,
                      'TR_mid_ver': mid_tr_version,
                      'TR_map_ver': map_tr_version,
                      'eyeq_version': eyeq_version,
                      'tracker_version': tracker_version,
                      'sata_version': sata_version,
                      'SPP_version': spp_version,
                      'VSE_version': vse_version,
                      'map_manager_version': map_version,
                      'localization_version': localization_version,
                      'motion_model_version': Motion_Model_version,
                      'lw2_version': lw2_version,
                      'vs_version': vs_version,
                      'ca_version': ca_version,
                      'hmi_version': hmi_version,
                      'ow_version': ow_version,
                      'mrm_version': mrm_version,
                      'lc_version': lc_version,
                      'maph_version': maph_version,
                      'tpc_version': tpc_version,
                      'mmc_version': mmc_version,
                      'sdm_version': sdm_version
                      }

    return sw_ver_no_dict


def get_tavi_path(log_path):
    """
    This function is used to get the tavi path from the same directory in which mat/mudp files are present.
    """
    if "2-Sim" in log_path and "USER_DATA" in log_path:
        ls_log_path = log_path.split("/")
        ls_tavi_path_1 = ls_log_path[:6]
        ls_tavi_path_2 = ls_log_path[10:13]
        tavi_path_1 = "/".join(ls_tavi_path_1)
        tavi_path_2 = "/".join(ls_tavi_path_2)
        tavi_path = tavi_path_1 + "/1-Rwup/" + tavi_path_2
        print("Tavi path", tavi_path)

    else:
        tavi_path = "tavi path does not exist"
        print(tavi_path)
    return tavi_path


def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)


def int_to_mac(macint):
    macint = int(macint)
    if type(macint) != int:
        raise ValueError('invalid integer')
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(macint))]*2)])


def _stream_def_pickle(root_stream_def_path,
                       str_def_types_pickle_path,
                       regenerate=False,
                       single_file_path=None,
                       ):

    def _map_properties(row,
                        mapping_dict,
                        pattern1,
                        property_type='type',
                        file_name=''):

        if row['index'] in mapping_dict.keys():
            return_val = mapping_dict.get(row['index'])[property_type]

        else:
            # print(f'index is {row.name}')

            pattern_list = re.findall(pattern1,
                                      row['index'])
            assert len(pattern_list) == 2, 'fixed point arithmetic fails here'

            if 's' in row['index'].lower():

                print(f'FILENAME : {file_name}')

                pattern_list[0] = str(int(pattern_list[0]) + 1)
                dtype_fxp = 'S' + '.'.join(pattern_list)
                size_fxp = (sum(map(int, pattern_list)) + 1)//8
            elif 'u' in row['index'].lower():

                print(f'FILENAME : {file_name}')

                dtype_fxp = 'U' + '.'.join(pattern_list)
                size_fxp = sum(map(int, pattern_list))//8

            # fxp_obj = Fxp(dtype=dtype_fxp)
            fxp_obj = Fxp_2(dtype_fxp=dtype_fxp)
            if property_type == 'type':
                return_val = fxp_obj
            elif property_type == 'size':
                return_val = size_fxp

        return return_val

    def _fill_pad(row, columns):

        if pd.isnull(row[columns[0]]):

            return_val = '.'.join(row[columns[1]].split('.')[
                :-1]) + '.' + row[columns[2]]

            if return_val.startswith("."):
                return_val = return_val[1:]

        else:
            return_val = row[columns[0]]

        return return_val

    pattern1 = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

    RE_D = re.compile('\d')

    error_stream_defs = []

    if not os.path.isfile(str_def_types_pickle_path):

        str_def_types_pickle_path = _stream_def_data_types_template_pickle(
            os.path.split(
                str_def_types_pickle_path)[0])

    with open(str_def_types_pickle_path, 'rb') as handle:
        mapping_dict = pickle.load(handle)

    if single_file_path is not None:
        database_filelist = [single_file_path]
        file_only_list = [os.path.split(single_file_path)[1][:-4]]
    else:
        database_filelist = []
        file_only_list = []
        for root, dirs, files in os.walk(root_stream_def_path):
            for file in files:
                if (file.endswith('.txt')):
                    # append the file name to the list
                    database_filelist.append(os.path.join(root, file))
                    file_only_list.append(file[:-4])

    for file_idx, (def_file, file_only) in enumerate(
            zip(database_filelist, file_only_list)):

        if os.path.isfile(os.path.join(root_stream_def_path,
                                       file_only + '.pickle')) and not regenerate:
            continue

        print(f'{file_idx}, \t{file_only}')

        req_str_def_path = def_file

        try:

            str_def_df = pd.read_csv(req_str_def_path,
                                     # delim_whitespace=True,
                                     sep='\s+',)
        except:
            str_def_df = pd.read_csv(req_str_def_path,)
            str_def_df[['index',
                        str_def_df.columns[0]]] = \
                str_def_df[str_def_df.columns[0]].str.split(" ", n=1,
                                                            expand=True)

        col_name = list(str_def_df.columns)[0]

        bytes_needed = int(col_name)

        str_def_df = str_def_df.reset_index()

        str_def_df['stream_path'] = str_def_df[col_name].shift(periods=0)

        indices_repeat_end = [index
                              for index, val in zip(str_def_df.index,
                                                    str_def_df['index'])
                              if 'REPEAT' in val and not bool(RE_D.search(val))]
        indices_repeat_end.sort()

        indices_repeat_start = [index
                                for index, val in zip(str_def_df.index,
                                                      str_def_df['index'])
                                if 'REPEAT' in val and bool(RE_D.search(val))]

        indices_repeat_start.sort()

        extend_list = []

        # extend_list.append([idx, item_type,  item_name])

        req_cols = ['index', 'stream_path']

        current_idx = 0

        index_append = []

        repeat_append = np.array([['none', -1, -1, -1, -1, -1]])

        is_start_outer = False
        outer_end = False

        while current_idx < len(str_def_df['index']):

            # print(current_idx)
            # if current_idx == 20:

            #     print('HHH')
            index_append.append(current_idx)

            is_start_outer = True if outer_end else False

            if current_idx in indices_repeat_start:
                repeat_check_item = str_def_df.loc[current_idx, 'index']
                repeat_times_outer = int(re.findall(pattern1,
                                                    repeat_check_item)[0])

                current_idx = current_idx+1
                outer_idx_start = current_idx
                # current_idx = current_idx-1

                outer_repeat_counter = 0
                is_after_inner = False
                while outer_repeat_counter < repeat_times_outer:

                    if is_after_inner:

                        req_check_idx = current_idx+1
                        is_after_inner = False
                    else:
                        req_check_idx = current_idx

                    is_start_outer = False
                    while True:
                        if is_start_outer:
                            repeat_check_item = str_def_df.loc[current_idx,
                                                               'index']
                        else:
                            repeat_check_item = str_def_df.loc[req_check_idx,
                                                               'index']
                            is_start_outer = True

                        if 'END_' in repeat_check_item:

                            outer_idx_end = current_idx

                            current_idx = outer_idx_start
                            break

                        # print(
                        #     f'Outer {outer_repeat_counter}, {repeat_times_outer}')

                        repeat_append = np.append(repeat_append, [
                            ['Outer',
                             outer_repeat_counter,
                             repeat_times_outer,
                             current_idx,
                             str_def_df.loc[current_idx,
                                            'index'],
                             str_def_df.loc[current_idx, 'stream_path']]],
                            axis=0)

                        index_append.append(current_idx)

                        if current_idx in indices_repeat_start:

                            repeat_check_item = str_def_df.loc[current_idx, 'index']
                            repeat_times_inner = int(re.findall(pattern1,
                                                                repeat_check_item)[0])
                            current_idx = current_idx+1
                            inner_idx_start = current_idx
                            # current_idx = current_idx-1

                            inner_repeat_counter = 0

                            while inner_repeat_counter < repeat_times_inner:

                                while True:

                                    repeat_check_item = str_def_df.loc[current_idx, 'index']

                                    if 'END_' in repeat_check_item:

                                        inner_idx_end = current_idx

                                        current_idx = inner_idx_start
                                        break
                                    # print(
                                    #     f'Inner {inner_repeat_counter}, {repeat_times_inner}')

                                    repeat_append = np.append(repeat_append, [
                                        ['Inner',
                                         inner_repeat_counter,
                                         repeat_times_inner,
                                         current_idx,
                                         str_def_df.loc[current_idx,
                                                        'index'],
                                         str_def_df.loc[current_idx, 'stream_path']]],
                                        axis=0)
                                    item = np.append(
                                        np.array([current_idx,
                                                 repeat_times_inner,
                                                 repeat_times_outer,
                                                 2,]
                                                 ),
                                        str_def_df.loc[current_idx,
                                                       req_cols].values)

                                    extend_list.append(item)

                                    current_idx = current_idx+1

                                inner_repeat_counter = inner_repeat_counter+1
                                if inner_repeat_counter == repeat_times_inner:

                                    is_after_inner = True
                                    current_idx = inner_idx_end + 1

                        else:

                            item = np.append(np.array([current_idx,
                                                      1,
                                                      repeat_times_outer,
                                                      1,]
                                                      ),
                                             str_def_df.loc[current_idx,
                                                            req_cols].values)

                            extend_list.append(item)
                            current_idx = current_idx+1

                    outer_repeat_counter = outer_repeat_counter+1

                    if outer_repeat_counter == repeat_times_outer:

                        outer_end = True
                        current_idx = outer_idx_end

            else:

                if not is_start_outer:

                    repeat_append = np.append(repeat_append, [
                        ['None',
                         0,
                         0,
                         current_idx,
                         str_def_df.loc[current_idx,
                                        'index'],
                         str_def_df.loc[current_idx, 'stream_path']]
                    ],
                        axis=0)

                    item = np.append(np.array([current_idx,
                                              0,
                                              0,
                                              0,]
                                              ),
                                     str_def_df.loc[current_idx,
                                                    req_cols].values)

                    extend_list.append(item)

                outer_end = False
                current_idx = current_idx+1

        extend_arr = np.array(extend_list)

        str_def_df_2 = pd.DataFrame(extend_arr[:, 1:], columns=[
            'inner_loop_size',
            'outer_loop_size',
            'len_map_size_list',
        ]+req_cols)

        str_def_df_2['type_shape'] = list(zip(str_def_df_2['inner_loop_size'],
                                              str_def_df_2['outer_loop_size']))

        idx_to_remove = str_def_df_2[str_def_df_2['index']
                                     .str.contains("REPEAT")].index
        copy_str_def_df_2 = str_def_df_2.copy(deep=True)
        str_def_df_2 = str_def_df_2.drop(idx_to_remove)

        # str_def_df_2['lag'] = str_def_df_2['stream_path'].shift(periods=1)

        str_def_df_2['lag'] = str_def_df_2['stream_path'].ffill()

        str_def_df_2['req_col'] = str_def_df_2.apply(_fill_pad,
                                                     axis=1,
                                                     columns=['stream_path',
                                                              'lag', 'index'])

        # str_def_df_2['map_type'] = str_def_df_2['index'].apply(
        #     lambda x: mapping_dict.get(x)['type'])

        # str_def_df_2['map_size'] = str_def_df_2['index'].apply(
        #     lambda x: mapping_dict.get(x)['size'])

        str_def_df_2['map_type'] = str_def_df_2.apply(_map_properties,
                                                      axis=1,
                                                      mapping_dict=mapping_dict,
                                                      pattern1=pattern1,
                                                      property_type='type',
                                                      file_name=file_only)
        str_def_df_2['map_size'] = str_def_df_2.apply(_map_properties,
                                                      axis=1,
                                                      mapping_dict=mapping_dict,
                                                      pattern1=pattern1,
                                                      property_type='size')

        req_df = str_def_df_2[['req_col', 'map_type',
                               'map_size', 'type_shape',
                               'len_map_size_list']]

        req_dict = req_df.to_dict(orient='list')
        req_dict['expected_stream_length'] = bytes_needed

        # supp_df = str_def_df_2

        # req_dict['supp_dict'] = supp_dict
        if bytes_needed != sum(req_df['map_size']):
            error_stream_defs.append(file_only)
            continue

        with open(os.path.join(root_stream_def_path, file_only + '.pickle'), 'wb') as handle:
            pickle.dump(req_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    return error_stream_defs


def _stream_def_data_types_template_pickle(output_dir, ):

    list_1 = ['single', 'float', 'float32',
              'float32_t', 'real32_t', 'f360_fpn_t']
    list_1 = list_1 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_1 if len(val.split('_')[-1]) == 1]
    list_1_subs = [np.float32, 4]

    list_2 = ['double', 'float64', 'float64_t', 'real64_t', 'f360_dpn_t']
    list_2 = list_2 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_2 if len(val.split('_')[-1]) == 1]
    list_2_subs = [np.float64, 8]

    list_3 = ['uint8', 'uint8_t', 'unsigned char', 'bool', 'boolean',
              'boolean_t', 'unsigned8_t',
              'f360_ui8n_t', 'f360_booln_t', 'u8p0_T', 'u8p0_t',
              'bitfield8_t',]
    list_3 = list_3 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_3 if len(val.split('_')[-1]) == 1]
    list_3_subs = [np.uint8, 1]

    list_4 = ['int8', 'sint8', 'int8_t', 'sint8_t',
              'signed char', 'char', 'signed8_t', 'f360_si8n_t',
              's7p0_t', 's7p0_T',
              'PADDING1']
    list_4 = list_4 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_4 if len(val.split('_')[-1]) == 1]
    list_4_subs = [np.int8, 1]

    list_5 = ['uint16', 'uint16_t', 'unsigned short',
              'unsigned16_t', 'f360_ui16n_t', 'u16p0_T',
              'u16p0_t', 'PADDING2', 'bitfield16_t',]
    list_5 = list_5 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_5 if len(val.split('_')[-1]) == 1]
    list_5_subs = [np.uint16, 2]

    list_6 = ['int16', 'sint16', 'int16_t',
              'sint16_t', 'signed short', 'short', 'signed16_t',
              'f360_si16n_t',
              # 's14p1_T', 's14p1_T'
              ]
    list_6 = list_6 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_6 if len(val.split('_')[-1]) == 1]
    list_6_subs = [np.int16, 2]

    list_7 = ['uint32', 'uint32_t', 'unsigned int', 'unsigned32_t', 'f360_ui32n_t',
              'PADDING4', 'bitfield32_t', ]
    list_7 = list_7 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_7 if len(val.split('_')[-1]) == 1]
    list_7_subs = [np.uint32, 4]

    list_8 = ['int32', 'sint32', 'int32_t', 'sint32_t', 'signed int', 'int', 'signed32_t',
              'f360_si32n_t']
    list_8 = list_8 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_8 if len(val.split('_')[-1]) == 1]
    list_8_subs = [np.int32, 4]

    list_9 = ['uint64', 'uint64_t', 'unsigned64_t', 'f360_ui64n_t', 'PADDING8']
    list_9 = list_9 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                       for val in list_9 if len(val.split('_')[-1]) == 1]
    list_9_subs = [np.uint64, 8]

    list_10 = ['int64', 'sint64', 'int64_t',
               'sint64_t', 'signed64_t', 'f360_si64n_t']
    list_10 = list_10 + [val.split('_')[0] + '_' + val.split('_')[-1].upper()
                         for val in list_10 if len(val.split('_')[-1]) == 1]
    list_10_subs = [np.int64, 8]

    list_11 = ['PADDING' + str(i+1) for i in range(1, 16)]
    # list_11_subs = [np.int64, 8]

    lists = [list_1, list_2, list_3, list_4, list_5,
             list_6, list_7, list_8, list_9, list_10]

    type_list = [np.float32, np.float64, np.uint8, np.int8, np.uint16,
                 np.int16, np.uint32, np.int32, np.uint64, np.int64]

    mapping_dict = {name: {'type': type_,
                           'size': np.dtype(type_).itemsize
                           }
                    for sub_list, type_ in zip(lists, type_list)
                    for name in sub_list

                    }

    mapping_dict_2 = {name: {'type': np.dtype((np.void, idx+2)),
                             'size': np.dtype((np.void, idx+2)).itemsize
                             }
                      for idx, name in enumerate(list_11)
                      }

    mapping_dict = {**mapping_dict_2, **mapping_dict, }

    output_file_path = os.path.join(output_dir,
                                    'datatypes_mapping.pickle')

    with open(output_file_path, 'wb') as handle:
        pickle.dump(mapping_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_file_path


def merge_dicts(a, b):
    # for k in a:
    #     a.get(k).update(b.get(k, {}))

    # return_dict = {k: dict(a.get(k, {}), **v) for k, v in b.items()}

    # merged_dict = defaultdict(dict)
    # merged_dict.update(a)
    # for key, nested_dict in b.items():
    #     if isinstance(nested_dict, str) and 'not' in nested_dict:
    #         continue
    #     elif (key in merged_dict.keys()
    #           and isinstance(merged_dict[key], str)
    #           and 'not' in merged_dict[key]):
    #         merged_dict[key].update(nested_dict)

    # return_dict = dict(merged_dict)

    dict_1 = flatten_nested_dict(a)
    dict_2 = flatten_nested_dict(b)

    merged_dict = defaultdict(dict)
    merged_dict.update(dict_1)

    for key, val in dict_1.items():

        if (any([True if key in dkey2 else False
                     for dkey2 in dict_2.keys()])
                    # and not isinstance(dict_2[key], str)
                    and isinstance(dict_1[key], str)
                and ('not' in dict_1[key]
                     or 'missing' in dict_1[key])
                ):

            del merged_dict[key]

    for key, val in dict_2.items():

        # if (key in dict_1.keys()
        #     and isinstance(dict_1[key], str)
        #         and 'not' in dict_1[key]):

        #     merged_dict[key] = val

        merged_dict[key] = val

    return_dict = nest_flatten_dict(dict(merged_dict))

    return return_dict


def flatten_nested_dict(nested_dict, separator='.', prefix=''):
    res = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            res.update(flatten_nested_dict(value,
                                           separator,
                                           prefix + key + separator))
        else:
            res[prefix + key] = value
    return res


def nest_flatten_dict(flat_dict, sep='.'):
    """Return nested dict by splitting the keys on a delimiter.

    >>> from pprint import pprint
    >>> pprint(nest_dict({'title': 'foo', 'author_name': 'stretch',
    ... 'author_zipcode': '06901'}))
    {'author': {'name': 'stretch', 'zipcode': '06901'}, 'title': 'foo'}
    """
    tree = {}
    for key, val in flat_dict.items():
        t = tree
        prev = None
        for part in key.split(sep):
            if prev is not None:
                t = t.setdefault(prev, {})
            prev = part
        else:
            t.setdefault(prev, val)
    return tree


# def merge_dicts2(d, u):
#     for k, v in u.items():
#         if isinstance(v, Mapping):
#             d[k] = merge_dicts2(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

def merge_dicts2(dict_1, other):

    d = copy.deepcopy(dict_1)
    for k, v in other.items():
        d_v = d.get(k)
        if isinstance(v, Mapping) and isinstance(d_v, Mapping):
            merge_dicts2(d_v, v)
        else:
            d[k] = v  # or d[k] = v if you know what you're doing


def deepupdate(original, update):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    if not isinstance(original, Mapping):
        return update
    for key, value in update.items():
        if isinstance(value, Mapping):
            original[key] = deepupdate(original.get(key, {}), value)
        else:
            original[key] = value
    return original


# def deep_merge(d, u):

#     stack = [(d,u)]
#     while stack:
#         d,u = stack.pop(0)
#         for k,v in u.items():
#             if not isinstance(v, collections.Mapping):
#                 d[k] = v
#             else:
#                 if k not in d:
#                     d[k] = v
#                 elif not isinstance(d[k], Mapping):
#                     d[k] = v
#                 else:
#                     stack.append((d[k], v))


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


# def _user_events_hot_key_mapping

def _get_numerical_signals_difference_matrix(
        signals_dict,
        scaling_dict,
        raw_data,
        drop_threshold):

    signals = {key: stream_check(raw_data, val)*scaling_dict[key]
               if isinstance(val, str)
               else np.add.reduce([stream_check(raw_data, array)
                                   * scaling_dict[key][i]
                                   for i, array in enumerate(val)]
                                  )
               for key, val in signals_dict.items()
               }

    min_max_dict = {key: np.array([np.min(val), np.max(val)])
                    if not isinstance(val, str)
                    else np.array([np.nan, np.nan])
                    for key, val in signals.items()
                    }
    min_max_dict['type'] = ['min_diff', 'max_diff']

    min_max_df = pd.DataFrame(min_max_dict).set_index('type')

    arr = min_max_df.values

    # column_combos = combinations(range(arr.shape[1]), 2)

    result = np.concatenate([(arr.T - arr.T[x])[x+1:]
                             for x in range(arr.shape[1])]).T

    columns = list(map(lambda x: x[1]+' - ' + x[0],
                       combinations(min_max_df.columns, 2)))

    min_max_diff_df = pd.DataFrame(result,
                                   columns=columns,
                                   index=min_max_df.index)
    min_max_diff_df_t = min_max_diff_df.T

    x_t = min_max_diff_df_t[
        min_max_diff_df_t.abs() > drop_threshold].dropna(
            axis=0, how='all')

    non_nan_indices_t = {col: list(x_t.query(f'`{col}` == `{col}`').index)
                         for col in x_t.columns}

    return non_nan_indices_t


def _generic_event_extraction(df: pd.DataFrame,
                              condition_cols: list,
                              condition_list: list,
                              event_cols: list,
                              condition_type_list: list = ['and', ''],
                              cTime_column: str = None,
                              ) -> pd.DataFrame:
    '''This is a simple functio to extract events based on 
    conditions applied on columns of a dataframe. 

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe with data in all columns (note that all columns as 
        mentioned in 'condition_cols' and 'event_cols' is mandatory. 
        If required, cTime column can be given as an  input )
    condition_cols : list
        Columns on which conditions are applied. 
        These columns decide whethere there is an event or not.
        E.g.: ['lateral_velocity', 'is_brake_applied', ]
    condition_list : list
        List of conditions that are to apply on 'condition_cols'. 
        Must match in length with the 'condition_cols'
        E.g.: ['> 4', '== True', ]
    event_cols : list
        All the list of columns that are to be extracted to be analysed later. 
    condition_type_list : list, optional
        The type of condiitons applied when multiple 'condition_cols' exist, 
        by default ['and', '']. 
        note that a single closed quotes is mandatorily 
        present as a last item in this arg.
        E.g.: ['and', '']
    cTime_column : str
        cTime column. by default None value

    Returns
    -------
    pd.DataFrame
        The output is a dataframe with column names same as earlier 
        but each row comprisng all the data for an event for that 
        column and number of rows equalling number of events.
    '''

    check_cols = condition_cols + event_cols

    if cTime_column is not None:

        check_cols = check_cols + cTime_column

    if not set(check_cols).issubset(set(df.columns)):
        return 'check if condition_cols and event_cols (and cTime col) are in df'

    query = ' '.join([' '.join(x)
                      for x in zip(condition_cols,
                                   condition_list,
                                   condition_type_list)])

    df2 = df.query(query)[check_cols]

    event_indices = list(df2.index)

    s = pd.Series(event_indices)
    event_start_end_groups = s.groupby(s.diff().ne(1)
                                       .cumsum()).apply(lambda x:
                                                        [x.iloc[0], x.iloc[-1]]
                                                        if len(x) >= 2
                                                        else [x.iloc[0], x.iloc[0]]
                                                        ).tolist()

    dict_out_list = [df2.loc[start: end, event_cols].to_dict(orient='list')
                     for start, end in event_start_end_groups]

    out_dict = {
        k: [d.get(k) for d in dict_out_list if k in d]
        for k in set().union(*dict_out_list)
    }

    # out_dict = {key: list() for key in event_cols}

    out_df = pd.DataFrame(out_dict)

    return out_df


def checksum_md5(path, block_size=256*128, hr=True):
    '''
    Block size directly depends on the block size of your filesystem
    to avoid performances issues
    Here I have blocks of 4096 octets (Default NTFS)
    '''
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            md5.update(chunk)
    if hr:
        return md5.hexdigest()
    return md5.digest()


def _calc_derivative(y_arr, x_arr, smoothen_y_data, smoothen_window_len):

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    if smoothen_y_data:

        y_arr = _calc_smoothened_data(y_arr,
                                      smoothen_window_len)

    dy = np.diff(y_arr, n=1, )
    dx = np.diff(x_arr, n=1, ).astype(dy.dtype)
    # dy = np.concatenate([[0], dy])
    # dx = np.concatenate([[0], dx])

    dy_dx = np.divide(dy, dx,
                      out=np.zeros_like(dx),
                      where=dx != 0)
    dy_dx = np.concatenate([[0], dy_dx])

    return dy_dx


def _calc_smoothened_data(y_arr, window_length=25):

    y_arr_new = pd.Series(y_arr).rolling(window=window_length)\
        .mean().iloc[window_length-1:].to_numpy(
        # dtype=float
    )

    req_val = y_arr_new[-1]

    req_length_to_fill = window_length - 1
    y_arr_new = np.concatenate((y_arr_new, [req_val]*req_length_to_fill))

    return y_arr_new


def patch_asscalar(a):

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if a.size == 0:
        return_val = np.nan
    else:
        return_val = a.item()
    # assert np.array(return_val).ndim == 0, 'assertion failed'
    # if np.array(return_val).ndim != 0:
    #     return_val = np.nan

    return return_val


def _list_of_dicts_to_dict_of_arrays(LD: List[dict]):

    DA = {k: np.array([int(dic[k]) for dic in LD]) for k in LD[0]}

    return DA


def _start_end_cTime_to_start_end_time(cTime_array: np.array,
                                       start_cTime: float,
                                       end_cTime: float,
                                       ):

    start_time_idx = (np.abs(np.array(cTime_array) - start_cTime)).argmin()
    start_time = cTime_array[start_time_idx] - cTime_array[0]

    end_time_idx = (np.abs(np.array(cTime_array) - end_cTime)).argmin()
    end_time = cTime_array[end_time_idx] - cTime_array[0]

    return start_time, end_time


def ignore_nan_inf(worksheet, row, col, number, format=None):
    if math.isnan(number):
        return worksheet.write_blank(row, col, 'nan', format)
    elif math.isinf(number):
        return worksheet.write_blank(row, col, 'inf', format)
    else:
        # Return control to the calling write() method for any other number.
        return None


def _write_gifs_to_excel(job_out_excel_path,
                         events_sheet_name,
                         excel_output_path,
                         start_cTime_col,
                         end_cTime_col,
                         log_path_col,
                         log_name_col,
                         video_cTimes_array_signal_and_multiplier_list: list,
                         log_name_replace_list: list = ['dma', 'v01'],
                         video_extension: str = '.webm',
                         prepend_to_start_time: float = 0.0,
                         append_to_end_time: float = 0.0,
                         req_fps: int = 5,
                         is_bw_required: bool = True,
                         gif_resize_factor: float = 0.25,
                         gif_output_path: os.path.join = None,
                         total_df: pd.DataFrame = None,

                         ):

    if total_df is None:
        total_df = pd.read_excel(job_out_excel_path,
                                 sheet_name=events_sheet_name)

    total_df.replace([np.inf, -np.inf], 'inf', inplace=True)

    # events_df = events_df.fillna(pd.NA).replace({pd.NA: None})

    # pd.set_option('use_inf_as_na', True)

    nr_of_rows_per_excel = 15

    # list_df = np.array_split(total_df,
    #                          math.ceil(
    #                              len(total_df)/nr_of_rows_per_excel))

    list_df = [total_df[i:i+nr_of_rows_per_excel]
               for i in range(0, total_df.shape[0],
                              nr_of_rows_per_excel)]

    for df_idx, events_df in enumerate(list_df):

        events_df = events_df.reset_index()

        excel_output_path_only, excel_name = os.path.split(excel_output_path)

        base_name, suffix = excel_name.split('.')
        print(base_name)
        excel_output_name_iter = \
            base_name + '_' + str(df_idx) + '.' + suffix
        excel_output_path_iter_only = os.path.join(
            excel_output_path_only,
            'gif_excel_data',
        )

        if not os.path.isdir(excel_output_path_iter_only):

            os.makedirs(excel_output_path_iter_only,
                        exist_ok=True
                        )

        excel_output_path_iter = os.path.join(excel_output_path_iter_only,
                                              excel_output_name_iter)
        workbook = xlsxwriter.Workbook(
            excel_output_path_iter)  # excel_output_path
        worksheet = workbook.add_worksheet(events_sheet_name)

        col_to_insert_at = events_df.shape[1]
        num_rows = events_df.shape[0]

        header = list(events_df.columns) + ['Event Video']

        for col in range(col_to_insert_at + 1):
            worksheet.write(0, col, header[col])

        cell_format = workbook.add_format()
        cell_format.set_align('vcenter')

        events_df.replace([np.inf, -np.inf], 'inf', inplace=True)
        events_df.replace([np.nan, pd.NA], 'nan', inplace=True)
        # workbook.nan_inf_to_errors = True
        # worksheet.add_write_handler(float, ignore_nan_inf)
        # worksheet.add_write_handler(np.inf, ignore_nan_inf)

        # video_formats = ['webm', 'mp4', 'avi', 'tavi']
        # row = 0
        for (row,
             log_path,
             log_name,
             start_cTime,
             end_cTime) in events_df[[log_path_col,
                                      log_name_col,
                                      start_cTime_col,
                                      end_cTime_col]].itertuples():

            print('Processing row ' + str(row+1) +
                  ' of ' + str(col_to_insert_at) + '...')

            for col in range(0, col_to_insert_at):
                worksheet.write(row+1, col,
                                events_df.iloc[row][col],
                                cell_format,
                                # nan_inf_to_errors=True
                                )

            try:

                sub_string_list = ['ADAS247',
                                   'FCA-CADM',
                                   'STLA-THUNDER',
                                   'MCIP',
                                   'GPO-E2E'
                                   ]
                signal_list = ['mudp.avi.flc.s104.common.header.cTime',
                               'mudp.eyeq.TimeSync.header.cTime',
                               'mudp.AVI_Message.header.time',
                               'mudp.avi.flc.tsr.header.time',
                               'mudp.avi.flc.tsr.header.time',
                               ]
                video_extension_list = ['.avi',
                                        '.tavi',
                                        '.webm',
                                        '.mp4',
                                        '.mp4'
                                        ]
                log_name_replace_list_types = [['', ''],
                                               ['', ''],
                                               ['dma', 'v01'],
                                               ['p01', 'v01'],
                                               ['p01', 'v01'],
                                               ]
                boolean_substring = list(map(log_path.__contains__,
                                             sub_string_list))

                boolean_index = np.argwhere(boolean_substring).flatten()

                if not bool(log_name_replace_list):

                    log_name_replace_list = log_name_replace_list_types[boolean_index[0]]

                if not bool(video_extension.strip()):

                    video_extension = video_extension_list[boolean_index[0]]

                video_file_path = os.path.join(log_path,
                                               log_name.replace(
                                                   *log_name_replace_list
                                               ).split('.')[0] + video_extension
                                               )
                if not os.path.isfile(video_file_path):

                    raise ValueError(
                        f'Video file not found at {video_file_path}')

                if not bool(video_cTimes_array_signal_and_multiplier_list[0].strip()):

                    video_cTimes_array_signal = signal_list[boolean_index[0]]
                else:
                    video_cTimes_array_signal = video_cTimes_array_signal_and_multiplier_list[0]

                raw_data = loadmat(os.path.join(log_path,
                                                log_name))

                video_cTimes_array_multiplier = float(
                    video_cTimes_array_signal_and_multiplier_list[1])

                data_df = transform_df({'cTime': stream_check(raw_data,
                                                              video_cTimes_array_signal)}
                                       )*video_cTimes_array_multiplier

                raw_data = None

                video_cTimes_array = data_df['cTime'].values

                jobout_path, excel_name = os.path.split(job_out_excel_path)

                output_path = os.path.join(jobout_path, 'gif_data', )

                gif_out_path, video_width, video_height = _add_gif_to_events(video_file_path,
                                                                             video_cTimes_array,
                                                                             start_cTime,
                                                                             end_cTime,
                                                                             prepend_to_start_time,
                                                                             append_to_end_time,
                                                                             req_fps,
                                                                             is_bw_required,
                                                                             gif_resize_factor,
                                                                             output_path,
                                                                             )

                worksheet.set_row(row=row+1, height=video_height)

                # worksheet.set_column(first_col  = col_to_insert_at,
                #                      last_col = col_to_insert_at,
                #                      width = video_height)

                worksheet.insert_image(row=row+1,
                                       col=col_to_insert_at,
                                       filename=gif_out_path,
                                       )
            except:
                logging.exception("Sheet: " + events_sheet_name +
                                  ", Row: " + str(row+2))
                worksheet.write(row+1, col_to_insert_at,
                                f'Video file not found at {video_file_path}',
                                cell_format)

            # row = row+1

        print('Writing output...')

        workbook.close()


def _add_gif_to_events(video_file_path,
                       video_cTimes_array: np.array,
                       start_cTime,
                       end_cTime,
                       prepend_to_start_time: float = 0.5,
                       append_to_end_time: float = 0.5,
                       req_fps: int = 5,
                       is_bw_required: bool = True,
                       gif_resize_factor: float = 0.25,
                       output_path: os.path.join = None):

    start_time, end_time = _start_end_cTime_to_start_end_time(video_cTimes_array,
                                                              start_cTime,
                                                              end_cTime)

    gif_out_path, video_width, video_height = _save_video_to_gif(video_file_path,
                                                                 start_time,
                                                                 end_time,
                                                                 prepend_to_start_time,
                                                                 append_to_end_time,
                                                                 req_fps,
                                                                 is_bw_required,
                                                                 gif_resize_factor,
                                                                 output_path,
                                                                 )

    return gif_out_path, video_width, video_height


def _save_video_to_gif(video_file_path,
                       start_time: float,
                       end_time: float,
                       prepend_to_start_time: float = 0.5,
                       append_to_end_time: float = 0.5,
                       req_fps: int = 5,
                       is_bw_required: bool = True,
                       gif_resize_factor: float = 0.25,
                       output_path: os.path.join = None,
                       to_write: bool = False):

    video_path, video_name = os.path.split(video_file_path)

    video_clip_obj = VideoFileClip(video_file_path)

    # duration = video_clip_obj.duration
    end_time_video = video_clip_obj.end
    fps = video_clip_obj.fps

    video_width = video_clip_obj.w*gif_resize_factor
    video_height = video_clip_obj.h*gif_resize_factor

    if end_time + append_to_end_time > end_time_video:

        if end_time > end_time_video:

            end_time = end_time_video
        else:
            end_time = end_time
    else:
        end_time = end_time + append_to_end_time

    if start_time - prepend_to_start_time < 0:

        if start_time < 0:

            start_time = 0
        else:

            start_time = start_time

    else:

        start_time = start_time - prepend_to_start_time

    if req_fps > fps:

        req_fps = fps

    if gif_resize_factor <= 0 or gif_resize_factor > 1:

        gif_resize_factor = 1

    clip = (
        video_clip_obj
        .subclipped(start_time, end_time)
        # .with_volume_scaled(0.8)
        .resized(new_size=gif_resize_factor)
    )

    if is_bw_required:

        clip = BlackAndWhite().apply(clip)

    if output_path is None:

        output_path = video_path

    video_base_name = video_name.split('.')[0]

    if not os.path.isdir(output_path):

        os.makedirs(output_path,
                    exist_ok=True
                    )

    all_gif_files = [file[:-4]
                     for file in os.listdir(output_path)
                     if file.endswith(".gif")
                     and video_base_name in file
                     ]
    # print('*************')
    # print(output_path)
    # print('&&&&&&&&&&&&&&')
    if len(all_gif_files) > 0:

        all_gif_files.sort(reverse=False)

        print(all_gif_files)

        gif_enum = int(all_gif_files[-1].split('.')[-1][-3:]) + 1

    else:

        gif_enum = 0

    gif_out_path = os.path.join(output_path,
                                video_base_name +
                                f'_{gif_enum:03}' + '.gif')

    # if os.path.isfile(gif_out_path):

    #     gif_enum = int(gif_out_path.split('.')[-2][-4:-1]) + 1
    #     gif_out_path = os.path.join(output_path,
    #                                 video_name.split('.')[0] +
    #                                 f'_{gif_enum:03}' + '.gif')

    clip.write_gif(gif_out_path, fps=req_fps)

    return gif_out_path, video_width, video_height


def _video_to_images(video_path,
                     output_frames_dir,
                     req_fps=None,
                     image_extension: str = 'jpeg'
                     ):

    if not image_extension in ['jpeg', 'png']:

        image_extension = 'jpeg'

    name_format = os.path.join(output_frames_dir,
                               'frame%04d.' + image_extension)

    os.makedirs(os.path.join(output_frames_dir,

                             ),
                exist_ok=True)

    clip_obj = VideoFileClip(video_path)

    fps = clip_obj.fps

    if req_fps is not None and req_fps > fps:

        req_fps = fps

    output_image_list = clip_obj.write_images_sequence(
        name_format=name_format,
        fps=req_fps)

    return output_image_list


def _resim_path_to_orig_path(log_name,
                             log_path,
                             is_path: bool = True):

    reg_exp = re.compile(r'(rA|rF|rS|rC|rO)')

    if bool(reg_exp.search(log_name)):

        split_path_list = Path(log_path).parts

        root_path = os.path.join(*split_path_list[:5+1])

        resim_tag = split_path_list[-1]

        log_specific_path = os.path.join(*split_path_list[-1-3:-1])

        orig_log_name = log_name.replace('_' + resim_tag, '')

        full_log_path = os.path.join(root_path,
                                     '1-Rwup',
                                     log_specific_path)
    else:

        full_log_path = log_path
        orig_log_name = log_name

    if is_path:

        return_val = full_log_path

    else:

        return_val = orig_log_name

    return return_val


def create_mysql_engine_fn(
    host: str = "10.192.229.101",  # "usmidet-db001.aptiv.com",  #  #10.192.229.101
    user: str = 'aptiv_db_algo_team',
    password: str = 'DKwVd5Le',
                    database: str = "aptiv_production",  # 'ds_team_na'
):
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{ip}/{db}".format(
            user=user,
            pw=password,
            ip=host,
            db=database,
            # echo=False
        )
    )

    if type(engine) == str:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',
              'Something wrong with connection Database',
              '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',)
        # sys.exit(-1)
        return None
    else:
        return engine


def update_sql_table(sql_engine_object: create_engine,
                     df: pd.DataFrame,
                     table_name: str,
                     to_do_if_table_exists: str = 'append'  # 'replace'
                     ):

    df.to_sql(table_name,
              con=sql_engine_object,
              if_exists=to_do_if_table_exists,
              index=False
              )

    return_val = sql_engine_object.url

    return return_val


def alter_sql_table(
        sql_engine_object: create_engine,
    df: pd.DataFrame,
    table_name: str,
    host: str = "10.192.229.101",  # "usmidet-db001.aptiv.com",  #  #10.192.229.101
    user: str = 'aptiv_db_algo_team',
    password: str = 'DKwVd5Le',
    database: str = "aptiv_production",  # 'ds_team_na',

):

    try:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',
              'This method is not recommended. ',
              'I hope you know what you are doing'
              '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',)
        connection = sql_engine_object.connect()
        description_tuple = connection.execute(
            text(f"Select * from {database}.{table_name} LIMIT 5")
        ).cursor.description
        # a list of the column names
        headers = [k[0] for k in description_tuple]
        new_cols = list(set(list(df.columns)) - set(headers))

        if len(new_cols) > 0:

            new_cols_str = " TEXT, ".join(new_cols + [''])[:-2]

            with connection as conn:

                conn.execute(
                    text(f"ALTER TABLE {database}.{table_name} ADD ({new_cols_str})"
                         ))
        else:

            print('No new columns present as of now. DB is safe from altering')

    except:

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',
              'Something wrong with connection. Check Database immediately',
              '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',)


def _time_duration_to_indices_length(out_df,
                                     time_duration):

    assert 'cTime' in out_df.columns, \
        'Assertion Error : cTime column is not in input df'

    start_index = 0
    start_cTime = float(out_df.loc[start_index, 'cTime'])
    delta_cTime = time_duration
    end_cTime = start_cTime + delta_cTime
    end_index = int(out_df[out_df['cTime']
                           <= end_cTime]['cTime'].idxmax())
    if end_index == start_index:
        end_index = start_index+1

    indices_length = end_index - start_index

    return indices_length


def find_closest_index(value, df, colname):

    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return [exactmatch.index]
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]


def is_monotonic(arr):

    increasing_list = np.array([arr[i] <= arr[i + 1]
                                for i in range(len(arr) - 1)])
    increasing = all(increasing_list) \
        if increasing_list.ndim and increasing_list.size else False
    non_increasing_count = np.sum(~increasing_list) \
        if increasing_list.ndim and increasing_list.size else 0

    decreasing_list = np.array([arr[i] >= arr[i + 1]
                                for i in range(len(arr) - 1)])
    decreasing = all(decreasing_list) \
        if decreasing_list.ndim and decreasing_list.size else False
    non_decreasing_count = np.sum(~decreasing_list) \
        if decreasing_list.ndim and decreasing_list.size else 0

    return increasing, decreasing, non_increasing_count, non_decreasing_count


def find_ranges_in_iterable(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0], group[0]+1
        else:
            yield group[0], group[-1]


def _check_for_obj_instance(obj_instance, iterable):

    indices = [item for item in iterable
               if isinstance(item, obj_instance)]
    boolean_out = True if len(indices) > 0 else False

    return boolean_out, indices


def duplicate_keys_to_tuples_dict(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return {key: locs for key, locs in tally.items()
            # if len(locs) > 1
            }


def ndix_unique(x):
    """
    Returns an N-dimensional array of indices
    of the unique values in x
    ----------
    x: np.array
       Array with arbitrary dimensions
    Returns
    -------
    - 1D-array of sorted unique values
    - Array of arrays. Each array contains the indices where a
      given value in x is found
    """
    x_flat = x.ravel()
    ix_flat = np.argsort(x_flat)
    u, ix_u = np.unique(x_flat[ix_flat], return_index=True)
    ix_ndim = np.unravel_index(ix_flat, x.shape)
    ix_ndim = np.c_[ix_ndim] if x.ndim > 1 else ix_flat
    return u, np.split(ix_ndim, ix_u[1:])


def _create_base_name_list_from_file_list(log_path_list,

                                          sub_string_list=[
                                              '_b01', '_b02',
                                              '_b03', '_b04', '_b05',
                                              '_p01',
                                              '_r03',
                                          ]):

    log_path_name_split_tuple = [
        os.path.split(log_path)
        for log_path in log_path_list
    ]

    names_list = [item[-1] for item in log_path_name_split_tuple]

    extension_list = [item.split('.')[-1] for item in names_list]

    reg_resim = re.compile(
        r'_rR.*\d+|_rM.{,2}\d+|_rIE.*\d+|_rFVC.*\d+|_rHBA.*\d+'
        + r'|_rFLR.*\d+|_r.*SRR.*\d+|_rA.*\d+')

    longest_first = sorted(sub_string_list, key=len, reverse=True)
    compiled_strings = re.compile(r'(?:{})'
                                  .format('|'.join(
                                      map(re.escape, longest_first))))

    rTag_list = [
        reg_resim.search(name).group(0)
        if bool(reg_resim.search(name)) else 'None Available'
        for name in names_list
    ]

    name_only_no_rTag_list = [
        name.replace(rTag, '').split('.')[0]
        for name, rTag in zip(names_list, rTag_list)
    ]

    base_name_list = [
        re.sub(compiled_strings, '',
               name_only_no_rTag)
        for name_only_no_rTag in name_only_no_rTag_list
    ]

    seq_path_list = \
        [os.path.join(*os.path.normpath(
            log_path).split(os.path.sep)[:6]
            + ['1-Rwup'] + basename.split('_')[1:-1])
         for log_path, basename in zip(log_path_list, base_name_list)
         ]

    rTag_list = [item.lstrip('_') for item in rTag_list]

    rTag_list = [item if item != '' else 'None Available'
                 for item in rTag_list]

    original_log_name_list = [
        name_only + '.' + extension

        for name_only, extension in zip(name_only_no_rTag_list,
                                        extension_list)
    ]

    log_path_list = [item[0] for item in log_path_name_split_tuple]

    return (log_path_list, base_name_list,
            rTag_list, seq_path_list, original_log_name_list)


def _create_flat_file_list(cont_list_name,
                           SIMULATE_LINUX=False,
                           digits_log_enum=4,
                           sub_string_list=[
                               '_b01', '_b02',
                               '_b03', '_b04', '_b05',
                               '_p01',
                               '_r03',
                           ]):

    cont_list_name_split = cont_list_name.split(',')
    full_path_cont = cont_list_name_split[0]
    count_cont_req = int(cont_list_name_split[1])
    log_path, name = os.path.split(full_path_cont)  # [0-9]+

    reg_resim = re.compile(
        r'_rR.*\d+|_rM.{,2}\d+|_rIE.*\d+|_rFVC.*\d+|_rHBA.*\d+'
        + r'|_rFLR.*\d+|_r.*SRR.*\d+|_rA.*\d+')

    compiled_regex = re.compile(r"\[(.*?)\]")

    log_path_split = log_path

    log_start_num = int(compiled_regex.search(
        name).group(0).strip('[]'))
    reg_search_resim = reg_resim.search(name)

    if bool(reg_search_resim):

        rTag = reg_search_resim.group(0)

    else:
        rTag = ''

    name_no_rTag = name.replace(rTag, '')

    name_split_no_rTag = name_no_rTag.split('.')
    extension = name_split_no_rTag[-1]

    name_only_no_rTag = name_split_no_rTag[0]

    longest_first = sorted(sub_string_list, key=len, reverse=True)
    compiled_strings = re.compile(r'(?:{})'
                                  .format('|'.join(
                                      map(re.escape, longest_first))))

    string_search_for_type = compiled_strings.search(
        name_only_no_rTag)
    name_only_no_rTag_type = string_search_for_type.group(0)
    # base_name_no_rTag_no_Type = string_search_for_type.start(0)

    name_only_no_rTag_no_type = re.sub(compiled_strings, '',
                                       name_only_no_rTag)

    name_only_no_rTag_split = name_only_no_rTag_no_type.split('_')

    name_only_no_rTag_pre = name_only_no_rTag_split[:-1]

    name_only_no_rTag_enum = name_only_no_rTag_split[-1].strip('[]')
    name_only_no_rTag_enum = [
        f'{int(name_only_no_rTag_enum)+count_enum:0{digits_log_enum}d}'
        for count_enum in range(count_cont_req)
    ]

    name_list = [
        '_'.join(name_only_no_rTag_pre)
        + '_' + enum
        + name_only_no_rTag_type + rTag + '.' + extension

        for enum in name_only_no_rTag_enum
    ]

    original_log_name_list = [
        '_'.join(name_only_no_rTag_pre)
        + '_' + enum
        + name_only_no_rTag_type + '.' + extension

        for enum in name_only_no_rTag_enum
    ]

    base_name_list = [
        '_'.join(name_only_no_rTag_pre)
        + '_' + enum

        for enum in name_only_no_rTag_enum
    ]

    if (('MCIP' in name or 'GPO-IFV7XX' in log_path)
        or
            ('E2EML' in name or 'GPO-E2E' in log_path)):

        if "lin" in sys.platform or SIMULATE_LINUX:

            log_path_split = os.path.split(log_path)
            log_path_pre = log_path_split[0]
            log_path_post_01 = log_path_split[1].split('_')
            log_path_post = [
                '_'.join(
                    log_path_post_01[:-1]
                    + [f'{int(log_path_post_01[-1])+count_enum:0{digits_log_enum}d}'])
                for count_enum in range(count_cont_req)]
            log_path = [os.path.join(log_path_pre, item)
                        for item in log_path_post]

        else:

            log_path = [log_path]*count_cont_req

    elif ('TNDR1' in name or 'STLA-THUNDER' in log_path):

        log_path = [log_path]*count_cont_req

    else:

        log_path = [log_path]*count_cont_req

    log_path_list = [
        os.path.join(log_path_iter, name_iter)
        for log_path_iter, name_iter in zip(log_path, name_list)
    ]

    rTag_list = [rTag.lstrip('_')]*len(base_name_list)
    rTag_list = [item if item != '' else 'None Available'
                 for item in rTag_list]

    seq_path_list = \
        [os.path.join(*os.path.normpath(
            log_path).split(os.path.sep)[:6]
            + ['1-Rwup'] + basename.split('_')[1:-1])
         for log_path, basename in zip(log_path_list, base_name_list)
         ]

    return (log_path_list, base_name_list,
            rTag_list, seq_path_list, original_log_name_list)


def _create_flat_file_list2(cont_list_name, SIMULATE_LINUX=False):

    cont_list_name_split = cont_list_name.split(',')
    full_path_cont = cont_list_name_split[0]
    count_cont_req = int(cont_list_name_split[1])
    log_path, name = os.path.split(full_path_cont)  # [0-9]+

    # reg_resim = re.compile(
    #     r'rR.*\d+|rM.{,2}\d+|rFLR.*\d+|_r.*SRR.*\d+|rA.*\d+')

    reg_resim = re.compile(
        r'rR.*\d+|rM.{,2}\d+|rIE.*\d+|rFVC.*\d+|rFLR.*\d+|_r.*SRR.*\d+|rA.*\d+')

    # reg_resim = re.compile(
    #     r'(rR.*\d+)|(rM.{,2}\d+)|(rIE.*\d+)|(rFVC.*\d+)|(rFLR.*\d+)|(_r.*SRR.*\d+)|(rA.*\d+)')

    compiled_regex = re.compile(r"\[(.*?)\]")

    # if reg_resim.search(log_path):

    #     if "lin" in sys.platform:

    #         log_path = os.path.join(log_path,
    #                                 '_'.join(
    #                                     name.split('_')[:-1]+
    #                                     [name.split('_')[-1].strip('[]')]))

    log_path_split = log_path

    log_start_num = int(compiled_regex.search(
        name).group(0).strip('[]'))
    reg_search_resim = reg_resim.search(log_path)

    if ('MCIP' in name or 'GPO-IFV7XX' in log_path):

        if bool(reg_search_resim):

            if "lin" in sys.platform or SIMULATE_LINUX:
                name_pre = '_'.join([item.strip('[]')
                                     for item in name.split('_')])
                # name_pre_use =

                log_path = os.path.join(log_path,
                                        name_pre)
                log_path_split = os.path.split(log_path)

            name = name + '_p01_' + reg_search_resim.group(0) + '.mat'

        elif "lin" in sys.platform or SIMULATE_LINUX:

            log_path_split = os.path.split(log_path)
    elif ('TNDR1' in name or 'STLA-THUNDER' in log_path) and (
            "lin" in sys.platform or SIMULATE_LINUX):
        log_path_split = log_path
    else:
        log_path_split = log_path
    # log_start_num = int(log_path_split[1].split('_')[-1])

    # log_path_list = [os.path.join(log_path_split[0],
    #                               '_'.join(log_path_split[1].split('_')[:-1])
    #                               + f"_{(log_start_num+iter_num):04d}",
    #                               # '_'.join(log_path_split[1].split('_')[:-1])
    #                               # + f"_{(log_start_num+iter_num):04d}_p01.mat"
    #                               # name.replace('[','').replace(']',
    #                               #                              f"_{(log_start_num+iter_num):04d}")
    #                               name.replace(compiled_regex.search(
    #                                   name).group(0),
    #                                   f"{(log_start_num+iter_num):04d}")
    #                               ) if isinstance(log_path_split, tuple)
    #                  and not isinstance(log_path_split, str)
    #                  else os.path.join(log_path_split,
    #                                    name.replace(compiled_regex.search(
    #                                        name).group(0),
    #                                        f"{(log_start_num+iter_num):04d}")
    #                                    )
    #                  for iter_num in range(count_cont_req)]

    log_path_list = []

    for iter_num in range(count_cont_req):
        if isinstance(log_path_split, tuple):
            additional_path = ('_'.join(log_path_split[1].split('_')[:-1])
                               + f"_{(log_start_num+iter_num):04d}")
            first_arg = log_path_split[0]
            if bool(reg_search_resim) and (
                    not SIMULATE_LINUX):
                additional_path = ''

        elif isinstance(log_path_split, str):
            additional_path = ''
            first_arg = log_path_split

        log_path_iter = os.path.join(first_arg,
                                     additional_path,

                                     name.replace(compiled_regex.search(
                                         name).group(0),
                                         f"{(log_start_num+iter_num):04d}")
                                     )
        log_path_list.append(log_path_iter)

    return log_path_list


def _get_bearing_from_lat_long(lat1, lat2, long1, long2):

    bearing_angle = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return bearing_angle
