# -*- coding: utf-8 -*-

"""
Created on Wed Jan 24 18:51:11 2024

@author: mfixlz
"""


from asammdf import MDF
import os
import numpy as np
import scipy as sp
import calendar
from collections.abc import Iterable
import pandas as pd
import sys
import copy
from itertools import chain
import more_itertools as mit
import warnings
import re
import pickle
from fxpmath import Fxp
from collections import ChainMap
import json
from numba import jit
import yaml
from decimal import Decimal
from collections import defaultdict
import argparse
from scipy.signal import resample
import xml.etree.ElementTree

#############
import types
from functools import partial

import ctypes
from io import IOBase


from sbp.msg import SBP
from sbp.table import dispatch

try:
    from ProtocolTools import SerializerAPI

    try:
        from ProtocolTools import DecoderAPI

    except:
        print('No Decoder API can be imported')
except:
    print('No ProtocolTools package is available')

if __package__ is None:
    print('Here at none package')
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(to_change_path)
    print(f'Current dir: {os.getcwd()}, to change : {to_change_path}')
    from utils_generic import (mac_to_int, int_to_mac,
                               _stream_def_pickle, merge_dicts,
                               deep_update, sort_list, stream_check,
                               _list_of_dicts_to_dict_of_arrays,
                               Fxp_2,
                               _check_for_obj_instance,
                               duplicate_keys_to_tuples_dict,
                               )
    from utils_mdf import _to_dataframe, setup_fs, _to_dataframe2
    from extract_SPI import extract_SPI_data
    from run_ad5_decode import stream_stats_main
    # from arxml_decoder_core import decode_eth_channel_by_arxml
    from arxml_decoder_core_mux import decode_eth_channel_by_arxml
    import mdf_iter
    import can_decoder

    from fibex_parser_convertor.fibex_parser import FibexParser
    from fibex_parser_convertor.configuration_to_text import SimpleConfigurationFactory
    from dlt.header.header import DltHeader
    from dlt.payload.payload import DltPayload

else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    to_change_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(to_change_path)
    print(f'Current dir: {os.getcwd()}, to change : {to_change_path}')
    from utils_generic import (mac_to_int, int_to_mac,
                               _stream_def_pickle, merge_dicts,
                               deep_update, sort_list, stream_check,
                               _list_of_dicts_to_dict_of_arrays,
                               Fxp_2,
                               _check_for_obj_instance,
                               duplicate_keys_to_tuples_dict,
                               )
    from utils_mdf import _to_dataframe, setup_fs, _to_dataframe2
    from extract_SPI import extract_SPI_data
    from run_ad5_decode import stream_stats_main
    # from arxml_decoder_core import decode_eth_channel_by_arxml
    from arxml_decoder_core_mux import decode_eth_channel_by_arxml
    import mdf_iter
    import can_decoder

    from fibex_parser_convertor.fibex_parser import FibexParser
    from fibex_parser_convertor.configuration_to_text import SimpleConfigurationFactory
    from dlt.header.header import DltHeader
    from dlt.payload.payload import DltPayload


class TrimbleGNSSDataframeParser:
    def __init__(self, trimble_config_path):

        with open(trimble_config_path) as stream:
            self.msg_def_text = yaml.safe_load(stream)

        self.msg_keys_regex = [(msg_name, "\$"+msg_name)
                               for msg_name in list(self.msg_def_text.keys())]
        self.talker_id_capture_filter = re.compile(
            "\$(?P<talkerid>[A-Z]+?),", re.MULTILINE)

        self.TRIMBLE_UDP_DEST_PORT = 28002

    def return_matched_msg_key(self, message):
        # Returns regex which matches the message in packet content
        for msg_key_regex in self.msg_keys_regex:
            if re.compile(msg_key_regex[1]).search(message):
                return msg_key_regex[0]
        return None

    def return_message_def(self):
        # Returns message definitions loaded from yaml file
        return self.msg_def_text

    def get_msg_data(self, msg_id, data):
        # Returns data struct with signal content for each message based on
        # the span set in the yaml file.
        # Signal needs to still be decoded into the right datatype - not done yet
        msg_def = self.msg_def_text[msg_id]
        data_struct_out = dict()
        for signal in msg_def.keys():
            signal_span = msg_def[signal]["span"]
            if len(signal_span) == 1:
                data_struct_out[signal] = data[signal_span[0]]
            else:
                data_struct_out[signal] = data[signal_span[0]:signal_span[1]]
        return data_struct_out

    def parse_gps_df_row(self, df_row, cache, log_start_time):
        row_series = df_row[1]
        # data seems to have a 28 byte header

        udp_header_bytes = row_series["ETH_Frame.ETH_Frame.DataBytes"][20:28]
        udp_destination_port = udp_header_bytes.view(np.dtype(">H"))[1]

        packet_data = dict()
        packet_groups = set()
        packet_talker_ids = set()
        if udp_destination_port != self.TRIMBLE_UDP_DEST_PORT:

            return packet_data, packet_groups, packet_talker_ids, cache
        payload_bytes = row_series["ETH_Frame.ETH_Frame.DataBytes"].tobytes()[
            28:]

        packet_talker_ids = set(self.talker_id_capture_filter.findall(
            payload_bytes.decode("utf-8")))
        timestamp = row_series['timestamps']
        packet_data["timestamp"] = row_series['timestamps']  # + log_start_time
        packet_data["raw"] = list()
        packet_data["parsed"] = list()
        packet_data["debug"] = list()
        # Packet data contains multiple lines, with one message per line
        for lineToParse in payload_bytes.decode("utf-8").splitlines():
            if cache:
                lineToParse = cache+lineToParse
                cache = ""
            if not self.talker_id_capture_filter.findall(lineToParse):
                continue
            # Parse each line separately and extract signal content
            packet_data["raw"].append(lineToParse)
            # split line at every comma, and split last element into signal and checksum based on '*' location
            data_to_parse = lineToParse.split(",")
            if len(data_to_parse[-1].split("*")) == 2:
                # checksum present
                data_to_parse[-1], checksum = data_to_parse[-1].split("*")
            else:
                cache = lineToParse
                continue

            msg_key = self.return_matched_msg_key(lineToParse)
            if msg_key:
                packet_groups.add(msg_key)
                # Add timestamp of message and talker id
                # data contains only the signal content, debug contains information on how the data was extracted
                msg_data_dict = {"timestamp": timestamp,
                                 "talker_id": data_to_parse[0]}
                msg_debug_dict = {
                    "timestamp": timestamp,
                    "talker_id": data_to_parse[0],
                    "capture_regex": msg_key}
                # extract and add data for signals in message
                msg_data_dict.update(self.get_msg_data(
                    msg_key, data_to_parse[1:]))
                msg_data_dict["checksum"] = checksum
                packet_data["parsed"].append(msg_data_dict)
                packet_data["debug"].append(msg_debug_dict)
        return packet_data, packet_groups, packet_talker_ids, cache

    def process_mdf_dataframe(self, mdf_dataframe, log_start_time):
        mdf_dataframe = mdf_dataframe.reset_index()
        parsed_data = dict()
        parsed_data["msg_regex_groups"] = set()
        parsed_data["unique_talker_ids"] = set()
        parsed_data["data"] = list()
        cache = ""
        for row in mdf_dataframe.iterrows():

            (packet_data,
             packet_groups,
             packet_talker_ids, cache) = self.parse_gps_df_row(
                row, cache, log_start_time)
            # append signals from each packet to parsed data struct
            if packet_data:

                parsed_data["data"].append(packet_data)
                parsed_data["msg_regex_groups"].update(packet_groups)
                parsed_data["unique_talker_ids"].update(packet_talker_ids)
            # else:
            #     pass
        return parsed_data

    def get_trimble_data(self, mdf_dataframe, log_start_time):
        # find which groups are in the message.
        # create skeleton dataframe
        self.gnss_dataframe = {group_id: pd.DataFrame(columns=["timestamp",
                                                               "talker_id"]
                                                      + list(
            self.msg_def_text[group_id].keys())+["checksum"])
            for group_id in self.msg_def_text.keys()
        }
        mdf_dataframe = mdf_dataframe.reset_index()
        # mdf_dataframe['timestamps'] = mdf_dataframe['timestamps'].apply(Decimal) + \
        #     log_start_time
        mdf_dataframe['timestamps'] = mdf_dataframe['timestamps'] + \
            log_start_time
        # data imported in self.parsed_data
        parsed_data = self.process_mdf_dataframe(mdf_dataframe, log_start_time)
        for data in parsed_data["data"]:
            for parsed_data_iter, debug_data in zip(data["parsed"], data["debug"]):
                df_name = debug_data["capture_regex"]
                # Add parsed data to dataframe as new row
                self.gnss_dataframe[df_name].loc[len(
                    self.gnss_dataframe[df_name]), :] = parsed_data_iter
        # Need to initialize keys list outside to prevent errors
        msg_list = list(self.gnss_dataframe.keys())
        for msg_key in msg_list:
            if self.gnss_dataframe[msg_key].empty:
                # if dataframe is empty, remove it from the final dictionary
                self.gnss_dataframe.pop(msg_key)

        out_dict = {key: df.to_dict(orient='list')
                    for key, df in
                    self.gnss_dataframe.items()}
        return out_dict


class thunderSignalExtraction:

    def __init__(self,
                 # stream_def_dir_path,
                 # trimble_config_path
                 ):

        warnings.filterwarnings("ignore")

        # self.trimble_obj = TrimbleGNSSDataframeParser(trimble_config_path)
        # self.stream_def_dir_path = stream_def_dir_path
        self.window_length_to_search = 50
        self.me_SW_version_major = -1
        self.me_SW_version_minor = -1

        self._mudp_stream_mapping = {
            16: "VSE",
            18: "Feature_10ms",
            32: "Feature_20ms",
            80: "RadarIF",
            81: "Object_Fusion",
            82: "Tracker_MRR",
            83: "Tracker_FLR",
            85: "SATA_SPP_OTP_VEH",
            86: "AVI_Message",
            87: "AVI_Objectlist",
            33: 'OLP',
            # 	"strdef_src023_str017": "VEH_CALIB", 'PROXI_CAR_CFG'
            17: "Debug_Stream1",
            84: "Debug_Stream2",
            144: "Debug_Stream3",
            145: 'PROXI_CAR_CFG',
            146: 'VEH_CALIB',
        }

        self._srr_stream_mapping = {
            0: 'core_0',
            2: 'core_1',
            3: 'core_2',

        }

        self.mudp_header_mapping = {'header.sourceInfo': 'aptiv_udp_source_info',
                                    'header.customerId': 'aptiv_udp_customer_ID',
                                    'header.sensorId': 'aptiv_udp_sensor_ID',
                                    'header.sensorStatus': 'aptiv_udp_sensor_status',
                                    'header.detectionCnt': 'aptiv_udp_detection_cnt',
                                    'header.mode': 'aptiv_udp_mode',
                                    'header.streamRefIndex': 'aptiv_udp_stream_ref_index',
                                    'header.utcTime': 'aptiv_udp_utc_time',
                                    'header.timeStamp': 'aptiv_udp_time_stamp',
                                    'header.streamLength': 'aptiv_udp_stream_length',
                                    'header.streamNumber': 'aptiv_udp_stream_number',
                                    'header.streamTxCnt': 'aptiv_udp_stream_tx_cnt',
                                    'header.streamVersion': 'aptiv_udp_stream_version',
                                    'header.streamChunks': 'aptiv_udp_stream_chunks',
                                    'header.streamChunkIdx': 'aptiv_udp_stream_chunk_idx',
                                    'header.streamChunkLen': 'aptiv_udp_stream_chunk_len',
                                    'header.sourceTxCnt': 'aptiv_udp_source_tx_cnt',
                                    'header.sourceTxTime': 'aptiv_udp_source_tx_time',
                                    'header.headerLength': 'aptiv_udp_header_length',
                                    'header.version': 'aptiv_udp_header_version',
                                    'header.versionInfo': 'aptiv_udp_version_info',
                                    'header.bigEndian': 'aptiv_udp_big_endian',
                                    'header.streamDataLen': 'aptiv_udp_stream_data_length',
                                    'header.time': 'timestamps',

                                    }

        self.mudp_eth_dest_ports = np.array(
            [5003, 5555, 10002, 49001, 49002, 49003, 49004, 49005,
             30490, ])

        self.time_unit_conversion_factor = 1E-3

        self.req_channels_bus = [int('0x800638', 16),
                                 int('0xe0000', 16),
                                 int('0x1c', 16),
                                 int('0x1d', 16),
                                 int('0x1e', 16),
                                 int('0x3c', 16),
                                 int('0x3d', 16),
                                 int('0x1a', 16),
                                 int('0x1b', 16),
                                 int('0x11', 16),
                                 int('0x12', 16),
                                 ]
        self.req_channels_ref = [int('0xe0000', 16),
                                 int('0xe0001', 16),
                                 int('0xe0002', 16),
                                 int('0xe0005', 16),
                                 int('0xe000a', 16),
                                 ]
        self.req_channels_deb = [int('0x2c', 16),
                                 int('0x2a', 16),
                                 int('0x2b', 16),
                                 int('0x29', 16),
                                 int('0x10101', 16),
                                 int('0x10107', 16),
                                 int('0x10108', 16),
                                 int('0x10109', 16),
                                 int('0x1010a', 16),
                                 int('0x10111', 16),
                                 ]

        self.debug_str = ''

        self.CAN_flat = False
        self.SPI_flat = False
        self.FLR = False

        self.is_updated_udp_parser = True
        self._remove_unused_vars = self.is_updated_udp_parser  # True

        self.stream_check_dict = {}
        self.can_check_dict = {}
        self.is_decoding = True

        self.busID = 'unknown'

        self._min_aptiv_udp_header_len = 80

    def main(self, bus_dict, deb_dict, ref_dict, log_path,
             run_mode: int = -999, **kwargs):

        sub_string_list = ['_bus_', '_deb_', '_ref_']
        boolean_substring = list(map(log_path.__contains__,
                                     sub_string_list))

        boolean_index = np.argwhere(boolean_substring).flatten()

        if not bool(list(boolean_index)):

            return_dict = {'Error message':
                           'File is not of deb, bus or ref type. cannot be decoded'}

            self.debug_str = 'File is not of deb, bus or ref type. cannot be decoded'
            return return_dict, self.debug_str

        sub_string_list_tuple = [(sub_string_list[boolean_index[0]], item)
                                 for item in sub_string_list]

        path_list = [log_path.replace(*iter_tuple)
                     for iter_tuple in sub_string_list_tuple]

        ref_out, bus_out, deb_out, dvl_ext_only = {}, {}, {}, {}

        bus_run_type = True if (run_mode == -999
                                or run_mode == 1
                                or run_mode == -1
                                or run_mode == -3
                                ) else False
        deb_run_type = True if (run_mode == -999
                                or run_mode == 2
                                or run_mode == -1
                                or run_mode == -2
                                or run_mode == 1606
                                or run_mode == 1902
                                ) else False
        ref_run_type = True if (run_mode == -999
                                or run_mode == 3
                                or run_mode == -2
                                or run_mode == -3
                                or run_mode == 1012
                                or run_mode == 1902
                                ) else False

        dvl_only_run_type = True if (run_mode == 312
                                     or run_mode == 1606
                                     or run_mode == 1012
                                     or run_mode == 1902) else False

        if dvl_only_run_type:

            # assert 'mat_data' in kwargs.keys(), \
            #     'send .mat file as input kwargs, i.e., "kwargs["mat_data"]" '

            if 'mat_data' in kwargs.keys():

                is_mat_data = isinstance(kwargs['mat_data'], dict)

                if is_mat_data:

                    dvl_ext_only = self._dvl_ext_only(
                        kwargs['mat_data'], bus_dict, log_path)
                else:
                    dvl_ext_only = {'dvl_ext': {
                        'ERR_MSG': 'Cannot process dvlext. ' +
                                    'check if mat_data file present '}
                                    }

                    self.debug_str = self.debug_str + \
                        '\n ERROR in dvlext only processing, mat file has errors'

            else:

                dvl_ext_only = {'dvl_ext': {
                    'ERR_MSG': 'Cannot process dvlext. ' +
                                'check if mat_data is sent as input ' +
                                'or sw_version from mudp is incorrect'}
                                }

                self.debug_str = self.debug_str + \
                    ('\n ERROR in dvlext only processing, ' +
                     'mat file path is not sent or sw_version from mudp is incorrect')

        if (('trimble_input_dict' in bus_dict)
                    and boolean_substring[2]
                and
                ('trimble_input_dict' in ref_dict)
                ):
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('GPS data exists in ref file disregarding bus file for GPS data')
            del bus_dict['trimble_input_dict']

        if os.path.isfile(path_list[2]) and bool(ref_dict) and ref_run_type:
            existing_channels_ref_CAN = list(self._read_raw_data_mdf(path_list[2],
                                                                     False).keys())
            existing_channels_ref_ETH = list(self._read_raw_data_mdf(path_list[2],
                                                                     True).keys())

            existing_channels_ref = existing_channels_ref_CAN + existing_channels_ref_ETH
            missing_channels = list(set(self.req_channels_ref)
                                    .difference(existing_channels_ref))
            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'ref log missing channels {missing_channels} \n'

            ref_out = self.main_ref(ref_dict, path_list[2])

        if os.path.isfile(path_list[0]) and bool(bus_dict) and bus_run_type:
            existing_channels_bus_CAN = list(self._read_raw_data_mdf(path_list[0],
                                                                     False).keys())
            existing_channels_bus_ETH = list(self._read_raw_data_mdf(path_list[0],
                                                                     True).keys())
            existing_channels_bus = existing_channels_bus_CAN + existing_channels_bus_ETH
            missing_channels = list(set(self.req_channels_bus)
                                    .difference(existing_channels_bus))
            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'bus log missing channels {missing_channels} \n'
            bus_out = self.main_bus(bus_dict, path_list[0])

        if os.path.isfile(path_list[1]) and bool(deb_dict) and deb_run_type:
            existing_channels_deb_CAN = list(self._read_raw_data_mdf(path_list[1],
                                                                     False).keys())
            existing_channels_deb_ETH = list(self._read_raw_data_mdf(path_list[1],
                                                                     True).keys())
            existing_channels_deb = existing_channels_deb_CAN + existing_channels_deb_ETH
            missing_channels = list(set(self.req_channels_deb)
                                    .difference(existing_channels_deb))
            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'debug log missing channels {missing_channels} \n'
            deb_out = self.main_deb(deb_dict, path_list[1])

        # return_dict = {**bus_out, **deb_out, **ref_out}

        return_dict = merge_dicts(bus_out, deb_out)
        return_dict = merge_dicts(return_dict, ref_out)
        return_dict = merge_dicts(return_dict, dvl_ext_only)

        return return_dict, self.debug_str

    def main_ref(self, ref_dict, log_path):

        self._log_type = 'ref'

        self.stream_def_dir_path = ref_dict['stream_def_dir_path']

        trimble_input_dict = ref_dict['trimble_input_dict']

        can_input_dict = ref_dict['can_input_dict']

        group_data_dict = self._read_all_eth_data(log_path)

        can_dict = {}

        self.CAN_flat = True

        _, can_dict, self.can_log_start_time, can_dict_2 = \
            self._extract_thunder_CAN(
                can_input_dict['can_db_path'],
                can_input_dict['db_signal_pairs'],
                can_input_dict['channel_name_pairs'],
                log_path,
            )

        trimble_dict = {}
        if trimble_input_dict['bus_channel'] in group_data_dict.keys():
            trimble_dict = self._get_trimble_data(
                group_data_dict[trimble_input_dict['bus_channel']],
                trimble_input_dict['trimble_config_path'])
        else:

            self.debug_str = self.debug_str \
                + f"Trimble channel {trimble_input_dict['bus_channel']} missing \n"

        trimble_dict['bfname'] = os.path.split(log_path)[1]

        user_event_dict = self._get_user_event_data(MDF(log_path))
        user_event_dict['log_path'] = os.path.split(log_path)[0]
        # user_event_dict['log_name'] = os.path.split(log_path)[1]
        user_event_dict['bfname'] = os.path.split(log_path)[1]

        # return_dict = {}

        return_dict = {
            'trimble_gps': trimble_dict,
            'user_event_dict': user_event_dict,
            'can': can_dict,

        }

        return return_dict

    def main_deb(self, deb_dict: dict,
                 log_path):

        self._log_type = 'deb'

        self.stream_def_dir_path = deb_dict['stream_def_dir_path']

        srr_input_dict = deb_dict['srr_dict']
        spi_input_dict = deb_dict['spi_input_dict']

        group_data_dict = self._read_all_eth_data(log_path)

        srr_dict = {}

        if bool(srr_input_dict):

            for channel_name, bus_channel in \
                    srr_input_dict['bus_channel_dict'].items():

                self.busID = bus_channel

                srr_dict_iter = self._extract_thunder_udp(
                    group_data_dict[bus_channel],
                    is_tcp=srr_input_dict['is_tcp'])

                srr_dict[channel_name] = srr_dict_iter

        srr_dict['bfname'] = os.path.split(log_path)[1]

        SPI_eyeQ_dict = {}

        if bool(spi_input_dict):

            self.CAN_flat = False

            self.SPI_flat = True

            # SPI_eyeQ_dict = self._SPI_data_parser(log_path,
            #                                       protocol_type_='SPI',
            #                                       dbc_path=spi_input_dict['dbc_path'])
            dbc_pickle_path_dma = spi_input_dict['dbc_pickle_path_dma']

            SPI_eyeQ_dict, _ = self._SPI_data_parser2(
                log_path, dbc_pickle_path_dma)

        SPI_eyeQ_dict['bfname'] = os.path.split(log_path)[1]

        return_dict = {'srr': srr_dict,
                       'SPI_eyeQ': SPI_eyeQ_dict,
                       }

        return return_dict

    def main_bus(self,
                 bus_dict: dict,
                 log_path,
                 ):

        self._log_type = 'bus'

        self.stream_def_dir_path = bus_dict['stream_def_dir_path']

        can_input_dict = bus_dict['can_input_dict']
        mudp_input_dict = bus_dict['mudp_input_dict']
        flr4_input_dict = bus_dict['flr4_input_dict']

        if 'trimble_input_dict' in bus_dict.keys():
            trimble_input_dict = bus_dict['trimble_input_dict']
        else:
            trimble_input_dict = {}

        group_data_dict = self._read_all_eth_data(log_path)

        mudp_dict = {}

        if mudp_input_dict['bus_channel'] in group_data_dict.keys():

            if 'bus_channel_flr' in mudp_input_dict.keys():
                req_data_mudp = pd.concat([group_data_dict[
                    mudp_input_dict['bus_channel']],
                    group_data_dict[
                    mudp_input_dict['bus_channel_flr']]
                ],
                    axis=0)
            else:
                req_data_mudp = group_data_dict[mudp_input_dict['bus_channel']]
                self.debug_str = self.debug_str \
                    + f"mudp channel {mudp_input_dict['bus_channel_flr']} missing \n"
            # self.stream_check_dict[
            #     f"busID_{mudp_input_dict['bus_channel']}"] = True

            self.busID = mudp_input_dict['bus_channel']

            mudp_dict = self._extract_thunder_udp(
                req_data_mudp,
                is_tcp=mudp_input_dict['is_tcp'])
        else:

            self.debug_str = self.debug_str \
                + f"mudp channel {mudp_input_dict['bus_channel']} missing \n"
            # self.stream_check_dict[
            #     f"busID_{mudp_input_dict['bus_channel']}"] = False

        mudp_dict['bfname'] = os.path.split(log_path)[1]

        if ('Debug_Stream3' in mudp_dict.keys()
                    and
                    isinstance(mudp_dict['Debug_Stream3'], dict)
                and
                'sw_version_data' in mudp_dict['Debug_Stream3'].keys()
                ):
            root_path = mudp_dict['Debug_Stream3']['sw_version_data']

            model_year = hex(np.unique(root_path['model_year'])[-1])[2:]
            vehicle_platform = hex(
                np.unique(root_path['vehicle_platform'])[-1])[2:]
            hardware_config = hex(
                np.unique(root_path['hardware_config'])[-1])[2:]
            calender_year = hex(np.unique(root_path['calender_year'])[-1])[2:]
            pi_sprint = hex(np.unique(root_path['pi_sprint'])[-1])[2:]
            patch_level = hex(np.unique(root_path['patch_level'])[-1])[2:]

            sw_version_list = [f'{model_year:0>2s}',
                               f'{vehicle_platform:0>2s}',
                               f'{hardware_config:0>2s}',
                               f'{calender_year:0>2s}',
                               f'{pi_sprint:0>2s}',
                               f'{patch_level:0>2s}', ]

            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            # print(sw_version_list)
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

            sw_version = '.'.join(sw_version_list)

            vehicle_line = mudp_dict['PROXI_CAR_CFG'][
                'proxi_data']['Vehicle_Line_Configuration_msg']
            vehicle_line = vehicle_line[-1]

            ################################
            # # RB 29082024 18:30 After discussion with Vinay
            # mudp_dict['sw_version_edited'] = f'YES. 29082024 18:42, {sw_version}'
            # sw_version = '24.01.01.24.36.91'

            ###########################################

            req_path = log_path
            # reg_resim = re.compile('rFLR.+?_r.SRR.+?_rM[0-9]+?_rV.+?_rA.+?_')
            reg_resim = re.compile('rA.+?_')

            is_resim = bool(reg_resim.search(req_path))

            if is_resim:

                sw_version_resim = re.findall(r'rA(\d+)', log_path)[0][:-2]
                sw_version_resim = '.'.join(a+b
                                            for a, b in
                                            zip(sw_version_resim[::2],
                                                sw_version_resim[1::2]))
                mudp_dict['sw_version_Debug_Stream3'] = sw_version

            else:
                sw_version_resim = sw_version

            mudp_dict['sw_version'] = sw_version_resim

            if vehicle_platform in ['05', '06', '07', '08'] or vehicle_line == 127:

                can_input_dict['db_signal_pairs'][6] = \
                    ('PLB24_E3A_R1_CANFD3.dbc', 26)
                can_input_dict['db_signal_pairs'][5] = \
                    ('PLB24_E3A_R1_CANFD14.dbc', 27)

                # can_input_dict['channel_name_pairs']['CAN26'] = 'FDCAN14_LB_ADCAM'
                # can_input_dict['channel_name_pairs']['CAN27'] = 'FDCAN3_LB_ADCAM'

            if mudp_dict['sw_version'] in can_input_dict['sw_ver_dbc_mapping'].keys():

                vals = can_input_dict['sw_ver_dbc_mapping'][
                    mudp_dict['sw_version']]
                can_input_dict['db_signal_pairs'][0] = (
                    vals[0], 29)
                can_input_dict['db_signal_pairs'][1] = (
                    vals[1], 30)

                can_input_dict['db_signal_pairs'][5] = (
                    vals[2], 27)
                can_input_dict['db_signal_pairs'][6] = (
                    vals[3], 26)

                if vehicle_line == 101:

                    can_input_dict['db_signal_pairs'][2] = \
                        (vals[3], 61)
                    can_input_dict['db_signal_pairs'][3] = \
                        (vals[2], 60)

                    can_input_dict['channel_name_pairs']['CAN60'] = 'FDCAN14_Mule'
                    can_input_dict['channel_name_pairs']['CAN61'] = 'FDCAN3_Mule'

            else:

                can_input_dict['db_signal_pairs'][0] = (
                    'CANSB1_S3_01_22_2024_Ver_12.dbc', 29)
                can_input_dict['db_signal_pairs'][1] = (
                    'CANSB2_S3_01_22_2024_Ver_12.dbc', 30)

                can_input_dict['db_signal_pairs'][5] = (
                    'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc', 27)
                can_input_dict['db_signal_pairs'][6] = (
                    'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc', 26)

                if vehicle_line == 101:

                    can_input_dict['db_signal_pairs'][2] = \
                        ('Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc', 61)
                    can_input_dict['db_signal_pairs'][3] = \
                        ('Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc', 60)

                    can_input_dict['channel_name_pairs']['CAN60'] = 'FDCAN14_Mule'
                    can_input_dict['channel_name_pairs']['CAN61'] = 'FDCAN3_Mule'

            if mudp_dict['sw_version'] in flr4_input_dict[
                    'flr4_sw_ver_dbc_mapping'].keys():

                flr4_dbc = os.path.join(
                    flr4_input_dict['flr4_arxml_root_path'],
                    flr4_input_dict['flr4_sw_ver_dbc_mapping'][
                        mudp_dict['sw_version']]
                )
            else:
                flr4_dbc = os.path.join(
                    flr4_input_dict['flr4_arxml_root_path'],
                    # 'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml',
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                )

        else:
            sw_version = 'Cannot be determined. Debug_Stream3 data is not present'
            mudp_dict['sw_version'] = 'sw version cannot be determined. ' +\
                'Debug_Stream3 data is not present'

            # can_input_dict['db_signal_pairs'][0] = [
            #     'CANSB1_S3_01_22_2024_Ver_12.dbc', 29]
            # can_input_dict['db_signal_pairs'][1] = [
            #     'CANSB2_S3_01_22_2024_Ver_12.dbc', 30]

            can_input_dict['db_signal_pairs'][0] = (
                'CANSB1_S3_01_22_2024_Ver_12.dbc', 29)
            can_input_dict['db_signal_pairs'][1] = (
                'CANSB2_S3_01_22_2024_Ver_12.dbc', 30)

            can_input_dict['db_signal_pairs'][5] = (
                'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc', 27)
            can_input_dict['db_signal_pairs'][6] = (
                'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc', 26)

            can_input_dict['db_signal_pairs'][2] = \
                ('Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc', 61)
            can_input_dict['db_signal_pairs'][3] = \
                ('Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc', 60)

            can_input_dict['channel_name_pairs']['CAN60'] = 'FDCAN14_Mule'
            can_input_dict['channel_name_pairs']['CAN61'] = 'FDCAN3_Mule'

            flr4_dbc = os.path.join(
                flr4_input_dict['flr4_arxml_root_path'],
                # 'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml',
                'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
            )

        can_dict = {}

        self.CAN_flat = True

        # print('*********************',
        #       f"{can_input_dict['db_signal_pairs']}, \n",
        #       f"{can_input_dict['channel_name_pairs']}")
        print(f"\n Software version : {mudp_dict['sw_version']}")
        print(
            f"\n Software version according to  Debug_Stream3 : {sw_version}")

        _, can_dict, self.can_log_start_time, return_val_2 = \
            self._extract_thunder_CAN(
                can_input_dict['can_db_path'],
                can_input_dict['db_signal_pairs'],
                can_input_dict['channel_name_pairs'],
                log_path,
            )
        can_dict['bfname'] = os.path.split(log_path)[1]

        if ('Debug_Stream3' in mudp_dict.keys()
                    and
                    isinstance(mudp_dict['Debug_Stream3'], dict)
                and
                'sw_version_data' in mudp_dict['Debug_Stream3'].keys()
                ):

            if vehicle_platform in ['05', '06', '07', '08'] or vehicle_line == 127:

                can_dict['vehicle_line'] = 'LB'
            else:
                # isinstance(can_dict['FDCAN14_Mule'], str):
                if vehicle_line == 101:
                    can_dict['vehicle_line'] = 'WL MULE'
                elif vehicle_line in [125, 126]:
                    can_dict['vehicle_line'] = 'KX'
                    # if 'FDCAN14_Mule' in can_dict:
                    #     can_dict['FDCAN14_Mule'] = \
                    #         'The channel 60 is not supposed to be in KX vehicle. so removed'
                    # if 'FDCAN3_Mule' in can_dict:
                    #     can_dict['FDCAN14_Mule'] = \
                    #         'The channel 61 is not supposed to be in KX vehicle. so removed'

                    # can_dict['DBC_used'] = vals
                else:
                    can_dict['vehicle_line'] = 'UNKNOWN'
        else:

            can_dict['vehicle_line'] = 'UNKNOWN. Debug_Stream3 data is not present'

        flr4_dict = {}
        flr4_dict_dvl_ext = {}

        try:
            self.FLR = True
            self.req_arxml_name = flr4_dbc
            # print('Before here *****************')

            # flr4_dict = self._flr4_ad5_parser(log_path)
            flr4_dict, flr4_dict_dvl_ext = self._flr4_ad5_parser2(log_path)
            # print('After here *****************')
            self.FLR = False
        except:
            print('Error in FLR4 decoding.')
            self.FLR = False

        flr4_dict['bfname'] = os.path.split(log_path)[1]
        flr4_dict_dvl_ext['bfname'] = os.path.split(log_path)[1]

        trimble_dict = {}
        if not bool(trimble_input_dict):

            self.debug_str = self.debug_str \
                + f"Trimble channel {int('0xe0000', 16)} "\
                + " might be present in ref file and not processed here \n"

        elif trimble_input_dict['bus_channel'] in group_data_dict.keys():
            trimble_dict = self._get_trimble_data(
                group_data_dict[trimble_input_dict['bus_channel']],
                trimble_input_dict['trimble_config_path'])
        else:

            self.debug_str = self.debug_str \
                + f"Trimble channel {trimble_input_dict['bus_channel']} missing \n"\

        trimble_dict['bfname'] = os.path.split(log_path)[1]

        return_dict = {'mudp': mudp_dict,
                       'can': can_dict,
                       'flr4_AD5': flr4_dict,
                       'trimble_gps': trimble_dict,

                       }

        return_dict_2 = copy.deepcopy(return_dict)
        return_dict_2['can'] = return_val_2
        return_dict_2['flr4_AD5'] = flr4_dict_dvl_ext

        dvl_data_dict_path = bus_dict['dvl_data_dict_path']

        dvl_ext = self._dvl_ext_data(dvl_data_dict_path, return_dict_2)

        return_dict['dvl_ext'] = dvl_ext

        return return_dict

    def _dvl_ext_only(self, mat_data, bus_dict, log_path):

        # reg_resim = re.compile('rA.+?_')

        # bf_name = mat_data['mudp']['bfname']

        # is_resim = bool(reg_resim.search(bf_name))

        sw_version = mat_data['mudp']['sw_version']

        can_input_dict = bus_dict['can_input_dict']
        # mudp_input_dict = bus_dict['mudp_input_dict']
        flr4_input_dict = bus_dict['flr4_input_dict']

        vehicle_line = mat_data['mudp']['PROXI_CAR_CFG'][
            'proxi_data']['Vehicle_Line_Configuration_msg']
        vehicle_line = vehicle_line[-1]

        if sw_version in can_input_dict['sw_ver_dbc_mapping'].keys():

            vals = can_input_dict['sw_ver_dbc_mapping'][
                sw_version]
            can_input_dict['db_signal_pairs'][0] = (
                vals[0], 29)
            can_input_dict['db_signal_pairs'][1] = (
                vals[1], 30)

            can_input_dict['db_signal_pairs'][5] = (
                vals[2], 27)
            can_input_dict['db_signal_pairs'][6] = (
                vals[3], 26)

            if vehicle_line == 101:

                can_input_dict['db_signal_pairs'][2] = \
                    (vals[3], 61)
                can_input_dict['db_signal_pairs'][3] = \
                    (vals[2], 60)

                can_input_dict['channel_name_pairs']['CAN60'] = 'FDCAN14_Mule'
                can_input_dict['channel_name_pairs']['CAN61'] = 'FDCAN3_Mule'

        else:

            can_input_dict['db_signal_pairs'][0] = (
                'CANSB1_S3_01_22_2024_Ver_12.dbc', 29)
            can_input_dict['db_signal_pairs'][1] = (
                'CANSB2_S3_01_22_2024_Ver_12.dbc', 30)

            can_input_dict['db_signal_pairs'][5] = (
                'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc', 27)
            can_input_dict['db_signal_pairs'][6] = (
                'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc', 26)

            if vehicle_line == 101:

                can_input_dict['db_signal_pairs'][2] = \
                    ('Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc', 61)
                can_input_dict['db_signal_pairs'][3] = \
                    ('Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc', 60)

                can_input_dict['channel_name_pairs']['CAN60'] = 'FDCAN14_Mule'
                can_input_dict['channel_name_pairs']['CAN61'] = 'FDCAN3_Mule'

        if sw_version in flr4_input_dict[
                'flr4_sw_ver_dbc_mapping'].keys():

            flr4_dbc = os.path.join(
                flr4_input_dict['flr4_arxml_root_path'],
                flr4_input_dict['flr4_sw_ver_dbc_mapping'][
                    sw_version]
            )
        else:

            flr4_dbc = os.path.join(
                flr4_input_dict['flr4_arxml_root_path'],
                # 'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml',
                'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
            )

        self.CAN_flat = True

        _, can_dict, self.can_log_start_time, return_val_2 = \
            self._extract_thunder_CAN(
                can_input_dict['can_db_path'],
                can_input_dict['db_signal_pairs'],
                can_input_dict['channel_name_pairs'],
                log_path,
            )

        self.FLR = True
        self.req_arxml_name = flr4_dbc
        # print('Before here *****************')

        # flr4_dict = self._flr4_ad5_parser(log_path)
        flr4_dict, flr4_dict_dvl_ext = self._flr4_ad5_parser2(log_path)
        # print('After here *****************')
        self.FLR = False

        return_dict_2 = copy.deepcopy(mat_data)
        return_dict_2['can'] = return_val_2
        return_dict_2['flr4_AD5'] = flr4_dict_dvl_ext

        dvl_ext = self._dvl_ext_data(dvl_data_dict_path, return_dict_2)

        mat_data['dvl_ext_2'] = dvl_ext

        return mat_data

    def _dvl_ext_data(self,
                      dvl_data_dict_path: os.path.join,
                      raw_data: dict):

        df = pd.read_excel(dvl_data_dict_path, sheet_name='data_dict')

        df2 = df[['message_path_dvl_ext',
                  'signal_path_to_copy_from']]

        df2.loc[:, 'signal_path_to_copy_from'] = \
            df2['signal_path_to_copy_from'].fillna(
                'Signal path not yet available')

        dvlExt_flat = {key: val
                       if 'Signal path not yet available' in val
                       else stream_check(raw_data, val)
                       for key, val in
                       zip(df2['message_path_dvl_ext'],
                           df2['signal_path_to_copy_from'])
                       }

        dvlExt = self.nest_dict(dvlExt_flat)
        return dvlExt

    def _get_user_event_data(self,
                             mdf_obj,
                             ):

        header_comment_xml = mdf_obj.header.comment
        events_list = mdf_obj.events
        events_meta_data_comments = [event.comment
                                     for event in events_list]
        event_types = [event.event_type
                       for event in events_list]
        event_tx_names = [event.name
                          for event in events_list]

        event_addresses_hex = [hex(event.address)
                               for event in events_list]

        event_sync_types = [event.sync_type
                            for event in events_list]

        event_range_types = [event.range_type
                             for event in events_list]

        event_causes = [event.cause
                        for event in events_list]

        event_time_stamp_bases = [event.sync_base
                                  for event in events_list]

        event_time_stamp_factor = [event.sync_factor
                                   for event in events_list]

        user_event_dict = {
            'header_comment_xml': header_comment_xml,
            'events_count': len(events_list),
            'events_meta_data_comments': events_meta_data_comments,
            'event_hot_key_enums': event_types,
            'event_tx_names': event_tx_names,
            'event_addresses': event_addresses_hex,
            'event_sync_types': event_sync_types,
            'event_range_types': event_range_types,
            'event_causes': event_causes,
            'event_time_stamp_bases': event_time_stamp_bases,
            'event_time_stamp_factor': event_time_stamp_factor,

        }

        return user_event_dict

    def _get_trimble_data(self,
                          trimble_raw_data_df,
                          trimble_config_path):

        trimble_obj = TrimbleGNSSDataframeParser(trimble_config_path)

        trimble_dict = trimble_obj.get_trimble_data(trimble_raw_data_df,
                                                    self.log_start_time)

        trimble_dict = {key.replace('.', '')
                        .replace(',', '_')
                        .replace(" ", ""): val
                        for key, val in trimble_dict.items()
                        }

        return trimble_dict

    def _thunder_CAN_db_mapping(self, can_db_path,
                                db_signal_pairs: tuple):

        can_db_map = {'CAN': [(os.path.join(can_db_path, db_name), channel)
                              for db_name, channel in db_signal_pairs
                              ]
                      }

        return can_db_map

    def _extract_thunder_CAN2(self, can_db_path,
                              db_signal_pairs,
                              channel_name_pairs,
                              log_path):

        # req_path = log_path

        group_data_dict = self._read_all_CAN_data(log_path)

        fs = setup_fs()

        # with fs.open(log_path, "rb") as handle:
        #     # Open the file and extract a dataframe with the raw CAN records.
        #     mdf_file = mdf_iter.MdfFile(handle)

        #     df = mdf_file.get_data_frame()

        can_db_map = self._thunder_CAN_db_mapping(can_db_path,
                                                  db_signal_pairs)

        CAN_decoded = {}

        can_msg_id_dict = {}

        for can_dbc_path, can_channel in can_db_map['CAN']:

            db = can_decoder.load_dbc(can_dbc_path,
                                      use_custom_attribute="SPN"
                                      )
            dataframe_decoder = can_decoder.DataFrameDecoder(db)

            if not can_channel in group_data_dict.keys():

                CAN_decoded[can_channel] = f'The channel {can_channel} is not available'
            else:

                id_name_dict = {}

                Lines = open(can_dbc_path,
                             encoding='cp1252',
                             errors='replace').readlines()

                req_id_lines = [line.split()
                                for line in Lines if 'BO_ ' in line]
                req_id_lines = [
                    item for item in req_id_lines if item[0] == 'BO_']

                req_id_lines_idx = {line_nr: line.split()
                                    for line_nr, line in enumerate(Lines)
                                    if 'BO_ ' in line}
                req_id_lines_idx = {line_nr: item[1]
                                    for line_nr, item in req_id_lines_idx.items()
                                    if item[0] == 'BO_'}
                line_nr_list = list(req_id_lines_idx.keys())

                line_nr_range_tuple = [(start, end)
                                       for start, end in zip(line_nr_list,
                                                             line_nr_list[1:])]

                # TODO: need to add while loop to determine the end of last message

                for start, end in line_nr_range_tuple:

                    line_x = Lines[start:end]

                for id_line in req_id_lines:

                    id_name_dict[int(id_line[1])] = id_line[2].replace(':', '')

                can_msg_id_dict[can_channel] = id_name_dict

                channel_data = dataframe_decoder.decode_frame(
                    group_data_dict[can_channel],
                    # columns_to_drop=["Raw Value"]
                )
                channel_data['can_id'] = channel_data['CAN ID'].apply(
                    lambda x: id_name_dict.get(x))

                channel_data['signal_path'] = \
                    channel_data['can_id'] + '.' + channel_data['Signal']

                channel_data = channel_data[['signal_path',
                                             'Physical Value',
                                             'CAN ID']]

                channel_data = [y[['signal_path', 'Physical Value']]
                                for x, y in channel_data.groupby('CAN ID')]

                channel_data = [df_x.pivot_table(index=df_x.index,
                                                 columns='signal_path',
                                                 values='Physical Value',
                                                 aggfunc='first')
                                for df_x in channel_data]

                CAN_decoded[can_channel] = \
                    dataframe_decoder.decode_frame(
                        group_data_dict[can_channel],
                        # columns_to_drop=["Raw Value"]
                )

                print(f'Decoded channel {can_channel}')

        return CAN_decoded

    def _extract_thunder_CAN(self, can_db_path,
                             db_signal_pairs,
                             channel_name_pairs,
                             log_path):

        req_path = log_path
        reg_resim = re.compile('rFLR.+?_r.SRR.+?_rM[0-9]+')
        is_resim = bool(reg_resim.search(req_path))
        if is_resim:
            MDF.to_dataframe = _to_dataframe  # Monkey patching
        yop = MDF(req_path)

        can_db_map = self._thunder_CAN_db_mapping(can_db_path,
                                                  db_signal_pairs)

        can_data = yop.extract_bus_logging(database_files=can_db_map)

        # log_start_time = calendar.timegm(
        #     can_data.header.start_time.timetuple())

        log_start_time = can_data.header.start_time.timestamp()
        # log_start_time = Decimal(can_data.header.abs_time)/Decimal(1e9)

        can_data_generator = can_data.iter_groups(use_display_names=True,
                                                  # raster=0.001
                                                  time_from_zero=False,
                                                  )
        can_data_list = []
        for data_ in can_data_generator:
            # can_data_dict[key] = data_
            data_2 = data_.reset_index()

            col_prefixes = list(data_2.columns)
            col_prefixes = [col.split('.') for col in data_2.columns]
            col_prefix = [list_[:-1] for list_ in col_prefixes
                          if len(list_) > 1][0]
            new_time_col = '.'.join(col_prefix) + '.cTime'
            data_2 = data_2.rename({'timestamps': new_time_col, },
                                   axis=1, )
            # data_2[new_time_col] = data_2[new_time_col].apply(
            #     Decimal) + log_start_time
            data_2[new_time_col] = data_2[new_time_col] + log_start_time
            can_data_list.append(data_2)

        ########################################################################

        can_data_list_dtypes_dict = [item.dtypes.to_dict()
                                     for item in can_data_list]

        can_data_list_dict = [item.to_dict(orient='list')
                              for item in can_data_list]

        can_data_out_dict = dict(ChainMap(*can_data_list_dict))

        can_data_out_dtypes_dict = dict(ChainMap(*can_data_list_dtypes_dict))

        can_data_out_dict = {key: np.array(val,
                                           dtype=can_data_out_dtypes_dict[key])
                             for key, val in can_data_out_dict.items()
                             }

        #########################################################################
        #########################################################################

        can_data_out_dict_2 = self._process_flat_dict(can_data_list,
                                                      sub_string_list=[['_\d{3}_\d{3}',
                                                                        r'_\\d{3}_\\d{3}'],
                                                                       ['_\d+$',
                                                                        r'_\\d+$']
                                                                       ]
                                                      )

        can_data_out_dict_2 = dict(ChainMap(*can_data_out_dict_2))

        can_data_out_dict_2, debug_str_append = self._process_flat_dict_all(
            can_data_out_dict_2,
            sub_string_list=[['_\d{3}_\d{3}',
                              r'_\\d{3}_\\d{3}'],
                             ['_\d+$',
                              r'_\\d+$']
                             ]
        )

        can_data_out_dict_2 = self.nest_dict(can_data_out_dict_2)

        return_val_2 = {}

        for replaced_key, replaced_val in channel_name_pairs.items():

            if replaced_key in can_data_out_dict_2.keys():

                return_val_2[replaced_val] = can_data_out_dict_2[replaced_key]
            else:

                return_val_2[replaced_val] = \
                    f'The channel {replaced_key} is not available'

        if bool(debug_str_append):

            for key, val in channel_name_pairs.items():

                debug_str_append = debug_str_append.replace(key, val)

            self.debug_str = self.debug_str \
                + '\n CAN decoding warning (dvlExt):' + debug_str_append

        ########################################################################

        can_data_out_dict = self.nest_dict(can_data_out_dict)

        return_val = {}

        for replaced_key, replaced_val in channel_name_pairs.items():

            if replaced_key in can_data_out_dict.keys():

                return_val[replaced_val] = can_data_out_dict[replaced_key]
            else:

                return_val[replaced_val] = \
                    f'The channel {replaced_key} is not available'

        return can_data_out_dict, return_val, log_start_time, return_val_2

    def _extract_thunder_udp(self, mudp_raw_data, is_tcp: bool = False):

        eth_df = self._read_raw_eth_data(mudp_raw_data)

        _, ipv4_df = self._read_raw_ipv4_data(eth_df)

        if is_tcp:
            tcp_df = self._read_raw_tcp_data(ipv4_df)

        udp_df = self._read_raw_udp_data(ipv4_df)

        # self.udp_df = udp_df

        mudp_output_dict = self._read_raw_aptiv_udp_data(
            udp_df, self.is_decoding)

        return mudp_output_dict

    def _read_raw_data_mdf(self, log_path, ethernet_only=True):

        MDF.to_dataframe = _to_dataframe2

        yop = MDF(log_path)
        self.log_start_time = yop.header.start_time.timestamp()

        # self.log_start_time = Decimal(yop.header.abs_time)/Decimal(1e9)

        group_data_generator = yop.iter_groups(use_display_names=True,
                                               # raster=0.001
                                               raw=True,
                                               keep_arrays=True,
                                               time_from_zero=False,

                                               )

        group_data_mdf = {}

        if ethernet_only:

            for data in group_data_generator:

                if ((not data.empty) and
                        ('ETH_Frame.ETH_Frame.BusChannel' in data.columns)):
                    data = data.dropna(
                        subset=['ETH_Frame.ETH_Frame.BusChannel'], axis=0, )
                    assert len(data['ETH_Frame.ETH_Frame.BusChannel'].unique()) == 1, \
                        'multiple channels exist in dataframe. check'

                    bus_channel = int(data['ETH_Frame.ETH_Frame.BusChannel'].unique(
                    ).item())

                    group_data_mdf[bus_channel] = data
        else:
            for data in group_data_generator:
                if not data.empty:
                    if 'CAN_DataFrame.CAN_DataFrame.BusChannel' in data.columns:
                        unique_vals = data[
                            'CAN_DataFrame.CAN_DataFrame.BusChannel'].unique(
                        )
                        unique_vals = unique_vals[~pd.isnull(unique_vals)]
                        bus_channel = int(unique_vals.item())
                        group_data_mdf[bus_channel] = data

                    # elif 'ETH_Frame.ETH_Frame.BusChannel' in data.columns:
                    #     bus_channel = int(data[
                    #         'ETH_Frame.ETH_Frame.BusChannel'].unique(
                    #     ).item())
                    # group_data_mdf[bus_channel] = data

                    # # elif 'PLP_Raw_Data.PLP_Raw_Data.BusChannel' in data.columns:
                    # #     bus_channel = int(data[
                    # #         'PLP_Raw_Data.PLP_Raw_Data.BusChannel'].unique(
                    # #     ).item())

                    # else:
                    #     continue

        return group_data_mdf

    def _read_raw_aptiv_udp_data(self, udp_df, is_decoding: bool = True):

        udp_df['is_aptiv_udp'] = False
        aptiv_udp_indices = udp_df.query('udp_port_filter == True ' +
                                         'and udp_payload_length >= 24').index
        udp_df.loc[aptiv_udp_indices, 'is_aptiv_udp'] = True

        aptiv_udp_df = udp_df.query(
            'is_aptiv_udp == True').reset_index(drop=True)

        non_aptiv_udp_cols = list(udp_df.columns)

        if not aptiv_udp_df.empty:

            HEADER_CONSTANT_MASK = int('F0', 16)
            HEADER_CONSTANT_FLAG = int('A0', 16)
            HEADER_VERSION_MASK = int('0F', 16)

            aptiv_udp_header_bytes = pd.DataFrame(
                aptiv_udp_df.apply(self._helper_header_bytes,
                                   axis=1,
                                   col_name='udp_payload',
                                   row_end=2).to_list()).values

            aptiv_udp_df['aptiv_udp_header_bytes'] = pd.Series(
                list(aptiv_udp_header_bytes))

            aptiv_udp_df['aptiv_udp_version_info'] = aptiv_udp_df.apply(
                self._helper_typecast,
                axis=1,
                col_name='aptiv_udp_header_bytes',
                row_start_end=[
                    1-1, 2],
                conversion_type=np.uint16,
                is_big_endian=False,
            ).values
            aptiv_udp_df['aptiv_udp_version'] = aptiv_udp_header_bytes[:, 2-1]
            aptiv_udp_df['aptiv_udp_header_length'] = aptiv_udp_header_bytes[:, 1-1]

            little_endian = np.bitwise_and(aptiv_udp_header_bytes[:, 2-1],
                                           HEADER_CONSTANT_MASK) == HEADER_CONSTANT_FLAG

            big_endian = np.bitwise_and(aptiv_udp_header_bytes[:, 1-1],
                                        HEADER_CONSTANT_MASK) == HEADER_CONSTANT_FLAG

            aptiv_udp_df['aptiv_udp_big_endian'] = False
            aptiv_udp_df.loc[big_endian, 'aptiv_udp_big_endian'] = True

            aptiv_udp_df['aptiv_udp_little_endian'] = False
            aptiv_udp_df.loc[little_endian, 'aptiv_udp_little_endian'] = True

            if bool(np.sum(big_endian)):
                aptiv_udp_df.loc[big_endian, 'aptiv_udp_version'] = \
                    aptiv_udp_header_bytes[big_endian, 1-1]
                aptiv_udp_df.loc[big_endian, 'aptiv_udp_header_length'] = \
                    aptiv_udp_header_bytes[big_endian, 2-1]

            to_compare_audp = np.logical_not(np.logical_or(
                aptiv_udp_df['aptiv_udp_big_endian'].values,
                aptiv_udp_df['aptiv_udp_little_endian'].values
            ).astype(bool))
            # to_compare_bool_arr_audp = np.sum(to_compare_audp, ).astype(bool)
            to_compare_audp_2 = np.greater(aptiv_udp_df['aptiv_udp_header_length']
                                           .values,
                                           self._min_aptiv_udp_header_len)
            indices_to_drop = np.flatnonzero(
                np.logical_or(to_compare_audp,
                              to_compare_audp_2))

            if len(indices_to_drop) > 0:
                aptiv_udp_df = aptiv_udp_df.drop(
                    index=indices_to_drop).reset_index()
                if np.sum(to_compare_audp) > 0:
                    out_val_map_key = 'In Aptiv mudp decoding, ' +\
                        f'there are {np.sum(to_compare_audp)} ' + \
                        ' timestamps with header being neither big endian ' + \
                        'nor small endian. Skipping them for now.'
                    self.debug_str = self.debug_str + out_val_map_key + '\n'
                big_endian = aptiv_udp_df['aptiv_udp_big_endian']
                little_endian = aptiv_udp_df['aptiv_udp_little_endian']

            min_header_len = aptiv_udp_df['aptiv_udp_header_length'].min()
            max_header_len = aptiv_udp_df['aptiv_udp_header_length'].max()

            aptiv_udp_header_bytes = pd.DataFrame(aptiv_udp_df.apply(
                self._helper_header_bytes,
                axis=1,
                col_name='udp_payload',
                row_end=max_header_len)
                .to_list()).values

            aptiv_udp_df['aptiv_udp_header_bytes'] = pd.Series(
                list(aptiv_udp_header_bytes))

            aptiv_udp_df['aptiv_udp_payload'] = aptiv_udp_df.apply(
                self._helper_extract_payload,
                axis=1,
                col_name_or_names='udp_payload',
                row_start=min_header_len+1-1,
                is_payload_edit=False,
            )

            # audp_payload=np.array(aptiv_udp_df['aptiv_udp_payload'].to_list()) #just to compare

            aptiv_udp_df['aptiv_udp_extra_bytes'] = \
                aptiv_udp_df['aptiv_udp_header_length'] - min_header_len

            aptiv_udp_df['aptiv_udp_payload_length'] = \
                aptiv_udp_df['udp_payload_length']-min_header_len

            indices_extra_bytes = aptiv_udp_df.query(
                'aptiv_udp_extra_bytes > 0').index

            aptiv_udp_df.loc[indices_extra_bytes,
                             'aptiv_udp_payload_length'] =  \
                aptiv_udp_df.loc[indices_extra_bytes,
                                 'aptiv_udp_payload_length'] - \
                aptiv_udp_df.loc[indices_extra_bytes,
                                 'aptiv_udp_extra_bytes']

            # FIXME: Here

            aptiv_udp_df['aptiv_udp_payload'] = \
                aptiv_udp_df.apply(self._helper_extract_payload,
                                   axis=1,
                                   col_name_or_names=['aptiv_udp_payload',
                                                      'aptiv_udp_extra_bytes',
                                                      'aptiv_udp_payload_length'
                                                      ],
                                   row_start=min_header_len+1-1,
                                   is_payload_edit=True,
                                   )

            aptiv_udp_df['aptiv_udp_payload_length'] = aptiv_udp_df.apply(
                lambda x: x['aptiv_udp_payload_length']
                - x['aptiv_udp_extra_bytes']
                if x['aptiv_udp_extra_bytes'] > 0
                else x['aptiv_udp_payload_length'],
                axis=1
            )

            if self._remove_unused_vars:

                udp_df = None  # RB 04082025_1130
                aptiv_udp_df['udp_payload'] = None  # RB 04082025_1130

            aptiv_udp_df['aptiv_udp_header_version'] = np.bitwise_and(
                aptiv_udp_df['aptiv_udp_version'],
                HEADER_VERSION_MASK)

            aptiv_udp_df['aptiv_udp_source_info'] = None
            aptiv_udp_df['aptiv_udp_source_tx_cnt'] = None
            aptiv_udp_df['aptiv_udp_source_tx_time'] = None
            aptiv_udp_df['aptiv_udp_stream_number'] = None
            aptiv_udp_df['aptiv_udp_stream_version'] = None
            aptiv_udp_df['aptiv_udp_stream_ref_index'] = None
            aptiv_udp_df['aptiv_udp_stream_tx_cnt'] = None
            aptiv_udp_df['aptiv_udp_stream_chunks'] = None
            aptiv_udp_df['aptiv_udp_stream_chunk_idx'] = None
            aptiv_udp_df['aptiv_udp_stream_chunk_len'] = None

            aptiv_udp_df['aptiv_udp_sensor_ID'] = None
            aptiv_udp_df['aptiv_udp_customer_ID'] = None

            aptiv_udp_df['aptiv_udp_detection_cnt'] = None
            aptiv_udp_df['aptiv_udp_mode'] = None
            aptiv_udp_df['aptiv_udp_sensor_status'] = None
            aptiv_udp_df['aptiv_udp_utc_time'] = None
            aptiv_udp_df['aptiv_udp_stream_length'] = None
            aptiv_udp_df['aptiv_udp_time_stamp'] = None

            aptiv_udp_df['aptiv_udp_reserved_src_01'] = None
            aptiv_udp_df['aptiv_udp_reserved_src_02'] = None
            aptiv_udp_df['aptiv_udp_platform_id'] = None
            aptiv_udp_df['stream_chunks_per_cycle'] = None

            # a123_headers_bool = (aptiv_udp_df['aptiv_udp_header_version'] <= 3
            #                      | aptiv_udp_df['aptiv_udp_header_version'] == 5)
            a123_headers_bool_index = aptiv_udp_df.query(
                'aptiv_udp_header_version <=3 ' +
                'or aptiv_udp_header_version == 5').index
            aptiv_udp_df['aptiv_udp_a123_headers_bool'] = False
            aptiv_udp_df.loc[a123_headers_bool_index,
                             'aptiv_udp_a123_headers_bool'] = True
            a123_headers_bool = aptiv_udp_df['aptiv_udp_a123_headers_bool'].values
            a123_headers_indices = aptiv_udp_df.query(
                'aptiv_udp_header_version <= 3 or aptiv_udp_header_version == 5 ').index

            dtype_aptiv_udp_headers = np.dtype(aptiv_udp_header_bytes.dtype)

            if bool(np.sum(a123_headers_bool)):

                req_columns_headers = ['aptiv_udp_stream_number',
                                       'aptiv_udp_stream_version',
                                       'aptiv_udp_stream_tx_cnt',
                                       'aptiv_udp_stream_chunks',
                                       'aptiv_udp_stream_chunk_idx',
                                       # 'aptiv_udp_sensor_ID',
                                       # 'aptiv_udp_customer_ID',
                                       # 'aptiv_udp_source_info',
                                       # 'aptiv_udp_sensor_status',
                                       # 'aptiv_udp_detection_cnt',
                                       # 'aptiv_udp_mode'
                                       ]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_source_info'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 9-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_source_tx_cnt'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a123_headers_indices, 3-1:4]),
                        index=a123_headers_indices
                )

                aptiv_udp_df['aptiv_udp_source_tx_cnt'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_source_tx_cnt',
                    row_start_end=[
                        0, 2],
                    conversion_type=np.uint16,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_source_tx_time'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a123_headers_indices, 5-1:8]),
                        index=a123_headers_indices
                )

                aptiv_udp_df['aptiv_udp_source_tx_time'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_source_tx_time',
                    row_start_end=[
                        0, 4],
                    conversion_type=np.uint32,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_number'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 20-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_version'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 21-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_ref_index'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a123_headers_indices, 13-1:16]),
                        index=a123_headers_indices
                )

                aptiv_udp_df['aptiv_udp_stream_ref_index'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_stream_ref_index',
                    row_start_end=[
                        0, 4],
                    conversion_type=np.uint32,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_tx_cnt'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 19-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_chunks'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 22-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_chunk_idx'] = \
                    aptiv_udp_header_bytes[a123_headers_indices, 23-1]

                aptiv_udp_df.loc[a123_headers_indices,
                                 'aptiv_udp_stream_chunk_len'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a123_headers_indices, 17-1:18]),
                        index=a123_headers_indices
                )

                aptiv_udp_df['aptiv_udp_stream_chunk_len'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_stream_chunk_len',
                    row_start_end=[
                        0, 2],
                    conversion_type=np.uint16,
                    is_big_endian=False,
                ).values

                big_endian_bool2 = np.logical_and(
                    a123_headers_bool, big_endian)

                aptiv_udp_df['aptiv_udp_big_endian2'] = big_endian_bool2

                big_endian_indices2 = aptiv_udp_df.query(
                    'aptiv_udp_big_endian2 == True').index

                aptiv_udp_df.loc[big_endian_indices2,
                                 'aptiv_udp_source_tx_cnt'] = \
                    aptiv_udp_df.loc[big_endian_indices2,
                                     'aptiv_udp_source_tx_cnt']\
                    .to_numpy(dtype=np.uint16).byteswap()

                aptiv_udp_df.loc[big_endian_indices2,
                                 'aptiv_udp_source_tx_time'] = \
                    aptiv_udp_df.loc[big_endian_indices2,
                                     'aptiv_udp_source_tx_time']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices2,
                                 'aptiv_udp_stream_ref_index'] = \
                    aptiv_udp_df.loc[big_endian_indices2,
                                     'aptiv_udp_stream_ref_index']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices2,
                                 'aptiv_udp_stream_chunk_len'] = \
                    aptiv_udp_df.loc[big_endian_indices2,
                                     'aptiv_udp_stream_chunk_len']\
                    .to_numpy(dtype=np.uint16).byteswap()

                aptiv_udp_df['aptiv_udp_sensor_ID'] = np.uint8(0)
                aptiv_udp_df['aptiv_udp_customer_ID'] = np.uint8(0)

                a1_headers_indices = aptiv_udp_df.query(
                    'aptiv_udp_header_version == 1').index

                aptiv_udp_df.loc[a1_headers_indices,
                                 'aptiv_udp_sensor_ID'] = \
                    aptiv_udp_header_bytes[a1_headers_indices,
                                           24-1]

                aptiv_udp_df.loc[a1_headers_indices,
                                 'aptiv_udp_reserved_src_01'] = \
                    aptiv_udp_header_bytes[a1_headers_indices,
                                           10-1]
                aptiv_udp_df.loc[a1_headers_indices,
                                 'aptiv_udp_reserved_src_02'] = \
                    aptiv_udp_header_bytes[a1_headers_indices,
                                           11-1]
                if (aptiv_udp_df['aptiv_udp_reserved_src_01'] > 1).any():

                    aptiv_udp_df['aptiv_udp_platform_id'] = aptiv_udp_df[
                        'aptiv_udp_reserved_src_01'].values

                if (aptiv_udp_df['aptiv_udp_reserved_src_02'] > 1).any():

                    aptiv_udp_df['aptiv_udp_sensor_ID'] = aptiv_udp_df[
                        'aptiv_udp_reserved_src_02'].values

                a2_headers_indices = aptiv_udp_df.query(
                    'aptiv_udp_header_version >= 2 ' +
                    'and aptiv_udp_header_version < 4').index

                aptiv_udp_df.loc[a2_headers_indices,
                                 'aptiv_udp_sensor_ID'] = \
                    aptiv_udp_header_bytes[a2_headers_indices,
                                           10-1]
                aptiv_udp_df.loc[a2_headers_indices,
                                 'aptiv_udp_customer_ID'] = \
                    aptiv_udp_header_bytes[a2_headers_indices,
                                           24-1]

                a5_headers_bool = aptiv_udp_df['aptiv_udp_header_version'] == 5

                if bool(np.sum(a5_headers_bool)):

                    a5_headers_indices = aptiv_udp_df.query(
                        'aptiv_udp_header_version == 5').index

                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_source_info_01'] = \
                        aptiv_udp_header_bytes[a5_headers_indices, 10-1]
                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_source_info_02'] = \
                        aptiv_udp_header_bytes[a5_headers_indices, 11-1]
                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_source_info_02'] = \
                        aptiv_udp_header_bytes[a5_headers_indices, 12-1]
                    aptiv_udp_df.loc[a5_headers_indices,
                                     'stream_chunks_per_cycle'] = \
                        aptiv_udp_header_bytes[a5_headers_indices,
                                               22-1]
                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_sensor_ID'] = \
                        aptiv_udp_header_bytes[a5_headers_indices,
                                               28-1]

                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_stream_chunks'] = \
                        pd.Series(
                            list(
                                aptiv_udp_header_bytes[a5_headers_indices, 23-1:24]),
                            index=a5_headers_indices
                    )

                    aptiv_udp_df['aptiv_udp_stream_chunks'] = aptiv_udp_df.apply(
                        self._helper_typecast,
                        axis=1,
                        col_name='aptiv_udp_stream_chunks',
                        row_start_end=[
                            0, 2],
                        conversion_type=np.uint16,
                        is_big_endian=False,
                    ).values

                    aptiv_udp_df.loc[a5_headers_indices,
                                     'aptiv_udp_stream_chunk_idx'] = \
                        pd.Series(
                            list(
                                aptiv_udp_header_bytes[a5_headers_indices, 23-1:24]),
                            index=a5_headers_indices
                    )

                    aptiv_udp_df['aptiv_udp_stream_chunk_idx'] = aptiv_udp_df.apply(
                        self._helper_typecast,
                        axis=1,
                        col_name='aptiv_udp_stream_chunk_idx',
                        row_start_end=[
                            0, 2],
                        conversion_type=np.uint16,
                        is_big_endian=False,
                    ).values

                    big_endian_bool5 = np.logical_and(
                        a5_headers_bool, big_endian)

                    aptiv_udp_df['aptiv_udp_big_endian5'] = big_endian_bool5

                    big_endian_indices5 = aptiv_udp_df.query(
                        'aptiv_udp_big_endian5 == True').index

                    aptiv_udp_df.loc[big_endian_indices5,
                                     'aptiv_udp_stream_chunks'] = \
                        aptiv_udp_df.loc[big_endian_indices5,
                                         'aptiv_udp_stream_chunks']\
                        .to_numpy(dtype=np.uint16).byteswap()

                    aptiv_udp_df.loc[big_endian_indices5,
                                     'aptiv_udp_stream_chunk_idx'] = \
                        aptiv_udp_df.loc[big_endian_indices5,
                                         'aptiv_udp_stream_chunk_idx']\
                        .to_numpy(dtype=np.uint16).byteswap()

                # for req_columns_header in req_columns_headers:
                #     print(f'\n @@@@@@@@@@@@@@@@@@@@ {req_columns_header}')
                #     aptiv_udp_df[req_columns_header] = aptiv_udp_df[
                #         req_columns_header].astype(
                #         dtype_aptiv_udp_headers)

                # aptiv_udp_df[req_columns_headers] = aptiv_udp_df[
                #     req_columns_headers].astype(
                #     dtype_aptiv_udp_headers)

            a4_headers_indices = aptiv_udp_df.query(
                'aptiv_udp_header_version == 4').index
            a4_headers_bool = aptiv_udp_df['aptiv_udp_header_version'] == 4
            aptiv_udp_df['aptiv_udp_a4_headers_bool'] = a4_headers_bool

            if bool(np.sum(a4_headers_bool)):

                req_columns_headers = ['aptiv_udp_stream_number',
                                       'aptiv_udp_stream_version',
                                       'aptiv_udp_stream_tx_cnt',
                                       'aptiv_udp_stream_chunks',
                                       'aptiv_udp_stream_chunk_idx',
                                       'aptiv_udp_sensor_ID',
                                       'aptiv_udp_customer_ID',
                                       'aptiv_udp_source_info',
                                       'aptiv_udp_sensor_status',
                                       'aptiv_udp_detection_cnt',
                                       'aptiv_udp_mode']

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_source_info'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 3-1]
                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_sensor_ID'] = \
                    aptiv_udp_header_bytes[a4_headers_indices,
                                           5-1]
                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_customer_ID'] = \
                    aptiv_udp_header_bytes[a4_headers_indices,
                                           4-1]
                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_sensor_status'] = \
                    aptiv_udp_header_bytes[a4_headers_indices,
                                           6-1]
                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_detection_cnt'] = \
                    aptiv_udp_header_bytes[a4_headers_indices,
                                           7-1]
                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_mode'] = \
                    aptiv_udp_header_bytes[a4_headers_indices,
                                           8-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_ref_index'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 9-1:12]),
                        index=a4_headers_indices
                )
                aptiv_udp_df['aptiv_udp_stream_ref_index'] = \
                    aptiv_udp_df.apply(self._helper_typecast,
                                       axis=1,
                                       col_name='aptiv_udp_stream_ref_index',
                                       row_start_end=[
                                           0, 4],
                                       conversion_type=np.uint32,
                                       is_big_endian=False,
                                       ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_utc_time'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 13-1:16]),
                        index=a4_headers_indices
                )
                aptiv_udp_df['aptiv_udp_utc_time'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_utc_time',
                    row_start_end=[
                        0, 4],
                    conversion_type=np.uint32,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_time_stamp'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 17-1:20]),
                        index=a4_headers_indices
                )
                aptiv_udp_df['aptiv_udp_time_stamp'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_time_stamp',
                    row_start_end=[
                        0, 4],
                    conversion_type=np.uint32,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_length'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 21-1:22]),
                        index=a4_headers_indices
                )
                aptiv_udp_df['aptiv_udp_stream_length'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_stream_length',
                    row_start_end=[
                        0, 2],
                    conversion_type=np.uint16,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_number'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 23-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_tx_cnt'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 24-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_version'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 25-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_chunks'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 26-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_chunk_idx'] = \
                    aptiv_udp_header_bytes[a4_headers_indices, 27-1]

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_stream_chunk_len'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 29-1:30]),
                        index=a4_headers_indices
                )

                aptiv_udp_df['aptiv_udp_stream_chunk_len'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_stream_chunk_len',
                    row_start_end=[
                        0, 2],
                    conversion_type=np.uint16,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_source_tx_cnt'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 31-1:32]),
                        index=a4_headers_indices
                )

                aptiv_udp_df['aptiv_udp_source_tx_cnt'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_source_tx_cnt',
                    row_start_end=[
                        0, 2],
                    conversion_type=np.uint16,
                    is_big_endian=False,
                ).values

                aptiv_udp_df.loc[a4_headers_indices,
                                 'aptiv_udp_source_tx_time'] = \
                    pd.Series(
                        list(
                            aptiv_udp_header_bytes[a4_headers_indices, 33-1:36]),
                        index=a4_headers_indices
                )
                aptiv_udp_df['aptiv_udp_source_tx_time'] = aptiv_udp_df.apply(
                    self._helper_typecast,
                    axis=1,
                    col_name='aptiv_udp_source_tx_time',
                    row_start_end=[
                        0, 4],
                    conversion_type=np.uint32,
                    is_big_endian=False,
                ).values

                big_endian_bool3 = np.logical_and(a4_headers_bool, big_endian)

                aptiv_udp_df['aptiv_udp_big_endian3'] = big_endian_bool3

                big_endian_indices3 = aptiv_udp_df.query(
                    'aptiv_udp_big_endian3 == True').index

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_stream_ref_index'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_stream_ref_index']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_utc_time'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_utc_time']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_time_stamp'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_time_stamp']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_stream_length'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_stream_length']\
                    .to_numpy(dtype=np.uint16).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_source_tx_cnt'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_source_tx_cnt']\
                    .to_numpy(dtype=np.uint16).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_source_tx_time'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_source_tx_time']\
                    .to_numpy(dtype=np.uint32).byteswap()

                aptiv_udp_df.loc[big_endian_indices3,
                                 'aptiv_udp_stream_chunk_len'] = \
                    aptiv_udp_df.loc[big_endian_indices3,
                                     'aptiv_udp_stream_chunk_len']\
                    .to_numpy(dtype=np.uint16).byteswap()

                # for req_columns_header in req_columns_headers:

                #     aptiv_udp_df[req_columns_header] = aptiv_udp_df[
                #         req_columns_header].astype(
                #         dtype_aptiv_udp_headers)

                # aptiv_udp_df[req_columns_headers] = aptiv_udp_df[
                #     req_columns_headers].astype(
                #     dtype_aptiv_udp_headers)
            other_req_casting_cols = ['aptiv_udp_source_tx_cnt',
                                      'aptiv_udp_source_tx_time',
                                      'aptiv_udp_stream_ref_index',
                                      'aptiv_udp_stream_chunk_len',
                                      'aptiv_udp_utc_time',
                                      'aptiv_udp_stream_length',
                                      'aptiv_udp_time_stamp',
                                      ]

            aptiv_udp_df[other_req_casting_cols] = aptiv_udp_df[
                other_req_casting_cols].apply(pd.to_numeric,
                                              errors='ignore',
                                              downcast='unsigned')

            aptiv_udp_df[req_columns_headers] = aptiv_udp_df[
                req_columns_headers].apply(pd.to_numeric,
                                           errors='ignore',
                                           downcast='unsigned')

            window_length = self.window_length_to_search

            non_time_cols = list(
                set(aptiv_udp_df.columns).difference(set(non_aptiv_udp_cols)))

            req_non_time_cols = aptiv_udp_df[non_time_cols].map(lambda x:
                                                                isinstance(x, Iterable)).all()
            req_non_time_cols = req_non_time_cols.index[~req_non_time_cols].tolist(
            )

            duplicated_bool = aptiv_udp_df.duplicated(
                subset=req_non_time_cols,
                keep=False
            )
            duplicated_indices = aptiv_udp_df.index[duplicated_bool]
            grplist = [list(group) for group in
                       mit.consecutive_groups(duplicated_indices)]

            grp_list_1 = [item for item in grplist
                          if len(item) <= window_length]

            grp_list_2 = [self.divide_chunks(item, window_length)
                          for item in grplist if len(item) > window_length]

            grp_list_2 = [item for group in grp_list_2 for item in group]

            not_req_indices = np.sort(np.unique([item[-1]
                                                 for item in
                                                 grp_list_1 + grp_list_2]))

            req_indices = np.sort(np.unique([0] +
                                            list(set(aptiv_udp_df.index)
                                                 .difference(set(not_req_indices))
                                                 )
                                            ))
            aptiv_udp_df_req = aptiv_udp_df.loc[req_indices, :]

            source_sensor_stream_grouped_df = aptiv_udp_df_req.groupby(
                ['aptiv_udp_stream_number',
                 'aptiv_udp_source_info',
                 'aptiv_udp_sensor_ID',
                 ], group_keys=True)

            group_index_dict = source_sensor_stream_grouped_df.groups
            group_values = list(group_index_dict.keys())

            me_SW_version_major = self.me_SW_version_major
            me_SW_version_minor = self.me_SW_version_minor

            mudp_output_dict = {}

            if self._log_type in ['bus', 'p01', ]:

                stream_mapping = self._mudp_stream_mapping
            elif self._log_type == ['deb',  'b05', ]:
                stream_mapping = self._srr_stream_mapping

            group_values.sort(key=lambda x: x[1])

            self.group_df_list = []

            #   # [(144, 23, 16), (145, 23, 16)]:  #(33, 23, 16)
            for keys in group_values:  # [(81, 23, 0)]:  #

                print(f'group keys : {keys}\t',)

                self.stream_source_sid = keys

                if keys[0] in stream_mapping.keys():

                    if (keys[0] in mudp_output_dict.keys()
                            or stream_mapping[keys[0]] in mudp_output_dict.keys()
                            ):
                        print('Stream already existing, refer source\t',
                              f"{'stream_' + str(keys[0]) + 'source_' + str(keys[1])}")
                        map_key = 'source_' + str(keys[1]) \
                            + '_sid_' + str(keys[2]) + \
                            '_stream_' + str(keys[0])

                    else:
                        # print(
                        #     f"{'stream_' + str(keys[0]) + 'source_' + str(keys[1])}")
                        map_key = stream_mapping[keys[0]]

                else:

                    map_key = 'source_' + str(keys[1]) \
                        + '_sid_' + str(keys[2]) + '_stream_' + str(keys[0])

                self.stream_check_dict[
                    f"busID_{self.busID}" +
                    f"_source_{str(keys[1])}" +
                    f"_stream_{keys[0]} "
                ] = True

                group_df = source_sensor_stream_grouped_df.get_group(keys)

                unique_stream_chunks_group = group_df['aptiv_udp_stream_chunks'].unique(
                )

                if len(unique_stream_chunks_group) > 1:
                    warnings.warn(f"group of [str, src, sns] = {keys} has " +
                                  'multiple number of total chunks and will not be processed.',
                                  DeprecationWarning)
                    out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                        ' has multiple number of total chunks'
                    mudp_output_dict[map_key] = out_val_map_key

                    self.debug_str = self.debug_str + out_val_map_key + '\n'

                    self.stream_check_dict[
                        f"busID_{self.busID}" +
                        f"_source_{str(keys[1])}" +
                        f"_stream_{keys[0]} "
                    ] = 'multiple_chunk_lengths'
                    continue

                stream_chunks_group = unique_stream_chunks_group.item()

                if stream_chunks_group > 1:

                    zero_chunks_group = np.array(group_df.reset_index(drop=True).query(
                        'aptiv_udp_stream_chunk_idx ==0').index)

                    zero_chunks_group2 = np.array(group_df.query(
                        'aptiv_udp_stream_chunk_idx ==0').index)

                    if len(zero_chunks_group) == 0:
                        warnings.warn(f"group of [str, src, sns] = {keys} " +
                                      'does not have any chunks with ' +
                                      'aptiv_udp_stream_chunk_idx == 0 and ' +
                                      'will not be processed',
                                      DeprecationWarning)

                        out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                            ' does not have any chunks with ' + \
                            'aptiv_udp_stream_chunk_idx == 0 '
                        mudp_output_dict[map_key] = out_val_map_key

                        self.debug_str = self.debug_str + out_val_map_key + '\n'

                        self.stream_check_dict[
                            f"busID_{self.busID}" +
                            f"_source_{str(keys[1])}" +
                            f"_stream_{keys[0]} "
                        ] = 'missing_start_chunk'

                        continue

                    # to_drop_indices_0_chunk = np.argwhere((zero_chunks_group
                    #                                        + stream_chunks_group - 1)
                    #                                       > len(group_df[
                    #                                           'aptiv_udp_stream_chunk_idx'])
                    #                                       ).flatten()

                    to_keep_indices_0_chunk = np.argwhere((zero_chunks_group
                                                           + stream_chunks_group - 1)
                                                          <= len(group_df[
                                                              'aptiv_udp_stream_chunk_idx'])
                                                          ).flatten()

                    zero_chunks_group2 = zero_chunks_group2[
                        to_keep_indices_0_chunk]

                    chunk_idx_df = pd.DataFrame(group_df['aptiv_udp_stream_chunk_idx']
                                                )

                    # val_index_list = [[np.array(list(group1)),
                    #                    np.array(list(group2))]
                    #                   for group1, group2 in
                    #                   zip(mit.consecutive_groups(chunk_idx_df[
                    #                       'aptiv_udp_stream_chunk_idx']),
                    #                   mit.consecutive_groups(chunk_idx_df.index))
                    #                   ]

                    val_index_list = chunk_idx_df[
                        'aptiv_udp_stream_chunk_idx'].groupby(
                        chunk_idx_df[
                            'aptiv_udp_stream_chunk_idx']
                        .diff().ne(1).cumsum()
                    ).apply(lambda x: [np.array(x),
                                       np.array(x.index)]
                            if len(x) >= 2
                            else None).tolist()
                    val_index_list = [item for item in val_index_list
                                      if item is not None]

                    # to_keep_indices_0_chunk_2_lists = [val[1]
                    #                                    for val in val_index_list
                    #                                    if (val[0][-1]+1 == len(val[0]))
                    #                                    and (val[0][-1]+1 == len(val[1]))
                    #                                    ]
                    val_index_list_2 = [val
                                        for val in val_index_list
                                        if (val[0][-1]+1 == len(val[0]))
                                        and (val[0][-1]+1 == len(val[1])
                                             == stream_chunks_group)
                                        ]

                    # to_keep_indices_0_chunk_2 = list(chain.from_iterable(
                    #     to_keep_indices_0_chunk_2_lists))

                    index_list = [val[1] for val in val_index_list_2]
                    zero_chunks_group2 = [idx[0] for idx in index_list]

                    to_keep_indices_group_df = list(chain.from_iterable(
                        [val[1]
                         for val in val_index_list_2
                         # if len(np.unique(val[0][::-1]
                         #                  + val[1])) == 1
                         ]
                    ))

                    group_df = group_df.copy().loc[to_keep_indices_group_df, :]

                    unique_chunk_idx = \
                        group_df['aptiv_udp_stream_chunk_idx'].unique().astype(
                            int)

                    if (len(unique_chunk_idx) == 0
                            or len(unique_chunk_idx) < stream_chunks_group):

                        warnings.warn(f"group of [str, src, sns] = {keys} " +
                                      'is missing one or more chunks ' +
                                      'and will not be processed ',
                                      DeprecationWarning)
                        out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                            ' is missing one or more chunks'
                        mudp_output_dict[map_key] = out_val_map_key

                        self.debug_str = self.debug_str + out_val_map_key + '\n'
                        self.stream_check_dict[
                            f"busID_{self.busID}" +
                            f"_source_{str(keys[1])}" +
                            f"_stream_{keys[0]} "
                        ] = 'missing_chunks'
                        continue

                    req_indices = [list(
                        group_df
                        .query(f'aptiv_udp_stream_chunk_idx == {chunk_idx}')
                        .index)[0]
                        for chunk_idx in unique_chunk_idx
                    ]

                    stream_data_length = np.sum(np.array(
                        [group_df.loc[idx, 'aptiv_udp_stream_chunk_len']
                         for idx in req_indices], dtype=np.uint16))

                    group_df['aptiv_udp_stream_data_length'] = stream_data_length

                    req_indices_combined_payload = zero_chunks_group2

                else:

                    group_df['aptiv_udp_stream_data_length'] = \
                        group_df.copy(deep=True)['aptiv_udp_stream_chunk_len'].astype(
                            np.uint16).values

                    req_indices_combined_payload = group_df.index

                if not group_df.empty:

                    stream_def_source = 'src{:>03d}'.format(keys[1])
                    stream_def_sensor_id = 'sid{:>03d}'.format(keys[2])

                    if bool(np.sum(group_df['aptiv_udp_source_info'] == 100)):

                        if (me_SW_version_major < 0) or (me_SW_version_minor < 0):

                            lane_advanced_index = \
                                list(group_df.query('aptiv_udp_source_info == 100 ' +
                                                    'and aptiv_udp_stream_number == 108').index)
                            if not lane_advanced_index:
                                warning = 'Mobileye Lane_Advanced stream 108 not found ' +\
                                    'cannot read ME SW version.'
                                warnings.warn(warning,
                                              DeprecationWarning)
                                mudp_output_dict[map_key] = warning
                                self.debug_str = self.debug_str + warning + '\n'
                                continue
                            else:
                                me_SW_version_major = \
                                    aptiv_udp_df_req.loc[lane_advanced_index[0],
                                                         'aptiv_udp_payload'][4-1]
                                me_SW_version_minor = \
                                    aptiv_udp_df_req.loc[lane_advanced_index[0],
                                                         'aptiv_udp_payload'][4-1]
                        else:

                            stream_def_pre = 'mecp'
                            stream_def_str = 'all'
                            stream_def_ver = 'sw{:>02d}p{}'.format(me_SW_version_major,
                                                                   me_SW_version_minor)
                            stream_def_extension = '.dbc'
                    else:

                        stream_def_pre = 'strdef'
                        stream_def_str = 'str{:>03d}'.format(keys[0])
                        stream_def_ver = 'ver{:>03d}'.format(
                            np.array(group_df['aptiv_udp_stream_version'])[0])
                        stream_def_extension = '.txt'

                    # group_df['aptiv_udp_payload_trimmed'] = group_df.apply(
                    #     self._helper_combine_payloads,
                    #     axis=1,

                    # )

                    req_db_path = self._db_file_helper(stream_def_pre,
                                                       stream_def_source,
                                                       stream_def_sensor_id,
                                                       stream_def_str,
                                                       stream_def_ver,
                                                       stream_def_extension

                                                       )
                    if ((req_db_path is None)
                            # or
                            # (not os.path.isfile(req_db_path[:-4] + '.pickle'))
                            ):
                        warning = f'Stream def source : {stream_def_source},' + \
                            f'stream : {stream_def_str},' + \
                            f' version : {stream_def_ver}, not found'
                        warnings.warn(warning,
                                      DeprecationWarning)
                        mudp_output_dict[map_key] = warning

                        self.debug_str = self.debug_str + warning + '\n'
                        self.stream_check_dict[
                            f"busID_{self.busID}" +
                            f"_source_{str(keys[1])}" +
                            f"_stream_{keys[0]} "
                        ] = 'stream_def_file_missing'
                        continue

                    req_db_path_orig = req_db_path
                    req_db_path = req_db_path[:-4] + '.pickle'

                    # print('&&&&&&&&&&&&&&&&&&&&&&',
                    #       f'{req_db_path}')

                    if not os.path.isfile(req_db_path):

                        print('\n&&&&& Creating streamdef pickles for ',
                              f'source : {stream_def_source}, ',
                              f'stream : {stream_def_str}, ',
                              f'version : {stream_def_ver}, &&&&\n',
                              f'\n Path to pickle file is {req_db_path} \n '
                              )

                        error_stream_defs = _stream_def_pickle(
                            self.stream_def_dir_path,
                            os.path.join(self.stream_def_dir_path,
                                         'datatypes_mapping.pickle'),
                            regenerate=False,
                            single_file_path=req_db_path_orig,
                        )

                        if (len(error_stream_defs) > 0
                            and
                            not os.path.isfile(req_db_path)
                            ):

                            warnings.warn(f"group of [str, src, sns] = {keys} " +
                                          'has a problem with stream def file ' +
                                          'and will not be processed ',
                                          DeprecationWarning)
                            out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                                ' is having error with stream def file'
                            mudp_output_dict[map_key] = out_val_map_key

                            self.debug_str = self.debug_str + out_val_map_key + '\n'
                            self.stream_check_dict[
                                f"busID_{self.busID}" +
                                f"_source_{str(keys[1])}" +
                                f"_stream_{keys[0]} "
                            ] = 'stream_def_file_error'
                            continue

                    group_df['aptiv_udp_payload_trimmed'] = group_df.apply(
                        self._helper_combine_payloads,
                        axis=1,

                    )

                    if stream_chunks_group > 1:

                        index_arr = np.array(index_list)

                        # self.group_df_list.append([index_list,
                        #                            group_df,
                        #                            req_indices_combined_payload])

                        req_group_df = self.combine_chunk_payloads(
                            index_list,
                            group_df,
                            req_indices_combined_payload)
                    else:

                        req_group_df = group_df.copy(deep=True)
                        req_group_df.loc[:,
                                         'aptiv_udp_combined_payload'] = \
                            group_df['aptiv_udp_payload_trimmed'].values

                        index_arr = None

                    stream_data_length_unique = \
                        req_group_df['aptiv_udp_stream_data_length'].unique()

                    if len(stream_data_length_unique) > 1:

                        warnings.warn(f"group of [str, src, sns] = {keys} " +
                                      'is having multiple length payloads ' +
                                      'and will not be processed ',
                                      DeprecationWarning)
                        out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                            ' is having multiple length payloads'
                        mudp_output_dict[map_key] = out_val_map_key

                        self.debug_str = self.debug_str + out_val_map_key + '\n'
                        self.stream_check_dict[
                            f"busID_{self.busID}" +
                            f"_source_{str(keys[1])}" +
                            f"_stream_{keys[0]} "
                        ] = 'multiple_payload_lengths'
                        continue

                    if len(stream_data_length_unique) == 1:

                        stream_data_length_unique = \
                            req_group_df['aptiv_udp_stream_data_length'].unique()[
                                0]

                        with open(
                                os.path.join(req_db_path), 'rb') as handle:
                            str_def_dict = pickle.load(handle)

                        if stream_data_length_unique != \
                                str_def_dict['expected_stream_length']:

                            warnings.warn(f"group of [str, src, sns] = {keys} " +
                                          'is stream length of combined payloads ' +
                                          f'{stream_data_length_unique} not matching ' +
                                          'with what the stream def file expects ' +
                                          f"{str_def_dict['expected_stream_length']}" +
                                          ' and will not be processed ',
                                          DeprecationWarning)
                            out_val_map_key = f'Stream {keys[0]}, source {keys[1]}' + \
                                f' payload stream length {stream_data_length_unique} ' +\
                                'mismatch with stream def files expectation ' +\
                                f"{str_def_dict['expected_stream_length']}"

                            mudp_output_dict[map_key] = out_val_map_key

                            self.debug_str = self.debug_str + out_val_map_key + '\n'
                            self.stream_check_dict[
                                f"busID_{self.busID}" +
                                f"_source_{str(keys[1])}" +
                                f"_stream_{keys[0]} "
                            ] = 'stream_len_mismatch_with_stream_def_file'
                            continue

                    if is_decoding:
                        if self.is_updated_udp_parser:

                            _parse_aptiv_udp_method = self._parse_aptiv_udp3
                        else:

                            _parse_aptiv_udp_method = self._parse_aptiv_udp

                        empty_dict_tree = _parse_aptiv_udp_method(req_group_df,
                                                                  group_df,
                                                                  index_arr,
                                                                  stream_chunks_group,
                                                                  req_db_path)

                        mudp_output_dict[map_key] = empty_dict_tree

        return mudp_output_dict

    def _parse_aptiv_udp2(self,
                          req_group_df,
                          group_df,
                          index_arr,
                          stream_chunks_group,
                          req_db_path):

        with open(
                os.path.join(req_db_path), 'rb') as handle:
            str_def_dict = pickle.load(handle)

        empty_dict = {key: list()
                      for key in str_def_dict['req_col']
                      }

        # empty_dict = {key: np.array([], dtype=dtype_)
        #               if not (np.dtype((np.void, size_)) == dtype_)
        #               else np.array([], dtype=np.uint8)
        #               for key, dtype_, size_ in zip(str_def_dict['req_col'],
        #                                             str_def_dict['map_type'],
        #                                             str_def_dict['map_size'])
        #               }

        # aptiv_udp_payload_combined = np.array(
        #     req_group_df['aptiv_udp_combined_payload'].to_list())

        supp_dict = pd.DataFrame(list(zip(str_def_dict['req_col'],
                                          str_def_dict['type_shape'],
                                          str_def_dict['len_map_size_list'])),
                                 columns=['req_col',
                                          'type_shape',
                                          'len_map_size_list']
                                 )

        supp_df_2d = supp_dict.query('len_map_size_list == 1')
        supp_2d_help = pd.DataFrame({'req_col':
                                     supp_df_2d['req_col'].unique()})
        supp_2d_help['type_shape'] = [
            list(set(supp_df_2d['type_shape'].loc[supp_df_2d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_2d_help.iterrows()]

        supp_df_3d = supp_dict.query('len_map_size_list == 2')

        supp_3d_help = pd.DataFrame({'req_col':
                                     supp_df_3d['req_col'].unique()})
        supp_3d_help['type_shape'] = [
            list(set(supp_df_3d['type_shape'].loc[supp_df_3d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_3d_help.iterrows()]

        header_dict = {}
        header_dict = {key: list()
                       for key in self.mudp_header_mapping.keys()}

        # group_df['timestamps_orig'] = group_df['timestamps']

        # group_df['timestamps'] = group_df[
        #     'timestamps'].diff(
        #         periods=1).ffill().fillna(0).cumsum()
        parsed_header_dict_list = req_group_df[
            self.mudp_header_mapping.values()].apply(
            self._helper_header_parser,
            # engine='numba',
            axis=1,

            **{'stream_chunks_group': stream_chunks_group,
               'index_arr': index_arr,
               'header_mapping': self.mudp_header_mapping,
               'group_df': group_df,
               'str_def_dict': str_def_dict,
               'log_start_time': self.log_start_time, }
        ).tolist()
# aptiv_udp_combined_payload, aptiv_udp_big_endian,
        parsed_payload_list = req_group_df[
            ['aptiv_udp_combined_payload', 'aptiv_udp_big_endian']].apply(
            self._helper_payload_parser,
            # engine='numba',
            axis=1,

            **{'stream_chunks_group': stream_chunks_group,
               'index_arr': index_arr,
               'header_mapping': self.mudp_header_mapping,
               'group_df': group_df,
               'str_def_dict': str_def_dict,
               'log_start_time': self.log_start_time, }
        ).tolist()

        empty_dict = self.merge_list_of_dicts(parsed_payload_list)
        header_dict = self.merge_list_of_dicts(parsed_header_dict_list)
        for col_name, dtype_, size_ in zip(str_def_dict['req_col'],
                                           str_def_dict['map_type'],
                                           str_def_dict['map_size']):

            # if 'padding' in col_name.lower():
            #     continue

            if not (np.dtype((np.void, size_)) == dtype_):
                if isinstance(dtype_, Fxp):

                    empty_dict[col_name] = np.array(empty_dict[col_name])
                    # continue

                else:
                    empty_dict[col_name] = np.array(empty_dict[col_name],
                                                    dtype=dtype_)
            else:
                empty_dict[col_name] = np.array(empty_dict[col_name],
                                                dtype=np.uint32)

        for idx_supp_3d in supp_3d_help.index:

            col_name = supp_3d_help.loc[idx_supp_3d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1,
                                     *supp_3d_help.loc[idx_supp_3d,
                                                       'type_shape'][0][::-1])),
                           order='C')

        for idx_supp_2d in supp_2d_help.index:
            col_name = supp_2d_help.loc[idx_supp_2d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1, supp_2d_help.loc[idx_supp_2d,
                                                           'type_shape'][0][1])),
                           order='C')

        header_dict = {key: np.array(val, )
                       for key, val in header_dict.items()}
        header_dict = {key: val
                       if (val.flatten() != None).all()
                       else 'Not a value'
                       for key, val in header_dict.items()
                       }

        for key, val in header_dict.items():
            if not isinstance(val, str) and len(np.unique(val)) > 1:
                header_dict[key] = val
            elif isinstance(val, str):
                header_dict[key] = val
            else:
                header_dict[key] = np.unique(val).item()

        empty_dict = {**header_dict, **empty_dict}

        empty_dict_tree = self.nest_dict(empty_dict)

        return empty_dict_tree

    def _parse_aptiv_udp3(self,
                          req_group_df,
                          group_df,
                          index_arr,
                          stream_chunks_group,
                          req_db_path):

        with open(
                os.path.join(req_db_path), 'rb') as handle:
            str_def_dict = pickle.load(handle)

        str_def_dict['req_col'] = np.char.strip(str_def_dict['req_col'])

        empty_dict = {key: list()
                      for key in str_def_dict['req_col']
                      }

        # empty_dict = {key: np.array([], dtype=dtype_)
        #               if not (np.dtype((np.void, size_)) == dtype_)
        #               else np.array([], dtype=np.uint8)
        #               for key, dtype_, size_ in zip(str_def_dict['req_col'],
        #                                             str_def_dict['map_type'],
        #                                             str_def_dict['map_size'])
        #               }

        # aptiv_udp_payload_combined = np.array(
        #     req_group_df['aptiv_udp_combined_payload'].to_list())

        supp_dict = pd.DataFrame(list(zip(str_def_dict['req_col'],
                                          str_def_dict['type_shape'],
                                          str_def_dict['len_map_size_list'])),
                                 columns=['req_col',
                                          'type_shape',
                                          'len_map_size_list']
                                 )

        supp_df_2d = supp_dict.query('len_map_size_list == 1')
        supp_2d_help = pd.DataFrame({'req_col':
                                     supp_df_2d['req_col'].unique()})
        supp_2d_help['type_shape'] = [
            list(set(supp_df_2d['type_shape'].loc[supp_df_2d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_2d_help.iterrows()]

        supp_df_3d = supp_dict.query('len_map_size_list == 2')

        supp_3d_help = pd.DataFrame({'req_col':
                                     supp_df_3d['req_col'].unique()})
        supp_3d_help['type_shape'] = [
            list(set(supp_df_3d['type_shape'].loc[supp_df_3d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_3d_help.iterrows()]

        header_dict = {}
        header_dict = {key: list()
                       for key in self.mudp_header_mapping.keys()}
        header_dtype_dict = {key: ''
                             for key in self.mudp_header_mapping.keys()}

        # group_df['timestamps_orig'] = group_df['timestamps']

        # group_df['timestamps'] = group_df[
        #     'timestamps'].diff(
        #         periods=1).ffill().fillna(0).cumsum()
        empty_dict_list = []
        for index_req_group, payload_trimmed_0, is_big_endian in \
            zip(req_group_df.index,
                req_group_df['aptiv_udp_combined_payload'],
                req_group_df['aptiv_udp_big_endian']):

            if is_big_endian:
                payload_trimmed_0 = np.flipud(payload_trimmed_0)

            if stream_chunks_group > 1:
                idx_req_iter = np.argwhere(index_arr[:, 0] ==
                                           index_req_group)[0]
                idx_vals_iter = index_arr[idx_req_iter].flatten()
            else:
                idx_vals_iter = index_req_group

            for key, key_rev in self.mudp_header_mapping.items():

                val_iter = group_df.loc[idx_vals_iter, key_rev]

                header_val_dtype = np.dtype(group_df[key_rev].dtypes)

                if 'header.time' == key:

                    # print(f'Before : {val_iter}, {idx_vals_iter}, {key_rev}',
                    #       f'start_cTime : {self.log_start_time}')

                    # val_iter = val_iter\
                    #     # *self.time_unit_conversion_factor \
                    # + self.log_start_time

                    # if isinstance(val_iter, Iterable):
                    #     val_iter = val_iter.apply(
                    #         Decimal) + self.log_start_time
                    # else:
                    #     val_iter = Decimal(val_iter) + self.log_start_time

                    val_iter = val_iter + self.log_start_time
                    # print(f'After : {val_iter}, ')

                header_dict[key].append(val_iter)
                header_dtype_dict[key] = header_val_dtype

            byte_counter = 0

            map_type_array = np.array(str_def_dict['map_type'])
            map_size_array = np.array(str_def_dict['map_size'],
                                      dtype=np.uint32)
            req_col_array = np.array(str_def_dict['req_col'])

            is_fxp, idx_fxp = _check_for_obj_instance(Fxp,
                                                      str_def_dict['map_type'])

            if is_fxp:

                map_type_array_orig = copy.deepcopy(map_type_array)
                map_type_array[idx_fxp] = [
                    np.dtype((np.void, map_size_iter))
                    for map_size_iter in map_size_array[idx_fxp]
                ]

                fxp_path_cols = [req_col_array[item]
                                 for item in idx_fxp]

            duplicated_cols_dict = duplicate_keys_to_tuples_dict(req_col_array)
            dtypes_arr2 = [('', dtype) for dtype in map_type_array]
            data_out = np.frombuffer(payload_trimmed_0, dtypes_arr2)[0]
            dtypes_arr2 = None
            # series_data_out = pd.Series([(col, data)
            #                              for col, data in
            #                              zip(req_col_array, data_out)])
            # new_col_list = ['signal_path', 'value',]
            # df_data_out_iter = pd.DataFrame.from_records(
            #     series_data_out, columns=new_col_list)

            # empty_dict = {key :
            #                   df_data_out_iter['value'][val].to_list()
            #                   for key, val in duplicated_cols_dict.items()
            #                   }

            # empty_dict_iter = {
            #     key: [data_out[item] for item in val]
            #     for key, val in duplicated_cols_dict.items()
            # }

            for key, val in zip(req_col_array, data_out):

                empty_dict[key].append(val)

            dtypes_arr2 = None

            if is_fxp:

                for fxp_col in fxp_path_cols:
                    empty_dict[fxp_col] = [np.uint8(0)]*len(
                        empty_dict[fxp_col])
                    for idx_enum_fxp, idx_fxp_iter in enumerate(
                            duplicated_cols_dict[fxp_col]):

                        byte_counter = np.cumsum(map_size_array[:idx_fxp_iter])
                        start_idx = byte_counter
                        end_idx = byte_counter + map_size_array[idx_fxp_iter]

                        x1 = map_type_array_orig[idx_fxp_iter](
                            payload_trimmed_0[start_idx: end_idx])

                        empty_dict[fxp_col][idx_enum_fxp] = x1

            # empty_dict_list.append(empty_dict_iter)

            # for map_size_, type_, col_name in zip(
            #         str_def_dict['map_size'],
            #         str_def_dict['map_type'],
            #         str_def_dict['req_col'],
            # ):

            #     start_idx = byte_counter
            #     end_idx = byte_counter + map_size_

            #     if (('PADDING' in col_name) and
            #             (np.dtype((np.void, map_size_)) == type_)):
            #         data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
            #             type_).item()
            #         data_iter = int.from_bytes(data_iter)

            #     else:

            #         if isinstance(type_, Fxp):
            #             # type_fxp = copy.deepcopy(type_)
            #             x1 = payload_trimmed_0[start_idx: end_idx].copy().view(
            #                 np.uint8)
            #             # binary_val = '0b'+''.join(map(str,
            #             #                               np.unpackbits(x1)))

            #             # data_iter = type_fxp.set_val(binary_val).get_val()

            #             data_iter = type_(x1)

            #         else:
            #             data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
            #                 type_).item()

            #         # data_iter = payload_trimmed_0[start_idx: end_idx].view(
            #         #     type_).item()

            #     empty_dict[col_name].append(data_iter)
            #     # empty_dict[col_name] = np.append(
            #     #     empty_dict[col_name], data_iter)

            #     byte_counter = byte_counter+map_size_

        col_name_list_repeat = []

        # empty_dict = {k: [dic[k]
        #                   for dic in empty_dict_list]
        #               for k in empty_dict_list[0]}

        for col_name, dtype_, size_ in zip(req_col_array,
                                           str_def_dict['map_type'],
                                           str_def_dict['map_size']):

            # if 'padding' in col_name.lower():
            #     continue

            if col_name in col_name_list_repeat:

                continue

            if not (np.dtype((np.void, size_)) == dtype_):
                if isinstance(dtype_, Fxp):

                    empty_dict[col_name] = np.array(empty_dict[col_name])
                    # continue

                else:
                    empty_dict[col_name] = np.array(empty_dict[col_name],
                                                    dtype=dtype_)
            else:
                # print('*********  Error handling\n',
                #       empty_dict[col_name])
                empty_dict[col_name] = np.array([0]*len(empty_dict[col_name]),
                                                dtype=bool
                                                # empty_dict[col_name],
                                                # dtype=np.uint32
                                                )

            col_name_list_repeat.append(col_name)

        for idx_supp_3d in supp_3d_help.index:

            col_name = supp_3d_help.loc[idx_supp_3d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1,
                                     *supp_3d_help.loc[idx_supp_3d,
                                                       'type_shape'][0][::-1])),
                           order='C')

        for idx_supp_2d in supp_2d_help.index:
            col_name = supp_2d_help.loc[idx_supp_2d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1, supp_2d_help.loc[idx_supp_2d,
                                                           'type_shape'][0][1])),
                           order='C')

        header_dict = {key: np.array(val, dtype=header_dtype_dict[key])
                       for key, val in header_dict.items()}
        header_dict = {key: val
                       if (val.flatten() != None).all()
                       else 'Not a value'
                       for key, val in header_dict.items()
                       }

        for key, val in header_dict.items():
            if not isinstance(val, str) and len(np.unique(val)) > 1:
                header_dict[key] = val
            elif isinstance(val, str):
                header_dict[key] = val
            else:
                header_dict[key] = np.unique(val).item()

        empty_dict = {**header_dict, **empty_dict}

        empty_dict_tree = self.nest_dict(empty_dict)

        return empty_dict_tree

    # @jit

    def _helper_payload_parser(self, row, stream_chunks_group,
                               index_arr, header_mapping, group_df,
                               str_def_dict, log_start_time,
                               ):

        empty_dict = {key: list()
                      for key in str_def_dict['req_col']
                      }

        index_req_group = row.name
        payload_trimmed_0 = row['aptiv_udp_combined_payload']
        is_big_endian = row['aptiv_udp_big_endian']
        if is_big_endian:
            payload_trimmed_0 = np.flipud(payload_trimmed_0)

        byte_counter = 0

        for map_size_, type_, col_name in zip(
                str_def_dict['map_size'],
                str_def_dict['map_type'],
                str_def_dict['req_col'],
        ):

            start_idx = byte_counter
            end_idx = byte_counter + map_size_

            if (('PADDING' in col_name) and
                    (np.dtype((np.void, map_size_)) == type_)):
                data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
                    type_).item()
                data_iter = int.from_bytes(data_iter)
                # if indicator == 2:
                #     data_iter = int.from_bytes(data_iter)

                #     data_iter_inner.append(data_iter)

                # else:
                #     data_iter = int.from_bytes(data_iter)
                #     data_iter_inner = data_iter

            else:

                if isinstance(type_, Fxp):
                    type_fxp = copy.deepcopy(type_)
                    x1 = payload_trimmed_0[start_idx: end_idx].copy().view(
                        np.uint8)
                    binary_val = '0b'+''.join(map(str,
                                                  np.unpackbits(x1)))

                    data_iter = type_fxp.set_val(binary_val).get_val()

                else:
                    data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
                        type_).item()

            empty_dict[col_name].append(data_iter)

            byte_counter = byte_counter+map_size_

        return empty_dict

    # @jit
    def _helper_header_parser(self, row, stream_chunks_group,
                              index_arr, header_mapping, group_df,
                              str_def_dict, log_start_time,
                              ):

        header_dict = {key: list()
                       for key in header_mapping.keys()}

        index_req_group = row.name

        # for row in range(aptiv_udp_payload_combined.shape[0]):
        #     payload_trimmed_0 = aptiv_udp_payload_combined[row, :]

        if stream_chunks_group > 1:
            idx_req_iter = np.argwhere(index_arr[:, 0] ==
                                       index_req_group)[0]
            idx_vals_iter = index_arr[idx_req_iter].flatten()
        else:
            idx_vals_iter = index_req_group

        for key, key_rev in header_mapping.items():

            val_iter = group_df.loc[idx_vals_iter, key_rev]

            if 'header.time' == key:

                val_iter = val_iter + log_start_time

            header_dict[key].append(val_iter)

        return header_dict

    def _parse_aptiv_udp(self,
                         req_group_df,
                         group_df,
                         index_arr,
                         stream_chunks_group,
                         req_db_path):

        with open(
                os.path.join(req_db_path), 'rb') as handle:
            str_def_dict = pickle.load(handle)

        empty_dict = {key: list()
                      for key in str_def_dict['req_col']
                      }

        # empty_dict = {key: np.array([], dtype=dtype_)
        #               if not (np.dtype((np.void, size_)) == dtype_)
        #               else np.array([], dtype=np.uint8)
        #               for key, dtype_, size_ in zip(str_def_dict['req_col'],
        #                                             str_def_dict['map_type'],
        #                                             str_def_dict['map_size'])
        #               }

        # aptiv_udp_payload_combined = np.array(
        #     req_group_df['aptiv_udp_combined_payload'].to_list())

        supp_dict = pd.DataFrame(list(zip(str_def_dict['req_col'],
                                          str_def_dict['type_shape'],
                                          str_def_dict['len_map_size_list'])),
                                 columns=['req_col',
                                          'type_shape',
                                          'len_map_size_list']
                                 )

        supp_df_2d = supp_dict.query('len_map_size_list == 1')
        supp_2d_help = pd.DataFrame({'req_col':
                                     supp_df_2d['req_col'].unique()})
        supp_2d_help['type_shape'] = [
            list(set(supp_df_2d['type_shape'].loc[supp_df_2d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_2d_help.iterrows()]

        supp_df_3d = supp_dict.query('len_map_size_list == 2')

        supp_3d_help = pd.DataFrame({'req_col':
                                     supp_df_3d['req_col'].unique()})
        supp_3d_help['type_shape'] = [
            list(set(supp_df_3d['type_shape'].loc[supp_df_3d['req_col']
                                                  == x['req_col']]))
            for _, x in supp_3d_help.iterrows()]

        header_dict = {}
        header_dict = {key: list()
                       for key in self.mudp_header_mapping.keys()}
        header_dtype_dict = {key: ''
                             for key in self.mudp_header_mapping.keys()}

        # group_df['timestamps_orig'] = group_df['timestamps']

        # group_df['timestamps'] = group_df[
        #     'timestamps'].diff(
        #         periods=1).ffill().fillna(0).cumsum()
        for index_req_group, payload_trimmed_0, is_big_endian in \
            zip(req_group_df.index,
                req_group_df['aptiv_udp_combined_payload'],
                req_group_df['aptiv_udp_big_endian']):

            if is_big_endian:
                payload_trimmed_0 = np.flipud(payload_trimmed_0)

            if stream_chunks_group > 1:
                idx_req_iter = np.argwhere(index_arr[:, 0] ==
                                           index_req_group)[0]
                idx_vals_iter = index_arr[idx_req_iter].flatten()
            else:
                idx_vals_iter = index_req_group

            for key, key_rev in self.mudp_header_mapping.items():

                val_iter = group_df.loc[idx_vals_iter, key_rev]

                header_val_dtype = np.dtype(group_df[key_rev].dtypes)

                if 'header.time' == key:

                    # print(f'Before : {val_iter}, {idx_vals_iter}, {key_rev}',
                    #       f'start_cTime : {self.log_start_time}')

                    # val_iter = val_iter\
                    #     # *self.time_unit_conversion_factor \
                    # + self.log_start_time

                    # if isinstance(val_iter, Iterable):
                    #     val_iter = val_iter.apply(
                    #         Decimal) + self.log_start_time
                    # else:
                    #     val_iter = Decimal(val_iter) + self.log_start_time

                    val_iter = val_iter + self.log_start_time
                    # print(f'After : {val_iter}, ')

                header_dict[key].append(val_iter)
                header_dtype_dict[key] = header_val_dtype

            byte_counter = 0

            for map_size_, type_, col_name in zip(
                    str_def_dict['map_size'],
                    str_def_dict['map_type'],
                    str_def_dict['req_col'],
            ):

                start_idx = byte_counter
                end_idx = byte_counter + map_size_

                if (('PADDING' in col_name) and
                        (np.dtype((np.void, map_size_)) == type_)):
                    data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
                        type_).item()
                    data_iter = int.from_bytes(data_iter)

                else:

                    if isinstance(type_, Fxp):
                        type_fxp = copy.deepcopy(type_)
                        x1 = payload_trimmed_0[start_idx: end_idx].copy().view(
                            np.uint8)
                        # binary_val = '0b'+''.join(map(str,
                        #                               np.unpackbits(x1)))

                        # data_iter = type_fxp.set_val(binary_val).get_val()
                        data_iter = type_fxp(x1)

                    else:
                        data_iter = payload_trimmed_0[start_idx: end_idx].copy().view(
                            type_).item()

                    # data_iter = payload_trimmed_0[start_idx: end_idx].view(
                    #     type_).item()

                empty_dict[col_name].append(data_iter)
                # empty_dict[col_name] = np.append(
                #     empty_dict[col_name], data_iter)

                byte_counter = byte_counter+map_size_

        col_name_list_repeat = []

        for col_name, dtype_, size_ in zip(str_def_dict['req_col'],
                                           str_def_dict['map_type'],
                                           str_def_dict['map_size']):

            # if 'padding' in col_name.lower():
            #     continue

            if col_name in col_name_list_repeat:

                continue

            if not (np.dtype((np.void, size_)) == dtype_):
                if isinstance(dtype_, Fxp):

                    empty_dict[col_name] = np.array(empty_dict[col_name])
                    # continue

                else:
                    empty_dict[col_name] = np.array(empty_dict[col_name],
                                                    dtype=dtype_)
            else:
                empty_dict[col_name] = np.array(empty_dict[col_name],
                                                # dtype=np.uint32
                                                )

            col_name_list_repeat.append(col_name)

        for idx_supp_3d in supp_3d_help.index:

            col_name = supp_3d_help.loc[idx_supp_3d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1,
                                     *supp_3d_help.loc[idx_supp_3d,
                                                       'type_shape'][0][::-1])),
                           order='C')

        for idx_supp_2d in supp_2d_help.index:
            col_name = supp_2d_help.loc[idx_supp_2d,
                                        'req_col']
            empty_dict[col_name] = \
                np.reshape(empty_dict[col_name],
                           newshape=((-1, supp_2d_help.loc[idx_supp_2d,
                                                           'type_shape'][0][1])),
                           order='C')

        header_dict = {key: np.array(val, dtype=header_dtype_dict[key])
                       for key, val in header_dict.items()}
        header_dict = {key: val
                       if (val.flatten() != None).all()
                       else 'Not a value'
                       for key, val in header_dict.items()
                       }

        for key, val in header_dict.items():
            if not isinstance(val, str) and len(np.unique(val)) > 1:
                header_dict[key] = val
            elif isinstance(val, str):
                header_dict[key] = val
            else:
                header_dict[key] = np.unique(val).item()

        empty_dict = {**header_dict, **empty_dict}

        empty_dict_tree = self.nest_dict(empty_dict)

        return empty_dict_tree

    def _read_raw_udp_data(self, ipv4_df):

        ipv4_df['is_udp_index'] = False  # udp_protocol
        udp_indices = ipv4_df.query('ipv4_protocol == 17 ' +
                                    'and ipv4_payload_length_ >= 8').index
        ipv4_df.loc[udp_indices, 'is_udp_index'] = True

        udp_df = ipv4_df.query('is_udp_index == True').reset_index(drop=True)

        if not udp_df.empty:
            udp_df['udp_payload'] = udp_df.apply(self._helper_extract_payload,
                                                 axis=1,
                                                 col_name_or_names='ipv4_payload_',
                                                 row_start=9-1,
                                                 is_payload_edit=False,
                                                 )
            # udp_payload=np.array(udp_df['udp_payload'].to_list()) #just to compare

            udp_df['udp_payload_length'] = udp_df['ipv4_payload_length_']-8

            udp_header_bytes = pd.DataFrame(udp_df.apply(self._helper_header_bytes,
                                                         axis=1,
                                                         col_name='ipv4_payload_',
                                                         row_end=8).to_list()).values

            udp_df['udp_header_bytes'] = pd.Series(list(udp_header_bytes))

            udp_df['udp_source_port'] = udp_df.apply(self._helper_typecast,
                                                     axis=1,
                                                     col_name='udp_header_bytes',
                                                     # start-1, end
                                                     row_start_end=[1-1, 2],
                                                     conversion_type=np.uint16,
                                                     ).values

            udp_df['udp_destination_port'] = udp_df.apply(self._helper_typecast,
                                                          axis=1,
                                                          col_name='udp_header_bytes',
                                                          row_start_end=[
                                                              3-1, 4],
                                                          conversion_type=np.uint16,
                                                          ).values

            udp_df['udp_length'] = udp_df.apply(self._helper_typecast,
                                                axis=1,
                                                col_name='udp_header_bytes',
                                                row_start_end=[5-1, 6],
                                                conversion_type=np.uint16,
                                                ).values

            udp_df['udp_checksum'] = udp_df.apply(self._helper_typecast,
                                                  axis=1,
                                                  col_name='udp_header_bytes',
                                                  row_start_end=[7-1, 8],
                                                  conversion_type=np.uint16,
                                                  ).values

            if self._remove_unused_vars:

                ipv4_df = None
                udp_df['ipv4_payload_'] = None

            port_filter = np.zeros((len(udp_df),), dtype=bool)
            eth_source_ports = np.array([])
            eth_dest_ports = self.mudp_eth_dest_ports

            port_list = copy.deepcopy(eth_source_ports)
            if bool(port_list.size):
                to_compare_ports = np.equal(port_list[:, np.newaxis],
                                            udp_df['udp_source_port'].values).T
                to_compare_bool_arr = np.sum(
                    to_compare_ports, axis=1).astype(bool)
                port_filter = np.logical_or(port_filter, to_compare_bool_arr)

            port_list = np.append(port_list, eth_dest_ports)
            if bool(port_list.size):
                to_compare_ports = np.equal(port_list[:, np.newaxis],
                                            udp_df['udp_destination_port'].values).T
                to_compare_bool_arr = np.sum(
                    to_compare_ports, axis=1).astype(bool)
                port_filter = np.logical_or(port_filter, to_compare_bool_arr)

            udp_df['udp_port_filter'] = port_filter

        return udp_df

    def _read_raw_tcp_data(self, ipv4_df):

        ipv4_df['is_tcp_index'] = False  # tcp_protocol

        tcp_indices = ipv4_df.query('ipv4_protocol == 6 ' +
                                    'and ipv4_payload_length_ >= 20').index
        ipv4_df.loc[tcp_indices, 'is_tcp_index'] = True

        tcp_df = ipv4_df.query('is_tcp_index == True').reset_index(drop=True)

        if not tcp_df.empty:
            tcp_df['tcp_payload'] = tcp_df.apply(self._helper_extract_payload,
                                                 axis=1,
                                                 col_name_or_names='ipv4_payload_',
                                                 row_start=21-1,
                                                 is_payload_edit=False,
                                                 )

            tcp_df['tcp_payload_length'] = tcp_df['ipv4_payload_length_']-20

            tcp_header_bytes = pd.DataFrame(tcp_df.apply(self._helper_header_bytes,
                                                         axis=1,
                                                         col_name='ipv4_payload_',
                                                         row_end=20).to_list()).values

            tcp_df['tcp_header_bytes'] = pd.Series(list(tcp_header_bytes))

            tcp_df['tcp_source_port'] = tcp_df.apply(self._helper_typecast,
                                                     axis=1,
                                                     col_name='tcp_header_bytes',
                                                     # start-1, end
                                                     row_start_end=[1-1, 2],
                                                     conversion_type=np.uint16,
                                                     ).values

            tcp_df['tcp_destination_port'] = tcp_df.apply(self._helper_typecast,
                                                          axis=1,
                                                          col_name='tcp_header_bytes',
                                                          row_start_end=[
                                                              3-1, 4],
                                                          conversion_type=np.uint16,
                                                          ).values

            tcp_df['tcp_data_offset'] = np.right_shift(
                tcp_header_bytes[:, 13-1], 4)
            tcp_df['tcp_seq_number'] = tcp_df.apply(self._helper_typecast,
                                                    axis=1,
                                                    col_name='tcp_header_bytes',
                                                    row_start_end=[5-1, 8],
                                                    conversion_type=np.uint32,
                                                    ).values
            tcp_df['tcp_ack_number'] = tcp_df.apply(self._helper_typecast,
                                                    axis=1,
                                                    col_name='tcp_header_bytes',
                                                    row_start_end=[9-1, 12],
                                                    conversion_type=np.uint32,
                                                    ).values

            tcp_df['tcp_ns'] = ipv4_df['tcp_header_bytes'].apply(lambda x:
                                                                 (x[:, 13-1] >> (1-1)) & 1)
            tcp_df['tcp_flags'] = tcp_header_bytes[:, 14-1]

            tcp_df['tcp_cwr'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (8-1)) & 1)
            tcp_df['tcp_ece'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (7-1)) & 1)
            tcp_df['tcp_urg'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (6-1)) & 1)
            tcp_df['tcp_ack'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (5-1)) & 1)
            tcp_df['tcp_psh'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (4-1)) & 1)
            tcp_df['tcp_rst'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (3-1)) & 1)
            tcp_df['tcp_syn'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (2-1)) & 1)
            tcp_df['tcp_fin'] = tcp_df['tcp_flags'].apply(lambda x:
                                                          (x >> (1-1)) & 1)
            tcp_df['tcp_window_size'] = tcp_df.apply(self._helper_typecast,
                                                     axis=1,
                                                     col_name='tcp_header_bytes',
                                                     row_start_end=[15-1, 16],
                                                     conversion_type=np.uint16,
                                                     ).values
            tcp_df['tcp_checksum'] = tcp_df.apply(self._helper_typecast,
                                                  axis=1,
                                                  col_name='tcp_header_bytes',
                                                  row_start_end=[17-1, 18],
                                                  conversion_type=np.uint16,
                                                  ).values
            tcp_df['tcp_urg_pointer'] = tcp_df.apply(self._helper_typecast,
                                                     axis=1,
                                                     col_name='tcp_header_bytes',
                                                     row_start_end=[19-1, 20],
                                                     conversion_type=np.uint16,
                                                     ).values

            tcp_df['tcp_extra_bytes'] = (
                (tcp_df['tcp_data_offset'] - 5)*4).values

            tcp_df['tcp_payload'] = tcp_df.apply(self._helper_extract_payload,
                                                 axis=1,
                                                 col_name_or_names=['tcp_payload',
                                                                    'tcp_extra_bytes',
                                                                    'tcp_payload_length'
                                                                    ],
                                                 row_start=21-1,
                                                 is_payload_edit=True,
                                                 )

            return tcp_df

    def _read_raw_ipv4_data(self, eth_df):

        # eth_df = self._read_raw_eth_data()

        eth_df['is_ipv4_index'] = False

        ipv4_indices = eth_df.query(f'`ETH_Frame.ETH_Frame.EtherType` == {int("0800", 16)} ' +
                                    'and `ETH_Frame.ETH_Frame.DataLength` >= 20').index

        eth_df.loc[ipv4_indices, 'is_ipv4_index'] = True

        ipv4_df = eth_df.query('is_ipv4_index == True').reset_index(drop=True)

        ipv4_df['ipv4_payload'] = ipv4_df.apply(self._helper_extract_payload,
                                                axis=1,
                                                col_name_or_names='ETH_Frame.ETH_Frame.DataBytes',
                                                row_start=21-1,
                                                is_payload_edit=False,
                                                )

        ipv4_df['ipv4_payload_length'] = ipv4_df['ETH_Frame.ETH_Frame.DataLength']-20

        ipv4_header_bytes = pd.DataFrame(ipv4_df.apply(self._helper_header_bytes,
                                                       axis=1,
                                                       col_name='ETH_Frame.ETH_Frame.DataBytes',
                                                       row_end=20).to_list()).values

        ipv4_df['ipv4_header_bytes'] = pd.Series(list(ipv4_header_bytes))

        ipv4_df['ipv4_ihl'] = np.bitwise_and(ipv4_header_bytes[:, 1-1], 15)

        ipv4_df['ipv4_length'] = ipv4_df.apply(self._helper_typecast,
                                               axis=1,
                                               col_name='ipv4_header_bytes',
                                               row_start_end=[3-1, 4],
                                               conversion_type=np.uint16,
                                               )

        ipv4_df['ipv4_identification'] = ipv4_df.apply(self._helper_typecast,
                                                       axis=1,
                                                       col_name='ipv4_header_bytes',
                                                       row_start_end=[5-1, 6],
                                                       conversion_type=np.uint16,
                                                       )

        ipv4_df['ipv4_flags'] = ipv4_header_bytes[:, 7-1]

        ipv4_df['ipv4_more_fragments'] = ipv4_df['ipv4_flags'].apply(lambda x:
                                                                     (x >> (6-1)) & 1)

        fragment_offset_0 = ipv4_df.apply(self._helper_typecast,
                                          axis=1,
                                          col_name='ipv4_header_bytes',
                                          row_start_end=[7-1, 8],
                                          conversion_type=np.uint16,
                                          ).values

        ipv4_df['ipv4_fragment_offset'] = np.bitwise_and(fragment_offset_0,
                                                         int('1FFF', 16))*8

        ipv4_df['ipv4_protocol'] = ipv4_header_bytes[:, 10-1]

        ipv4_df['ipv4_source_IP'] = ipv4_df.apply(self._helper_list_to_string,
                                                  axis=1,
                                                  col_name='ipv4_header_bytes',
                                                  row_start_end=[13-1, 16],
                                                  string_delimiter='.'
                                                  )

        ipv4_df['ipv4_dest_IP'] = ipv4_df.apply(self._helper_list_to_string,
                                                axis=1,
                                                col_name='ipv4_header_bytes',
                                                row_start_end=[17-1, 20],
                                                string_delimiter='.'
                                                )

        ipv4_df['ipv4_version'] = np.right_shift(ipv4_header_bytes[:, 1-1], 4)
        ipv4_df['ipv4_dscp'] = np.left_shift(ipv4_header_bytes[:, 2-1], 2)
        ipv4_df['ipv4_ecn'] = np.bitwise_and(ipv4_header_bytes[:, 2-1], 3)

        ipv4_df['ipv4_dont_fragment'] = ipv4_df['ipv4_flags'].apply(lambda x:
                                                                    (x >> (7-1)) & 1)

        ipv4_df['ipv4_ttl'] = ipv4_header_bytes[:, 10-1]

        ipv4_df['ipv4_checksum'] = ipv4_df.apply(self._helper_typecast,
                                                 axis=1,
                                                 col_name='ipv4_header_bytes',
                                                 row_start_end=[11-1, 12],
                                                 conversion_type=np.uint16,
                                                 ).values

        ipv4_df['ipv4_extra_bytes'] = ((ipv4_df['ipv4_ihl'] - 5)*4).values

        ipv4_df['ipv4_payload'] = ipv4_df.apply(self._helper_extract_payload,
                                                axis=1,
                                                col_name_or_names=['ipv4_payload',
                                                                   'ipv4_extra_bytes',
                                                                   'ipv4_payload_length'
                                                                   ],
                                                row_start=21-1,
                                                is_payload_edit=True,
                                                )

        ipv4_df['ipv4_payload_length'] = ipv4_df.apply(
            lambda x: x['ipv4_payload_length']
            - x['ipv4_extra_bytes']
            if x['ipv4_payload_length'] >
            x['ipv4_extra_bytes']
            else x['ipv4_payload_length'],
            axis=1
        )

        if self._remove_unused_vars:

            eth_df = None
            ipv4_df['ETH_Frame.ETH_Frame.DataBytes'] = None

        hash_source_ip = ipv4_df.apply(self._helper_typecast,
                                       axis=1,
                                       col_name='ipv4_header_bytes',
                                       row_start_end=[13-1, 16],
                                       conversion_type=np.uint32,
                                       is_big_endian=False
                                       )

        hash_source_ip = np.uint64(hash_source_ip)

        hash_dest_ip = ipv4_df.apply(self._helper_typecast,
                                     axis=1,
                                     col_name='ipv4_header_bytes',
                                     row_start_end=[17-1, 20],
                                     conversion_type=np.uint32,
                                     is_big_endian=False
                                     )

        hash_dest_ip = np.uint64(hash_dest_ip)

        ipv4_df['ipv4_hash'] = (np.left_shift(hash_source_ip, 48)
                                + np.left_shift(hash_dest_ip, 16)
                                + np.uint64(ipv4_df['ipv4_identification'].values)
                                )

        # complete_packets_indices = \
        #     np.argwhere((ipv4_df['ipv4_fragment_offset']
        #                  + np.uint16(ipv4_df['ipv4_more_fragments'])) == 0)

        complete_packets_indices = ipv4_df.query('(ipv4_fragment_offset ' +
                                                 '+ ipv4_more_fragments) == 0').index

        ipv4_df.loc[complete_packets_indices, 'ipv4_hash'] = 0

        # incomplete_packets_indices = list(ipv4_df.query('ipv4_hash != 0').index)

        incomplete_packets_indices = np.setdiff1d(list(ipv4_df.index),
                                                  complete_packets_indices)

        if len(incomplete_packets_indices) > 0:
            ipv4_df_defrag, ipv4_df = self._helper_defragment(ipv4_df, hash_col='ipv4_hash',
                                                              fragment_offset_col='ipv4_fragment_offset')

            to_drop_indices = True
        else:
            ipv4_df['defrag_helper_tuple'] = None
            ipv4_df['ipv4_payload_length_'] = ipv4_df['ipv4_payload_length'].values
            ipv4_df['ipv4_payload_'] = ipv4_df['ipv4_payload'].values
            ipv4_df['ipv4_previous_zero_index'] = np.nan

            to_drop_indices = False
            ipv4_df_defrag = ipv4_df

        complete_packets_indices = np.setdiff1d(complete_packets_indices,
                                                ipv4_df.query(
                                                    'ipv4_previous_zero_index != -1')
                                                .index)

        incomplete_packets_indices = np.setdiff1d(list(ipv4_df.index),
                                                  complete_packets_indices)

        ipv4_df['ipv4_relevant_indices'] = True
        ipv4_df.loc[incomplete_packets_indices,
                    'ipv4_relevant_indices'] = False  # rows to be discarded

        # if to_drop_indices:
        #     ipv4_df = ipv4_df.drop(index=incomplete_packets_indices)
        # self.ipv4_cols = list(ipv4_df_defrag.columns)
        # self.ipv4_df_defrag = ipv4_df_defrag

        return ipv4_df, ipv4_df_defrag

    def _read_all_CAN_data(self, log_path):

        group_data_dict = self._read_raw_data_mdf(log_path,
                                                  ethernet_only=False)

        return group_data_dict

    def _read_all_eth_data(self, log_path):

        group_data_dict = self._read_raw_data_mdf(log_path,
                                                  ethernet_only=True)

        return group_data_dict

    def _read_raw_eth_data(self, eth_df):

        eth_df = eth_df.reset_index()

        # eth_df[
        #     'ETH_Frame.ETH_Frame.DataLength'] = eth_df[
        #         'ETH_Frame.ETH_Frame.DataLength'].astype(int)

        eth_df = eth_df.astype({'ETH_Frame.ETH_Frame.DataLength': int,
                                'ETH_Frame.ETH_Frame.Source': int,
                                'ETH_Frame.ETH_Frame.Destination': int,
                                })

        eth_df['source_mac'] = eth_df.apply(self._helper_macint,
                                            axis=1,
                                            col_name='ETH_Frame.ETH_Frame.Source')

        eth_df['destination_mac'] = eth_df.apply(self._helper_macint,
                                                 axis=1,
                                                 col_name='ETH_Frame.ETH_Frame.Destination')

        vlan_indices = eth_df[eth_df['ETH_Frame.ETH_Frame.EtherType']
                              == int('8100', 16)].index

        # payload = eth_df[['ETH_Frame.ETH_Frame.DataBytes', 'timestamps', ]]

        # payload_true = payload.loc[vlan_indices, :]

        EtherType_new = eth_df.apply(self._helper_typecast,
                                     axis=1,
                                     col_name='ETH_Frame.ETH_Frame.DataBytes',
                                     row_start_end=[3-1, 4],
                                     conversion_type=np.uint16,
                                     )

        eth_df.loc[vlan_indices, 'ETH_Frame.ETH_Frame.EtherType'] = \
            EtherType_new[vlan_indices]

        eth_df.loc[vlan_indices, 'ETH_Frame.ETH_Frame.DataLength'] = \
            eth_df.loc[vlan_indices, 'ETH_Frame.ETH_Frame.DataLength']-4

        vlan_indices_new = eth_df[eth_df['ETH_Frame.ETH_Frame.EtherType']
                                  == int('8100', 16)].index

        eth_df['vlan_indices'] = False
        # rows to be discarded
        eth_df.loc[vlan_indices_new, 'vlan_indices'] = True

        eth_df = eth_df.drop(index=vlan_indices_new)

        eth_df['ETH_Frame.ETH_Frame.DataBytes'] = \
            eth_df.apply(self._helper_payload,
                         axis=1,
                         col_name_or_names='ETH_Frame.ETH_Frame.DataBytes',
                         row_start=5-1,
                         vlan_indices=vlan_indices,
                         )
        eth_df = eth_df.drop(vlan_indices_new, axis=1)

        return eth_df

    def _helper_macint(self, row, col_name: str):
        return int_to_mac(row[col_name])

    def _db_file_helper(self,
                        stream_def_pre,
                        stream_def_source,
                        stream_def_sensor_id,
                        stream_def_str,
                        stream_def_ver,
                        stream_def_extension='.txt'

                        ):

        database_filelist = []

        for root, dirs, files in os.walk(self.stream_def_dir_path):
            for file in files:
                if (file.endswith(stream_def_extension)):
                    # append the file name to the list
                    database_filelist.append(
                        os.path.join(root, file))

        file_name_only_list = [os.path.basename(file_path)
                               for file_path in database_filelist]

        substring_list_1 = [stream_def_pre,
                            stream_def_source,
                            stream_def_sensor_id,
                            stream_def_str,
                            stream_def_ver,
                            stream_def_extension,
                            ]

        out_search_results_1 = [re.search('.*'.join(substring_list_1),
                                          sample_str)
                                for sample_str in file_name_only_list]
        out_search_results_1 = [result.group()
                                for result in out_search_results_1
                                if result is not None]

        substring_list_2 = [stream_def_pre,
                            stream_def_source,
                            stream_def_str,
                            stream_def_ver,
                            stream_def_extension,
                            ]

        out_search_results_2 = [re.search('.*'.join(substring_list_2),
                                          sample_str)
                                for sample_str in file_name_only_list]
        out_search_results_2 = [result.group()
                                for result in out_search_results_2
                                if result is not None]

        out_search_results = list(set(out_search_results_1).union(
            set(out_search_results_2)))

        if len(out_search_results) == 0:

            req_db_path = None
        else:

            if len(out_search_results) > 1:
                req_name = out_search_results_1[0]
            else:
                req_name = out_search_results[0]

            req_db_path = database_filelist[file_name_only_list.index(
                req_name)]

        return req_db_path

    def nest_dict(self, flat_dict, sep='.'):
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

    def divide_chunks(self, l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _helper_combine_payloads(self, row, ):

        # index = row.name

        stream_length = row['aptiv_udp_stream_data_length']

        return_val = row['aptiv_udp_payload'][:stream_length]

        return return_val

    def combine_chunk_payloads(self, index_list, group_df, req_indices_combined_payload):

        out_df = group_df.copy(deep=True).loc[req_indices_combined_payload, :]
        # pd.Series(dtype='object')
        out_df['aptiv_udp_combined_payload'] = None

        # print('here I am ********************************')
        payload_list = []

        for idx, (indices, req_idx) in enumerate(zip(index_list,
                                                     req_indices_combined_payload)):  # [index_list[1]]:  #
            dtype_ = np.dtype(group_df.loc[indices[0],
                                           'aptiv_udp_payload_trimmed'].dtype)
            idxs_ends = group_df.loc[indices,
                                     'aptiv_udp_stream_chunk_len'].values

            # vals = np.array(
            #     group_df.loc[indices, 'aptiv_udp_payload_trimmed'].to_list(),
            #     dtype=dtype_)

            vals = group_df.loc[indices,
                                'aptiv_udp_payload_trimmed'].to_list()

            payload = np.array(list(
                chain.from_iterable([dd[:end]
                                     for dd, end in zip(vals, idxs_ends)
                                     ]
                                    )
            ),
                dtype=dtype_)

            payload_list.append(payload.tolist())

        # req_path_pickle = os.path.join(r'C:\Users\mfixlz\OneDrive - Aptiv\Documents\DM_A\Aravind\Projects\2025\KW_29\BCO-14819',
        #                                'payload_list_2.pkl')

        # print('**********************************',
        #       req_path_pickle)

        # with open(req_path_pickle, 'wb') as file:

        #     pickle.dump([payload_list,
        #                  index_list,
        #                  group_df,
        #                  req_indices_combined_payload,
        #                  dtype_], file)

        payload_lengths_list = np.array([len(item) for item in payload_list])
        most_probable_payload_length = sp.stats.mode(payload_lengths_list)[0]
        indices_to_change = [idx
                             for idx, item in enumerate(payload_lengths_list)
                             if item != most_probable_payload_length]

        indices_to_change.sort()

        indices_to_copy = [idx-1 if idx != 0 else 1
                           for idx in indices_to_change]

        if len(indices_to_change) > 0:

            cTime_arr = group_df.loc[
                list(chain.from_iterable(
                    [index_list[item]
                     for item in indices_to_change])),
                'timestamps'].values

            out_val_map_key = f'Stream {self.stream_source_sid[0]}, ' + \
                f'source {self.stream_source_sid[1]}' +\
                ' is having multiple length payload chunks at cTimes' + \
                f'\n{cTime_arr}\n' +\
                              'and will be replaced by previous payload'
            warnings.warn(out_val_map_key,
                          DeprecationWarning)

            self.debug_str = self.debug_str + out_val_map_key + '\n'

            for (index, replacement) in zip(indices_to_change, indices_to_copy):
                payload_list[index] = payload_list[replacement]

        out_df['aptiv_udp_combined_payload'] = pd.Series(
            payload_list, dtype=dtype_,
            index=req_indices_combined_payload).apply(
                lambda x: np.array(x, dtype=dtype_))

        return out_df

    def _helper_defragment(self,
                           df, hash_col: str,
                           fragment_offset_col: str,
                           other_cols: list = None):

        hash_grouped_df = df.groupby(hash_col, group_keys=True)

        intermediate_val = hash_grouped_df[[fragment_offset_col,
                                            ]
                                           ].apply(self._prev_0_index,
                                                   req_cols=[
                                                       fragment_offset_col],

                                                   ).reset_index()
        # print(f'^^^^^^^^^^^^  \n {intermediate_val.columns} \n %%%%%%%%%%%%%%%%%%')
        df['ipv4_previous_zero_index'] = \
            intermediate_val[['level_1', 0]].set_index(
            'level_1').sort_index(
        ).rename({0: 'rev'}, axis=1).values

        hash_grouped_df = df.groupby(hash_col, group_keys=True)
        group_index_dict = hash_grouped_df.groups

        total_df_list = []

        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&& 3520')
        iteration = 0

        for group_key, group_indices in group_index_dict.items():

            # print(f'%%%%%%%%%%%%%%%%%%%%%% 3525 *********** {iteration}')

            group_df = hash_grouped_df.get_group(group_key)

            if group_key == 0:
                group_df_req = group_df
                group_df_req['ipv4_payload_'] = group_df['ipv4_payload']
                group_df_req['ipv4_payload_length_'] = group_df['ipv4_payload_length']
            else:

                if np.all(np.diff(group_df['ipv4_fragment_offset'], 2) == 0):

                    # concat_arrays_list = group_df['ipv4_payload'].tolist()

                    # concat_arrays_length = group_df['ipv4_payload_length'].tolist()

                    concat_arrays_list = [item[:length]
                                          for item, length in
                                          zip(group_df['ipv4_payload'].tolist(),
                                              group_df['ipv4_payload_length'].tolist())
                                          ]
                    ipv4_payload_ = np.concatenate(
                        concat_arrays_list, axis=0)

                    ipv4_payload_length_ = len(ipv4_payload_)

                    # group_df_req = pd.DataFrame(columns=group_df.columns)

                    data_cols = {'ipv4_payload_': [ipv4_payload_],
                                 'ipv4_payload_length_': [ipv4_payload_length_]
                                 }

                    for column_group in group_df.columns:

                        data_cols[column_group] = [group_df.loc[
                            group_indices[0], column_group]]

                    group_df_req = pd.DataFrame(
                        data_cols,
                        index=[group_indices[0]]
                    )

                else:

                    continue

            total_df_list.append(group_df_req)
            iteration += 1

        total_df = pd.concat(total_df_list,
                             axis=0).apply(pd.to_numeric, errors='ignore')

        return total_df.sort_index(), df

    def _helper_defragment_old(self, df, hash_col: str,
                               fragment_offset_col: str,
                               other_cols: list = None):

        hash_grouped_df = df.groupby(hash_col, group_keys=True)

        intermediate_val = hash_grouped_df[[fragment_offset_col,
                                            ]
                                           ].apply(self._prev_0_index,
                                                   req_cols=[
                                                       fragment_offset_col],

                                                   ).reset_index()
        # print(f'^^^^^^^^^^^^  \n {intermediate_val.columns} \n %%%%%%%%%%%%%%%%%%')
        df['ipv4_previous_zero_index'] = \
            intermediate_val.set_index(
            'level_1').sort_index(
        ).rename({0: 'rev'}, axis=1)

        hash_grouped_df = df.groupby(hash_col, group_keys=True)
        group_index_dict = hash_grouped_df.groups

        total_df_list = []

        for group_key, group_indices in group_index_dict.keys():

            group_df = hash_grouped_df.get_group(group_key)

            group_df['defrag_helper_tuple'] = group_df.apply(self._helper_defragment_sub,
                                                             axis=1, df=group_df)

            group_df['ipv4_payload_length_'] = group_df.apply(self._helper_defragment_main,
                                                              axis=1,
                                                              is_payload=False)
            group_df['ipv4_payload_'] = group_df.apply(self._helper_defragment_main,
                                                       axis=1,
                                                       is_payload=True)
            total_df_list.append(group_df)

        total_df = pd.concat(total_df_list,
                             axis=1)  # .apply(pd.to_numeric, errors='ignore')

        return total_df.sort_index()

    def _prev_0_index(self, df, req_cols: list):

        return_val = df.apply(self._helper_prev_0_index,
                              axis=1,
                              req_cols=req_cols)

        return return_val

    def _helper_prev_0_index(self, row, req_cols: list):

        index = row.name
        if row[req_cols[0]] == 0:
            return_val = index
        else:
            return_val = -1

        return return_val

    def _helper_defragment_sub(self, row, df):

        # index = row.name

        return_val = (0, 0)
        if (row['ipv4_previous_zero_index'] > 0
                and row['ipv4_fragment_offset'] != 0):
            prev_payload_len_0_idx = df.loc[row['ipv4_previous_zero_index'],
                                            'ipv4_payload_length']
            if (row['ipv4_fragment_offset']
                    == prev_payload_len_0_idx):

                current_ipv4_payload_length = prev_payload_len_0_idx
                fragment_ipv4_payload_length = row['ipv4_payload_length']
                new_payload_length = (current_ipv4_payload_length
                                      + fragment_ipv4_payload_length)

                payload = [current_ipv4_payload_length+1,
                           new_payload_length+1,
                           row['ipv4_payload'][:fragment_ipv4_payload_length]
                           ]

                return_val = (payload,
                              new_payload_length)

            return return_val

    def _helper_defragment_main(self, row, is_payload: bool = True):

        req_tuple = row['defrag_helper_tuple']
        if is_payload:
            if sum(req_tuple) == 0:

                return_val = row['ipv4_payload']

            else:
                return_val = row['ipv4_payload']
                return_val[req_tuple[0]:
                           req_tuple[1]] = req_tuple[2]
        else:
            if sum(req_tuple) == 0:

                return_val = row['ipv4_payload_length']

            else:

                return_val = req_tuple[2]

        return return_val

    def _helper_list_to_string(self, row, col_name,
                               row_start_end,
                               string_delimiter: str = '.'):

        return_val = f'{string_delimiter}'.join(map(str,
                                                    row[col_name]
                                                    [row_start_end[0]:
                                                     row_start_end[1]]))

        return return_val

    def _helper_extract_payload(self, row, col_name_or_names,
                                row_start: int = 20,
                                is_payload_edit: bool = False):
        if is_payload_edit:
            # if row[col_name_or_names[2]] > row[col_name_or_names[1]]:
            if row[col_name_or_names[1]] > 0:
                return_val = row[col_name_or_names[0]][
                    row[col_name_or_names[1]]+1-1:]
            else:
                return_val = row[col_name_or_names[0]]
        else:
            return_val = row[col_name_or_names][row_start:]

        return return_val

    def _helper_header_bytes(self, row, col_name, row_end: int = 20):

        return_val = row[col_name][:row_end]

        return return_val

    def _helper_typecast(self, row, col_name,
                         row_start_end: list,
                         conversion_type: np.dtypes = np.uint16,
                         is_big_endian: bool = True):

        row_val = row[col_name]

        if isinstance(row_val, Iterable):

            if is_big_endian:

                return_val = np.array(row_val[row_start_end[0]:
                                              row_start_end[1]][::-1]).view(
                                                  conversion_type)[0]  # .item()

            else:
                return_val = np.array(row_val[row_start_end[0]:
                                              row_start_end[1]]).view(
                                                  conversion_type)[0]  # .item()
        else:
            return_val = row_val

        return return_val

    def _helper_payload(self, row, col_name_or_names,
                        row_start: int,
                        vlan_indices: list):
        index = row.name
        if index in vlan_indices:
            return_val = row[col_name_or_names][row_start:]
        else:
            return_val = row[col_name_or_names]

        return return_val

    def _existence_channels(self, log_path, channel_list, ):

        group_data = self._read_all_eth_data(log_path)

        existing_channels = list(group_data.keys())

        missing_channels = set(channel_list).difference(existing_channels)

        return list(missing_channels)

    def _SPI_data_parser2(self, log_path, dbc_pickle_path_dma):

        # mapper_json_path = os.path.join(dbc_pickle_path_dma,
        #                     'eyeq_message_protocol_version_mapper.json')

        with open(dbc_pickle_path_dma) as fp:

            spi_mapper_json = json.load(fp)

        # with open(dbc_pickle_path_dma, 'rb') as fp:

        #     spi_dbc = pickle.load(fp)

        SPI_eyeQ_dict, _ = extract_SPI_data(log_path,
                                            spi_mapper_json,
                                            os.path.split(
                                                dbc_pickle_path_dma)[0]
                                            )

        SPI_flat = self.flatten_dict(SPI_eyeQ_dict)
        SPI_flat_non_empty = {key: val
                              for key, val in SPI_flat.items()
                              if bool(val)
                              }
        SPI_flat_empty = {key: val
                          for key, val in SPI_flat.items()
                          if not bool(val)
                          }
        sub_string_list = [['_\d+$',
                            r'_\\d+$'],
                           ['\d+$',
                            r'\\d+$']
                           ]

        SPI_flat_non_empty_processed, debug_str_append = \
            self._process_flat_dict_all(SPI_flat_non_empty,
                                        sub_string_list)

        SPI_flat_processed = {**SPI_flat_empty,
                              **SPI_flat_non_empty_processed}

        SPI_eyeQ_dict_return = self.nest_dict(SPI_flat_processed,)

        return SPI_eyeQ_dict_return, SPI_eyeQ_dict

    def pop_dict_vals(self, in_dict, edit_=False):

        in_dict_copy = copy.deepcopy(in_dict)

        if edit_:

            for key, val in in_dict_copy.items():

                if '.Decoded.' in key:

                    del in_dict[key]

                if '.Raw' in key:

                    key_split = key.split('.')
                    key_split.remove('Raw')
                    key_edited = '.'.join(key_split)

                    in_dict[key_edited] = in_dict[key]

                    del in_dict[key]

                if '.Key' in key:

                    del in_dict[key]

                if '.Value' in key:

                    key_split = key.split('.')
                    key_split.remove('Value')
                    key_edited = '.'.join(key_split)

                    in_dict[key_edited] = in_dict[key]

                    del in_dict[key]

        return_val = in_dict

        return return_val

    def _SPI_data_parser(self,
                         log_path,
                         protocol_type_: str = 'SPI',
                         dbc_path: os.path.join = None):

        protocol_type = DecoderAPI.ProtocolType(protocol_type_)
        protocol_path = dbc_path

        req_file_list = []
        for root, dirs, files in os.walk(protocol_path):
            for file in files:
                if (file.endswith('.dbc')):
                    req_file_list.append(os.path.join(root, file))

        decoding_settings = DecoderAPI.DecodingSettings(
            protocol_type=protocol_type,
            protocol_path=protocol_path)

        decoder = DecoderAPI.DecoderAPI(
            file_path=log_path,
            decoding_settings=decoding_settings,)

        id_name_dict = {}
        req_Lines = [open(file, encoding='cp1252', errors='replace').readlines()
                     for file in req_file_list]

        for Lines in req_Lines:

            req_id_lines = [line.split() for line in Lines if 'BO_ ' in line]

            for id_line in req_id_lines:

                id_name_dict[int(id_line[1])] = id_line[2].replace(':', '')

        ser_msgs_ids = [[{**msg.get_signal([]),
                          **{'cTime': msg.get_timestamp(),
                        'ID': msg.get_id()}
                          },  msg.get_id()]for msg in decoder]

        ser_msgs = np.array([item[0] for item in ser_msgs_ids])
        ser_ids = [item[1] for item in ser_msgs_ids]

        id_idx_dict = self.grouping_unique_vals_indices(ser_ids)

        name_idx_dict = {val: id_idx_dict[key]
                         for key, val in id_name_dict.items()
                         }

        name_val_dict = {key: ser_msgs[val]
                         for key, val in name_idx_dict.items()
                         }

        flatten_name_val_dict2 = {}

        merged_name_val_dict2 = {}

        nested_name_val_dict2 = {}

        for key, val in name_val_dict.items():

            flatten_name_val_dict2[key] = []

            for item in val:

                flat_dict = self.flatten_dict2(item, separator='.')

                flatten_name_val_dict2[key].append(
                    flat_dict)

            flatten_name_val_dict2[key] = \
                [self.pop_dict_vals(item_dict, edit_=True)
                 for item_dict in flatten_name_val_dict2[key]]

            # merged_name_val_dict2[key] = self.merge_list_of_dicts(
            #     flatten_name_val_dict2[key])

            nested_name_val_dict2[key] = self.nest_dict(
                self.merge_list_of_dicts(flatten_name_val_dict2[key]))

        return nested_name_val_dict2  # , ser_msgs_ids

    def _flr4_ad5_parser2(self, log_path):

        df = stream_stats_main(log_path).sort_values(by=['timestamps'],
                                                     ascending=True)
        flr4_dict = decode_eth_channel_by_arxml(df,
                                                self.req_arxml_name)
        try:
            arxml_version = float(re.search(r'[\d]*[.][\d]+',
                                            os.path.split(
                                                self.req_arxml_name)[1]
                                            .partition('_Ver_')[2])[0])

        except:
            arxml_version = float(re.search(r'[\d]*[.]',
                                            os.path.split(
                                                self.req_arxml_name)[1]
                                            .partition('_Ver_')[2])[0])
        if arxml_version < 13:

            # self.CAN_flat = False

            print('&&&&&&&&&&&&&&&&&&&&& ',
                  f'arxml version is {arxml_version} and it is < 13, ',
                  'proceeding to process the detections')

            flr_dma_list_flat = {key: self.flatten_dict2(val, separator='.')
                                 if isinstance(val, dict)
                                 else [val]
                                 for key, val in flr4_dict.items()
                                 }
            flr_dma_list_flat = [{key + '.' + key2: item2
                                  for key2, item2 in val.items()}
                                 if isinstance(val, dict)
                                 else {key: val}
                                 for key, val in flr_dma_list_flat.items()
                                 ]

            flr_dma_list_flat_df = [pd.DataFrame(item)
                                    for item in flr_dma_list_flat]

            flr4_dma_flat = self._process_flat_dict(flr_dma_list_flat_df,
                                                    sub_string_list=[['_\d{3}_\d{3}',
                                                                      r'_\\d{3}_\\d{3}'],
                                                                     ['_\d+$',
                                                                      r'_\\d+$']
                                                                     ]
                                                    )

            flr4_dma_flat = dict(ChainMap(*flr4_dma_flat))

            # CAN_flat_local = self.CAN_flat

            # self.CAN_flat = True

            flr4_dma_out, debug_str_append = self._process_flat_dict_all(
                flr4_dma_flat,
                sub_string_list=[['_\d{3}_\d{3}',
                                  r'_\\d{3}_\\d{3}'],
                                 ['_\d+$',
                                  r'_\\d+$']
                                 ]
            )

            # self.CAN_flat = CAN_flat_local

            flr4_dict_dvl_ext = self.nest_dict(flr4_dma_out)
            # self.CAN_flat = True
        else:

            flr4_dict_dvl_ext = flr4_dict
        return flr4_dict, flr4_dict_dvl_ext

    def grouping_unique_vals_indices(self, a):
        d = defaultdict(list)
        for i, j in enumerate(a):
            d[j].append(i)
        return d

    def _process_flat_dict(self, flat_dict,
                           sub_string_list: str = [['\d{3}_\d{3}',
                                                    r'\\d{3}_\\d{3}'],
                                                   ['\d+$',
                                                    r'\\d+$']
                                                   ]):

        def mysplit(s):
            head = s.rstrip('0123456789')
            tail = s[len(head):]
            return head, tail

        compile_pattern = re.compile(sub_string_list[0][0])
        compile_pattern2 = re.compile(sub_string_list[1][0])
        map_dict_list = []
        for i, dict_ in enumerate(flat_dict):  # enumerate([flat_dict[3]]):#
            # print(i)

            if isinstance(dict_, pd.DataFrame):
                dum_df = copy.deepcopy(dict_)

            else:
                with warnings.catch_warnings(action='ignore',
                                             category=FutureWarning):
                    dum_df = pd.DataFrame(dict_, index=[0]).apply(
                        pd.to_numeric, errors='ignore')

            req_list = dum_df.columns

            val = list(filter(compile_pattern2.search, req_list))
            # val.sort()
            y = list(set([compile_pattern2.sub(r'$', item)
                          for item in val]))

            y_ = list(set([compile_pattern2.sub('', item)
                          for item in val]))
            y.sort()
            y_.sort()
            z = [compile_pattern.sub('', item) for item in y_]

            sort_substrings = [item.split('.')[-1] for item in z]
            # y1 = list(set([compile_pattern2.sub(r'\\d+$', item)
            #                for item in val]))
            y1 = list(set([compile_pattern2.sub(sub_string_list[1][1], item)
                           for item in val]))
            y1 = [item for item2 in sort_substrings
                  for item in y1 if item2 == item.split('.')[-1][:
                                                                 -len(sub_string_list[1][1]
                                                                      )+1]]
            # y1.sort()
            # z1 = [compile_pattern.sub(r'\\d{3}_\\d{3}', item) for item in y1]
            # z2 = [compile_pattern.sub(r'\\d{3}_\\d{3}', item) for item in y]
            z1 = [compile_pattern.sub(sub_string_list[0][1], item)
                  for item in y1]
            z2 = [compile_pattern.sub(sub_string_list[0][1], item)
                  for item in y]
            map_dict = {}
            total_iterals = []
            for item1, item2, item3 in zip(z1, z2, y_):

                pattern1 = re.compile(item1)
                pattern2 = re.compile(item2)
                iter_val1 = list(filter(pattern1.search, req_list))
                iter_val2 = list(filter(pattern2.search, req_list))
                iter_val = list(set(iter_val1 + iter_val2))
                # iter_val = list(set(iter_val1).union(set(iter_val2)))

                numerical_part_2 = [mysplit(item.split('.')[-1])[-1]
                                    for item in iter_val]
                numerical_part_2 = [int(item) if len(item) != 0 else 0
                                    for item in numerical_part_2]

                sort_index_2 = np.argsort(numerical_part_2)
                iter_val = np.array(iter_val)[sort_index_2]

                # iter_val = sort_list(iter_val)
                # dum_arr = dum_df.loc[0, iter_val].to_numpy()

                if isinstance(dict_, pd.DataFrame):
                    dum_arr = dum_df.loc[:, iter_val]
                else:
                    dum_arr = dum_df.loc[0, iter_val]
                with warnings.catch_warnings(action='ignore',
                                             category=FutureWarning):
                    dum_arr = dum_arr.apply(
                        pd.to_numeric, errors='ignore'
                    ).to_numpy(
                        # dtype=dtypes_dum_arr
                    )
                # dum_arr = np.array(dum_df[iter_val])

                # map_dict[item3] = iter_val
                map_dict[item3] = dum_arr

                total_iterals.append(iter_val.tolist())

            # np.array(total_iterals).flatten()
            total_iterals = list(chain(*total_iterals))

            remaining_keys = list(set(dum_df.columns) - set(total_iterals))

            rem_dict = dum_df[remaining_keys].to_dict(orient='list')

            map_dict = {**map_dict, **rem_dict}

            map_dict_list.append(map_dict)

        return map_dict_list

    def _process_flat_dict_all(self, flat_dict_all_data,
                               sub_string_list: str = [['\d{3}_\d{3}',
                                                        r'\\d{3}_\\d{3}'],
                                                       ['\d+$',
                                                        r'\\d+$']
                                                       ]):

        def stack_padding(list_of_np_arrays, add_col_index):

            def pad_(row, size_req, add_index):

                if not isinstance(row, np.ndarray):
                    row = np.array(row)

                if len(row) == np.size(row):

                    row = np.reshape(row, (len(row), 1))

                if len(row) != size_req:
                    if add_index == 0:
                        row = np.insert(row,
                                        add_index,
                                        0,  # row[-1, :],
                                        axis=0)
                    elif add_index == -1:

                        data_type = row[0].dtype.type
                        if data_type is np.str_:
                            fill_val = np.array(['']*len(row[0]),
                                                dtype=row[0].dtype)
                        elif (data_type is np.float_
                              or data_type is np.int_):
                            fill_val = row[0]*0

                        else:
                            fill_val = row[0]*0

                        row = np.append(row,
                                        (fill_val).reshape((1,
                                                            np.shape(row)[1])),
                                        axis=0,
                                        )

                return row

            row_length = max(list_of_np_arrays, key=len).__len__()

            mat = [pad_(row, row_length, add_index)
                   for row, add_index in zip(list_of_np_arrays, add_col_index)]

            return mat

        compile_pattern = re.compile(sub_string_list[0][0])

        if self.CAN_flat:
            val = list(filter(compile_pattern.search,
                       flat_dict_all_data.keys()))

        elif self.SPI_flat:
            parent_paths = ['.'.join(key.split('.')[:-1])
                            for key in flat_dict_all_data.keys()
                            if 'Sync_ID' in key.split('.')[-1]]
            req_signals = [key for parent in parent_paths
                           for key in flat_dict_all_data.keys() if parent in key]
            val = list(filter(compile_pattern.search, req_signals))

        z = list(set([compile_pattern.sub('', item) for item in val]))

        if self.CAN_flat:
            sort_substrings = [item.split('.')[-1] for item in z
                               if ('cTime' not in item)
                               # or ('timestamp' not in item)
                               ]
        elif self.SPI_flat:
            sort_substrings = ['.'.join(item.split('.')[-2:]) for item in z
                               if ('cTime' not in item)
                               # or ('timestamp' not in item)
                               ]

        map_dict = {}
        # ['LrrfEthDetAzimuth']:#
        debug_str_append = ''
        for enum, item in enumerate(sort_substrings):

            #

            if self.CAN_flat:
                key = [item_z for item_z in z if item ==
                       item_z.split('.')[-1]][0]
                key_val = [item_val for item_val in val if item ==
                           item_val.split('.')[-1]]
                # key = [item_z for item_z in z if item == item_z.split('.')[-1]][0]

            elif self.SPI_flat:
                key = [item_z for item_z in z
                       if item == '.'.join(item_z.split('.')[-2:])][0]
                compile_pattern_iter = re.compile(item + sub_string_list[0][0])
                key_val = list(filter(compile_pattern_iter.search, val))

            else:
                key_val = [item_val for item_val in val if item in
                           item_val.split('.')[-1]]

            key_val.sort()

            # map_dict[key] = []

            # for values in key_val:
            #     map_dict[key].append(flat_dict_all_data[values])

            # if self.FLR:

            #     to_create_dict_path = 'ERR_DEBUG_DATA'

            #     req_pdu_id_list = ['0x100'+str(item).zfill(2)
            #                        for item in range(1, 12)
            #                        ]
            #     final_pdu_id = '0x10012'

            #     final_pdu_name = 'PDU_LRRF_to_ADCAM_LRRF_Header_Timestamps'

            #     final_pdu_dict = {final_pdu_id: final_pdu_name}

            #     path_list = [values.split('.')[0]
            #                  for values in key_val]

            #     pdu_id_list = [np.unique(
            #         flat_dict_all_data[path + '.header.pdu_id'])[0]
            #         for path in path_list]

            #     req_cols_dict = {pdu_id: col + '.header.time'
            #                      for pdu_id, col in zip(pdu_id_list, path_list)
            #                      if pdu_id in req_pdu_id_list}

            #     to_compare_col = final_pdu_name + '.header.time'

            #     sein_pdu_col_list = list(req_cols_dict.keys())
            #     sein_pdu_col_list.sort()
            #     Erste_schluessel = sein_pdu_col_list[0]

            #     req_indices_dict = {}

            #     for (key_req, col_req), key_val_item in zip(req_cols_dict.items(), key_val):
            #         # !!!: assumption is that lengths of to_compare_col and col are same
            #         bool_array = np.greater(
            #             flat_dict_all_data[to_compare_col],
            #             flat_dict_all_data[col_req])

            #         req_indices_dict[key_req] = bool_array

            #         flat_dict_all_data[key_val_item] = np.array(
            #             flat_dict_all_data[key_val_item])

            #         flat_dict_all_data[key_val_item][np.argwhere(
            #             np.logical_not(bool_array))] = -999

            #     # flat_dict_all_data[to_create_dict_path +
            #     #                    '.LRRF_ETH_LOOK_INDEX'] = \
            #     #     flat_dict_all_data[final_pdu_name + '.LRRF_ETH_LOOK_INDEX']

            #     # flat_dict_all_data[to_create_dict_path +
            #     #                    '.GROUP_DIFF_TIME'] = \
            #     #     (np.array(flat_dict_all_data[to_compare_col]) -
            #     #      np.array(flat_dict_all_data[req_cols_dict[Erste_schluessel]
            #     #                                  ]
            #     #               ))

            list_vals = [flat_dict_all_data[values] for values in key_val]
            list_vals = [np.reshape(row, (len(row), 1))
                         if len(row) == np.size(row)
                         else row
                         for row in list_vals]

            # unique_vals_cond = max(list_vals,
            #                        key=len).__len__() != min(list_vals, key=len).__len__()
            # unique_vals_len = len(np.unique(
            #     [np.shape(item) for item in list_vals], axis=0))
            # if unique_vals_cond and unique_vals_len > 1:
            #     print(f'{key},\t {unique_vals_len}')
            # print('&&&&&&&&& \n', f'{key}',  np.unique(
            #     [np.shape(item) for item in list_vals], axis=0),
            # )
            # print(f'&&&&&&&&&&&&& {item}\n')

            counter = 0
            while max(list_vals,
                      key=len).__len__() != min(list_vals, key=len).__len__():

                # RB 27082024, on discussion with Vinay and Aditya

                if self.CAN_flat and not self.FLR:

                    try:

                        check_str_list = ['MRR_FL_',
                                          'MRR_FR_', 'MRR_RL_', 'MRR_RR_',]

                        check_str_dict = {'MRR_FL_': 'MRR_FL_Header_Status_Radar.cTime',
                                          'MRR_FR_': 'MRR_FR_Header_Status_Radar.cTime',
                                          'MRR_RL_': 'MRR_RL_Header_Status_Radar.cTime',
                                          'MRR_RR_': 'MRR_RR_Header_Status_Radar.cTime',
                                          }
                        boolean_substring = list(map(key.__contains__,
                                                     check_str_dict.keys()))
                        boolean_index = np.argwhere(
                            boolean_substring).flatten()
                        req_placholder = check_str_list[boolean_index[0]]

                        req_path = key.split(
                            '.')[0] + '.' + check_str_dict[req_placholder]
                        comp_path_list = ['.'.join(raw_path.split('.')[:-1]) + '.cTime'
                                          for raw_path in key_val]
                        c_diff = []
                        for comp_path in comp_path_list:

                            # print('\n*********',
                            #       f'req_path :{req_path} \t comp_path : {comp_path}')
                            # assert req_path in flat_dict_all_data.keys(), \
                            #     f'ERROR HERE {req_path}'
                            append_item = np.abs(flat_dict_all_data[req_path][0]
                                                 - flat_dict_all_data[comp_path][0])
                            c_diff.append(append_item)

                        # c_diff = [flat_dict_all_data[req_path][0]
                        #           - flat_dict_all_data[comp_path][0]
                        #           for comp_path in comp_path_list]
                        add_col_index = [0 if c_diff_item > 0.01 else -1 for
                                         c_diff_item in c_diff]
                    except:
                        add_col_index = [-1 for _ in range(len(list_vals))]

                elif not self.CAN_flat and self.FLR:

                    try:
                        check_str_dict = {'FLR_':
                                          'PDU_LRRF_to_ADCAM_LRRF_Header_Timestamps.header.time', }

                        req_placholder = 'FLR_'

                        req_path = key.split(
                            '.')[0] + '.' + check_str_dict[req_placholder]

                        comp_path_list = ['.'.join(raw_path.split('.')[:-1]) + '.time'
                                          for raw_path in key_val]
                        c_diff = []
                        for comp_path in comp_path_list:

                            # print('\n*********',
                            #       f'req_path :{req_path} \t comp_path : {comp_path}')
                            # assert req_path in flat_dict_all_data.keys(), \
                            #     f'ERROR HERE {req_path}'
                            append_item = np.abs(flat_dict_all_data[req_path][0]
                                                 - flat_dict_all_data[comp_path][0])
                            c_diff.append(append_item)

                        # c_diff = [flat_dict_all_data[req_path][0]
                        #           - flat_dict_all_data[comp_path][0]
                        #           for comp_path in comp_path_list]
                        add_col_index = [0 if c_diff_item > 0.05 else -1 for
                                         c_diff_item in c_diff]
                    except:

                        add_col_index = [-1 for _ in range(len(list_vals))]

                else:

                    # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')

                    add_col_index = [-1 for _ in range(len(list_vals))]
                ######################################################

                list_vals = stack_padding(list_vals, add_col_index)

                if counter == 0:
                    debug_str_append = debug_str_append + '\n' +\
                        '\n multiple message lengths are found at signal ' + key \
                        + '\n padded shorter arrays for now'
                counter = counter+1

            if not self.CAN_flat and self.FLR:

                to_create_dict_path = 'ERR_DEBUG_DATA'

                req_pdu_id_list = ['0x100'+str(item).zfill(2)
                                   for item in range(1, 12)
                                   ]
                final_pdu_id = '0x10012'

                final_pdu_name = 'PDU_LRRF_to_ADCAM_LRRF_Header_Timestamps'

                final_pdu_dict = {final_pdu_id: final_pdu_name}

                path_list = [values.split('.')[0]
                             for values in key_val]

                pdu_id_list = [np.unique(
                    flat_dict_all_data[path + '.header.pdu_id'])[0]
                    for path in path_list]

                req_cols_dict = {pdu_id: col + '.header.time'
                                 for pdu_id, col in zip(pdu_id_list, path_list)
                                 if pdu_id in req_pdu_id_list}

                to_compare_col = final_pdu_name + '.header.time'

                sein_pdu_col_list = list(req_cols_dict.keys())
                sein_pdu_col_list.sort()
                Erste_schluessel = sein_pdu_col_list[0]

                req_indices_dict = {}

                for (key_req, col_req), key_val_item in zip(req_cols_dict.items(),
                                                            key_val):
                    # !!!: assumption is that lengths of to_compare_col and col are same
                    bool_array = np.greater(
                        flat_dict_all_data[to_compare_col],
                        flat_dict_all_data[col_req])

                    req_indices_dict[key_req] = bool_array

                    flat_dict_all_data[key_val_item] = np.array(
                        flat_dict_all_data[key_val_item])

                    flat_dict_all_data[key_val_item][np.argwhere(
                        np.logical_not(bool_array))] = -999

            if np.size(list_vals[0]) == len(list_vals[0]) and len(list_vals[0]) == 1:

                axis_req = 0

            else:
                axis_req = 1

            map_dict[key] = np.concatenate(list_vals, axis=axis_req)

        if self.CAN_flat:

            check_str_list = ['MRR_FL_Detection',
                              'MRR_FR_Detection',
                              'MRR_RL_Detection',
                              'MRR_RR_Detection',]

            check_str_dict = {'MRR_FL_Detection':
                              'MRR_FL_Header_Status_Radar.cTime',
                              'MRR_FR_Detection':
                                  'MRR_FR_Header_Status_Radar.cTime',
                              'MRR_RL_Detection':
                                  'MRR_RL_Header_Status_Radar.cTime',
                              'MRR_RR_Detection':
                                  'MRR_RR_Header_Status_Radar.cTime',
                              }

            req_keys_flat_list = list(set(['.'.join(item.split('.')[:2]) + '.cTime'
                                           for item in flat_dict_all_data.keys()
                                           for item2 in check_str_list if item2 in item]))
            req_indices_dict = {'FL': '',
                                'FR': '',
                                'RL': '',
                                'RR': '',
                                }

            for (key_req, col_req), check_key in zip(req_indices_dict.items(),
                                                     check_str_list):

                bool_array_list = []
                if key_req not in check_key:
                    check_key = [item_check
                                 for item_check in check_str_list
                                 if key_req in item_check][0]
                # print(check_key)

                to_compare_col = check_str_dict[check_key]

                look_index_col = [col_head
                                  for col_head in flat_dict_all_data.keys()
                                  if (to_compare_col.split('.')[0] in col_head)
                                  and ('CAN_LOOK_INDEX' in col_head)]

                if len(look_index_col) > 0:
                    look_index_col = look_index_col[0]

                    # print(to_compare_col)
                    to_compare_col = [item_comp
                                      for item_comp in flat_dict_all_data.keys()
                                      if to_compare_col in item_comp][0]

                    to_create_dict_path = to_compare_col.split(
                        '.')[0] + '.ERR_DEBUG_DATA'

                    # print('\n\n', to_compare_col)
                    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

                    all_req_cols = [req_col_
                                    for req_col_ in req_keys_flat_list
                                    if check_key in req_col_]

                    all_req_cols.sort()
                    # print(all_req_cols)
                    for req_col_ in all_req_cols:
                        if (len(flat_dict_all_data[to_compare_col])
                                != len(flat_dict_all_data[req_col_])):

                            # max_len = max(len(flat_dict_all_data[to_compare_col]),
                            #               len(flat_dict_all_data[req_col_]))

                            max_len = len(flat_dict_all_data[to_compare_col])

                            min_len = min(len(flat_dict_all_data[to_compare_col]),
                                          len(flat_dict_all_data[req_col_]))

                            bool_array = np.zeros(max_len, dtype=bool)

                            col_req_vals = flat_dict_all_data[req_col_]

                            bool_array[:min_len] = np.greater(
                                flat_dict_all_data[to_compare_col][:min_len],
                                col_req_vals[:min_len])

                            # continue
                        else:

                            col_req_vals = flat_dict_all_data[req_col_]
                            bool_array = np.greater(
                                flat_dict_all_data[to_compare_col],
                                col_req_vals)
                            bool_array_list.append(bool_array)

                    req_indices_dict[key_req] = \
                        np.logical_and.reduce(bool_array_list)

                    flat_dict_all_data[to_create_dict_path +
                                       f'.{key_req}_' + 'IS_ALL_GROUP'] = req_indices_dict[key_req]

                    flat_dict_all_data[to_create_dict_path +
                                       f'.{key_req}_' + 'LOOK_INDEX'] = \
                        flat_dict_all_data[look_index_col]

                    # max_len = max(len(flat_dict_all_data[to_compare_col]),
                    #               len(flat_dict_all_data[all_req_cols[0]]))

                    max_len = len(flat_dict_all_data[to_compare_col])

                    min_len = min(len(flat_dict_all_data[to_compare_col]),
                                  len(flat_dict_all_data[all_req_cols[0]]))
                    group_diff_array = np.zeros(max_len, dtype=float)

                    group_diff_array[:min_len] = (
                        np.array(flat_dict_all_data[to_compare_col])[:min_len] -
                        np.array(flat_dict_all_data[all_req_cols[0]
                                                    ]
                                 )[:min_len]
                    )

                    flat_dict_all_data[to_create_dict_path +
                                       f'.{key_req}_' + 'GROUP_DIFF_TIME'] = \
                        group_diff_array

        if self.FLR:

            to_create_dict_path = 'ERR_DEBUG_DATA'

            req_pdu_id_list = ['0x100'+str(item).zfill(2)
                               for item in range(1, 12)
                               ]
            final_pdu_id = '0x10012'

            final_pdu_name = 'PDU_LRRF_to_ADCAM_LRRF_Header_Timestamps'

            final_pdu_dict = {final_pdu_id: final_pdu_name}

            # path_list = [values.split('.')[0]
            #              for values in key_val]

            path_list = [col for col in flat_dict_all_data.keys()
                         if '.header.pdu_id' in col]

            pdu_id_list = [np.unique(
                flat_dict_all_data[path])[0]
                for path in path_list]

            req_cols_dict = {pdu_id: col.split('.')[0] + '.header.time'
                             for pdu_id, col in zip(pdu_id_list, path_list)
                             if pdu_id in req_pdu_id_list}

            to_compare_col = final_pdu_name + '.header.time'

            sein_pdu_col_list = list(req_cols_dict.keys())
            sein_pdu_col_list.sort()
            Erste_schluessel = sein_pdu_col_list[0]

            req_indices_dict = {}

            for key_req, col_req in req_cols_dict.items():
                # !!!: assumption is that lengths of to_compare_col and col are same

                # print(key_req, col_req)

                # print(len(flat_dict_all_data[to_compare_col]),
                #       len(flat_dict_all_data[col_req]))
                if (len(flat_dict_all_data[to_compare_col])
                        != len(flat_dict_all_data[col_req])):

                    # col_req_vals = resample(flat_dict_all_data[col_req],
                    #                         len(flat_dict_all_data[to_compare_col]))

                    continue
                    # print('&&&&&', len(flat_dict_all_data[to_compare_col]),
                    #       len(col_req_vals))
                else:

                    col_req_vals = flat_dict_all_data[col_req]

                    bool_array = np.greater(
                        flat_dict_all_data[to_compare_col],
                        col_req_vals)

                    req_indices_dict[key_req] = bool_array

            flat_dict_all_data[to_create_dict_path +
                               '.LRRF_ETH_LOOK_INDEX'] = \
                flat_dict_all_data[final_pdu_name + '.LRRF_ETH_LOOK_INDEX']

            flat_dict_all_data[to_create_dict_path +
                               '.GROUP_DIFF_TIME'] = \
                (np.array(flat_dict_all_data[to_compare_col]) -
                 np.array(flat_dict_all_data[req_cols_dict[Erste_schluessel]
                                             ]
                          ))

            flat_dict_all_data[to_create_dict_path +
                               '.IS_ALL_GROUP'] = \
                np.logical_and.reduce(list(req_indices_dict.values()))

        for k in val:
            flat_dict_all_data.pop(k, None)

        flat_dict_all_data = {**flat_dict_all_data, **map_dict}

        return flat_dict_all_data, debug_str_append

    def _flr4_ad5_parser(self, log_path):

        available_protocols = SerializerAPI.get_avaiable_protocolLib_versions()
        # protocol_version = [item
        #                     for item in available_protocols
        #                     if 'ENET_AD5_S1_01_05_2023' in item][0]

        protocol_version = [item for item in available_protocols
                            if item in
                            'THUNDER_' +
                            os.path.split(self.req_arxml_name)[
                                1].replace('-', '_')
                            ][0]

        # if 'THUNDER_ENET_AD5_ECU_Composition_S1_11_28_2023'
        # in item][0]
        decoding_settings = [SerializerAPI.DecodingSettings(
            protocol_type=SerializerAPI.ProtocolType('AUTOSARPDU'),
            protocol_version=protocol_version
        )]

        ser_msgs = []
        messages = [*SerializerAPI.Serializer(
            in_path=log_path,
            decoding_settings=decoding_settings)]

        for msg in messages:

            try:
                ser_msgs.append(json.loads(msg))
            except:
                ser_msgs.append(yaml.load(msg,
                                          Loader=yaml.Loader))

        # ser_msgs = [
        #     json.loads(msg) for msg in SerializerAPI.Serializer(
        #         in_path=log_path,
        #         decoding_settings=decoding_settings)]

        # flat_dict = [self.flatten_dict(item, separator='.')
        #              for item in ser_msgs]

        flat_dict = [self.flatten_dict2(item, separator='.')
                     for item in ser_msgs]

        flat_dict = [self.pop_dict_vals(item_dict, edit_=True)
                     for item_dict in flat_dict]

        unique_data = [np.unique(['.'.join(item.split('.')[:2])
                                  for item in dict_in.keys()
                                  if len(item.split('.')[:2]) > 1])
                       for dict_in in flat_dict]

        req_list_orig = [
            'ID',
            'Protocol'
        ]
        time_col = ['Timestamp']

        req_list = time_col + req_list_orig

        _ = [item.update({key + '.header.'+list_item: item[list_item]
                          for key in item2 for list_item in req_list})
             for item, item2 in zip(flat_dict, unique_data)]

        flat_dict = [{key: value for key, value in list_dict.items()
                      if key not in req_list} for list_dict in flat_dict]

        flat_dict = self._process_flat_dict(flat_dict)

        flat_dict_all_data = self.merge_list_of_dicts(flat_dict)

        flat_dict_all_data = {key: np.unique(val)
                              if any(substring in key
                                     for substring in req_list_orig)
                              else val
                              for key, val in flat_dict_all_data.items()

                              }

        flat_dict_all_data = {key.replace('.Timestamp', '.cTime'): val
                              if any(substring in key
                                     for substring in time_col)
                              else val
                              for key, val in flat_dict_all_data.items()

                              }

        flat_dict_all_data, debug_str_append = \
            self._process_flat_dict_all(flat_dict_all_data)

        if bool(debug_str_append):

            self.debug_str = self.debug_str \
                + '\n FLR4 AD5 error :' + debug_str_append

        flr4_dict = self.nest_dict(flat_dict_all_data)

        return flr4_dict  # , flat_dict, flat_dict_all_data, ser_msgs

    def flatten_dict(self, nested_dict, separator='.', prefix=''):
        res = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                res.update(self.flatten_dict(value,
                                             separator, prefix + key + separator))
            else:
                res[prefix + key] = value
        return res

    def flatten_dict2(self, nested_dict, separator='.', prefix=''):
        res = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                out_ = self.flatten_dict2(value,
                                          separator,
                                          prefix + key + separator)

                res.update(out_)

            elif isinstance(value, list or tuple or pd.Series or np.ndarray):

                # dum_dict = {}
                dum_list = []

                for i, val in enumerate(value):

                    dum_list.append(self.flatten_dict2(val,
                                                       separator,
                                                       prefix + key + separator))

                out_ = pd.DataFrame(dum_list).to_dict(orient="list")

                out_ = self.pop_dict_vals(out_, edit_=True)

                res.update(out_)

            else:

                res[prefix + key] = value

        # return_val = {key: self.pop_dict_vals(val, edit_=True)
        #               for key, val in res.items()}
        return_val = res

        return return_val

    def merge_list_of_dicts(self, list_of_dicts):
        # First, figure out which keys are present.
        keys = set().union(*list_of_dicts)
        # Build a dict with those keys, using a list comprehension to
        # pull the values from the source dicts.
        return {
            k: np.array([d[k] for d in list_of_dicts if k in d])
            for k in keys
        }


class swiftNavGps:

    def __init__(self, ):

        self.req_bus_channel = int('0x16f32', 16)
        self.check_dict_swift_nav = {}

    def _decode_sbp(self,
                    row,
                    col_name='ETH_Frame.DataBytes',
                    cTime_col='timestamps'):

        udp_base = 28

        row_val = row[col_name]
        time_stamp = row[cTime_col]

        if row_val[udp_base] == 0x55:

            L = row_val[udp_base+5]
            payload = bytes(row_val[udp_base:udp_base+L+6+2])
            msg = SBP.unpack(payload)
            msg = dispatch(msg)

            decoded_dict = msg.to_json_dict()

            if 'payload' in decoded_dict:
                del decoded_dict["payload"]

            decoded_dict['cTime'] = time_stamp

            msg_type = decoded_dict['msg_type']

            return_val = [decoded_dict, msg_type]

        else:
            return_val = [{}, np.nan]

        return return_val

    def main(self,
             req_group_msg: list,
             file_path,
             group_msg_mapping_dict: dict = None):

        yop = MDF(file_path,)
        log_start_time = yop.header.start_time.timestamp()

        group_data_generator = yop.iter_groups(use_display_names=True,
                                               # raster=0.001
                                               # raw=True,
                                               # keep_arrays=True,
                                               time_from_zero=False,

                                               )
        group_data = []
        for data in group_data_generator:
            if ((not data.empty) and
                    ('ETH_Frame.ETH_Frame.BusChannel' in data.columns)):
                data = data.dropna(
                    subset=['ETH_Frame.ETH_Frame.BusChannel'], axis=0, )
                assert len(data['ETH_Frame.ETH_Frame.BusChannel'].unique()) == 1, \
                    'multiple channels exist in dataframe. check'
                if (data['ETH_Frame.ETH_Frame.BusChannel'].unique().item()
                        == self.req_bus_channel):
                    group_data.append(data)

        eth_df = group_data[0]
        eth_df = eth_df.reset_index()
        eth_df['timestamps'] = eth_df['timestamps'] + log_start_time

        # eth_df['decoded_msg_dicts'], eth_df['msg_type'] = \
        #     zip(*eth_df.apply(self._decode_sbp,
        #                       axis=0,
        #                       col_name = 'ETH_Frame.ETH_Frame.DataBytes' ))

        eth_df[['decoded_msg_dicts', 'msg_type']] = \
            eth_df.apply(self._decode_sbp,
                         axis=1,
                         result_type="expand",
                         col_name='ETH_Frame.ETH_Frame.DataBytes')

        req_df = eth_df[['timestamps',
                         'decoded_msg_dicts',
                         'msg_type']]

        _grouped_df = req_df.groupby('msg_type', group_keys=True)
        group_index_dict = _grouped_df.groups

        message_dict_all = {}

        self.check_dict_swift_nav = {key: True
                                     if key in list(group_index_dict.keys())
                                     else False
                                     for key in req_group_msg
                                     }

        # if int(group_key) in to_log_msg_IDs:

        #     self.swift_nav_check_dict[
        #         'swift_nav_msg_' + str(group_key)] = [True]

        for group_key, group_indices in group_index_dict.items():

            # print(f'%%%%%%%%%%%%%%%%%%%%%% 3525 *********** {iteration}')

            if group_key in req_group_msg:

                group_df = _grouped_df.get_group(group_key)

                list_of_dicts = group_df['decoded_msg_dicts'].tolist()

                if group_msg_mapping_dict is not None:

                    req_key = int(group_key)

                    if not req_key in group_msg_mapping_dict:

                        req_key = 'msg_' + str(group_key)
                    else:
                        req_key = group_msg_mapping_dict[int(group_key)]

                    message_dict_all[req_key] = {
                        k: np.array([dic[k] for dic in list_of_dicts])
                        for k in list_of_dicts[0]
                    }
                else:

                    message_dict_all[
                        'msg_' + str(group_key)] = {
                            k: np.array([dic[k] for dic in list_of_dicts])
                            for k in list_of_dicts[0]
                    }

        return message_dict_all, self.check_dict_swift_nav

    def main2(self,
              req_group_msg: list,
              file_path,
              group_msg_mapping_dict: dict = None):

        mf4_file = os.path.join(file_path)
        # Load the MDF file
        mdf = MDF(mf4_file)  # Replace with your actual file path

        # Extract relevant signals
        timestamps = mdf.get('TimeStamp').samples
        # sources = mdf.get('ETH_Frame.Source').samples
        # destinations = mdf.get('ETH_Frame.Destination').samples
        # ethertypes = mdf.get('ETH_Frame.EtherType').samples
        data_bytes_list = mdf.get('ETH_Frame.DataBytes').samples

        # Function to convert Ethernet data to SBP forma
        sbp_message = b''
        udp_base = 28
        for d1 in data_bytes_list:
            # sbp_message = bytes(d1)
            if d1[udp_base] == 0x55:
                L = d1[udp_base+5]
                sbp_message += bytes(d1[udp_base:udp_base+L+6+2])

        # Parse SBP messages
        message_dict_all = {}

        buf = memoryview(sbp_message)
        offset = 0
        while offset < len(buf):
            if buf[offset] != 0x55:
                offset += 1
                continue
            try:
                msg = SBP.unpack(buf[offset:])
                msg = dispatch(msg)
                message_dict = msg.to_json_dict()
                if 'payload' in message_dict:
                    del message_dict["payload"]
                if ('msg_type' in message_dict
                        and message_dict['msg_type'] in message_dict_all):
                    message_dict_all[
                        message_dict['msg_type']].append(message_dict)
                elif ('msg_type' in message_dict
                      and 'msg_type' not in message_dict_all):
                    message_dict_all[message_dict['msg_type']] = [message_dict]
                # elif 'msg_type' not in message_dict:
                #     continue

                offset += 6 + msg.length + 2  # header + payload + CRC
            except Exception as e:
                offset += 1

        # group_msgs = [258, 259, 522, 529, 536, 521, 532, 526,
        #               530, 525, 533, 540, 545, 65283, 65286, 65294, 535]

        out_dict = {'msg_' + str(key): {k: [dic[k]
                                            for dic in val]
                                        for k in val[0]}
                    for key, val in message_dict_all.items()
                    if key in req_group_msg}

        if group_msg_mapping_dict is not None:

            out_dict = {group_msg_mapping_dict[int(key.split('_')[-1])]
                        if int(key.split('_')[-1])
                        in list(group_msg_mapping_dict.keys())
                        else key: val
                        for key, val in out_dict.items()}

        return out_dict,


class mcipSignalExtraction(thunderSignalExtraction):

    def __init__(self, ):

        super().__init__()

        # import swiftNavGps
        self.swiftNavGps_obj = swiftNavGps()

        self.check_dict_swift_nav = {}

        self.swift_navgroup_msg_mapping_dict = {
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

        self.req_channels_b05 = [int('0x44d', 16),
                                 int('0x44e', 16),
                                 int('0x44f', 16),
                                 int('0x450', 16),
                                 int('0x460', 16),
                                 int('0x47e', 16),
                                 ]

        self.req_channels_b04 = [int('0x835', 16),
                                 int('0x836', 16),
                                 int('0x837', 16),
                                 int('0x848', 16),
                                 ]
        self.req_channels_b03 = [int('0x83f', 16),
                                 int('0x840', 16),
                                 int('0x8a3', 16),
                                 int('0x8a4', 16),
                                 int('0x7d2', 16),
                                 int('0x7d3', 16),
                                 # int('0xbb9', 16), #FlexRay
                                 int('0x3ea', 16),

                                 ]

        self.req_channels_r03 = [int('0x16f31', 16),
                                 int('0x16f32', 16),
                                 ]

        self.req_channels_p01 = [int('0x3e9', 16),
                                 int('0xf4664', 16),
                                 # int('0x5f6f680', 16),  # resim effect
                                 # int('0x7d1', 16),
                                 ]

        self.mudp_source = 23
        self.avi_source = 104

        self.colloquial_channel_map = {self.req_channels_p01[1]: 'p01 (b01)',
                                       self.req_channels_p01[0]: 'p01 (b01)',
                                       # self.req_channels_p01[2]: 'p01 (b01)',
                                       }

        self.colloquial_source_map = {self.mudp_source: 'mUDP',
                                      self.avi_source: 'AVI',
                                      }

        self.channel_source_stream_map_dict = {
            self.req_channels_p01[1]: {
                self.mudp_source:
                    {
                        16: "vse",
                        18: "ff_10ms",
                        # 32: "Feature_20ms",
                        19: 'ff_20ms',
                        80: "radar_FLR_IF",
                        81: "object_fusion",
                        82: "tracker_srr",
                        83: "tracker_FLR",
                        85: "sata_spp_otp",
                        86: "AVI_Message",
                        87: "AVI_Objectlist",
                        # 33: 'OLP',
                        114: 'olp',
                        115: 'trkr_internals_srr',
                        116: 'trkr_internals_flr',
                        117: 'rdr_srr_if',
                        # 17: "Debug_Stream1",
                        17: 'debug_stream_1',
                        # 84: "Debug_Stream2",
                        144: "debug_stream_3",
                        # 145: 'PROXI_CAR_CFG',
                        # 146: 'VEH_CALIB',
                    },
                self.avi_source: {
                    1: 'car_sns',
                        20: 'cam_cfg',
                        24: 'common',
                        25: 'ca',
                        26: 'failsafe',
                        31: 'ln_adj',
                        32: 'ln_app',
                        33: 'ln_host',
                        34: 'ln_re',
                        37: 'objects',
                        41: 'tsr',
                        40: 'tfl',
                        42: 'ca_debug',
                        43: 'failsafe_debug',
                        44: 'ln_adj_debug',
                        45: 'ln_app_debug',
                        46: 'ln_host_debug',
                        47: 'ln_re_debug',
                        50: 'objects_debug',
                        51: 'tsr_debug',
                        52: 'calib',
                        53: 'freespace',
                        54: 'tfl_debug',
                        56: 'parking_spaces'
                    }
            }
        }

        self.channel_source_stream_map_dict[
            self.req_channels_p01[0]] = self.channel_source_stream_map_dict[
                self.req_channels_p01[1]]

        # self.channel_source_stream_map_dict[
        #     self.req_channels_p01[2]] = self.channel_source_stream_map_dict[
        #         self.req_channels_p01[1]]

        self._mudp_stream_mapping = \
            self.channel_source_stream_map_dict[
                self.req_channels_p01[1]][self.mudp_source]

        self.avi_streams = \
            self.channel_source_stream_map_dict[
                self.req_channels_p01[1]][self.avi_source]

        self._srr_stream_mapping = {
            0: 'core_0',
            2: 'core_1',
            3: 'core_2',

        }

        self. combine_CAN_data_dets = False
        # self.debug_str = ''
        # self.CAN_flat = False
        # self.SPI_flat = False
        # self.FLR = False

        self.mudp_eth_dest_ports = np.array(
            [5003, 5141, 5555, 5556, 5557, 5558,
             10002, 49001, 49002, 49003, 49004, 49005, 30490])
        self.is_updated_udp_parser = True

    def main(self,
             b03_dict,
             b04_dict,
             b05_dict,
             p01_dict,
             r03_dict,
             log_path,
             run_mode: int = -999, **kwargs):

        sub_string_list = [
            '_b01', '_b02',
            '_b03', '_b04', '_b05',
            '_p01',
            '_r03', '_g03',
        ]

        log_path_only, log_name_only = os.path.split(log_path)

        boolean_substring = list(map(log_name_only.__contains__,
                                     sub_string_list))

        boolean_index = np.argwhere(boolean_substring).flatten()

        if not bool(list(boolean_index)):

            return_dict = {'error_message':
                           'File is not of bxx or p01 or r03 type. cannot be decoded'}

            self.debug_str = 'File is not of bxx or p01 or r03 type type. cannot be decoded'
            return return_dict, self.debug_str

        sub_string_list_tuple = [(sub_string_list[boolean_index[0]], item)
                                 for item in sub_string_list]

        path_list = [os.path.join(log_path_only,
                                  log_name_only.replace(*iter_tuple))
                     for iter_tuple in sub_string_list_tuple]

        b01_out, b02_out, b03_out, b04_out, b05_out = {}, {}, {}, {}, {}

        p01_out, r03_out = {}, {}

        if os.path.isfile(path_list[6]):
            req_swift_nav_path = path_list[6]
        if os.path.isfile(path_list[7]):
            req_swift_nav_path = path_list[7]

        swift_nav_only = True if run_mode == 312 else False

        if ((os.path.isfile(path_list[6])
            or os.path.isfile(path_list[7]))
                and bool(r03_dict)):

            existing_channels_r03_ETH = list(
                self._read_raw_data_mdf(req_swift_nav_path, True).keys())
            existing_channels_r03 = existing_channels_r03_ETH

            missing_channels = list(set(self.req_channels_r03)
                                    .difference(existing_channels_r03))

            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'r03 log missing channels {missing_channels} \n'
            r03_out, check_dict_swift_nav_instance = self.main_r03(r03_dict,
                                                                   req_swift_nav_path,
                                                                   self.swiftNavGps_obj
                                                                   )
            self.check_dict_swift_nav = {**self.check_dict_swift_nav,
                                         **check_dict_swift_nav_instance}

        p01_run_type = True if (run_mode
                                in [-999, 4, -3, -4, -6]
                                ) else False

        if os.path.isfile(path_list[5]):
            req_mudp_path = path_list[5]
        elif os.path.isfile(path_list[0]):
            req_mudp_path = path_list[0]

        if (os.path.isfile(path_list[5]) or
            os.path.isfile(path_list[0])) \
                and bool(p01_dict) and p01_run_type:

            existing_channels_p01_ETH = list(
                self._read_raw_data_mdf(req_mudp_path, True).keys())
            existing_channels_p01 = existing_channels_p01_ETH

            missing_channels = list(set(self.req_channels_p01)
                                    .difference(existing_channels_p01))
            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'p01 (b01) log missing channels {missing_channels} \n'
            p01_out = self.main_p01(p01_dict, req_mudp_path)

        b03_run_type = True if (run_mode
                                in [-999, 1, -1, -4, -5]
                                ) else False
        if os.path.isfile(path_list[2]) and bool(b03_dict) and b03_run_type:
            existing_channels_b03_CAN = list(self._read_raw_data_mdf(path_list[2],
                                                                     False).keys())
            existing_channels_b03_ETH = list(self._read_raw_data_mdf(path_list[2],
                                                                     True).keys())
            existing_channels_b03 = existing_channels_b03_CAN \
                + existing_channels_b03_ETH
            missing_channels = list(set(self.req_channels_b03)
                                    .difference(existing_channels_b03))

            check_dict_b03 = {key: True
                              if key in existing_channels_b03_CAN
                              else False
                              for key in self.req_channels_b03
                              }
            self.can_check_dict = {**self.can_check_dict, **check_dict_b03}

            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'b03 log missing channels {missing_channels} \n'
            b03_out, log_start_time = self.main_b03(b03_dict, path_list[2])

        b04_run_type = True if (run_mode
                                in [-999, 2, -1, -2, -6]
                                ) else False
        if os.path.isfile(path_list[3]) and bool(b04_dict) and b04_run_type:
            existing_channels_b04_CAN = list(self._read_raw_data_mdf(path_list[3],
                                                                     False).keys())
            existing_channels_b04_ETH = list(self._read_raw_data_mdf(path_list[3],
                                                                     True).keys())
            existing_channels_b04 = existing_channels_b04_CAN \
                + existing_channels_b04_ETH
            missing_channels = list(set(self.req_channels_b04)
                                    .difference(existing_channels_b04))
            check_dict_b04 = {key: True
                              if key in existing_channels_b04_CAN
                              else False
                              for key in self.req_channels_b04
                              }
            self.can_check_dict = {**self.can_check_dict, **check_dict_b04}
            if len(missing_channels) > 0:
                self.debug_str = self.debug_str \
                    + f'b04 log missing channels {missing_channels} \n'
            b04_out, log_start_time = self.main_b04(b04_dict, path_list[3])

        return_dict = merge_dicts(b03_out, b04_out)
        return_dict = merge_dicts(return_dict, p01_out)
        return_dict = merge_dicts(return_dict, r03_out)
        # return_dict = {**return_dict, **r03_out}

        return return_dict, self.debug_str

    def main_b03(self, b03_dict, log_path):

        can_input_dict = b03_dict['can_input_dict']

        mudp_dict = {'b03_ETH_decoding_notYetImplemented': ''}

        self.CAN_flat = True
        self.SPI_flat = False

        can_dict, can_dict_edited_keys, log_start_time, can_dict_2 = \
            self._extract_thunder_CAN(
                can_input_dict['can_db_path'],
                can_input_dict['db_signal_pairs'],
                can_input_dict['channel_name_pairs'],
                log_path,
            )

        can_data = can_dict_2 if self. combine_CAN_data_dets \
            else can_dict_edited_keys

        return_dict = {'mudp': {
            'mcip_demo': mudp_dict,
        },
            'can': can_data,
        }

        return return_dict, log_start_time

    def main_b04(self, b04_dict, log_path):

        can_input_dict = b04_dict['can_input_dict']

        self.CAN_flat = True
        self.SPI_flat = False

        can_dict, can_dict_edited_keys, log_start_time, can_dict_2 = \
            self._extract_thunder_CAN(
                can_input_dict['can_db_path'],
                can_input_dict['db_signal_pairs'],
                can_input_dict['channel_name_pairs'],
                log_path,
            )

        can_data = can_dict_2 if self. combine_CAN_data_dets \
            else can_dict_edited_keys

        return_dict = {
            'can': can_data,

        }

        return return_dict, log_start_time

    def main_p01(self,
                 p01_dict: dict,
                 log_path,
                 ):

        self._log_type = 'p01'

        self.stream_def_dir_path = p01_dict['stream_def_dir_path']

        mudp_input_dict = p01_dict['mudp_input_dict']

        group_data_dict = self._read_all_eth_data(log_path)

        mudp_dict = {}

        if mudp_input_dict['bus_channel'] in group_data_dict.keys():

            self.busID = mudp_input_dict['bus_channel']

            # self.stream_check_dict[
            #     f"busID_{mudp_input_dict['bus_channel']}"] = True

            req_data_mudp = group_data_dict[mudp_input_dict['bus_channel']]
            mudp_dict = self._extract_thunder_udp(
                req_data_mudp,
                is_tcp=mudp_input_dict['is_tcp'])
        else:

            self.debug_str = self.debug_str \
                + f"mudp channel {mudp_input_dict['bus_channel']} missing \n"
            # self.stream_check_dict[
            #     f"busID_{mudp_input_dict['bus_channel']}"] = False

        mudp_dict['bfname'] = os.path.split(log_path)[1]

        req_avi_keys = [key
                        for key in mudp_dict.keys()
                        if 'source_' + str(self.avi_source) in key]
        req_avi_keys_map = [self.avi_streams[int(key.split('_')[-1])]
                            for key in req_avi_keys]
        all_avi_data = {key_map:
                        mudp_dict.pop(key,
                                      'source_' +
                                      str(self.avi_source)
                                      + '_'+'stream_'
                                      + key_map + '_data_not_found'
                                      )
                        for key, key_map in zip(req_avi_keys,
                                                req_avi_keys_map)
                        }

        return_dict = {'mudp': {
            'mcip_demo': mudp_dict,
            'avi': {'flc': all_avi_data,
                    },
        },

        }

        return return_dict

    def main_r03(self,
                 r03_dict,
                 log_path,
                 swiftNavGps_obj):

        req_group_msg = r03_dict['req_group_msg']
        group_msg_mapping_dict = r03_dict['group_msg_mapping_dict']

        if group_msg_mapping_dict is None:

            group_msg_mapping_dict = self.swift_navgroup_msg_mapping_dict

        return_dict = {}

        (return_dict['swift_nav_gps'],
         check_dict_swift_nav_instance) = swiftNavGps_obj.main(
            req_group_msg,
            log_path,
            group_msg_mapping_dict)

        return return_dict, check_dict_swift_nav_instance


class bmwSignalExtraction(thunderSignalExtraction):

    def __init__(self, ):
        super().__init__()

        self.endianness_dict = {0: False,  # Little Endian
                                1: True  # Big Endian
                                }

        self.endianness_symbol_map = {0: '<',
                                      1: '>'
                                      }

        self.msg_ID_endianess_map = {}

        self.header_obj_dlt = DltHeader(
            # msbf = False
        )

        self.data_type_mapping = {
            'UINT8': np.uint8,
            'UINT16': np.uint16,
            'UINT32': np.uint32,
            'UINT64': np.uint64,
            'SINT8': np.int8,
            'SINT16': np.int16,
            'SINT32': np.int32,
            'SINT64': np.int64,
            'FLOA16': np.float16,
            'FLOA32': np.float32,
            'FLOA64': np.float64,
            'STRG_UTF8': np.str_,
        }

        self.NVDLT_HEADER_SIZE_BYTES = 16

    def _parse_xml(self, req_file_path):

        conf_factory = SimpleConfigurationFactory()

        ecu_name_replacement = None
        plugin_file = None
        fibex_parser_obj = FibexParser(plugin_file, ecu_name_replacement)
        verbose = True

        tree = xml.etree.ElementTree.parse(req_file_path)

        root = tree.getroot()

        fibex_parser_obj.parse_file(
            conf_factory, req_file_path, verbose=verbose)

        all_frames = root.findall(
            './/fx:FRAMES/fx:FRAME', fibex_parser_obj.__ns__)

        all_frames_id_order_dict = {frame.attrib['ID']: idx
                                    for idx, frame in enumerate(all_frames)}

        all_frames_length_dict = {int(frame.attrib['ID'][3:]):
                                  int(frame.find('./fx:BYTE-LENGTH',
                                                 fibex_parser_obj.__ns__).text)
                                  for frame in all_frames}

        total_data_length = np.sum(list(all_frames_length_dict.values()))

        frame_data_pairs_3 = {(all_frames_id_order_dict[frame.__id__],
                               frame.__id__,
                               frame.name()):
                              [(pdu_key.pdu().__id__,
                               pdu_key.pdu().__short_name__,
                                pdu_key.pdu().__byte_length__,
                               pdu_key.pdu().signal_instances_sorted_by_bit_position()[
                                  0].__signal__.__name__
                               if len(pdu_key.pdu().signal_instances_sorted_by_bit_position()) > 0
                               else None)
                              for pdu_key in frame.pdu_instances().values()]
                              for frame in fibex_parser_obj.__frames__.values()}

        message_ID_value_pairs = dict()

        for key, val in frame_data_pairs_3.items():
            message_ID = int(key[1][3:])
            message_ID_value_pairs[message_ID] = []
            for item in val:

                if item[3] is not None:

                    message_ID_value_pairs[message_ID].append((f'{key[2]}.{item[1]}',
                                                               self.data_type_mapping[item[3][2:]]
                                                               ))

                else:
                    continue

        return message_ID_value_pairs, all_frames_length_dict

    def _read_header_data(self, payload, req_msg_ID: bool = False):

        if payload[0] == 0:
            header_type = ''.join(np.array(
                list(np.binary_repr(
                    payload[0]).zfill(8))).astype(str))
        else:

            header_type = bin(payload[0])[2:]
        # from Log and Trace Protocol Specification AUTOSAR FO R20-11
        # endianness = int(header_type[::-1][1]) #big endian like in binary conversion
        # little endian like in binary conversion
        endianness = int(header_type[1])

        header_data = self.header_obj_dlt.create_from(f=ArrayUploader(payload),
                                                      msbf=self.endianness_dict[endianness])

        if req_msg_ID:

            header_start = 4

            if header_data.standard.has_weid():

                header_start += 4

            if header_data.standard.has_wsid():

                header_start += 4

            if header_data.standard.has_wtms():

                header_start += 4

            return_val = payload[header_start:header_start +
                                 4].view(np.uint32)[0]

            self.msg_ID_endianess_map[return_val] = \
                self.endianness_symbol_map[endianness]

        else:

            return_val = header_data

        return return_val

    def _create_buffer_dtype(self, req_ID,
                             message_ID_value_pairs,
                             # endianness: str = '<'
                             ):

        req_value_data = message_ID_value_pairs[req_ID]

        # req_dtype_list = [endianness + dtype[1] for dtype in req_value_data]

        # complete_dtype = ','.join(req_dtype_list)

        complete_dtype = req_value_data

        # complete_dtype = [(item[0], item[1]) for item in req_value_data]

        return complete_dtype

    def _demux_recursive(self,
                         payload,
                         mux_series,
                         append_list,
                         all_frames_length_dict
                         ):
        if len(payload) == 0:
            return 0
        else:
            try:
                message_ID = self._read_header_data(payload, True)
                debug_msg = f'DEMUX MESSAGE ID {message_ID} '
                print(f'&&&&&&&&&&& {debug_msg}')
                message_len = all_frames_length_dict[message_ID]
                payload_demux = payload[:self.NVDLT_HEADER_SIZE_BYTES+message_len]

                series_copy = mux_series.copy()
                series_copy["message_ID"] = message_ID
                series_copy["udp_payload"] = payload_demux
                series_copy["udp_payload_length"] = message_len

                append_list.append(series_copy)

                payload = payload[self.NVDLT_HEADER_SIZE_BYTES+message_len:]

                self._demux_recursive(payload,
                                      mux_series,
                                      append_list,
                                      all_frames_length_dict)

            except:

                debug_msg = f'cannot be demuxed. ' + \
                    'So all subsequent demuxing is being skipped'
                print(f'&&&&&&&&&&& {debug_msg}')
                self.debug_str += debug_msg
                payload = []
                self._demux_recursive(payload,
                                      mux_series,
                                      append_list,
                                      all_frames_length_dict)

    def _helper_demux_udp_data(self,
                               series_row,
                               append_list,
                               all_frames_length_dict
                               ):

        payload = series_row['udp_payload']

        self._demux_recursive(payload,
                              series_row,
                              append_list,
                              all_frames_length_dict)

        return

    def _demux_udp_data(self,
                        udp_df,
                        all_frames_length_dict):

        udp_df['udp_frame_bytes_present'] = udp_df['udp_payload'].apply(
            lambda x: x.nbytes - self.NVDLT_HEADER_SIZE_BYTES)

        udp_df['udp_frame_bytes_expected'] = udp_df['message_ID'].apply(
            lambda x: all_frames_length_dict[x] if x in all_frames_length_dict else -999)

        udp_df_mux = udp_df.query('udp_frame_bytes_present != ' +
                                  'udp_frame_bytes_expected ' +
                                  'and udp_frame_bytes_expected != -999')

        demux_req_indices = udp_df_mux.index

        if len(list(demux_req_indices)) > 0:

            demux_append_list = []
            udp_df_mux.apply(lambda x: self._helper_demux_udp_data(
                x, demux_append_list, all_frames_length_dict), axis=1)

            demux_append_list_df = pd.DataFrame(demux_append_list)

            demux_append_list = []

            udp_df = udp_df.drop(demux_req_indices)

            udp_df = pd.concat([udp_df,
                                demux_append_list_df],
                               axis=0).sort_values(
                                   ["timestamps", "message_ID"])

            udp_df = udp_df.reset_index().drop(columns=[
                "index"])

        return udp_df

    def _read_raw_bmw_udp_data(self, udp_df,
                               to_parse_xml: bool = False,
                               req_xml_file_path: str = None,
                               **kwargs):

        if (to_parse_xml
           and (('message_value_pairs_path' in kwargs
                and
                'all_frames_length_dict_path' in kwargs
                 )
                and not os.path.isfile(kwargs['message_value_pairs_path'])
                and not os.path.isfile(kwargs['all_frames_length_dict_path']))):

            assert os.path.isfile(req_xml_file_path), \
                'xml file path is not correct or not provided. please check'

            (message_ID_value_pairs,
             all_frames_length_dict) = self._parse_xml(req_xml_file_path)

            if ('message_value_pairs_path' in kwargs
                and
                'all_frames_length_dict_path' in kwargs
                ):

                with open(kwargs['message_value_pairs_path'], 'wb') as file:

                    # A new file will be created
                    pickle.dump(message_ID_value_pairs, file)

                with open(kwargs['all_frames_length_dict_path'], 'wb') as file:

                    # A new file will be created
                    pickle.dump(all_frames_length_dict, file)

        else:
            if ('message_value_pairs_path' in kwargs
                and
                'all_frames_length_dict_path' in kwargs
                ):

                assert os.path.isfile(kwargs['message_value_pairs_path']), \
                    f'{kwargs["message_value_pairs_path"]} is not present'
                assert os.path.isfile(kwargs['all_frames_length_dict_path']), \
                    f'{kwargs["all_frames_length_dict_path"]} is not present'

                with open(kwargs['message_value_pairs_path'], 'rb') as file:

                    # A new file will be created
                    message_ID_value_pairs = pickle.load(file)

                with open(kwargs['all_frames_length_dict_path'],
                          'rb') as file:

                    # A new file will be created
                    all_frames_length_dict = pickle.load(file)

            else:

                message_ID_value_pairs = kwargs['message_ID_value_pairs']
                all_frames_length_dict = kwargs['all_frames_length_dict']

        udp_df = self._demux_udp_data(udp_df, all_frames_length_dict)

        buffer_dtype_dict = {}

        for msg_ID in udp_df['message_ID'].unique():

            if msg_ID in message_ID_value_pairs.keys():

                buffer_dtype_dict[msg_ID] = \
                    self._create_buffer_dtype(msg_ID,
                                              message_ID_value_pairs)

        msg_ID_grouped_df = udp_df.groupby('message_ID', group_keys=True)
        group_index_dict = msg_ID_grouped_df.groups

        decoded_data_dict_final = {}

        for group_key, group_indices in group_index_dict.items():

            print(f'&&&&&&&&&&&&&&&&&&&&&&&  {group_key}')

            if not group_key in message_ID_value_pairs.keys():

                debug_str = f'\n Frame/Message ID {group_key} ' +\
                    'is not present in xml, skipping for now \n'

                print(debug_str)

                self.debug_str += debug_str

                continue

            group_df = msg_ID_grouped_df.get_group(group_key)

            buffer_dtype = buffer_dtype_dict[group_key]

            if group_key in self.msg_ID_endianess_map:
                # buffer_dtype = np.dtype(buffer_dtype).newbyteorder(
                #     self.msg_ID_endianess_map[group_key])
                buffer_dtype = np.dtype(buffer_dtype).newbyteorder(
                    '<')
            else:
                debug_str = \
                    f'Cannot determine the endianness of message ID {group_key}' +\
                    'decoding as big endian'
                print(debug_str)
                self.debug_str += debug_str
                buffer_dtype = np.dtype(buffer_dtype).newbyteorder(
                    '>')

            decode_data_names = [name[0]
                                 for name in
                                 message_ID_value_pairs[group_key]]

            # decoded_data_dict = {}

            try:

                decoded_data = group_df['udp_payload'].apply(
                    lambda x: self._decode_payload_dlt(x,
                                                       buffer_dtype,
                                                       decode_data_names)
                ).tolist()

                decoded_data_dict_msg_ID = \
                    {k: [dic[k] for dic in decoded_data]
                        for k in decoded_data[0]}

                decoded_data_dict_final[group_key] = decoded_data_dict_msg_ID
            except Exception as e:

                decoded_data_dict_final[group_key] = f'Cannot decode. Error is {e}'

        return decoded_data_dict_final

    def _decode_payload_dlt(self, row, buffer_dtype, decode_data_names):

        decoded_data = \
            np.frombuffer(row[self.NVDLT_HEADER_SIZE_BYTES:].tobytes(),
                          dtype=buffer_dtype)[0]

        decoded_data_dict = {key: val
                             for key, val in
                             zip(decode_data_names, decoded_data)
                             }

        return decoded_data_dict

    def _extend_udp_data(self, udp_df):

        udp_df['message_ID'] = udp_df['udp_payload'].apply(
            lambda x: self._read_header_data(x, req_msg_ID=True))

        return udp_df

    def _extract_bmw_udp(self, mudp_raw_data,
                         is_tcp: bool = False,
                         to_parse_xml: bool = False,
                         req_xml_file_path: str = None,
                         **kwargs
                         ):

        eth_df = self._read_raw_eth_data(mudp_raw_data)

        _, ipv4_df = self._read_raw_ipv4_data(eth_df)

        if is_tcp:
            tcp_df = self._read_raw_tcp_data(ipv4_df)

        udp_df = self._read_raw_udp_data(ipv4_df)

        udp_df = self._extend_udp_data(udp_df)

        output_dict = self._read_raw_bmw_udp_data(udp_df,
                                                  to_parse_xml,
                                                  req_xml_file_path,
                                                  **kwargs)

        return output_dict


class ArrayUploader(IOBase):
    # set this up as a child of IOBase because boto3 wants an object
    # with a read method.
    def __init__(self, array):
        # get the number of bytes from the name of the data type
        # this is a kludge; make sure it works for your case
        dbits = re.search('\d+', str(np.dtype(array.dtype))).group(0)
        dbytes = int(dbits) // 8
        self.nbytes = array.size * dbytes
        self.bufferview = (ctypes.c_char*(self.nbytes)
                           ).from_address(array.ctypes.data)
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, pos):
        self._pos = pos

    def read(self, size=-1):
        if size == -1:
            return self.bufferview.raw[self._pos:]
        old = self._pos
        self._pos += size
        return self.bufferview.raw[old:self._pos]


if __name__ == '__main__':
    import time
    from functools import reduce
    import psutil
    import warnings
    import contextlib
    import joblib
    from joblib import Parallel, delayed, parallel_backend, parallel_config
    from tqdm import tqdm
    import scipy as sp
    warnings.filterwarnings("ignore")

    is_cmd_line = False

    if is_cmd_line:

        parser = argparse.ArgumentParser(
            prog='MATGEN',
            description='Extracts a .mat from data logged in an MF4 file',
            epilog='Author: revanth.bhattaram@aptiv.com')

        parser.add_argument(
            '-c', '--config_path', help='Full path to where input config file is present')
        parser.add_argument('-f', '--file', help='Full path of input MF4 file')
        parser.add_argument('-r', '--run_mode',
                            help='''run mode required.
                            run_mode == -999 -> runs all bus, deb and ref
                            run_mode == 1 -> runs bus only
                            run_mode == 2 -> runs deb only
                            run_mode == 3 -> runs ref only
                            run_mode == -1 -> runs bus and deb
                            run_mode == -2 -> runs deb and ref
                            run_mode == -3 -> runs ref and bus'''
                            )

        parser.add_argument('-m', '--mat_path', help='path to mat file',
                            required=False, default="")

        args = parser.parse_args()

        # print(f'Printing args {args}')
        configuration_path = args.config_path
        log_path = args.file
        run_mode = int(args.run_mode)
        mat_path = args.mat_path

        # print(f'Printing configuration_path {configuration_path}')

        assert os.path.isfile(configuration_path), \
            'Input config file path is incorrect or config file missing'

        with open(configuration_path) as stream:

            yaml_input_dict = yaml.safe_load(stream)

        bus_dict = yaml_input_dict['bus_dict']
        deb_dict = yaml_input_dict['deb_dict']
        ref_dict = yaml_input_dict['ref_dict']

        thun_obj = thunderSignalExtraction()

        kwargs_main = {}

        if run_mode == 312:

            # assert os.path.isfile(
            #     mat_path), 'Send the path to mat file as cmd arg'

            from scipy.io import loadmat as load_mat_scipy

            kwargs_load_mat_scipy = {  # 'struct_as_record' : False,
                # 'squeeze_me' : True,
                'simplify_cells': True,
                'verify_compressed_data_integrity': True
            }

            if os.path.isfile(mat_path):

                kwargs_main['mat_data'] = load_mat_scipy(
                    mat_path,
                    **kwargs_load_mat_scipy
                )
            else:

                run_mode = 1

        out_dict, debug_str = thun_obj.main(bus_dict,
                                            deb_dict,
                                            ref_dict,
                                            log_path,
                                            run_mode=run_mode,
                                            **kwargs_main)
        sp.io.savemat(log_path[:-4]+'.mat', out_dict,
                      **{'long_field_names': True,
                          'oned_as': 'column'},
                      do_compression=True
                      )

    else:

        def secondsToStr(t):
            return "%d:%02d:%02d.%03d" % \
                reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                       [(t*1000,), 1000, 60, 60])

        def process_memory():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss, mem_info.vms

        bmw_data = False
        thunder_data = False
        mcip_data = True

        if bmw_data:

            start_time = time.time()
            mem_before_phy, mem_before_virtual = process_memory()

            log_name = (
                # 'BMWSP2025_ENDURANCE_IPN_NVDLT_IPC_WBA21BY0307L57814_20250107_134451_0004.MF4'
                # '2024-06-11_16-44-30_2024-06-11_16-44-50_552348_9P52514_IPN_NVDLT_IPC_000017.MF4'
                '2025-01-07_12-52-29_2025-01-07_12-52-49_548398_9H57073_IPN_NVDLT_IPC_000010.MF4'
            )
            log_path = os.path.join(r'C:\Users\mfixlz\Downloads',
                                    log_name)
            bmw_obj = bmwSignalExtraction()

            req_xml_file_path = \
                os.path.join(r"C:\Users\mfixlz\Downloads\ip_next",
                             # 'ip_next.xml'
                             'ip_next_I460_57b678.xml'
                             )

            group_data_dict = bmw_obj._read_all_eth_data(log_path)

            # Non Variable Data Link Trace : NVDLT
            # bus_channel = int('0x105d1', 16)
            bus_channel = int('0x105d0', 16)
            req_data_mudp = group_data_dict[bus_channel]

            to_parse_xml: bool = True

            kwargs = {'message_value_pairs_path':
                      os.path.join(r"C:\Users\mfixlz\Downloads\ip_next",
                                   # 'ip_next.xml'
                                   'message_value_pairs.pickle'
                                   ),
                      'all_frames_length_dict_path':
                          os.path.join(r"C:\Users\mfixlz\Downloads\ip_next",
                                       # 'ip_next.xml'
                                       'all_frames_length_dict.pickle'
                                       ),
                      }

            output_dict = bmw_obj._extract_bmw_udp(req_data_mudp,
                                                   is_tcp=False,
                                                   to_parse_xml=to_parse_xml,
                                                   req_xml_file_path=req_xml_file_path,
                                                   **kwargs
                                                   )

        if thunder_data:

            run_mode = 1

            parallel_run = False
            back_end = 'ray'  # "loky"

            if back_end == 'ray' and parallel_run:
                from ray.util.joblib import register_ray
                register_ray()
            num_jobs = -1

            thun_obj = thunderSignalExtraction(
                # stream_def_dir_path,
                # trimble_config_path
            )

            log_names = [


                # 'TNDR1_DRUK_20240530_164859_WDC5_bus_0006.MF4',

                # 'MCIP_CANOEM_WS_b03_20250714_115736_0015.MF4',
                # 'TNDR1_KALU_20240801_200642_WDC5_bus_0001.MF4',


                'TNDR1_BEDG_20240426_012708_WDC5_rFLR240008243301_r4SRR240011243301_rM05_rVs05070011_rA24010124369190_bus_0061.MF4'

                # 'BMWSP2025_ENDURANCE_bus_RADETH_WBA21BY0307L57814_20241015_093825_0002.MF4',
            ]

            stream_def_dir_path = os.path.join(r'C:\Users\mfixlz\Downloads\DEXT_Aswin',
                                               r'DEXT v1.1.0_Patched\stream_definitions\stream_defs')

            trimble_config_path = os.path.join(stream_def_dir_path,
                                               'trimble_udp_messages_def.yaml')
            can_db_path = os.path.join(
                r'C:\Users\mfixlz\Downloads\Databases\Databases', )

            sw_ver_dbc_mapping = {
                '24.04.01.24.37.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.37.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.104': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.103': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.101': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.100': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.104': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.103': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.100': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                       'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                       'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.99': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.97': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.97': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.96': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.96': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.93': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.92': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.92': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.00.01.24.36.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.36.90': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.36.90': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.35.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.00.01.24.32.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.04.01.24.34.03': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.34.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.34.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.34.02': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.34.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.00.01.24.34.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.00.01.24.33.94': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.33.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.01.01.24.33.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R15_CANFD3.dbc'],
                '24.04.01.24.32.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.32.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.31.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.01.01.24.31.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.25.95': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.25.94': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.25.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.25.90': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                # '24.00.01.24.25.95': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                #                       'CANSB2_S3_01_22_2024_Ver_12.dbc'],
                '24.00.01.24.24.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.24.93': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.04.01.24.24.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.24.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.23.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.23.94': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.22.94': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.22.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.22.90': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.21.93': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.21.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.01.02.24.17.00': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.17.94': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                '24.00.01.24.17.01': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.17.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.17.90': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                # '24.00.01.24.17.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                #                       'CANSB2_S3_01_22_2024_Ver_12.dbc'],
                '24.00.01.24.16.91': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc'],
                ###
                '24.00.01.24.16.00': ['CANSB1_S3_01_22_2024_Ver_12.dbc',
                                      'CANSB2_S3_01_22_2024_Ver_12.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.91': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.14.91': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.14.00': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.14.93': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.00': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.01': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.90': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.03': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.04': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.03.24.15.00': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],
                '24.00.01.24.15.92': ['CANSB1_S3_11_16_2022_Ver_9.dbc',
                                      'CANSB2_S3_11_16_2022_Ver_9.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R6_CANFD14.dbc',
                                      'Sig_Grp_Slave_PKM_E2A_R14_CANFD3.dbc'],

            }

            flr4_sw_ver_dbc_mapping = {
                '24.00.01.24.24.91':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.23.01':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.23.94':
                    # 'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                    'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml',
                '24.00.01.24.22.94':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.22.91':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.22.90':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.21.93':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.21.91':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.01.02.24.17.00':
                    'ENET-AD5_S3_08_18_2023_Ver4.57.arxml',
                '24.00.01.24.17.94':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.17.01':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.17.00':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.17.90':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                # '24.00.01.24.17.91':
                #     'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.16.91':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.16.00':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.15.91':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.14.91':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',
                '24.00.01.24.14.00':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',
                '24.00.01.24.14.93':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',
                '24.00.01.24.15.00':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',
                '24.00.01.24.15.01':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.15.90':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',
                '24.00.01.24.15.03':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.01.24.15.04':
                    'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml',
                '24.00.03.24.15.00':
                    'ENET-AD5_S3_08_18_2023_Ver4.57.arxml',
                '24.00.01.24.15.92':
                    'ENET-AD5_S1_01_05_2023_Ver_10.arxml',

            }
            flr4_arxml_root_path = os.path.join(r'C:\Users\mfixlz\Downloads',
                                                'flr4_dbc')

            db_signal_pairs = [
                # ('CANSB1_S3_11_16_2022_Ver_9.dbc', 29),
                # ('CANSB2_S3_11_16_2022_Ver_9.dbc', 30),
                # ('mrrrl.dbc', 29),
                ('CANSB1_S3_01_22_2024_Ver_12.dbc', 29),
                ('CANSB2_S3_01_22_2024_Ver_12.dbc', 30),
                ('PKM_E2A_R14_CANFD3.dbc', 61),
                ('PKM_E2A_R6_CANFD14.dbc', 60),
                # ('PLB24_E3A_R1_CANFD3.dbc', 26),
                # ('PLB24_E3A_R1_CANFD14.dbc', 27),
                ('PKM_E2A_R3_CANFD1.dbc', 28),
                ('Sig_Grp_Slave_PKM_E3A_R1_CANFD14.dbc', 27),
                ('Sig_Grp_Slave_PKM_E3A_R4_CANFD3.dbc', 26),
                ('Range_Target.dbc', 31),
                ('RT3K_Host.dbc', 31),
                ('GPS_Veh_test.dbc', 31),
                ('PWS_E3A_R1_CANFD14.dbc', 2112),
                ('PWS_E3A_R12_CANFD3.dbc', 2111),
                ('PWS_E3A_R1_CANFD14.dbc', 2212),
                ('PWS_E3A_R12_CANFD3.dbc', 2211),
                ('TKEY_PCAN_SRR_Front_v2_00_64DETS.dbc', 2102),
                ('TKEY_PCAN_SRR_Rear_v2_00_64DETS.dbc', 2103),
                ('TKEY_PCAN_FLR_v2_00_128DETS.dbc', 2120),
            ]
            # edited line 451: C:\Users\mfixlz\py_venv\data_analytics
            # \Lib\site-packages\canmatrix\canmatrix.py
            ##########################################################################
            # REMOVE THE FOLLOWING LINE 450
            # raw_value = (self.float_factory(value) -
            #               self.float_factory(self.offset)) / self.float_factory(self.factor)

            # ADD THE FOLLOWING
            # if float(self.factor) == 0:
            #     raw_value = (self.float_factory(value) -
            #                   self.float_factory(self.offset)) / self.float_factory(1.0)
            # else:
            #     raw_value = (self.float_factory(value) -
            #                   self.float_factory(self.offset)) / self.float_factory(self.factor)
            ##########################################################################

            channel_name_pairs = {'CAN29': 'CANSB1',
                                  'CAN30': 'CANSB2',
                                  'CAN61': 'FDCAN3_Mule',
                                  'CAN26': 'FDCAN3_WL',
                                  'CAN28': 'FDCAN1_WL',
                                  'CAN60': 'FDCAN14_Mule',
                                  'CAN27': 'FDCAN14_WL',
                                  'CAN31': 'FDCAN_RT',
                                  'CAN2111': 'FDCAN3_MCIP',
                                  'CAN2112': 'FDCAN14_MCIP',
                                  'CAN2211': 'FDCAN3_MCIP_IFV',
                                  'CAN2212': 'FDCAN14_MCIP_IFV',
                                  'CAN2102': 'CANSB1',
                                  'CAN2103': 'CANSB2',
                                  'CAN2120': 'FDCAN_LRRF',
                                  }

            can_input_dict = {}
            can_input_dict['can_db_path'] = can_db_path
            can_input_dict['db_signal_pairs'] = db_signal_pairs
            can_input_dict['channel_name_pairs'] = channel_name_pairs
            ###
            can_input_dict['sw_ver_dbc_mapping'] = sw_ver_dbc_mapping

            mudp_input_dict = {}
            mudp_input_dict['bus_channel'] = int('0x800638', 16)
            mudp_input_dict['bus_channel_flr'] = int('0x12', 16)
            mudp_input_dict['is_tcp'] = False

            trimble_input_dict = {}
            trimble_input_dict['bus_channel'] = int('0xe0000', 16)
            trimble_input_dict['trimble_config_path'] = trimble_config_path

            flr4_input_dict = {}
            flr4_input_dict['flr4_arxml_root_path'] = flr4_arxml_root_path
            flr4_input_dict['flr4_sw_ver_dbc_mapping'] = flr4_sw_ver_dbc_mapping

            bus_dict = {}
            bus_dict['can_input_dict'] = can_input_dict
            bus_dict['mudp_input_dict'] = mudp_input_dict
            bus_dict['trimble_input_dict'] = trimble_input_dict
            bus_dict['stream_def_dir_path'] = stream_def_dir_path
            bus_dict['flr4_input_dict'] = flr4_input_dict

            dvl_data_dict_path = os.path.join(
                r"C:\Users\mfixlz\OneDrive - Aptiv\Documents\DM_A",
                r'PO_Chaitanya_K\Projects\GPO Data Mining Analysis',
                r'GPO_Data_Mining_Analysis\src\eventExtraction\data\Thunder',
                r'dvl_ext_data_dictionary.xlsx')
            bus_dict['dvl_data_dict_path'] = dvl_data_dict_path

            srr_input_dict = {}
            srr_input_dict['bus_channel_dict'] = {
                'Front_Right': int('0x29', 16),
                'Front_Left': int('0x2a', 16),
                'Rear_Right': int('0x2b', 16),
                'Rear_Left': int('0x2c', 16),
            }
            srr_input_dict['is_tcp'] = False

            spi_dbc_path = os.path.join(r'C:\Users\mfixlz\Downloads\DBC2')
            spi_dbc_pickle_path_dma = os.path.join(r"C:\Users\mfixlz\Downloads\Versioned\pickles",
                                                   'eyeq_message_protocol_version_mapper.json')
            spi_input_dict = {}
            spi_input_dict['dbc_path'] = spi_dbc_path
            spi_input_dict['dbc_pickle_path_dma'] = spi_dbc_pickle_path_dma

            deb_dict = {}
            deb_dict['srr_dict'] = srr_input_dict
            deb_dict['spi_input_dict'] = spi_input_dict
            deb_dict['stream_def_dir_path'] = stream_def_dir_path

            ref_dict = {}
            ref_dict['trimble_input_dict'] = trimble_input_dict
            ref_dict['can_input_dict'] = can_input_dict
            ref_dict['stream_def_dir_path'] = stream_def_dir_path

            start_time = time.time()
            mem_before_phy, mem_before_virtual = process_memory()

            if parallel_run:

                @ contextlib.contextmanager
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

                def parallel_wrapper_run(kwargs):

                    # bus_dict, deb_dict, ref_dict, log_path,
                    #          run_mode: int = -999

                    kwargs_main = kwargs['kwargs_main']
                    kwargs.pop('kwargs_main', None)
                    out_dict, debug_str = thun_obj.main(
                        **kwargs, **kwargs_main)

                    sp.io.savemat(kwargs['log_path'][:-4]+'.mat', out_dict,
                                  **{'long_field_names': True, 'oned_as': 'column'})

                    return out_dict

                kwargs_main = {}

                if run_mode == 312:

                    from scipy.io import loadmat as load_mat_scipy

                    kwargs_load_mat_scipy = {  # 'struct_as_record' : False,
                        # 'squeeze_me' : True,
                        'simplify_cells': True,
                        'verify_compressed_data_integrity': True
                    }

                    is_mat_file_bool_list = [os.path.isfile(
                        os.path.join(r'C:\Users\mfixlz\Downloads',
                                     log_name)[:-4] + '.mat')
                        for log_name in log_names]

                    kwargs_list_2 = [{'mat_data': load_mat_scipy(
                        os.path.join(r'C:\Users\mfixlz\Downloads',
                                      log_name)[:-4] + '.mat',
                        **kwargs_load_mat_scipy
                    )
                    } if bool_val else {}
                        for log_name, bool_val in zip(log_names,
                                                      is_mat_file_bool_list)]

                    # kwargs_main['mat_data'] = load_mat_scipy(
                    #     log_path[:-4] + '.mat',
                    #     **kwargs_load_mat_scipy
                    # )

                kwargs_list = [{'bus_dict': bus_dict,
                                'deb_dict': deb_dict,
                                'ref_dict': ref_dict,
                                'run_mode': run_mode,
                                'log_path': os.path.join(r'C:\Users\mfixlz\Downloads',
                                                         log_name),
                                'kwargs_main': kwargs_main
                                } if bool(kwargs_main)
                               else
                               {'bus_dict': bus_dict,
                                'deb_dict': deb_dict,
                                'ref_dict': ref_dict,
                                'run_mode': -999,
                                'log_path': os.path.join(r'C:\Users\mfixlz\Downloads',
                                                         log_name),
                                'kwargs_main': kwargs_main
                                }
                               for log_name, kwargs_main in zip(log_names,
                                                                kwargs_list_2)]

                args = kwargs_list

                with parallel_config(backend=back_end):

                    with tqdm_joblib(tqdm(desc="My calculation",
                                          total=len(args)
                                          )) as progress_bar:

                        results = Parallel(n_jobs=num_jobs,
                                           prefer='processes',
                                           # return_as="generator",
                                           )(delayed(
                                               parallel_wrapper_run)(a)
                                             for a in args)

            else:

                results = []
                for log_name in log_names:
                    log_path = os.path.join(r'C:\Users\mfixlz\Downloads',
                                            log_name)

                    kwargs_main = {}

                    if run_mode == 312:

                        from scipy.io import loadmat as load_mat_scipy

                        kwargs_load_mat_scipy = {  # 'struct_as_record' : False,
                            # 'squeeze_me' : True,
                            'simplify_cells': True,
                            'verify_compressed_data_integrity': True
                        }

                        if os.path.isfile(log_path[:-4] + '.mat'):

                            kwargs_main['mat_data'] = load_mat_scipy(
                                log_path[:-4] + '.mat',
                                **kwargs_load_mat_scipy
                            )
                        else:

                            run_mode = 1

                    ########################################################

                    # out_group_data = thun_obj._read_raw_data_mdf(log_path, False)

                    # out_dict = thun_obj.main_bus(bus_dict, log_path)
                    # out_dict = thun_obj.main_deb(deb_dict, log_path)
                    ########################################################

                    out_dict, debug_str = thun_obj.main(bus_dict,
                                                        deb_dict,
                                                        ref_dict,
                                                        log_path,
                                                        run_mode=run_mode,
                                                        **kwargs_main)

                    results.append(out_dict)
                    # sp.io.savemat(log_path[:-4]+'_corrected.mat', out_dict,
                    #               **{'long_field_names': True, 'oned_as': 'column'},
                    #               do_compression=True
                    #               )

                    ###############################################################

                    # thun_obj.CAN_flat = True
                    # thun_obj.SPI_flat = False
                    # can_dict, can_dict_edited_keys, log_start_time, can_dict_2 = \
                    #     thun_obj._extract_thunder_CAN(can_db_path,
                    #                                   db_signal_pairs,
                    #                                   channel_name_pairs,
                    #                                   log_path)
                    # FIXME: For MCIP data, in asamdf\mdf.py,
                    # around line 4920-4922,
                    # change to .astype("<u8") from .astype("<u1")

                    ###############################################################

                    # can_dict = thun_obj._extract_thunder_CAN2(can_db_path,
                    #                                           db_signal_pairs,
                    #                                           channel_name_pairs,
                    #                                           log_path)
                    ###############################################################

                    # thun_obj.CAN_flat = True
                    # thun_obj.FLR = True
                    # thun_obj.req_arxml_name = os.path.join(
                    #     flr4_input_dict['flr4_arxml_root_path'],
                    #     # 'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml',
                    #     'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml'
                    # )
                    # # flr4_dict = thun_obj._flr4_ad5_parser(
                    # #     log_path)
                    # flr4_dict_dma, flr4_dict_dvl_ext = thun_obj._flr4_ad5_parser2(
                    #     log_path)

                    ###############################################################

                    # (flr4_dict,
                    #  flat_dict, flat_dict_all_data, ser_msgs
                    #  ) = thun_obj._flr4_ad5_parser(
                    #     log_path)

                    # SPI_dict = thun_obj._SPI_data_parser(log_path,
                    #                                      protocol_type_='SPI',
                    #                                      dbc_path=spi_dbc_path)
                    ##############################################################
                    # thun_obj.CAN_flat = False
                    # thun_obj.SPI_flat = True
                    # SPI_eyeQ_dict_dma, SPI_eyeQ_dict = thun_obj._SPI_data_parser2(log_path,
                    #                                                               spi_dbc_pickle_path_dma)
        if mcip_data:
            results = []

            log_names = [

                # 'MCIP_CANOEM_WS_b03_20250714_115736_0015.MF4',
                # 'ThunderMCIP_WS11656_20250727_083502_0009_b03.MF4',
                'ThunderMCIP_WS11656_20250723_132447_0000_p01.MF4',


            ]

            ###########################################################
            # MCIP
            ###########################################################
            start_time = time.time()
            mem_before_phy, mem_before_virtual = process_memory()

            mcip_obj = mcipSignalExtraction()
            yaml_input_path = os.path.join(r"C:\Users\mfixlz",
                                           r'OneDrive - Aptiv\Documents',
                                           r'DM_A\Aravind\Projects\2024',
                                           r'KW_01_05\BCO-11009',
                                           'config_inputs_MCIP_win.yaml')

            can_db_path = os.path.join(
                r'C:\Users\mfixlz\Downloads\Databases\Databases', )

            with open(yaml_input_path) as stream:

                # yaml_input_dict = yaml.safe_load(stream)
                yaml_input_dict = yaml.load(stream, Loader=yaml.Loader)

            b03_dict = yaml_input_dict['b03_dict']
            b04_dict = yaml_input_dict['b04_dict']
            b05_dict = yaml_input_dict['b05_dict']
            p01_dict = yaml_input_dict['p01_dict']
            r03_dict = yaml_input_dict['r03_dict']

            b03_dict['can_input_dict']['can_db_path'] = can_db_path
            b04_dict['can_input_dict']['can_db_path'] = can_db_path
            run_mode = -999  # 4  #
            kwargs_main = {}

            for log_name in log_names:
                log_path = os.path.join(r'C:\Users\mfixlz\Downloads',
                                        log_name)

                out_dict, debug_str = mcip_obj.main(b03_dict,
                                                    b04_dict,
                                                    b05_dict,
                                                    p01_dict,
                                                    r03_dict,
                                                    log_path,
                                                    run_mode=run_mode,
                                                    **kwargs_main)

                results.append(out_dict)

                sp.io.savemat(log_path[:-4]+'_test.mat', out_dict,
                              **{'long_field_names': True,
                                 'oned_as': 'column'},
                              do_compression=True
                              )
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
