import numpy
import pandas
from asammdf import MDF
import json
import os
from comm_headers_from_array import IPHeader, UDPHeader, SPIHeader, AutosarPDUHeader
import pickle
import cantools
import scipy
import numpy as np
import struct
import time
import linecache
import sys

ETHERTYPE_BYTEOFFSETS = {2048: 0, 33024: 4}

AUTOSAR_PDU_SRC_444 = {
    "0x10000": "PDU_LRRF_to_ADCAM_LRRF_Active_Faults_Status",
    "0x10001": "PDU_LRRF_to_ADCAM_LRRF_Detection_001_022",
    "0x10002": "PDU_LRRF_to_ADCAM_LRRF_Detection_023_044",
    "0x10003": "PDU_LRRF_to_ADCAM_LRRF_Detection_045_066",
    "0x10004": "PDU_LRRF_to_ADCAM_LRRF_Detection_067_088",
    "0x10005": "PDU_LRRF_to_ADCAM_LRRF_Detection_089_110",
    "0x10006": "PDU_LRRF_to_ADCAM_LRRF_Detection_111_128",
    "0x10009": "PDU_LRRF_to_ADCAM_LRRF_Header_AlignmentState",
    "0x10010": "PDU_LRRF_to_ADCAM_LRRF_Header_EchoedSysStatus",
    "0x10011": "PDU_LRRF_to_ADCAM_LRRF_Header_SensorCoverage",
    "0x10012": "PDU_LRRF_to_ADCAM_LRRF_Header_Timestamps",
    "0x10007": "PDU_LRRF_to_ADCAM_LRRF_ETH_HTR",
    "0x10008": "PDU_LRRF_to_ADCAM_LRRF_HW_ID",
    "0x10013": "PDU_LRRF_to_ADCAM_LRRF_Radar_PartNumber",
    "0x10014": "PDU_LRRF_to_ADCAM_LRRF_Status_Radar",
    "0x10015": "PDU_LRRF_to_ADCAM_LRRF_Status_SwVersion",
    "0x10016": "PDU_LRRF_to_ADCAM_LRRF_Status_Temp_Volt",
    "0x20000": "PDU_ADCAM_to_LRRF_ADCAM_SWVersion",
    "0x20001": "PDU_ADCAM_to_LRRF_LRRF_ADCAM_TracksChange",
    "0x20002": "PDU_ADCAM_to_LRRF_LRRF_Radar_Cfg_Parameters",
    "0x20003": "PDU_ADCAM_to_LRRF_LRRF_System_TDBlockage",
    "0x20004": "PDU_ADCAM_to_LRRF_VEHICLE_STATE_MSG"
}

AUTOSAR_PDU_SRC_PORTS = {
    444: AUTOSAR_PDU_SRC_444
}

PLP_MSG_TYPES = {
    0: 'UNDEFINED',
    2: 'CAN',
    3: 'CAN-FD',
    4: 'LIN',
    8: 'Flexray',
    10: 'GPIO',
    16: 'UART/RS232_ASCII',
    32: 'Analog',
    128: 'Ethernet',
    144: 'TCP-DLT',
    160: 'XCP',
    257: 'MIPI-CSI2_Video',
    258: 'MIPI-CSI2_Lidar',
    259: 'SPI',
    260: 'I2C_7Bit',
    1024: 'Radar',
    40960: 'PLP_Raw',
    45056: 'PreLabel',
    49154: 'GigEVision'
}

EYEQ_MSGS = {
    13: "INIT_EOL_MAIN",
    18: "Core_Boot_Diagnostics_Message_EQ6_protocol",
    35: "INIT_CAL_MAIN",
    47: "Core_Objects_AnyO_protocol",
    49: "Camera_Init_main",
    50: "Camera_Init_narrow",
    51: "Camera_Init_fisheye",
    52: "Camera_Init_rear",
    53: "Camera_Init_rearCornerLeft",
    54: "Camera_Init_rearCornerRight",
    55: "Camera_Init_frontCornerLeft",
    56: "Camera_Init_frontCornerRight",
    57: "Camera_Init_parking_front",
    64: "Camera_Init_parking_rear",
    66: "Core_Application_Message_protocol",
    71: "Core_Optical_Path_Report_EQ5_protocol",
    72: "Core_Optical_Path_Error_EQ5_protocol",
    77: "Camera_Init_parking_left",
    78: "Camera_Init_parking_right",
    82: "Core_Common_protocol",
    87: "Core_Semantic_Marks_protocol",
    88: "Core_Semantic_Lines_protocol",
    90: "Core_DS_Traffic_Signs_protocol",
    92: "FFS_CAD_MAIN",
    95: "Core_Hazards_protocol",
    102: "Core_TFL_Spots_protocol",
    103: "Core_TFL_Structure_protocol",
    104: "Core_Construction_Area_protocol",
    108: "Lane_Advanced",
    113: "Core_Failsafe_protocol",
    116: "Core_Safety_Diagnostics_protocol",
    119: "Core_Calibration_Output_protocol",
    120: "Core_Calibration_To_Save_protocol",
    124: "Core_High_Low_Beam_protocol",
    125: "Core_Light_Scene_VCL_protocol",
    126: "Core_Light_Scene_RFL_protocol",
    130: "Core_SPC_Init",
    131: "Core_MTFV_Init",
    133: "Core_TAC2_Init",
    135: "Core_Semantic_Lanes_Description_protocol",
    138: "Core_Lanes_Road_Edge_protocol",
    139: "Core_Lateral_Departure_Warning_protocol",
    140: "Core_Lanes_Adjacent_protocol",
    141: "Core_Lanes_Applications_protocol",
    142: "Core_Lanes_Host_protocol",
    146: "Core_Vision_Init",
    154: "Core_Objects_protocol",
    157: "Core_Free_Space_EXT_protocol",
    162: "Core_Car_sensor",
    186: "FCF_CV_Alert_Init_L1",
    187: "FCF_CV_Alert_Init_L2",
    188: "FCF_CV_Alert_Init_L3",
    189: "FCF_CV_Alert_Init_L4",
    190: "FCF_CV_Alert_Init_L5",
    191: "FCF_CV_Alert_Init_L6",
    192: "IPB_Init_setH",
    193: "PCW_Init_L1",
    194: "PCW_Init_L2",
    195: "PCW_Init_L3",
    196: "PCW_Init_L4",
    197: "PCW_Init_L5",
    198: "PCW_Init_L6",
    199: "PCW_Init_L7",
    200: "PCW_Init_L8",
    201: "IPB_Init_setG",
    202: "IPB_Init",
    203: "IPB_Init_setB",
    204: "EyeQ4",
    205: "IPB_Init_setD",
    206: "IPB_Init_setE",
    207: "IPB_Init_setF",
    210: "Environment_Init_main",
    211: "General_Init",
    216: "Core_FCF_VD_DYN_protocol",
    217: "Core_FCF_VRU_DYN_protocol",
    218: "Light_Scene_Init",
    219: "LDW_Init",
    221: "Core_Debug_protocol",
    222: "Core_OCR_protocol",
    223: "Core_MTFV_protocol",
    224: "Core_FCF_CV_DYN_protocol",
    229: "Environment_Init_fisheye",
    248: "FCF_CV_Alert_Init_L7",
    249: "FCF_CV_Alert_Init_L8"
}

SPI_PACKET_SIZE = 128
SPI_PACKET_TO_LOG = [8]


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(
        filename, lineno, line.strip(), exc_obj))


def extract_SPI_data(mdf_file,
                     # spi_dbc
                     spi_mapper_json,
                     pickle_path
                     ):

    base_mdf = MDF(mdf_file)
    log_start_time = base_mdf.header.start_time.timestamp()
    if 'CAN_DataFrame' in base_mdf.channels_db.keys():
        can_dataframe_channel_groups = [
            channel_group[0] for channel_group in base_mdf.channels_db['CAN_DataFrame']]
    else:
        can_dataframe_channel_groups = []
    if 'ETH_Frame' in base_mdf.channels_db.keys():
        eth_frame_channel_groups = [channel_group[0]
                                    for channel_group in base_mdf.channels_db['ETH_Frame']]
    else:
        eth_frame_channel_groups = []
    if 'PLP_Raw_Data' in base_mdf.channels_db.keys():
        plp_frame_channel_groups = [channel_group[0]
                                    for channel_group in base_mdf.channels_db['PLP_Raw_Data']]
    else:
        plp_frame_channel_groups = []

    # Create the SPI dbc structure for decoding
    # with open('merged_spi.pickle', 'rb') as fp:
    #     spi_dbc = pickle.load(fp)

    spi_dbc = {}
    spi_dbc['tr_name'] = {}
    spi_dbc['dbcs'] = {}
    spi_dbc['dbc_struct'] = {}

    is_data = {}
    dd = {}

    not_available_msg_id = []
    pickle_file_path_list = set()
    base_channel_list = dict()
    for group_idx, group in enumerate(base_mdf.groups):
        if group.channel_group.acq_name and group.data_blocks:
            base_channel_list[group.channel_group.acq_name] = dict()
            # base_channel_list[group.channel_group.acq_name]["mdf_bus_idx"] = group_idx
            group_df = base_mdf.get_group(group_idx)
            # # PLP Data
            if group_idx in plp_frame_channel_groups:

                # Temp added #TBD
                # if group_idx==0:
                #     continue
                # Group by PLP message types
                plp_msg_types = {plp_type: group_df[group_df["PLP_Raw_Data.PLP_Raw_Data.MsgType"] == plp_type].copy(
                ) for plp_type in group_df["PLP_Raw_Data.PLP_Raw_Data.MsgType"].unique()}
                for msg_type in plp_msg_types.keys():
                    if msg_type in PLP_MSG_TYPES.keys():
                        if PLP_MSG_TYPES[msg_type] == 'SPI':
                            # Process SPI Header
                            spi_header_start = 0
                            spi_header_end = 13
                            base_channel_list[group.channel_group.acq_name]["SPI"] = dict(
                            )
                            plp_type_df = plp_msg_types[msg_type]
                            # Truncate payload based on data length here
                            # Extract data length and payload as different arrays
                            # Remove payloads starting with anything other than 0x8 for decoding
                            timestamps = plp_type_df.index.to_numpy()
                            datalengths = plp_type_df["PLP_Raw_Data.PLP_Raw_Data.DataLength"].to_numpy(
                            )
                            payloads = plp_type_df["PLP_Raw_Data.PLP_Raw_Data.DataBytes"].to_numpy(
                            )
                            payloads = [bytes[0:datalen] for bytes,
                                        datalen in zip(payloads, datalengths)]
                            timestamps_unpacked = []
                            payloads_unpacked = []
                            for timestamp, payload, datalength in zip(timestamps, payloads, datalengths):
                                if datalength < SPI_PACKET_SIZE:
                                    # Discard packets less than SPI size
                                    continue
                                elif datalength > SPI_PACKET_SIZE:
                                    payload_unpacked = numpy.reshape(
                                        payload, (-1, 128))
                                    [payloads_unpacked.append(
                                        packet) for packet in payload_unpacked if packet[0] in SPI_PACKET_TO_LOG]
                                    [timestamps_unpacked.append(
                                        timestamp) for packet in payload_unpacked if packet[0] in SPI_PACKET_TO_LOG]
                                else:
                                    if payload[0] in SPI_PACKET_TO_LOG:
                                        payloads_unpacked.append(payload)
                                        timestamps_unpacked.append(timestamp)
                            duplicate_timestamp_idx = numpy.array(
                                [idx + 1 for idx, timestamp in enumerate(timestamps_unpacked[2:-1]) if
                                 timestamp in timestamps_unpacked[0:idx] + timestamps_unpacked[idx + 1:]])
                            timestamps_unpacked = numpy.array(
                                timestamps_unpacked)
                            payloads_unpacked = numpy.array(payloads_unpacked)
                            payload_processed_data = numpy.array([SPIHeader(
                                x[spi_header_start:spi_header_end]).return_spi_header_info() for x in payloads_unpacked])
                            payload_processed_def = [
                                "single_check", "frame_num", "spi_app_id", "spi_size", "spi_transmit_id"]
                            # insert each byte of payload as a different array element
                            processed_payload_dict = {
                                "timestamp": timestamps_unpacked}
                            processed_payload_dict.update({definition: data_array for definition, data_array in zip(
                                payload_processed_def, payload_processed_data.T)})
                            payload_processed = pandas.DataFrame(
                                processed_payload_dict)
                            # Use payload_processed and payloads_unpacked to study SPI payload
                            # for message_id in payload_processed["spi_app_id"].unique():
                            try:
                                message_id_list = set()
                                missing_protocol_set = set()
                                message_id_out_dict = {}

                                counter_print_ = False
                                ctr = 0
                                p_iter = iter(payloads_unpacked)
                                while True:
                                    data_row = next(p_iter, 'end')
                                    # print(f'$$$$$$$$$$$$$$$$$$$ {ctr}')
                                    ctr += 1
                                    FRAME_IDX = 6

                                    if not type(data_row) == str:

                                        frame_type = data_row[FRAME_IDX]
                                        if frame_type == 0:
                                            SIZE_IDX = 7
                                            MSG_IDX = 8
                                            message_id = data_row[MSG_IDX]

                                            if not message_id in is_data:
                                                # ########################################
                                                # print(
                                                #     f'#### message ID:{message_id}, ',)

                                                if not str(
                                                        message_id) in spi_mapper_json[
                                                            'get_protocol_index']:
                                                    if (not message_id in message_id_list):
                                                        print(f'^^^^^^^ message ID:{message_id}, ',
                                                              'is not in mapper json for SPI decoding ',
                                                              'and shall be skipped for this message ID')

                                                        message_id_list.add(
                                                            message_id)
                                                        if not message_id in message_id_out_dict:
                                                            message_id_out_dict[message_id] = [
                                                            ]

                                                        message_id_out_dict[message_id].append(f' message ID:{message_id} '
                                                                                               + 'is not in mapper json for SPI decoding')

                                                    continue

                                                protocol_index_dict = \
                                                    spi_mapper_json['get_protocol_index'][str(
                                                        message_id)]
                                                # np.array(data_row[
                                                #     int(protocol_index_dict['start_bit']):
                                                #     int(
                                                #         protocol_index_dict['start_bit'])
                                                #     + int(protocol_index_dict['span_bits'] )

                                                # ][::-1]).view(np.uint32)[0]
                                                zz = ['{0:08b}'.format(
                                                    i) for i in data_row]
                                                rr = ''.join(zz)
                                                protocol_version = rr[72+int(protocol_index_dict['start_bit']):72+int(
                                                    protocol_index_dict['start_bit'])+int(protocol_index_dict['span_bits'])]
                                                protocol_version = int(
                                                    protocol_version, 2)

                                                dbc_pickle_dict = spi_mapper_json[
                                                    'get_dbc_pickle'][str(message_id)]

                                                if (len(dbc_pickle_dict) > 1
                                                        and isinstance(dbc_pickle_dict, dict)
                                                    ):

                                                    dbc_pickle_name = [val for item, val in dbc_pickle_dict.items()

                                                                       if (str(protocol_version) in item
                                                                       and 'high' in item) or
                                                                       (str(protocol_version) in item and
                                                                        (not ('high' in item or 'low' in item)))
                                                                       ]
                                                    if len(dbc_pickle_name) >= 1:

                                                        dbc_pickle_name = dbc_pickle_name[0]
                                                    else:

                                                        proto_string = f'{protocol_version}_{message_id}'
                                                        if not proto_string in missing_protocol_set:

                                                            print(f'^^^^^^^ Protocol version {protocol_version} ',
                                                                  f'for message ID:{message_id}, ',
                                                                  'does not have a pickle file for SPI decoding ')

                                                            if not message_id in message_id_out_dict:
                                                                message_id_out_dict[message_id] = [
                                                                ]

                                                            message_id_out_dict[message_id].append(f' 1.message ID:{message_id}, protocol version : {protocol_version} '
                                                                                                   + 'does not have a pickle file for SPI decoding')

                                                        missing_protocol_set.add(
                                                            proto_string)
                                                        continue

                                                elif isinstance(dbc_pickle_dict, dict):

                                                    if not str(
                                                            protocol_version) in dbc_pickle_dict:

                                                        proto_string = f'{protocol_version}_{message_id}'
                                                        if not proto_string in missing_protocol_set:
                                                            # print(f'$$$$$$$$$$$$$$$$$$$ {ctr}')
                                                            print(f'^^^^^^^ Protocol version {protocol_version} ',
                                                                  f'for message ID:{message_id}, ',
                                                                  'does not have a picke file for SPI decoding ')

                                                            if not message_id in message_id_out_dict:
                                                                message_id_out_dict[message_id] = [
                                                                ]

                                                            message_id_out_dict[message_id].append(f' 2.message ID:{message_id}, protocol version : {protocol_version} '
                                                                                                   + 'does not have a pickle file for SPI decoding')

                                                        missing_protocol_set.add(
                                                            proto_string)
                                                        continue

                                                    dbc_pickle_name = dbc_pickle_dict[str(
                                                        protocol_version)]

                                                else:

                                                    proto_string = f'{protocol_version}_{message_id}'
                                                    if not proto_string in missing_protocol_set:

                                                        if not message_id in message_id_out_dict:
                                                            message_id_out_dict[message_id] = [
                                                            ]

                                                        message_id_out_dict[message_id].append(f' message ID:{message_id}, protocol version : {protocol_version} '
                                                                                               + 'does not have a pickle file for SPI decoding')

                                                        print(
                                                            '\n^^^^^^^^^^^^^^^^^^^^^^^',
                                                            'No Pickle available. DBC not mapped',
                                                            f'message ID:{message_id}, ',
                                                            f'protocol version : {protocol_version}')
                                                    continue
                                                dbc_pickle_name = dbc_pickle_name \
                                                    + '.pickle'
                                                pickle_file_path = os.path.join(pickle_path,
                                                                                dbc_pickle_name)
                                                if not os.path.isfile(pickle_file_path):

                                                    if not pickle_file_path in pickle_file_path_list:
                                                        if not message_id in message_id_out_dict:
                                                            message_id_out_dict[message_id] = [
                                                            ]

                                                        message_id_out_dict[message_id].append(f' message ID:{message_id}, protocol version : {protocol_version} '
                                                                                               + 'does not have a pickle file for SPI decoding ' +
                                                                                               f'at : {pickle_file_path} ')
                                                        print(
                                                            f'@@@@@@@ pickle file at : {pickle_file_path} does not exist')
                                                    pickle_file_path_list.add(
                                                        pickle_file_path)
                                                    continue
                                                # else:
                                                #     print(
                                                #         f'@@@@@@@ pickle file at : {pickle_file_path}')

                                                with open(pickle_file_path, 'rb') as f:
                                                    dbc_dict = pickle.load(
                                                        f)
                                                    is_data[message_id] = True

                                                spi_dbc['tr_name'][message_id] = 'EyeQ'
                                                spi_dbc['dbcs'][message_id] = dbc_dict['dbcs']
                                                spi_dbc['dbc_struct'][message_id] = dbc_dict['dbc_struct']
                                                ################################################
                                            # struct_name = spi_dbc['dbcs'][message_id].get_message_by_frame_id(message_id)
                                            if spi_dbc['tr_name'][message_id] == 'EyeQ':
                                                L1 = data_row[SIZE_IDX] - 1
                                            else:
                                                L1 = data_row[SIZE_IDX]
                                            data = bytes(data_row[9:9+L1])
                                        else:
                                            spi_datasize_curr = 1
                                            while (spi_datasize_curr > 0):
                                                if frame_type == 1:
                                                    SIZE_IDX = 8
                                                    MSG_IDX = 12
                                                    L0 = bytes(
                                                        data_row[SIZE_IDX: SIZE_IDX + 4])
                                                    message_id = data_row[MSG_IDX]

                                                    if not message_id in is_data:
                                                        #########################################

                                                        if not str(
                                                                message_id) in spi_mapper_json[
                                                                    'get_protocol_index']:
                                                            if (not message_id in message_id_list):
                                                                print(f'^^^^^^^ message ID:{message_id}, ',
                                                                      'is not in mapper json for SPI decoding ',
                                                                      'and shall be skipped for this message ID')
                                                                message_id_list.add(
                                                                    message_id)
                                                                if not message_id in message_id_out_dict:
                                                                    message_id_out_dict[message_id] = [
                                                                    ]

                                                                message_id_out_dict[message_id].append(f'multiframe message ID:{message_id} '
                                                                                                       + 'is not in mapper json for SPI decoding')

                                                            break
                                                        protocol_index_dict = spi_mapper_json['get_protocol_index'][str(
                                                            message_id)]
                                                        # np.array(data_row[
                                                        #     int(protocol_index_dict['start_bit']):
                                                        #     int(
                                                        #         protocol_index_dict['start_bit'])
                                                        #     + int(protocol_index_dict['span_bits'] )

                                                        # ][::-1]).view(np.uint32)[0]
                                                        zz = ['{0:08b}'.format(
                                                            i) for i in data_row]
                                                        rr = ''.join(zz)
                                                        protocol_version = rr[104+int(protocol_index_dict['start_bit']):104+int(
                                                            protocol_index_dict['start_bit'])+int(protocol_index_dict['span_bits'])]
                                                        protocol_version = int(
                                                            protocol_version, 2)

                                                        dbc_pickle_dict = spi_mapper_json[
                                                            'get_dbc_pickle'][str(message_id)]

                                                        if (len(dbc_pickle_dict) > 1
                                                                and isinstance(dbc_pickle_dict, dict)
                                                            ):

                                                            dbc_pickle_name = [val for item, val in dbc_pickle_dict.items()

                                                                               if (str(protocol_version) in item
                                                                                   and 'high' in item) or
                                                                               (str(protocol_version) in item and
                                                                               (not ('high' in item or 'low' in item)))
                                                                               ]
                                                            if len(dbc_pickle_name) >= 1:

                                                                dbc_pickle_name = dbc_pickle_name[0]
                                                            else:

                                                                proto_string = f'{protocol_version}_{message_id}'
                                                                if not proto_string in missing_protocol_set:

                                                                    print(f'^^^^^^^ Protocol version {protocol_version} ',
                                                                          f'for message ID:{message_id}, ',
                                                                          'does not have a picke file for SPI decoding ')

                                                                    if not message_id in message_id_out_dict:
                                                                        message_id_out_dict[message_id] = [
                                                                        ]

                                                                    message_id_out_dict[message_id].append(f'1.multiframe message ID:{message_id}, protocol version : {protocol_version} '
                                                                                                           + 'does not have a pickle file for SPI decoding')

                                                                missing_protocol_set.add(
                                                                    proto_string)
                                                                break

                                                        elif isinstance(dbc_pickle_dict, dict):

                                                            if not str(
                                                                    protocol_version) in dbc_pickle_dict:

                                                                proto_string = f'{protocol_version}_{message_id}'
                                                                if not proto_string in missing_protocol_set:

                                                                    print(f'^^^^^^^ Protocol version {protocol_version} ',
                                                                          f'for message ID:{message_id}, ',
                                                                          'does not have a picke file for SPI decoding ')
                                                                    if not message_id in message_id_out_dict:
                                                                        message_id_out_dict[message_id] = [
                                                                        ]

                                                                    message_id_out_dict[message_id].append(f'2.multiframe message ID:{message_id}, protocol version : {protocol_version} '
                                                                                                           + 'does not have a pickle file for SPI decoding')

                                                                missing_protocol_set.add(
                                                                    proto_string)
                                                                break

                                                            dbc_pickle_name = dbc_pickle_dict[str(
                                                                protocol_version)]

                                                        else:
                                                            proto_string = f'{protocol_version}_{message_id}'
                                                            if not proto_string in missing_protocol_set:
                                                                print(
                                                                    '\n^^^^^^^^^^^^^^^^^^^^^^^',
                                                                    'No Pickle available. DBC not mapped',
                                                                    f'message ID:{message_id}, ',
                                                                    f'protocol version : {protocol_version}')
                                                                if not message_id in message_id_out_dict:
                                                                    message_id_out_dict[message_id] = [
                                                                    ]

                                                                message_id_out_dict[message_id].append(f'multiframe message ID:{message_id}, protocol version : {protocol_version} '
                                                                                                       + 'does not have a pickle file for SPI decoding')
                                                            break
                                                        dbc_pickle_name = dbc_pickle_name \
                                                            + '.pickle'

                                                        pickle_file_path = os.path.join(pickle_path,
                                                                                        dbc_pickle_name)
                                                        if not os.path.isfile(pickle_file_path):

                                                            if not pickle_file_path in pickle_file_path_list:
                                                                print(
                                                                    f'@@@@@@@ pickle file at : {pickle_file_path} does not exist')
                                                            pickle_file_path_list.add(
                                                                pickle_file_path)
                                                            break
                                                        # else:
                                                        #     print(
                                                        #         f'@@@@@@@ pickle file at : {pickle_file_path}')

                                                        with open(pickle_file_path, 'rb') as f:
                                                            dbc_dict = pickle.load(
                                                                f)
                                                            is_data[message_id] = True

                                                        spi_dbc['tr_name'][message_id] = 'EyeQ'
                                                        spi_dbc['dbcs'][message_id] = dbc_dict['dbcs']
                                                        spi_dbc['dbc_struct'][message_id] = dbc_dict['dbc_struct']

                                                    if spi_dbc['tr_name'][message_id] == 'EyeQ':
                                                        L1 = struct.unpack(
                                                            '<I', L0)[0]
                                                    else:
                                                        L1 = struct.unpack(
                                                            '<I', L0)[0] - 1

                                                    data = bytes(
                                                        data_row[13:128])
                                                    # spi_datasize_org = data_row[8]
                                                    spi_datasize_curr = L1 - 115
                                                    data_row = next(
                                                        p_iter, 'end')
                                                    ctr += 1
                                                    frame_type = data_row[6]
                                                    if type(data_row) == str:
                                                        break

                                                else:
                                                    if spi_datasize_curr > 117:
                                                        data += bytes(
                                                            data_row[8:128])
                                                        spi_datasize_curr = spi_datasize_curr - 120
                                                        data_row = next(
                                                            p_iter, 'end')
                                                        ctr += 1
                                                        frame_type = data_row[6]
                                                        if type(data_row) == str:
                                                            break
                                                    else:
                                                        data += bytes(
                                                            data_row[8:spi_datasize_curr+8])
                                                        spi_datasize_curr = 0

                                        # res = spi_dbc['dbcs'][message_id].decode_message(data)

                                        if message_id in spi_dbc['dbcs'].keys():
                                            m = spi_dbc['dbcs'][message_id]
                                            m1 = m.get_message_by_frame_id(
                                                message_id)
                                            # ** Handle dynamic protocol'
                                            dbc_msg_len = m1.length
                                            if dbc_msg_len > L1:
                                                pad_zero = bytes(
                                                    np.zeros((1, dbc_msg_len - L1), dtype=np.uint8))
                                                data += pad_zero
                                            # else:
                                            res = m1.decode(data)

                                            for k in res.keys():
                                                r1 = res[k]
                                                if isinstance(r1, cantools.database.namedsignalvalue.NamedSignalValue):
                                                    r1 = r1.value
                                                spi_dbc['dbc_struct'][message_id][m1.name][k].append(
                                                    r1)

                                            time1 = timestamps_unpacked[ctr -
                                                                        1] + log_start_time
                                            if 'timestamp' in spi_dbc['dbc_struct'][message_id][m1.name].keys():
                                                spi_dbc['dbc_struct'][message_id][m1.name]['timestamp'].append(
                                                    time1)
                                            else:
                                                spi_dbc['dbc_struct'][message_id][m1.name]['timestamp'] = [
                                                    time1]
                                        else:
                                            if message_id not in not_available_msg_id:
                                                not_available_msg_id.append(
                                                    message_id)
                                    else:
                                        break
                                # print('Decoded ', message_id)
                                kwargs = {'long_field_names': True,
                                          'oned_as': 'column'}
                                # res_file = 'spi_data_' + str(message_id) +'.mat'
                                res_file = mdf_file[0:-4] + '_7.mat'

                                for msg_id in message_id_out_dict:
                                    # if msg_id in spi_dbc['dbcs'].keys():
                                    #     m_ = spi_dbc['dbcs'][msg_id]
                                    #     m1_ = m_.get_message_by_frame_id(
                                    #         msg_id)

                                    #     spi_dbc['dbc_struct'][msg_id][m1_.name] \
                                    #         = ''.join(message_id_out_dict[msg_id])

                                    #     dd[m1_.name] = spi_dbc['dbc_struct'][msg_id][m1_.name]

                                    dd[EYEQ_MSGS[msg_id]] = ' and '.join(
                                        message_id_out_dict[msg_id])

                                    # del spi_dbc['dbcs'][msg_id]
                                    # del spi_dbc['dbc_struct'][msg_id]

                                for k1 in spi_dbc['dbc_struct'].keys():
                                    k2 = spi_dbc['dbc_struct'][k1]
                                    k3 = list(k2.keys())
                                    dd[k3[0]] = k2
                                # scipy.io.savemat(res_file, dd, **kwargs)
                            except Exception:

                                PrintException()

                        else:
                            base_channel_list[group.channel_group.acq_name][PLP_MSG_TYPES[msg_type]] = len(
                                plp_msg_types[msg_type])
                    else:
                        base_channel_list[group.channel_group.acq_name]["Unknown"] = len(
                            plp_msg_types[msg_type])
            # TAP data
            # V01 data
    return dd, base_channel_list  # , message_id_out_dict, spi_dbc


if __name__ == '__main__':

    import json
    import os
    # mdf_filename = r"C:\Users\wj1p7t\Desktop\Thunder_sample_logs\VSE_Issue\All_Files_Sample\TNDR1_MENI_20240511_203933_WDC5_deb_0008.MF4"
    t1 = time.time()
    mdf_filename = r"C:\Users\mfixlz\Downloads\TNDR1_DRUK_20240404_020921_WDC5_deb_0008.MF4"
    filepath, filename = os.path.split(mdf_filename)
    base_name = filename.split('.')[0]
    json_filename = os.path.join(
        r"C:\Users\mfixlz\Downloads\Versioned\pickles\eyeq_message_protocol_version_mapper.json")

    with open(json_filename) as f:

        spi_mapper_json = json.load(f)

    pickle_path = os.path.split(json_filename)[0]

    (SPI, mdf_bus_details
     # , message_id_out_dict, spi_dbc
     ) = extract_SPI_data(
        mdf_filename, spi_mapper_json, pickle_path)
    print('Finished ', time.time() - t1)
