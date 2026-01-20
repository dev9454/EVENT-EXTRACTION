import numpy as np
import pandas as pd
from asammdf import MDF
import os
import sys
from scipy.io import savemat
import re

# if __package__ is None:
#     print('Here at none package')
#     sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
#     to_change_path = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(to_change_path)

#     from arxml_decoder_core import decode_eth_channel_by_arxml

# else:
#     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#     to_change_path = os.path.dirname(os.path.abspath(__file__))
#     to_change_path = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(to_change_path)
#     from arxml_decoder_core import decode_eth_channel_by_arxml


def aptiv_stream_check(payload):
    # Function check first 2 bytes of payload for Aptiv header info, and provide a filter for Aptiv stream data
    # print('inside function')
    # print(payload[0:2])
    aptiv_stream_identifier = [hex(byte)
                               for byte in np.bitwise_and(payload[0:2], 0xf0)]
    if "0xa0" in aptiv_stream_identifier:
        return True
    else:
        return False


def return_mac_addr_str(mac_addr):
    dest_MAC_str = ""
    if isinstance(mac_addr, np.ndarray):
        dest_MAC = mac_addr
    else:
        mac_addr = int(mac_addr)
        dest_MAC = mac_addr.to_bytes(6, byteorder='big')
    dest_MAC_str = ":".join(hex(byte).replace("0x", "") for byte in dest_MAC)
    return dest_MAC_str


def get_ip_header_info(ip_header_array):
    # Function to return IP header info from IP header bytes
    source_address = ip_header_array[12:16].astype(np.uint8)
    source_address_str = ".".join(str(byte) for byte in source_address)
    destination_address = ip_header_array[16:20].astype(np.uint8)
    destination_address_str = ".".join(str(byte)
                                       for byte in destination_address)
    return source_address_str, destination_address_str


def get_udp_header_info(udp_header_array):
    # Function to return UDP header info from UDP header bytes
    udp_header = udp_header_array.view(np.dtype(">H"))
    source_port = udp_header[0].astype(np.dtype(">H"))
    destination_port = udp_header[1].astype(np.dtype(">H"))
    length = udp_header[2].astype(np.dtype(">H"))
    checksum = udp_header[3].astype(np.dtype(">H"))
    return source_port, destination_port, length, checksum


def extract_aptiv_header_ver(payload):
    # Function to decode Aptiv header version, endianness, and header length
    aptiv_header_hex = [hex(byte) for byte in payload[0:2]]
    aptiv_header_ver = ""
    aptiv_header_size = 0
    big_endian = False
    # print(payload)
    aptiv_header_endianness_check = [
        hex(byte) for byte in np.bitwise_and(payload[0:2], 0xf0)]
    # ['0x20', '0x40']
    if "0xa0" == aptiv_header_endianness_check[0]:
        # big endian
        big_endian = True
        aptiv_header_ver = aptiv_header_hex[0]
        aptiv_header_size = payload[1]
    elif "0xa0" == aptiv_header_endianness_check[1]:
        # little endian
        aptiv_header_ver = aptiv_header_hex[1]
        aptiv_header_size = payload[0]
    return np.array([aptiv_header_ver, aptiv_header_size, big_endian])


# DECODE aptiv header

def decode_aptiv_header(UDP_payload, aptiv_header_ver, aptiv_header_size, big_endian):
    A3_LITTLE_ENDIAN = [
        ("versionInfo", ">u2"),
        ("sourceTxCnt", ">u2"),
        ("sourceTxTime", ">u4"),
        ("sourceInfo", ">u1"),
        ("sensorId", ">u1"),
        ("res1", ">u1"),
        ("res2", ">u1"),
        ("streamRefIndex", ">u4"),
        ("streamDataLen", ">u2"),
        ("streamTxCnt", ">u1"),
        ("streamNumber", ">u1"),
        ("streamVersion", ">u1"),
        ("streamChunks", ">u1"),
        ("streamChunkIdx", ">u1"),
        ("customerId", ">u1"),
    ]

    A4_LITTLE_ENDIAN = [
        ("versionInfo", ">u2"),
        ("sourceInfo", ">u1"),
        ("customerId", ">u1"),
        ("sensorId", ">u1"),
        ("sensorStatus", ">u1"),
        ("detectionCnt", ">u2"),
        ("streamRefIndex", ">u4"),
        ("utcTime", ">u4"),
        ("sourceTxTime", ">u4"),
        ("streamDataLen", ">u2"),
        ("streamNumber", ">u1"),
        ("streamTxCnt", ">u1"),
        ("streamVersion", ">u1"),
        ("streamChunks", ">u1"),
        ("streamChunkIdx", ">u1"),
        ("mode", ">u1"),
        ("streamChunkSize", ">u2"),
        ("packetTxCnt", ">u2"),
        ("packetTxTime", ">u4"),
        ("calSource", ">u1"),
        ("calRunningCnt", ">u1"),
        ("totalCalChunkCnt", ">u1"),
        ("res1", ">u1"),
    ]

    A5_LITTLE_ENDIAN = [
        ("versionInfo", ">u2"),
        ("sourceTxCnt", ">u2"),
        ("sourceTxTime", ">u4"),
        ("sourceInfo_0", ">u1"),
        ("sourceInfo_1", ">u1"),
        ("sourceInfo_2", ">u1"),
        ("sourceInfo_3", ">u1"),
        ("streamRefIndex", ">u4"),
        ("streamDataLen", ">u2"),
        ("streamTxCnt", ">u1"),
        ("streamNumber", ">u1"),
        ("streamVersion", ">u1"),
        ("streamChunksPerCycle", ">u1"),
        ("streamChunks", ">u2"),
        ("streamChunkIdx", ">u2"),
        ("res1", ">u1"),
        ("sensorId", ">u1"),
    ]

    # Big endian versions
    A3_BIG_ENDIAN = [(x, y.replace(">", "<")) for x, y in A3_LITTLE_ENDIAN]
    A4_BIG_ENDIAN = [(x, y.replace(">", "<")) for x, y in A4_LITTLE_ENDIAN]
    A5_BIG_ENDIAN = [(x, y.replace(">", "<")) for x, y in A5_LITTLE_ENDIAN]

    result_series_list = []
    for i in range(len(aptiv_header_ver)):
        aptiv_header_version_val = aptiv_header_ver[i]
        aptiv_header_length_val = int(aptiv_header_size[i])
        big_endian_val = big_endian[i]
        aptiv_header_bytes = UDP_payload[i][: aptiv_header_length_val].astype(
            np.uint8)
        aptiv_header_check = aptiv_header_version_val in [
            "0xa1", "0xa2", "0xa3"]
        if aptiv_header_check and not big_endian_val:
            type_to_decode = A3_LITTLE_ENDIAN
        elif aptiv_header_version_val in ["0xa1", "0xa2", "0xa3"] and big_endian_val:
            type_to_decode = A3_BIG_ENDIAN
        elif aptiv_header_version_val == "0xa4" and not big_endian_val:
            type_to_decode = A4_LITTLE_ENDIAN
        elif aptiv_header_version_val == "0xa4" and big_endian_val:
            type_to_decode = A4_BIG_ENDIAN
        elif aptiv_header_version_val == "0xa5" and not big_endian_val:
            type_to_decode = A5_LITTLE_ENDIAN
        elif aptiv_header_version_val == "0xa5" and big_endian_val:
            type_to_decode = A5_BIG_ENDIAN
        elif not aptiv_header_version_val:
            # Not Aptiv UDP data. Fill with 0s
            aptiv_header_dict = {datatype[0]                                 : 0 for datatype in A3_LITTLE_ENDIAN}
        if aptiv_header_version_val:
            header_dtype = np.dtype(type_to_decode)
            aptiv_header_data = np.frombuffer(
                aptiv_header_bytes[0:header_dtype.itemsize], dtype=header_dtype)[0]
            aptiv_header_dict = {datatype[0]: decoded_value for datatype, decoded_value in
                                 zip(type_to_decode, aptiv_header_data)}
        a3_attributes = [datatype[0] for datatype in A3_LITTLE_ENDIAN]
        # Include A3 terms in result
        result_series = {attrib: (aptiv_header_dict[attrib] if attrib in aptiv_header_dict.keys() else 0) for attrib in
                         a3_attributes}
        result_series["aptiv_payload_len"] = len(
            UDP_payload[i]) - aptiv_header_length_val
        result_series_list.append(result_series)
    return_df = pd.DataFrame(result_series_list)
    return return_df


def process_mdf_eth_data(channel_df):
    result_df = pd.DataFrame()
    Src_MAC_Addr = []
    Dst_MAC_Addr = []
    src_array = np.array(channel_df["Source"])
    dst_array = np.array(channel_df["Destination"])
    return_mac_addr_str_vfunc = np.vectorize(
        return_mac_addr_str)  # vectorize the function
    Src_MAC_Addr = return_mac_addr_str_vfunc(src_array)
    Dst_MAC_Addr = return_mac_addr_str_vfunc(dst_array)

    # Extract IP UDP header data

    ethertype = np.array(channel_df["EtherType"])
    databytes = np.array(channel_df['DataBytes'])

    eth_payload_to_process_list = []
    ip_header_list = []
    udp_header_list = []
    UDP_payload_list = []
    IP_src_list, IP_dst_list = [], []
    UDP_src_port_list, UDP_dst_port_list, UDP_length_list, UDP_checksum_list = [], [], [], []
    for i in range(len(ethertype)):
        ether = ethertype[i]
        databytes_val = databytes[i]
        if ether == 2048:
            # IPv4 header
            eth_payload_to_process = databytes_val
        elif ether == 33024:
            # VLAN Header - remove 4 bytes
            eth_payload_to_process = databytes_val[4:]
        else:
            continue
        if eth_payload_to_process.size:
            ip_header = eth_payload_to_process[0:20]
            udp_header = eth_payload_to_process[20:28]
            UDP_payload = np.array(eth_payload_to_process[28:])
            IP_src, IP_dst = get_ip_header_info(ip_header)
            UDP_src_port, UDP_dst_port, UDP_length, UDP_checksum = get_udp_header_info(
                udp_header)
        eth_payload_to_process_list.append(eth_payload_to_process)
        ip_header_list.append(ip_header)
        udp_header_list.append(udp_header)
        UDP_payload_list.append(UDP_payload)
        IP_src_list.append(IP_src)
        IP_dst_list.append(IP_dst)
        UDP_src_port_list.append(UDP_src_port)
        UDP_dst_port_list.append(UDP_dst_port)
        UDP_length_list.append(UDP_length)
        UDP_checksum_list.append(UDP_checksum)

    if len(UDP_payload_list) == 0:
        return pd.DataFrame()

    # ID if channels have Aptiv data
    # print(UDP_payload_list)
    UDP_payload_array = np.array(UDP_payload_list, dtype=object)
    # print('after')
    # print(UDP_payload_array.shape)
    aptiv_stream_check_vfunc = np.vectorize(
        aptiv_stream_check, signature="(n) -> ()")
    aptiv_payload_check = aptiv_stream_check_vfunc(UDP_payload_array)

    # Extract header length, version, endianess
    extract_aptiv_header_ver_vfunc = np.vectorize(extract_aptiv_header_ver, signature="(n) -> ()",
                                                  otypes=[np.object_, np.uint8, np.bool_])
    # print( extract_aptiv_header_ver_vfunc(UDP_payload_array))
    # aptiv_header_ver, aptiv_header_size, big_endian = extract_aptiv_header_ver_vfunc(UDP_payload_array)
    temp = extract_aptiv_header_ver_vfunc(UDP_payload_array)

    aptiv_header_ver = [temp[i][0] for i in range(len(temp))]
    aptiv_header_size = [temp[i][1] for i in range(len(temp))]
    big_endian = [temp[i][2] for i in range(len(temp))]

    # constructine all into a dataframe

    result_df["timestamps"] = channel_df["timestamps"]
    result_df["BusChannel"] = channel_df["BusChannel"]
    result_df["Src_MAC_Addr"] = Src_MAC_Addr
    result_df["Dst_MAC_Addr"] = Dst_MAC_Addr

    result_df['IP_src'] = IP_src_list
    result_df['IP_dst'] = IP_dst_list
    result_df['UDP_src_port'] = UDP_src_port_list
    result_df['UDP_dst_port'] = UDP_dst_port_list
    result_df["UDP_Length"] = UDP_length_list
    result_df["UDP_Checksum"] = UDP_checksum_list
    result_df["UDP_Payload"] = UDP_payload_list

    result_df['aptiv_payload_check'] = aptiv_payload_check
    result_df["aptiv_header_ver"] = aptiv_header_ver
    result_df["aptiv_header_len"] = aptiv_header_size
    result_df["big_endian"] = big_endian

    decode_aptiv_df = decode_aptiv_header(
        UDP_payload_array, aptiv_header_ver, aptiv_header_size, big_endian)

    return_df = result_df.join(decode_aptiv_df)

    return return_df


def stream_stats_main(fileName):

    mdf4_file_handle = MDF(fileName)
    initial_timestamp = mdf4_file_handle.header.start_time
    channels_df_list = list()
    total_processing = 0
    # Detect channels with ETH data

    if 'ETH_Frame' in mdf4_file_handle.channels_db.keys():
        eth_frame_channel_groups = [
            channel_group[0] for channel_group in mdf4_file_handle.channels_db['ETH_Frame']]
    else:
        eth_frame_channel_groups = []

    df_process_list = []
    for group_idx, group in enumerate(mdf4_file_handle.groups):

        if not (group_idx in eth_frame_channel_groups):
            # Skip if not ETH channel
            continue
        # ID if channels have Aptiv data
        channel_df = mdf4_file_handle.get_group(group_idx)
        if not group.data_blocks:
            # Skip if channel has no data
            continue
        # RB 11072024
        # If bus channel not in IFV600 (AD5) or FLR4 (AD5) skip
        bus_channel = int(re.search(r'[\d]+',
                                    group.channel_group.acq_name)[0])

        if not bus_channel in [17, 18]:
            continue
        #######################################
        # Here channel is confirmed to be ETH and also have data
        channel_df = channel_df.reset_index()
        channel_df['timestamps'] = channel_df['timestamps'] + \
            initial_timestamp.timestamp()
        column_rename_mapping = dict()
        for column_name in channel_df.columns:
            if "." in column_name:
                column_rename_mapping[column_name] = column_name.split(".")[-1]
        channel_df = channel_df.rename(columns=column_rename_mapping)
        print("Processing channel: " + group.channel_group.acq_name)
        unique_payloads = list(channel_df['DataLength'].unique())
        for idx_, i in enumerate(unique_payloads):
            # print(
            #     f'*********************   {group.channel_group.acq_name}, {idx_}')
            process_df = channel_df.loc[channel_df['DataLength'] == i]
            channel_df_processed = process_mdf_eth_data(process_df)
            channel_df_processed['Processing channel'] = group.channel_group.acq_name
            df_process_list.append(channel_df_processed)
    #     channel_df_processed = channel_df.apply(process_mdf_eth_data, axis=1)

    final_df = pd.concat(df_process_list)
    return final_df


if __name__ == '__main__':
    import os
    # from arxml_decoder_core import decode_eth_channel_by_arxml
    from arxml_decoder_core_mux import decode_eth_channel_by_arxml

    # ARXML Based Decoding here

    hit = 1

    fileName = os.path.join(r'C:\Users\mfixlz\Downloads',
                            'TNDR1_KALU_20240801_200642_WDC5_bus_0001.MF4')
    flr4_arxml_root_path = os.path.join(r'C:\Users\mfixlz\Downloads',
                                        'flr4_dbc')
    logs = [fileName]
    # stream_stats_main(fileName)
    # fileName = r"local_filelist"

    # with open(fileName) as f:
    #     logs = [log.rstrip('\n') for log in f]

    list_df = []
    for i in logs:
        df = stream_stats_main(i)
        df['logname'] = fileName
        list_df.append(df)

    all_concat_df = pd.concat(list_df)
    final_df = all_concat_df.sort_values(by=['timestamps'], ascending=True)
    pdu_decode = decode_eth_channel_by_arxml(
        final_df,
        os.path.join(flr4_arxml_root_path,
                     'ENET-AD5_ECU_Composition_S1_11_28_2023_Ver_12.6.arxml'
                     # 'ENET-AD5_ECU_Composition_S1_02_05_2024_Ver_13.0.arxml'
                     # 'ENET-AD5_S1_01_05_2023_Ver_10.arxml'
                     )
    )
    savemat(os.path.join(flr4_arxml_root_path,
                         "result_12p6.mat"),
            pdu_decode, long_field_names=True,
            oned_as='column', do_compression=True)
    # final_df.to_excel('0000_output.xlsx')

    print('here')
