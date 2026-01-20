import scapy.all as scapy
from asammdf import MDF
from comm_headers_from_array import IPHeader, UDPHeader
import os


def mdf2pcap_converter(bus_id, mdf_filename, **kwargs):
    # args
    # srcip - source IP
    # dstip - destination IP
    # srcport - source port
    # dstport - destination port
    filepath, filename = os.path.split(mdf_filename)
    base_name = filename.split('.')[0]
    pcap_filename = os.path.join(filepath, str(base_name) + "_" + bus_id + '.pcap')
    mdf4 = MDF(mdf_filename)
    # Retrieve data from channel
    group_acq_name = bus_id
    initial_timestamp = mdf4.header.start_time
    packet_list = list()
    for group_idx, group in enumerate(mdf4.groups):
        if group.channel_group.acq_name == group_acq_name and group.data_blocks:
            channel_data = mdf4.get_group(group_idx)
            channel_data = channel_data.reset_index()
            # Add initalTimestamp to all timestamps
            channel_data['timestamps'] = channel_data['timestamps'] + initial_timestamp.timestamp()
            for frame in channel_data.iterrows():
                frame_data = frame[1]
                eth_payload = frame_data["ETH_Frame.ETH_Frame.DataBytes"]
                ip_header = IPHeader(eth_payload[0:20])
                udp_header = UDPHeader(eth_payload[20:28])

                if kwargs:
                    # skip appending packet if there is a mismatch
                    if "srcip" in kwargs.keys():
                        if kwargs["srcip"] != ip_header.source_address_str:
                            continue
                    if "dstip" in kwargs.keys():
                        if kwargs["dstip"] != ip_header.destination_address_str:
                            continue
                    if "srcport" in kwargs.keys():
                        if kwargs["srcport"] != udp_header.source_port:
                            continue
                    if "dstport" in kwargs.keys():
                        if kwargs["dstport"] != udp_header.destination_port:
                            continue
                
                ether = scapy.Ether(dst=":".join(hex(byte).replace("0x", "") for byte in
                                                 frame_data["ETH_Frame.ETH_Frame.Destination"].to_bytes(6,
                                                                                                        byteorder='big')),
                                    src=":".join(hex(byte).replace("0x", "") for byte in
                                                 frame_data["ETH_Frame.ETH_Frame.Source"].to_bytes(6, byteorder='big')),
                                    type=frame_data["ETH_Frame.ETH_Frame.EtherType"])
                packet = ether / eth_payload.tobytes()
                packet.time = frame[1].timestamps
                packet_list.append(packet)
                pass
    scapy.wrpcap(pcap_filename, packet_list)

GPS_ID = "ETH917504"
AURIX_ID = "ETH17"
LIDAR_ID = "ETH8390304"
bus_ID = "ETH917504"
mdf2pcap_converter(LIDAR_ID,
                   r"C:\Users\wj1p7t\Desktop\Thunder_sample_logs\Druk LIDAR-GPS test\TNDR1_DRUK_20240222_203647_WDC4_bus_0000.MF4")
