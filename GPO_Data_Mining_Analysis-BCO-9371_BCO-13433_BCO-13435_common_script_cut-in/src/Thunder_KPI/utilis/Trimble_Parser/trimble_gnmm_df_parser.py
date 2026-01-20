import re

import numpy as np
import yaml
from pathlib import Path

"""
Base class for parsing Trimble GNSS 
Version: v0.0 - Base Release

@author: wj1p7t
"""
TRIMBLE_UDP_DEST_PORT = 28002

class IPHeader:
    def __init__(self, data_bytes):
        #IP Header is the first 20 bytes
        ip_header = data_bytes
        self.version = ip_header[0] >> 4
        # length of IP header is measured in 32 bit words - 32 bit = 4 bytes
        self.header_length = (ip_header[0] & 0xf)*4
        self.tos = ip_header[1]
        self.total_length = ip_header[2:4].view(np.dtype(">H"))[0]
        self.identification = ip_header[4:6].view(np.dtype(">H"))[0]
        self.flags = ip_header[6:8].view(np.dtype(">H"))[0] >> 13
        self.fragment_offset = self.flags & 0x1fff
        self.ttl = ip_header[8]
        self.protocol = ip_header[9]
        self.header_checksum = ip_header[10:12].view(np.dtype(">H"))[0]
        self.source_address = ip_header[12:16].astype(np.uint8)
        self.source_address_str = ".".join(str(byte) for byte in self.source_address)
        self.destination_address = ip_header[16:20].astype(np.uint8)
        self.destination_address_str = ".".join(str(byte) for byte in self.destination_address)
        pass

class UDPHeader:
    def __init__(self, data_bytes):
        # UDP header is 8 bytes after IP header
        udp_header = data_bytes.view(np.dtype(">H"))
        self.source_port = udp_header[0]
        self.destination_port = udp_header[1]
        self.length = udp_header[2]
        self.checksum = udp_header[3]
        pass


class TrimbleGNSSDataframeParser:
    def __init__(self):
        self.msg_def_text = yaml.safe_load(Path("trimble_udp_messages_def.yaml").read_text())
        self.msg_keys_regex = [(msg_name, "\$"+msg_name) for msg_name in list(self.msg_def_text.keys())]
        self.talker_id_capture_filter = re.compile("\$(?P<talkerid>[A-Z]+?),", re.MULTILINE)

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
        # Returns data struct with signal content for each message based on the span set in the yaml file.
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

    def parse_gps_df_row(self, df_row, cache):
        row_series = df_row[1]
        # data seems to have a 28 byte header (IP Header(20 bytes) + UDP Header(8 Bytes))
        ipHeader = IPHeader(row_series["ETH_Frame.ETH_Frame.DataBytes"][0:20])
        udpHeader = UDPHeader(row_series["ETH_Frame.ETH_Frame.DataBytes"][20:28])
        payload_bytes = row_series["ETH_Frame.ETH_Frame.DataBytes"].tobytes()[28:]
        packet_data = dict()
        packet_groups = set()
        packet_talker_ids = set()
        timestamp = row_series['timestamps']
        if udpHeader.destination_port == TRIMBLE_UDP_DEST_PORT:
            # Packet data contains multiple lines, with one message per line
            packet_data["timestamp"] = timestamp
            packet_data["raw"] = list()
            packet_data["parsed"] = list()
            packet_data["debug"] = list()
            packet_talker_ids = set(self.talker_id_capture_filter.findall(payload_bytes.decode("utf-8")))
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
                    #checksum present
                    data_to_parse[-1], checksum = data_to_parse[-1].split("*")
                else:
                    cache = lineToParse
                    continue

                msg_key = self.return_matched_msg_key(lineToParse)
                if msg_key:
                    packet_groups.add(msg_key)
                    # Add timestamp of message and talker id
                    # data contains only the signal content, debug contains information on how the data was extracted
                    msg_data_dict = {"timestamp": timestamp, "talker_id": data_to_parse[0]}
                    msg_debug_dict = {"timestamp": timestamp, "talker_id": data_to_parse[0], "capture_regex": msg_key}
                    # extract and add data for signals in message
                    msg_data_dict.update(self.get_msg_data(msg_key, data_to_parse[1:]))
                    msg_data_dict["checksum"] = checksum
                    packet_data["parsed"].append(msg_data_dict)
                    packet_data["debug"].append(msg_debug_dict)
        else:
            # Data for other ports
            pass

        return packet_data, packet_groups, packet_talker_ids, cache

    def process_mdf_dataframe(self, mdf_dataframe):
        parsed_data = dict()
        parsed_data["msg_regex_groups"] = set()
        parsed_data["unique_talker_ids"] = set()
        parsed_data["data"] = list()
        cache = ""
        for row in mdf_dataframe.iterrows():
            packet_data, packet_groups, packet_talker_ids, cache = self.parse_gps_df_row(row, cache)
            # append signals from each packet to parsed data struct
            if packet_data:
                # Only append if packet data is present
                parsed_data["data"].append(packet_data)
                parsed_data["msg_regex_groups"].update(packet_groups)
                parsed_data["unique_talker_ids"].update(packet_talker_ids)
            else:
                pass
        return parsed_data
