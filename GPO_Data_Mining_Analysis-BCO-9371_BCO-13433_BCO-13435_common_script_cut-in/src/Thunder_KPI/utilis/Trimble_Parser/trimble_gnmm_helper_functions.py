from trimble_gnmm_df_parser import TrimbleGNSSDataframeParser
import pandas as pd

"""
Helper class for Trimble GNSS data
Version: v0.0 - Base Release

@author: wj1p7t
"""


class TrimbleGnssHelperFunctions:
    def __init__(self, parsed_data):
        parser_obj = TrimbleGNSSDataframeParser()
        self.msg_config = parser_obj.return_message_def()
        self.parsed_data = list()
        self.gnss_dataframe = dict()
        self.gnss_dictionary = dict()
        self.get_parsed_data(parsed_data)

    def get_parsed_data(self, parsed_data):
        self.parsed_data = parsed_data

    def return_dataframe(self):
        # Returns dictionary of dataframes, with one dataframe per regex message group.
        return self.gnss_dataframe

    def dump_data_to_dataframe(self):
        # find which groups are in the message.
        # create skeleton dataframe
        self.gnss_dataframe = {group_id: pd.DataFrame(columns=["timestamp", "talker_id"]+list(self.msg_config[group_id].keys())+["checksum"]) for group_id in self.msg_config.keys()}
        # data imported in self.parsed_data
        for data in self.parsed_data["data"]:
            for parsed_data, debug_data in zip(data["parsed"], data["debug"]):
                df_name = debug_data["capture_regex"]
                # Add parsed data to dataframe as new row
                self.gnss_dataframe[df_name].loc[len(self.gnss_dataframe[df_name]), :] = parsed_data
        # Need to initialize keys list outside to prevent errors
        msg_list = list(self.gnss_dataframe.keys())
        for msg_key in msg_list:
            if self.gnss_dataframe[msg_key].empty:
                # if dataframe is empty, remove it from the final dictionary
                self.gnss_dataframe.pop(msg_key)
        return self.gnss_dataframe

    def dump_data_to_dictionary(self):
        self.dump_data_to_dataframe()
        for msg_key in self.gnss_dataframe.keys():
            self.gnss_dictionary[msg_key] = self.gnss_dataframe[msg_key].to_dict()
        return self.gnss_dictionary

    def dump_data_to_excel(self, excel_path):
        with pd.ExcelWriter(excel_path) as ExcelWriter:
            for msg_key in self.gnss_dataframe.keys():
                # clean up regex in name
                sheet_name = msg_key.replace(".", "")
                self.gnss_dataframe[msg_key].to_excel(ExcelWriter, sheet_name=sheet_name, index=False)
        pass





# parser = TrimbleGnssUdpPacketParser()
# pcap_file = r"BX982_output.pcapng"
# parsedData = parser.process_pcap(pcap_file)
# helper = TrimbleGnssHelperFunctions()
# helper.get_parsed_data(parsedData)
# helper.dump_data_to_dataframe()
# helper.dump_data_to_excel("trimbleData.xlsx")

pass