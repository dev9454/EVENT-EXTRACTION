import pandas
from asammdf import MDF
from trimble_gnmm_df_parser import TrimbleGNSSDataframeParser
from trimble_gnmm_helper_functions import TrimbleGnssHelperFunctions
import pandas
import os

mdf4_folder = r"C:\Users\wj1p7t\Desktop\Thunder_sample_logs\0220\SHARK"
trimble_dataframe_list = []

for filename in os.listdir(os.fsencode(mdf4_folder)):
    filename = os.fsdecode(filename)
    if filename.endswith(".mf4") or filename.endswith(".MF4"):
        print("Processing "+filename+"...")
        full_filename = os.path.join(mdf4_folder, filename)
        mdf4 = MDF(full_filename)
        # Retrieve data from trimble GPS channel
        trimble_group_acq_name = "ETH917504"
        initial_timestamp = mdf4.header.start_time
        trimble_data = dict()
        for group_idx, group in enumerate(mdf4.groups):
            if group.channel_group.acq_name == trimble_group_acq_name and group.data_blocks:
                trimble_data = mdf4.get_group(group_idx)
                trimble_data = trimble_data.reset_index()
                # Add initalTimestamp to all timestamps
                trimble_data['timestamps'] = trimble_data['timestamps'] + initial_timestamp.timestamp()
                pass
        pass
        parser = TrimbleGNSSDataframeParser()
        if not trimble_data.empty:
            parsedData = parser.process_mdf_dataframe(trimble_data)
            helper = TrimbleGnssHelperFunctions(parsedData)
            trimble_dataframe = helper.dump_data_to_dataframe()
            trimble_dataframe["PTNL,GGK"]["log_path"] = mdf4_folder
            trimble_dataframe["PTNL,GGK"]["log_name"] = filename
            trimble_dataframe_list.append(trimble_dataframe["PTNL,GGK"])
pass
trimble_dataframe_full = pandas.concat(trimble_dataframe_list)
trimble_dataframe_full.to_excel(os.path.join(mdf4_folder, "gps.xlsx"), sheet_name = "PTNL,GGK")
pass



