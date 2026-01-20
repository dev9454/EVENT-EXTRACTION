from asammdf import MDF
from trimble_gnmm_df_parser import TrimbleGNSSDataframeParser
from trimble_gnmm_helper_functions import TrimbleGnssHelperFunctions

mdf4 = MDF(r"C:\Users\wj1p7t\Desktop\Thunder_sample_logs\0220\SHARK\TNDR1_SHRK_20240221_130741_WDC3_bus_0024.MF4")
# Retrieve data from trimble GPS channel
trimble_group_acq_name = "ETH917504"
initial_timestamp = mdf4.header.start_time
for group_idx, group in enumerate(mdf4.groups):
    if group.channel_group.acq_name == trimble_group_acq_name and group.data_blocks:
        trimble_data = mdf4.get_group(group_idx)
        trimble_data = trimble_data.reset_index()
        # Add initalTimestamp to all timestamps
        trimble_data['timestamps'] = trimble_data['timestamps'] + initial_timestamp.timestamp()
        pass
pass

parser = TrimbleGNSSDataframeParser()
parsedData = parser.process_mdf_dataframe(trimble_data)
helper = TrimbleGnssHelperFunctions(parsedData)
trimble_dataframe = helper.dump_data_to_dataframe()
trimble_dict = helper.dump_data_to_dictionary()
helper.dump_data_to_excel("trimbleData_from_MDF.xlsx")


