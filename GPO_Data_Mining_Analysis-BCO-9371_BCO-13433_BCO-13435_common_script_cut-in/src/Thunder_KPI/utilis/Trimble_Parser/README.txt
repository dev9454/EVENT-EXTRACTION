extract_gps_from_mdf_main - Main code that puts all the functions together to achieve parsing of trimble data out of Thunder logs. Run on WL Mule logs which have channel "ETH917504". Error handling and abstraction needs to be improved
trimble_gnmm_df_parser - Core parser code that takes input dataframe from MF4 and creates parsed struct of signals. Output structure has lots of debug information. Type casting has not been implemented yet
trimble_gnmm_helper_functions - Takes the raw output of parser and converts it into popular formats such as df, dictionary, excel sheet etc.
trimble_udp_messages_def - "Stream definiton" equivalent for Trimble data
plot_thunder_route_trimble - helper matlab code that can load a trimble excel sheet and plot the route on a basemap

Trimble GPS messages definition can be found here: https://receiverhelp.trimble.com/oem-gnss/index.html#NMEA-0183messages_MessageOverview.html?TocPath=Output%2520Messages%257CNMEA-0183%2520Messages%257C_____1