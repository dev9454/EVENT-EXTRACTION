import sys

if "lin" in sys.platform:
    print('platform :', sys.platform)
    sys.path.append(r'/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru')
    from mat_file_to_dict import loadmat
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.grid_gen import GridGen
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.support_functions import grid_decoder

    sys.path.append(
        r'/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru/repo/GPO_Data_Mining_Analysis/src/Lidar_KPI')
    from Lidartrackcompare import ABCompare

elif "win" in sys.platform:
    print('platform :', sys.platform)
    sys.path.append(r'C:\Users\pjp8lm\Desktop\Data_Mining_and_analystics\Repos')
    from ds_team_tools.hpc_frameworks.utils.mat_file_to_dict import loadmat
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.grid_gen import GridGen
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.support_functions import grid_decoder

    sys.path.append(r'C:\Users\pjp8lm\Desktop\Thunder\Repo\GPO_Data_Mining_Analysis\src\Lidar_KPI')
    from Lidartrackcompare import ABCompare

# from ds_team_tools.hpc_frameworks.utils.mat_file_to_dict import loadmat, load_large_mat_file
# from ds_team_tools.hpc_frameworks.utils.support_functions import search_stream
# from ds_team_tools.hpc_frameworks.utils.mat_file_to_dict import loadmat

from datetime import datetime, timezone
import os
import pandas as pd
import numpy as np
import time
from scipy.spatial import distance as dis
import openpyxl
import json
import warnings
from openpyxl import load_workbook

warnings.filterwarnings('ignore')


class ABcompareEuclidean:
    def correlation_ab(self, gt_data, log_data):
        # log_data = log_data[log_data['Longitudinal_position'] != 0] # just considering objects in front of the vehicle
        dfMatchedlog = pd.DataFrame()
        dfMatchedGT = pd.DataFrame()
        orphans = pd.DataFrame()
        # indexframes_persec = sum(vehicledata['Fusion_Index'] == vehicledata.Fusion_Index.iloc[1,])
        # Fusion_Indexes = np.unique(np.array(vehicledata.Fusion_Index))
        ctime = np.unique(log_data['cTime_string'])
        for timeframe in range(len(ctime)):  # len(Fusion_Indexes) ##testing

            ''' # Original code for comparing the CADM to RESIM objects
            #print(timeframe)
            # dataframe_1_part = a.iloc[ (timeframe * indexframes_persec):(indexframes_persec + timeframe * indexframes_persec)]
            # dataframe_2_part = b.iloc[(timeframe * indexframes_persec):(indexframes_persec + timeframe * indexframes_persec)]
            '''
            dataframe_1_part = gt_data[gt_data['cTime'] == ctime[timeframe]]
            dataframe_2_part = log_data[log_data['cTime_string'] == ctime[timeframe]]
            df1 = dataframe_1_part.loc[:, ['Longitudinal_position', 'Lateral_position']]
            df2 = dataframe_2_part.loc[:, ['Longitudinal_position', 'Lateral_position']]
            if not df1.empty and not df2.empty:
                matching_dist = dis.cdist(df1, df2,
                                          metric='euclidean')  # scipy.spatial.distance.cdist  returns 2D with ij refering eucd(df1[i],df2[j]) distance
                dfmatching = pd.DataFrame(matching_dist)  # 2D DataFrame - 96x2 size
                threshold_pos = 2.0  #####threshold for distance similarity######
                dfmatching = dfmatching.apply(self.candidate_pair_weightes,
                                              args=(threshold_pos, dataframe_1_part, dataframe_2_part), axis=1)
                # matching_df will be 96x1 size  - with value representing which resimObjIdx best match. If -1 then miss match
                missing_list = np.where(dfmatching == -1)[0]  # tuple so 1st element has all the list
                matchingGTIndices = np.where(dfmatching != -1)[0]
                matchingLogIndices = dfmatching[dfmatching != -1].values
                if matchingLogIndices.size != 0:
                    dfMatchedlog = pd.concat([dfMatchedlog, dataframe_2_part.iloc[matchingLogIndices,]])
                if matchingGTIndices.size != 0:
                    dfMatchedGT = pd.concat([dfMatchedGT, dataframe_1_part.iloc[matchingGTIndices,]])
                if missing_list.size != 0:
                    orphans = pd.concat([orphans, dataframe_1_part.iloc[missing_list,]])
            elif df2.empty:
                orphans = pd.concat([orphans, dataframe_1_part])
            else:
                continue

        dfMatchedGT_df = pd.DataFrame(dfMatchedGT)
        dfMatchedlog_df = pd.DataFrame(dfMatchedlog)
        orphans_df = pd.DataFrame(orphans)
        return dfMatchedGT_df, dfMatchedlog_df, orphans_df

    def candidate_pair_weightes(self, candidates, threshold, HARPFrame, GTFrame):
        row = int(candidates.name)
        candidates = candidates[(candidates < threshold)].index  # slice series to have values < threshold
        candidates = pd.Series(candidates)
        if candidates.empty:
            return -1
        if candidates.size == 1:
            return candidates[0]
        distances = [self.weighted_distance(HARPFrame.iloc[row,], GTFrame.iloc[int(GTindex),]) for GTindex in
                     candidates]
        candidates.apply(lambda GTindex: self.weighted_distance(HARPFrame.iloc[row,], GTFrame.iloc[int(GTindex),]))
        best = candidates[distances.index(min(distances))]
        return best

    def weighted_distance(self, a, b):  # a : HARP/tracker data series and b: lidar data series
        distance = 0
        weightes = [1, 1, 0.5, 0.5, 0.3, 0.2, 0.2, 0.07]
        features = ["Longitudinal_position", "Lateral_position", 'Longitudinal_velocity', 'Lateral_velocity', 'Heading',
                    'Target_Length', 'Target_Width', 'Age']
        for i in range(0, len(features)):
            distance = distance + weightes[i] * ((a[features[i]] - b[features[i]]) ** 2)
        return distance


def to_datetime(epoch_ts):
    """Convert `firstPackageTimestamp` and `reportedTimestamp` to datetime object"""
    # return datetime.fromtimestamp(int(epoch_ts) / 1000000, timezone.utc)
    return datetime.fromtimestamp(epoch_ts, timezone.utc)


def get_logdata(mat_data, fusion_ctime_corrected_string):
    # fusion_ctime = mat_data['mudp']['fus']['header']['cTime']  # units in microseconds

    # reading fusion time from object fusion header

    fusion_ctime = mat_data['mudp']['Object_Fusion']['header']['time'][:, 0]
    # fusion_ctime_ns = [str(i * 1000000000) for i in fusion_ctime]
    fusion_ctime_utc = [to_datetime(i).isoformat() for i in fusion_ctime]
    fusion_ctime_str = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]
    # log data

    vehicledata_filter = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['Fusion_output']
    # vehicledata_filter = mat_data['mudp']
    log_pos_vehi = vehicledata_filter['vcs_longposn']
    lat_pos_vehi = vehicledata_filter['vcs_latposn']
    log_vel_vehi = vehicledata_filter['vcs_longvel']
    lat_vel_vehi = vehicledata_filter['vcs_latvel']
    log_acc_vehi = vehicledata_filter['vcs_longaccel']
    lat_acc_vehi = vehicledata_filter['vcs_lataccel']
    speed_vehi = vehicledata_filter['speed']
    heading_vehi = vehicledata_filter['vcs_heading']
    # heading_rate_vehi = vehicledata_filter['heading_rate']
    tar_len_vehi = vehicledata_filter['length']
    tar_wid_vehi = vehicledata_filter['width']
    age_vehi = vehicledata_filter['age']
    fusion_index_vehi = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['fusion_index']
    try:
        reduced_id_array = vehicledata_filter['reducedID']
    except Exception as e:
        reduced_id_array = vehicledata_filter['reduced_id']
    cTime_vehi = fusion_ctime_str
    object_id_array = vehicledata_filter['id']

    Longitudinal_position = []
    Lateral_position = []
    Longitudinal_velocity = []
    Lateral_velocity = []
    Heading = []
    Target_length = []
    Target_width = []
    Age = []
    Fusion_Index = []
    Time = []
    Index = []
    cTime_string = []
    cTime = []
    reduced_ID = []
    corrected_ctime = []
    object_id_temp = []

    for time in range(0, len(fusion_index_vehi)):
        for index in range(0,  len(log_pos_vehi[0])):
            Time.append(time + 1)
            Index.append(index + 1)
            object_id_temp.append(object_id_array[time, index])
            Longitudinal_position.append(log_pos_vehi[time, index])
            Lateral_position.append(lat_pos_vehi[time, index])
            Longitudinal_velocity.append(log_vel_vehi[time, index])
            Lateral_velocity.append(lat_vel_vehi[time, index])
            Heading.append(heading_vehi[time, index])
            Target_length.append(tar_len_vehi[time, index])
            Target_width.append(tar_wid_vehi[time, index])
            Age.append(age_vehi[time, index])
            Fusion_Index.append(fusion_index_vehi[time])
            reduced_ID.append(reduced_id_array[time, index])
            cTime_string.append(cTime_vehi[time])
            cTime.append(fusion_ctime[time])
            corrected_ctime.append(fusion_ctime_corrected_string[time])
    columns_names = ['cTime', 'corrected_ctime', 'obj_ID', 'reduced_ID', 'Fusion_Index', 'Longitudinal_position',
                     'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity', 'Heading', 'Target_Length',
                     'Target_Width', 'Age']
    columns_data = [cTime, corrected_ctime, object_id_temp, reduced_ID, Fusion_Index, Longitudinal_position,
                    Lateral_position, Longitudinal_velocity, Lateral_velocity,
                    Heading, Target_length, Target_width, Age]
    zip_data = zip(columns_names, columns_data)
    dict_data = dict(zip_data)
    vehicledata1_df = pd.DataFrame(data=dict_data)
    return vehicledata1_df, fusion_ctime_str, fusion_ctime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def nexus_details():
    nexus_url = "https://nexus.aptiv.com/api/"
    nexus_pwd = "Test@1234"
    nexus_user = "GPO_DMA_NA@aptiv.com"
    return nexus_url, nexus_user, nexus_pwd


def convertToVcs(cuboidlist_sample, calibration_data):
    from pyquaternion import Quaternion
    if "lin" in sys.platform:
        sys.path.append(r'/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru')
        from autogt_tsel.quaternion_transform import QuaternionTransform
    elif "win" in sys.platform:
        sys.path.append(r'C:\Users\pjp8lm\Desktop\Data_Mining_and_analystics\Repos')
        from autogt_tsel.quaternion_transform import QuaternionTransform
    # code spliced from lines
    if len(cuboidlist_sample.cuboids) == 0:
        return cuboidlist_sample  # do nothing if cuboidList is empty
    else:
        for idx, cuboid in enumerate(cuboidlist_sample.cuboids):
            cub_rotation = Quaternion(**cuboid['r'])
            center = [cuboid['c']['x'], cuboid['c']['y'], cuboid['c']['z']]
            sizes = [cuboid['s']['x'], cuboid['s']['y'], cuboid['s']['z']]
            velocity = [cuboid.get('v', {}).get('x', 0.0), cuboid.get('v', {}).get('y', 0.0),
                        cuboid.get('v', {}).get('z', 0.0)]

            qt_cal = QuaternionTransform(rotation=Quaternion(**calibration_data['rotation']),
                                         translation=np.array(
                                             [calibration_data['position'][index] for index in ['x', 'y', 'z']]))

            new_center = (qt_cal @ np.array(center)).tolist()
            new_rot = list((qt_cal.rotation * cub_rotation).yaw_pitch_roll)
            new_rot = [new_rot[0] + np.pi / 2., new_rot[1], new_rot[2]]
            qt = QuaternionTransform.from_angles(yaw=new_rot[0], pitch=new_rot[1], roll=new_rot[2], order='ypr')
            qt_new = qt.rotation
            cuboid['r'] = {key: qt_new.q[index] for index, key in enumerate(cuboid['r'])}
            new_vel = (qt_cal.rotation.rotation_matrix @ np.array(velocity)).tolist()
            new_sizes = [sizes[1], sizes[0], sizes[2]]
            for index, key in enumerate(['x', 'y', 'z']):
                cuboid['c'][key] = new_center[index]
                cuboid['s'][key] = new_sizes[index]
                cuboid['v'][key] = new_vel[index]
        return cuboidlist_sample


def cubiodDimensions(dimensions):
    x = dimensions['x']
    y = dimensions['y']
    height = dimensions['z']

    if x > y:
        length = x
        width = y
    else:
        length = y
        width = x

    return {'length': length, 'width': width, 'height': height}


def convertGTCuboidsToDf(idx, sample):
    gtStreamData_df = pd.DataFrame({"frame": [],
                                    "timestamp": [],
                                    "track_id": [],
                                    "Longitudinal_position": [],
                                    "Lateral_position": [],
                                    "Longitudinal_velocity": [],
                                    "Lateral_velocity": [],
                                    "vcsHeading": [],
                                    "Target_Length": [],
                                    "Target_Width": [],
                                    "Age": [],
                                    "confidence": []
                                    })
    if sample.cuboids:
        # cuboid.c is center coordinates - measured to centroid of the cuboid
        # cuboid.s is size coordinates - full length and width - length > width
        # cuboid.r is the quaternion - will need conversion to euclidean yaw
        # cuboid.v is velocity

        # attribs.n.vcs_pointing is the yaw angle
        # attribs.n.uniqueID is the trackID
        # length = attribs.n.len1+attribs.n.len2
        # width = attribs.n.wid1+attribs.n.wid2

        # timestamp is sample.firstPackageTimestamp
        try:
            timestamp = [sample.reportedTimestamp for cuboid in sample.cuboids]
        except Exception as e:
            timestamp = [sample.timestamp for cuboid in sample.cuboids]
        track_id = [cuboid["uuid"] for cuboid in sample.cuboids]
        long_pos = [cuboid["c"]["x"] for cuboid in sample.cuboids]
        lat_pos = [cuboid["c"]["y"] for cuboid in sample.cuboids]
        long_vel = [cuboid["v"]["x"] for cuboid in sample.cuboids]
        lat_vel = [cuboid["v"]["y"] for cuboid in sample.cuboids]
        # QuaternionAxis_arr = [Quaternion(cuboid["r"].values()).axis for cuboid in sample.cuboids]# Rotational quaternion axis
        # heading = [np.arctan2(QuaternionAxis[1], QuaternionAxis[0])*2 for QuaternionAxis in QuaternionAxis_arr] #This gives VSE heading
        heading = [np.arctan2(2 * (cuboid["r"]["w"] * cuboid["r"]["z"] + cuboid["r"]["x"] * cuboid["r"]["y"]),
                              1 - 2 * (cuboid["r"]["y"] * cuboid["r"]["y"] + cuboid["r"]["z"] * cuboid["r"]["z"]))
                   for
                   cuboid in sample.cuboids]  # Wikipedia formula
        length = [cubiodDimensions(cuboid["s"])['length'] for cuboid in sample.cuboids]
        width = [cubiodDimensions(cuboid["s"])['width'] for cuboid in sample.cuboids]
        height = [cubiodDimensions(cuboid["s"])['height'] for cuboid in sample.cuboids]
        age = [1 for cuboid in sample.cuboids]  # Age is currently not logged in the stream
        confidence = [cuboid["confidence"] for cuboid in sample.cuboids]
        # Insert all data into df dictionary
        gtStreamData_df = pd.DataFrame({"frame": idx,
                                        "timestamp": timestamp,
                                        "track_id": track_id,
                                        "Longitudinal_position": long_pos,
                                        "Lateral_position": lat_pos,
                                        "Longitudinal_velocity": long_vel,
                                        "Lateral_velocity": lat_vel,
                                        "vcsHeading": heading,
                                        "Target_Length": length,
                                        "Target_Width": width,
                                        "Age": age,
                                        "confidence": confidence
                                        })

    return gtStreamData_df


def rotation_position_correction(df_1):
    import math
    df_1['Longitudinal_position'] = df_1['Longitudinal_position'] - 3.853
    return df_1


def assign_grid(df):
    # Creating Grid Gen
    g = GridGen()
    g.grid_positions()
    g.create_bounding_boxes()

    grid_used = []
    cell = []
    df['grid'] = np.NAN
    df['cell'] = np.NAN
    none_index = []
    none_array = [True] * len(df)
    lat_array = np.array(df['Lateral_position'])
    long_array = np.array(df['Longitudinal_position'])
    for i in range(len(df)):
        try:
            # lat = df['Lateral_position_GT'][i]
            # long = df['Longitudinal_position_GT'][i]
            # lat_array = np.array(df['Lateral_position_GT'])
            # long_array = np.array(df['Longitudinal_position_GT'])
            lat = lat_array[i]
            long = long_array[i]
            grid = g.box_index(lat, long)
            if grid is None:
                none_index.append(i)
                none_array[i] = False
                continue
            grid_used.append(grid)
            ccl = grid_decoder(grid)
            cell.append(ccl)
            df['grid'][i] = grid
            df['cell'][i] = ccl
        except Exception as e:
            print(e)
            print(df.iloc[i, :])
            none_index.append(i)
            none_array[i] = False
    # return_df = df.drop(index= none_index)
    return_df = df[none_array]
    # df['grid'] = grid_used
    # df['cell'] = cell
    return return_df


def gt_range_enum_apply(i):
    if i > 0 and i < 10:
        return '0.3 to 10'
    elif i > 10 and i < 20:
        return '10 to 20'
    elif i > 20 and i < 30:
        return '20 to 30'
    elif i > 30 and i < 40:
        return '30 to 40'
    elif i > 40 and i < 50:
        return '40 to 50'
    elif i > 50 and i < 60:
        return '50 to 60'
    elif i > 60 and i < 80:
        return '60 to 80'
    elif i > 80 and i < 100:
        return '80 to 100'
    elif i > 100 and i < 150:
        return '100 to 150'
    elif i > 150 and i < 200:
        return '150 to 200'
    elif i > 200 and i < 250:
        return '200 to 250'
    elif i > 250 and i < 300:
        return '250 to 300'
    elif i > 300:
        return '300+'
    else:
        return 'nan'


def gt_range_zones(i):
    if i > 0:
        return "_front"
    else:
        return "_rear"


def datapreparation_KPI(MatchedGT_df, MtachedThudner_df):
    MatchedGT_df.columns = [str(i) + '_GT' for i in list(MatchedGT_df.columns)]
    MtachedThudner_df.columns = [str(i) + '_Thunder' for i in list(MtachedThudner_df.columns)]
    MatchedGT_df.index = [idx for idx in range(0, len(MatchedGT_df))]
    MtachedThudner_df.index = [idx for idx in range(0, len(MtachedThudner_df))]
    joined_df = pd.concat([MatchedGT_df, MtachedThudner_df], axis=1)
    joined_df.rename(columns={'Heading_GT': 'vcsHeading_GT'}, inplace=True)
    joined_df.rename(columns={'Heading_Thunder': 'vcsHeading_Thunder'}, inplace=True)
    joined_df['track_eulicdean'] = np.sqrt(np.array(joined_df.loc[:, 'Longitudinal_position_GT']) ** 2 + np.array(
        joined_df.loc[:, 'Lateral_position_GT']) ** 2)
    joined_df['gt_range'] = joined_df.iloc[:, -1].apply(gt_range_enum_apply)
    joined_df['range'] = joined_df.loc[:, 'Longitudinal_position_GT'].apply(gt_range_zones)
    joined_df['gt_range'] = joined_df['gt_range'] + joined_df['range']
    del joined_df['range']
    joined_df['Longitudinal_position_error'] = abs(joined_df['Longitudinal_position_GT'] - joined_df[
        'Longitudinal_position_Thunder'])
    joined_df['Lateral_position_error'] = abs(joined_df['Lateral_position_GT'] - joined_df['Lateral_position_Thunder'])
    joined_df['Longitudinal_velocity_error'] = abs(joined_df['Longitudinal_velocity_GT'] - joined_df[
        'Longitudinal_velocity_Thunder'])
    joined_df['Lateral_velocity_error'] = abs(joined_df['Lateral_velocity_GT'] - joined_df['Lateral_velocity_Thunder'])
    joined_df['vcsHeading_error'] = abs(joined_df['vcsHeading_GT'] - joined_df['vcsHeading_Thunder'])
    joined_df['Target_Length_error'] = abs(joined_df['Target_Length_GT'] - joined_df['Target_Length_Thunder'])
    joined_df['Target_Width_error'] = abs(joined_df['Target_Width_GT'] - joined_df['Target_Width_Thunder'])

    KPI_dict = get_kpi_target(joined_df)

    return pd.DataFrame(KPI_dict, index=[0])


def get_kpi_target(joined_df):
    """

    :param joined_df: dataframe with matched objects simple metrics like position, velocity, heading, dimension error
    :return:
    """
    # Longitudinal position error
    long_pos_error_temp = np.array(joined_df['Longitudinal_position_error'])
    max_long_error_index = joined_df['Longitudinal_position_error'].idxmax()
    Fusion_index_long_max = joined_df['Fusion_Index_Thunder'][max_long_error_index]

    Longitudinal_position_error_min = np.min(joined_df['Longitudinal_position_error'])
    Longitudinal_position_error_max = np.max(joined_df['Longitudinal_position_error'])
    q3_long_error, q1_long_error = np.percentile(long_pos_error_temp, [75, 25])
    iqr_long_pos_error = q3_long_error - q1_long_error

    # Lateral position error
    lat_pos_error_temp = np.array(joined_df['Lateral_position_error'])
    max_lat_error_index = joined_df['Lateral_position_error'].idxmax()
    Fusion_index_max_lat = joined_df['Fusion_Index_Thunder'][max_lat_error_index]

    Lat_position_error_min = np.min(lat_pos_error_temp)
    Lat_position_error_max = np.max(lat_pos_error_temp)
    q3_lat_error, q1_lat_error = np.percentile(lat_pos_error_temp, [75, 25])
    iqr_lat_pos_error = q3_lat_error - q1_lat_error

    # Longitudinal velocity error
    long_vel_error_temp = np.array(joined_df['Longitudinal_velocity_error'])
    Longitudinal_vel_error_min = np.min(long_vel_error_temp)
    Longitudinal_vel_error_max = np.max(long_vel_error_temp)
    q3_long_error, q1_long_error = np.percentile(long_vel_error_temp, [75, 25])
    iqr_long_vel_error = q3_long_error - q1_long_error

    # Lateral velocity error
    lat_vel_error_temp = np.array(joined_df['Lateral_velocity_error'])
    Lat_vel_error_min = np.min(lat_vel_error_temp)
    Lat_vel_error_max = np.max(lat_vel_error_temp)
    q3_lat_error, q1_lat_error = np.percentile(lat_vel_error_temp, [75, 25])
    iqr_lat_vel_error = q3_lat_error - q1_lat_error

    # Heading error
    heading_error_temp = np.array(joined_df['vcsHeading_error'])
    heading_error_min = np.min(heading_error_temp)
    heading_error_max = np.max(heading_error_temp)
    q3_heading_error, q1_heading_error = np.percentile(heading_error_temp, [75, 25])
    iqr_heading_error = q3_heading_error - q1_heading_error

    # Length error
    length_error_temp = np.array(joined_df['Target_Length_error'])
    length_error_min = np.min(length_error_temp)
    length_error_max = np.max(length_error_temp)
    q3_length_error, q1_length_error = np.percentile(length_error_temp, [75, 25])
    iqr_length_error = q3_length_error - q1_length_error

    # Width error
    width_error_temp = np.array(joined_df['Target_Width_error'])
    width_error_min = np.min(width_error_temp)
    width_error_max = np.max(width_error_temp)
    q3_width_error, q1_width_error = np.percentile(width_error_temp, [75, 25])
    iqr_width_error = q3_width_error - q1_width_error

    return_dict = {'Longitudinal_position_error_min (m)': round(Longitudinal_position_error_min, 2),
                   'Longitudinal_position_error_max (m)': round(Longitudinal_position_error_max, 2),
                   'Fusion_index_max_long': Fusion_index_long_max,
                   'IQR_Longitudinal_position (m)': round(iqr_long_pos_error, 2),
                   'Lat_position_error_min (m)': round(Lat_position_error_min, 2),
                   'Lat_position_error_max (m)': round(Lat_position_error_max, 2),
                   'Fusion_index_max_lat': Fusion_index_max_lat,
                   'iqr_lat_pos_error (m)': round(iqr_lat_pos_error, 2),
                   'Longitudinal_vel_error_min': round(Longitudinal_vel_error_min, 2),
                   'Longitudinal_vel_error_max': round(Longitudinal_vel_error_max, 2),
                   'iqr_long_vel_error': round(iqr_long_vel_error, 2),
                   'Lat_vel_error_min': round(Lat_vel_error_min, 2),
                   'Lat_vel_error_max': round(Lat_vel_error_max, 2),
                   'iqr_lat_vel_error': round(iqr_lat_vel_error, 2),
                   'heading_error_min': round(heading_error_min, 2),
                   'heading_error_max': round(heading_error_max, 2),
                   'iqr_heading_error': round(iqr_heading_error, 2),
                   'length_error_min': round(length_error_min, 2),
                   'length_error_max': round(length_error_max, 2),
                   'iqr_length_error': round(iqr_length_error, 2),
                   'width_error_min': round(width_error_min, 2),
                   'width_error_max': round(width_error_max, 2),
                   'iqr_width_error': round(iqr_width_error, 2)
                   }
    return return_dict


def base_logname(log_name):
    filename = log_name.split('.')[0]
    elements = filename.split('_')
    filtered_elements = [element for element in elements if not element.startswith('r')]
    base_logname = '_'.join(filtered_elements)
    base_logname = base_logname.replace('_dma', '_bus')
    base_logname = base_logname.replace('.mat', '.MF4')
    return base_logname


def get_autoGT_pickle_file(event_logs):
    """
    :param event_logs: events logs
    :return: dict key as event log and value as AutoGT pickle file path.
    """
    try:
        ### get pickle file for the events logs

        import pandas as pd
        from sqlalchemy import create_engine

        try:
            engine_ds_team_na = create_engine('mysql+pymysql://aptiv_db_algo_team:DKwVd5Le@10.192.229.101/ds_team_na')
        except Exception as e:
            print(e)
            exit(-1)

        table_name = 'AutoGTSubmissionInfo'

        # Read the table into a DataFrame

        AutoGT_table = pd.read_sql_table(table_name, con=engine_ds_team_na)

        event_mf4_logs = []
        for i in event_logs:
            i = i + '.MF4'
            event_mf4_logs.append(i)

        log_name_path = AutoGT_table[['log_name', 'ATGT_pickle_path']]
        log_name_path_dict = log_name_path.set_index('log_name')['ATGT_pickle_path'].to_dict()

        # # loading the pickle file
        counter = 0
        event_pickle_dict = {}
        for i in range(len(event_mf4_logs)):
            event_log = event_mf4_logs[i]
            if event_log in log_name_path_dict.keys():
                pickle_file_path = log_name_path_dict[event_log]
                event_pickle_dict[event_log] = pickle_file_path
            else:
                counter += 1
                # print("Pickle file not found for log", event_log)

        print('Total event logs', len(event_mf4_logs))
        print('Missing pickle file logs', counter)
        engine_ds_team_na.dispose()
        return event_pickle_dict
    except Exception as e:
        print(e)
        exit(-1)


def to_datetime(epoch_ts):
    """Convert `firstPackageTimestamp` and `reportedTimestamp` to datetime object"""

    return datetime.fromtimestamp(epoch_ts, timezone.utc)


def timesync_fus(data, log_type):
    try:
        fusCtime_us = data['mudp']['Object_Fusion']['header']['time'][:, 0]
        fusiontimestamp = data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['object_list_timestamp']
        systemtimestamp = data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['timestamp_us']
        sourceTxTime_ms_0 = data['mudp']['Object_Fusion']['header']['sourceTxTime'][:, 0].astype(int)
        sourceTxTime_ms_last = data['mudp']['Object_Fusion']['header']['sourceTxTime'][:, -1].astype(int)

        fusiontimestamp = fusiontimestamp & int("FFFFFFFF", 16)
        systemtimestamp = systemtimestamp & int("FFFFFFFF", 16)

        if log_type == 'Original':
            transmission_delay_us = np.abs(sourceTxTime_ms_last - sourceTxTime_ms_0) * 1000
            fusion_process_consumed_time = sourceTxTime_ms_0 * 1e3 - systemtimestamp
        elif log_type == 'RESIM':
            transmission_delay_us = np.abs(sourceTxTime_ms_last - sourceTxTime_ms_0)
            fusion_process_consumed_time = sourceTxTime_ms_0 - systemtimestamp
        fusion_measurement_process_consumed_time = systemtimestamp - fusiontimestamp
        fusion_measurement_age_at_tx_us = fusion_process_consumed_time + fusion_measurement_process_consumed_time + transmission_delay_us
        if abs(max(fusion_measurement_age_at_tx_us)) > 1000000:
            fusion_measurement_age_at_tx_us = 200000
        else:
            pass
        fusCtime_us_corrected = fusCtime_us - fusion_measurement_age_at_tx_us * 1e-6
        fusion_ctime_utc = [to_datetime(i).isoformat() for i in fusCtime_us_corrected]
        fusion_ctime_corrected_string = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]

        return fusCtime_us_corrected, fusion_ctime_corrected_string
    except Exception as e:
        print(e)
        print('Time sync not done')
        fusion_ctime_utc = [to_datetime(i).isoformat() for i in fusCtime_us]
        fusion_ctime_corrected_string = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]
        return fusCtime_us, fusion_ctime_corrected_string

def log_type(logName):
    logName = logName.replace('_dma', '_bus')
    basename = logName.split('.')[0]
    basename_logname = "_".join([i for i in basename.split('_') if not i.startswith('r')])
    if len(basename_logname) == len(basename):
        return 'Original'
    else:
        return 'RESIM'

def GT_reference_KPI(events_sheet, config):
    # config details
    try:
        log_name = config.get('log_name')
    except Exception as e:
        print('Issue when reading log name input from config')
    try:
        log_path = config.get('log_path')
    except Exception as e:
        print('Issue when reading log name input from config')
    try:
        Reduced_ID = config.get('Reduced_ID')
    except Exception as e:
        print('Issue when reading log name input from config')
    try:
        Track_ID = config.get('Track_ID')
    except Exception as e:
        print('Issue when reading log name input from config')
    try:
        Event_start_index = config.get('Event_start_index')
    except Exception as e:
        print('Issue when reading log name input from config')
    try:
        long_position_upper_limit = config.get('long_position_front_limit')
    except Exception as e:
        print('long_position_front_limit is not available in config setting it to default 50m  ')
        long_position_upper_limit = 50

    try:
        long_position_lower_limit = config.get('long_position_lower_limit')
    except Exception as e:
        print('long_position_front_limit is not available in config setting it to default 50m  ')
        long_position_lower_limit = -30

    events_sheet['base_logname'] = events_sheet[log_name].apply(base_logname)
    events_sheet['full_path'] = events_sheet[log_path] + '/' + events_sheet[log_name]
    event_logs = events_sheet['base_logname']
    event_tracks = events_sheet[Reduced_ID]
    event_tracks_object_ID = events_sheet[Track_ID]
    event_fusion_indexes = events_sheet[Event_start_index]

    if "win" in sys.platform:
        ## local testing
        event_log_path = events_sheet['PC_path']
        log_name_list = events_sheet['log_name']
        try:
            pickle_file_dict = get_autoGT_pickle_file(event_logs)
        except Exception as e:
            print("Issue with retrieving pickle file info from database error in this function get_autoGT_pickle_file")
    elif "lin" in sys.platform:
        event_log_path = events_sheet['log_path']
        log_name_list = events_sheet['log_name']
        try:
            pickle_file_dict = get_autoGT_pickle_file(event_logs)
        except Exception as e:
            print("Issue with retrieving pickle file info from database error in this function get_autoGT_pickle_file")

    matched_percentage_list = []
    unique_obj_macthed_AutoGT_list = []
    GT_avail_list = []
    KPI_event_list = []

    null_df = pd.DataFrame({'Longitudinal_position_error_min (m)': [np.NAN],
                            'Longitudinal_position_error_max (m)': [np.NAN],
                            'Fusion_index_max_long': [np.NAN],
                            'IQR_Longitudinal_position (m)': [np.NAN],
                            'Lat_position_error_min (m)': [np.NAN],
                            'Lat_position_error_max (m)': [np.NAN],
                            'Fusion_index_max_lat': [np.NAN],
                            'iqr_lat_pos_error (m)': [np.NAN],
                            'Longitudinal_vel_error_min': [np.NAN],
                            'Longitudinal_vel_error_max': [np.NAN],
                            'iqr_long_vel_error': [np.NAN],
                            'Lat_vel_error_min': [np.NAN],
                            'Lat_vel_error_max': [np.NAN],
                            'iqr_lat_vel_error': [np.NAN],
                            'heading_error_min': [np.NAN],
                            'heading_error_max': [np.NAN],
                            'iqr_heading_error': [np.NAN],
                            'length_error_min': [np.NAN],
                            'length_error_max': [np.NAN],
                            'iqr_length_error': [np.NAN],
                            'width_error_min': [np.NAN],
                            'width_error_max': [np.NAN],
                            'iqr_width_error': [np.NAN]
                            })

    for i in range(len(events_sheet)):
        # Reading input data both vehicle log and pickle file
        if "win" in sys.platform:
            log_name = os.path.join(event_log_path[i], log_name_list[i])
            logName = log_name_list[i].replace('_dma', '_bus')
            pickle_file_name = logName.split('.')[0] + '_agt_077' + '.pickle'
            picklefile_path = os.path.join(event_log_path[i], pickle_file_name)
            picklefile_path = r'C:\Users\pjp8lm\Desktop\Thunder\ACC_AGT_Integration\kusthav_issue_log\TNDR1_ASHN_20240320_161241_WDC5_bus_0010_agt_077.pickle'
            print(log_name)
        elif "lin" in sys.platform:
            log_name = os.path.join(event_log_path[i], log_name_list[i])
            logName = log_name_list[i].replace('_dma', '_bus')
            logName_MF4 = logName.replace('.mat', '.MF4')
            logName_MF4_base = base_logname(logName_MF4) + '.MF4'
            if logName_MF4_base in pickle_file_dict.keys():
                picklefile_path = pickle_file_dict[logName_MF4_base]
            else:
                print('No pickle file found for log', logName_MF4)
                matched_percentage_list.append('NaN')
                unique_obj_macthed_AutoGT_list.append('NaN')
                GT_avail_list.append('NaN')
                KPI_event_list.append(null_df)
                continue

        # loading the vehicle data.
        data = loadmat(log_name)
        log_type_str = log_type(logName)
        corrected_fusion_ctime, fusion_ctime_corrected_string = timesync_fus(data, log_type_str)

        vehicledata_raw_df, fusion_ctime_str, fusion_ctime = get_logdata(data, fusion_ctime_corrected_string)
        # vehicledata_cleaned_df = vehicledata_raw_df[vehicledata_raw_df['reduced_ID'] != 0]
        event_track = event_tracks[i]
        event_object_ID = event_tracks_object_ID[i]
        event_fusion_index = event_fusion_indexes[i]
        # extract object information
        # object_group = vehicledata_cleaned_df.groupby(['reduced_ID', 'obj_ID'])
        object_group = vehicledata_raw_df.groupby(['reduced_ID', 'obj_ID'])
        event_objects_df = object_group.get_group((event_track, event_object_ID))
        events_bool_array = list(event_objects_df['Fusion_Index'] == event_fusion_index)
        event_objects_df['event'] = events_bool_array
        event_final_df = get_cleaned_event_df(event_objects_df, event_fusion_index, long_position_upper_limit,
                                              long_position_lower_limit)
        autoGT_df = get_AutoGTdf(picklefile_path, fusion_ctime_corrected_string)
        if len(autoGT_df) == 0 or len(event_final_df) == 0:
            # No data in AutoGT or Event related.
            matched_percentage_list.append('NaN')
            unique_obj_macthed_AutoGT_list.append('NaN')
            GT_avail_list.append('NaN')
            KPI_event_list.append(null_df)
            continue
        abcompare = ABCompare()

        dfMatchedlog_df, dfMatchedGT_df, orphans_veh_df, orphans_autogt_df = abcompare.run(autoGT_df, event_final_df)

        # dfMatchedGT_df, dfMatchedlog_df, orphans_df = abcompare.correlation_ab(autoGT_df, event_final_df)
        # dfMatchedGT_df, dfMatchedlog_df, orphans_df = abcompare.correlation_ab(autoGT_df, event_final_df)
        #### KPI's

        matched_percentage = (len(dfMatchedGT_df) / len(event_final_df)) * 100
        if len(dfMatchedlog_df) != 0:
            unique_obj_macthed_AutoGT = dfMatchedGT_df['track_id'].nunique()
            print('Percentage of object match with AutoGT data', (len(dfMatchedGT_df) / len(event_final_df)) * 100, '%')
            print('Number of unique objects matched w.r.t to AutoGT is:', dfMatchedGT_df['track_id'].nunique())
            print('During the event time is there a AutoGT object aligned with event target: ',
                  dfMatchedlog_df['event'].any())
            KPI_df = datapreparation_KPI(dfMatchedGT_df, dfMatchedlog_df)
            KPI_event_list.append(KPI_df)
            GT_analysis = dfMatchedlog_df['event_Thunder'].any()
            matched_percentage_list.append(matched_percentage)
            unique_obj_macthed_AutoGT_list.append(unique_obj_macthed_AutoGT)
            GT_avail_list.append(GT_analysis)
        else:
            KPI_event_list.append(null_df)
            GT_analysis = False
            matched_percentage_list.append(0)
            unique_obj_macthed_AutoGT_list.append(np.NaN)
            GT_avail_list.append(GT_analysis)

    events_sheet['GT_matched_percentage'] = matched_percentage_list
    events_sheet['GT_unique_objects_matched'] = unique_obj_macthed_AutoGT_list
    events_sheet['GT_verdict'] = GT_avail_list
    events_kpi = pd.concat(KPI_event_list)
    events_kpi = events_kpi.reset_index()
    return_event_sheet = pd.concat([events_sheet, events_kpi], axis=1)
    return return_event_sheet


def split_dataframe(df):
    dataframes = []
    current_df = []
    for idx, row in df.iterrows():
        if row['Age'] == 1 and current_df:
            dataframes.append(pd.DataFrame(current_df))
            current_df = []
        current_df.append(row)
    if current_df:
        dataframes.append(pd.DataFrame(current_df))

    # Resetting the index for each resulting data frame
    dataframes = [df.reset_index(drop=True) for df in dataframes]
    return dataframes


def get_cleaned_event_df(event_objects_df, event_fusion_index, long_position_upper_limit, long_position_lower_limit):
    split_dfs = split_dataframe(event_objects_df)
    if len(split_dfs) > 0:
        for i in split_dfs:
            if i['event'].any():
                event_df = i
                print(len(i))
    if long_position_upper_limit == None:
        long_position_upper_limit = 120
    if long_position_lower_limit == None:
        long_position_lower_limit = -50

    # event analysis
    object_age_event = event_df[event_df['Fusion_Index'] == event_fusion_index]['Age'].values[0]
    # print(event_df[event_df['Fusion_Index'] == event_fusion_index]['Age'])
    age_index_event = event_df['Age'][event_df['Age'] == object_age_event].index[0]
    min_object_age = min(event_df['Age'])
    age_index_event_min = event_df['Age'][event_df['Age'] == min_object_age].index[0]
    event_df = event_df.reset_index()
    event_final_df = event_df.iloc[age_index_event_min: age_index_event + 1, :]
    event_final_df = event_final_df[
        (event_final_df['Longitudinal_position'] < long_position_upper_limit) & (event_final_df[
                                                                                     'Longitudinal_position'] > long_position_lower_limit)]  ## select indexes only when traget is closer to host vehicle
    return event_final_df


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def get_AutoGTdf(picklefile_path, fusion_ctime_str):
    import pickle
    if os.path.exists(picklefile_path):
        print('The file exists')
    else:
        print('The file does not exist')
        return pd.DataFrame()
    try:
        with open(picklefile_path, 'rb') as f:
            pickle_data = pickle.load(f)
    except Exception as e:
        print('Unable to load the pickle file')
        return pd.DataFrame()
    print("Loaded pickle file")

    try:
        from nexus_toolkit.interpolation.stream_interpolator import upload_interpolated_stream, interpolate_cuboids
        import copy

        timestamps = pickle_data['frames'].keys()
        serial_cubiods = []
        for i in timestamps:
            try:
                cubs = pickle_data['frames'][i]
                cubs['timestamp'] = i
                cubs = dotdict(cubs)
                serial_cubiods.append(cubs)  # [{'time: xxxx', 'cubiods': []}, {}, {}]
            except Exception as e:
                print(e)
                continue
        nexus_url, nexus_user, nexus_pwd = nexus_details()
        interpolated_cuboids = interpolate_cuboids(serial_cubiods, fusion_ctime_str,
                                                   pi=None,
                                                   nexus_url=nexus_url,
                                                   nexus_user=nexus_user,
                                                   nexus_pw=nexus_pwd,
                                                   timing_offset=0,
                                                   # timestamp_type="reportedTimestamp",
                                                   timestamp_type='timestamp',
                                                   perform_sweep_comp=True,
                                                   sampling_rate=1)

        print("Done interpolation")
        lidarMountingCals = {'position': {'x': 1.181648716540159, 'y': -0.06774385394548615, 'z': -2.014510213476982},
                             'rotation': {'w': -0.003799397572359602, 'x': 0.6929092643722031, 'y': 0.7210009912854143,
                                          'z': -0.004459427172793808}}

        interpolated_data_vcs = copy.deepcopy(interpolated_cuboids)
        for idx, sample in enumerate(interpolated_cuboids):
            interpolated_data_vcs[idx] = convertToVcs(sample, lidarMountingCals)
        list_df = []
        for i in range(len(interpolated_data_vcs)):
            sample = interpolated_data_vcs[i]
            idx = i
            temp_df = convertGTCuboidsToDf(idx, sample)
            list_df.append(temp_df)
        autoGT_df = pd.concat(list_df, axis=0, ignore_index=True)
        autoGT_df.rename(columns={'vcsHeading': 'Heading'}, inplace=True)
        autoGT_df.rename(columns={'timestamp': 'cTime'}, inplace=True)
        autoGT_df = rotation_position_correction(autoGT_df)
        return autoGT_df
    except Exception as e:
        print('Error while interpolating or data preparation for AB comapre')
        return pd.DataFrame()


def main(config_file, input_workbook):
    # input_workbook = r'C:\Users\pjp8lm\Desktop\Thunder\ACC_AGT_Integration\extract_acc_events_thunder_dma_2024-05-09_05-17-06_0.xlsx'
    config = load_config(config_file)
    print("reading: ", config.get('_comment'))
    sheet_name = config.get('sheet_name')
    events_sheet = pd.read_excel(input_workbook, sheet_name=sheet_name)
    output_sheet = GT_reference_KPI(events_sheet, config)

    # Avoid modifying this part
    # book = load_workbook(input_workbook)
    output_Path, excel_name = os.path.split(input_workbook)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_Path, 'GT_labeled_Event_extraction_output_' + str(timestr) + '.xlsx')
    writer = pd.ExcelWriter(output_filename, engine='openpyxl')
    # writer.book = book
    # writer.sheets = {x.title: x for x in book.worksheets}
    output_sheet.to_excel(writer, index=False, sheet_name="AutoGT_labeled")
    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process Events Excel and JSON configuration files.')
    parser.add_argument('-config', required=True, help='Path to the configuration JSON file')
    parser.add_argument('-excel', required=True, help='Path to the Excel file')
    args = parser.parse_args()
    main(args.config, args.excel)
