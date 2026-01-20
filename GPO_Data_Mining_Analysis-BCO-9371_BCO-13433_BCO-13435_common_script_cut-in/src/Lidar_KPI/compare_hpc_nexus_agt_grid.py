from datetime import datetime, timezone
import os
import pandas as pd
import numpy as np
import time
import openpyxl
import sys

if "lin" in sys.platform:
    print('platform :', sys.platform)
    sys.path.append(r'/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru')
    from mat_file_to_dict import loadmat
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.grid_gen import GridGen
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.support_functions import grid_decoder
    # sys.path.append(r'/mnt/usmidet/projects/STLA-THUNDER/7-Tools/GPO_Data_Mining_Analysis/src/Lidar_KPI')
    sys.path.append(r'/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru/repo/GPO_Data_Mining_Analysis/src/Lidar_KPI')
    from Lidartrackcompare import ABCompare

elif "win" in sys.platform:
    print('platform :', sys.platform)
    sys.path.append(r'C:\Users\pjp8lm\Desktop\Data_Mining_and_analystics\Repos')
    from ds_team_tools.hpc_frameworks.utils.mat_file_to_dict import loadmat
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.grid_gen import GridGen
    from ds_team_tools.hpc_frameworks.event_extraction.src.utils.of_kpi_utils.support_functions import grid_decoder
    sys.path.append(r'C:\Users\pjp8lm\Desktop\Thunder\Repo\GPO_Data_Mining_Analysis\src\Lidar_KPI')
    from Lidartrackcompare import ABCompare
    from Lidartrackcompare import  ABcompareEuclidean


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Comparehpcnexusagtgrid:
    """
    User defined class which generates AB compare KPI's for Object fusion and Lidar objects.
        * Reads the tracks from mat file convert to dataframe table.
        * Reads the object cuboids from AGT pickle file convert to dataframe table.
        * Perform interpolation on AutoGT data to object fusion timestamp.
        * AB compare using matching logic from Lidartrackcompare.py -> mapper output
        * Pass the matched dataframes to KPI_sheet_generation and generate a Summary KPI.
        * Using the Grid templete generate the KPI in zones
    """

    def __init__(self):
        self._func_name = os.path.basename(__file__).replace(".py", "")

        self._headers = dict()

        self._headers['autogt_matched_data_OF'] = ['log_path',
                                                'log_name',
                                                'frame', 'timestamp', 'track_id', 'Longitudinal_position',
                                                'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity',
                                                'Heading', 'Target_Length', 'Target_Width', 'Age', 'confidence', 'ref_position', 'vertices']

        self._headers['vehicle_matched_objects_OF'] = ['log_path', 'log_name', 'cTime', 'corrected_ctime', 'obj_ID',
         'Fusion_Index', 'fus_vision_index', 'Longitudinal_position',
         'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity',
         'Heading', 'Target_Length', 'Target_Width', 'Age', 'ref_position', 'vertices']

        self._headers['autogt_matched_data_vision'] = ['log_path',
                                                'log_name',
                                                'log_name',
                                                'frame', 'timestamp', 'track_id', 'Longitudinal_position',
                                                'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity',
                                                'Heading', 'Target_Length', 'Target_Width', 'Age', 'confidence']


        self._headers['vehicle_matched_objects_vision'] = ['log_path',
                                                    'log_name', 'algo_vision_index', 'corrected_ctime', 'obj_ID',
                                                    'Longitudinal_position', 'Lateral_position',
                                                    'Longitudinal_velocity',
                                                    'Lateral_velocity', 'Heading', 'Target_Length', 'Target_Width']



        self._headers['orphans'] = ['log_path', 'log_name', 'frame', 'timestamp', 'track_id', 'Longitudinal_position',
                                    'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity',
                                    'Heading', 'Target_Length', 'Target_Width', 'Age',
                                    'confidence']

        self._headers['Run_mode'] = ['object_type']

        self._venv = '/mnt/usmidet/projects/FCA-CADM/11-Development/8-DMA_Reports/Scripts/DMA_env/DMA_ENV/bin/activate'

    def get_venv(self):
        # ADD THIS FUNCTION ONLY IF YOU WANT TO RUN MAPPER ON USER VIRTUAL ENV
        """
        Return mapper virtual env from constructor
        """
        return self._venv

    def get_func_name(self):
        """
        :return: Returns function name
        """
        return self._func_name

    def nexus_details(self):
        nexus_url = "https://nexus.aptiv.com/api/"
        nexus_pwd = 'Harika@11'
        nexus_user = "chaitanya.kondru@aptiv.com"
        return nexus_url, nexus_user, nexus_pwd

    def to_datetime(self, epoch_ts):
        """Convert `firstPackageTimestamp` and `reportedTimestamp` to datetime object"""
        # return datetime.fromtimestamp(int(epoch_ts) / 1000000, timezone.utc)
        return datetime.fromtimestamp(epoch_ts, timezone.utc)
        # return datetime.fromtimestamp(float(epoch_ts) / 1e9)

    def cubiodDimensions(self, dimensions):
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

    def get_vision_obj_data(self, mat_data, fus_vision_index_ctime_str_dict):

        vision_physical_state = mat_data['mudp']['AVI_Objectlist']['AVI_Objectlist_Buffer']['VIS_Obj_Element']['VIS_OBJ_Physical_State']
        obj_long_distance = vision_physical_state['Long_Distance']
        obj_lat_distance = vision_physical_state['Lat_Distance']
        obj_heading = vision_physical_state['Heading']
        obj_long_velocity = vision_physical_state['Abs_Long_Velocity']
        obj_lat_veloctiy = vision_physical_state['Abs_Lat_Velocity']
        obj_width = vision_physical_state['Width']
        obj_length = vision_physical_state['Length']
        obj_id = mat_data['mudp']['AVI_Objectlist']['AVI_Objectlist_Buffer']['VIS_Obj_Element']['VIS_OBJ_ID']
        algo_frame_id = mat_data['mudp']['AVI_Objectlist']['AVI_Objectlist_Buffer']['VIS_Obj_Header']['VIS_OBJ_Common_Header']['Algo_Frame_ID']
        Longitudinal_position = []
        Lateral_position = []
        Longitudinal_velocity = []
        Lateral_velocity = []
        Heading = []
        Target_length = []
        Target_width = []
        algo_vision_Index = []
        obj_index_list = []
        corrected_ctime = []
        missing_indexes = []

        for time in range(0, len(algo_frame_id)):
            for index in range(0, len(obj_long_distance[0])):
                try:
                    ctime = fus_vision_index_ctime_str_dict[algo_frame_id[time]]
                except Exception as e:
                    missing_indexes.append(algo_frame_id[time])
                    continue
                corrected_ctime.append(ctime)
                obj_index_list.append(obj_id[time, index])
                Longitudinal_position.append(obj_long_distance[time, index])
                Lateral_position.append(obj_lat_distance[time, index])
                Longitudinal_velocity.append(obj_long_velocity[time, index])
                Lateral_velocity.append(obj_lat_veloctiy[time, index])
                Heading.append(obj_heading[time, index])
                Target_length.append(obj_length[time, index])
                Target_width.append(obj_width[time, index])
                algo_vision_Index.append(algo_frame_id[time])

        columns_names = ['algo_vision_index','corrected_ctime', 'obj_ID',
                         'Longitudinal_position',
                         'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity', 'Heading', 'Target_Length',
                         'Target_Width']
        columns_data = [algo_vision_Index, corrected_ctime , obj_index_list , Longitudinal_position,
                        Lateral_position, Longitudinal_velocity, Lateral_velocity,
                        Heading, Target_length, Target_width]
        zip_data = zip(columns_names, columns_data)
        dict_data = dict(zip_data)
        vehicle_vision_data_df = pd.DataFrame(data=dict_data)
        return vehicle_vision_data_df

    def get_logdata(self, mat_data, fusion_ctime_corrected_string):

        fusion_ctime = mat_data['mudp']['Object_Fusion']['header']['time'][:, 0]

        fusion_ctime_utc = [self.to_datetime(i).isoformat() for i in fusion_ctime]
        fusion_ctime_str = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]
        # log data

        vehicledata_filter = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['Fusion_output']
        fus_vision_index = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['vision_index'][:, 0]

        log_pos_vehi = vehicledata_filter['vcs_longposn']
        lat_pos_vehi = vehicledata_filter['vcs_latposn']
        log_vel_vehi = vehicledata_filter['vcs_longvel']
        lat_vel_vehi = vehicledata_filter['vcs_latvel']
        heading_vehi = vehicledata_filter['vcs_heading']
        tar_len_vehi = vehicledata_filter['length']
        tar_wid_vehi = vehicledata_filter['width']
        age_vehi = vehicledata_filter['age']
        fusion_index_vehi = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['fusion_index']
        cTime_vehi = fusion_ctime_str
        fus_vision_index_ctime_str = dict(zip(fus_vision_index, fusion_ctime_corrected_string))

        # centriod_mode = mat_data['mudp']['Object_Fusion']['UDP_fus_objects']['Fus']['Fusion_output']['centroid_mode']
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
        cTime = []
        corrected_ctime = []
        fusion_vision_index_list = []
        # centriod_mode_list = []
        for time in range(0, len(fusion_index_vehi)):
            for index in range(0, len(log_pos_vehi[0])):
                Time.append(time + 1)
                Index.append(index + 1)
                Longitudinal_position.append(log_pos_vehi[time, index])
                Lateral_position.append(lat_pos_vehi[time, index])
                Longitudinal_velocity.append(log_vel_vehi[time, index])
                Lateral_velocity.append(lat_vel_vehi[time, index])
                Heading.append(heading_vehi[time, index])
                Target_length.append(tar_len_vehi[time, index])
                Target_width.append(tar_wid_vehi[time, index])
                # centriod_mode_list.append(centriod_mode[time, index])
                Age.append(age_vehi[time, index])
                Fusion_Index.append(fusion_index_vehi[time])
                fusion_vision_index_list.append(fus_vision_index[time])
                cTime.append(cTime_vehi[time])
                corrected_ctime.append(fusion_ctime_corrected_string[time])
        columns_names = ['cTime','corrected_ctime', 'obj_ID','Fusion_Index','fus_vision_index', 'Longitudinal_position',
                         'Lateral_position', 'Longitudinal_velocity', 'Lateral_velocity', 'Heading', 'Target_Length',
                         'Target_Width', 'Age']
        columns_data = [cTime, corrected_ctime, Index, Fusion_Index,fusion_vision_index_list, Longitudinal_position,
                        Lateral_position, Longitudinal_velocity, Lateral_velocity,
                        Heading, Target_length, Target_width, Age]
        zip_data = zip(columns_names, columns_data)
        dict_data = dict(zip_data)
        vehicledata1_df = pd.DataFrame(data=dict_data)
        return vehicledata1_df, fusion_ctime_str, fusion_ctime, fus_vision_index_ctime_str

    def convertToVcs(self, cuboidlist_sample, calibration_data):

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

    def convertGTCuboidsToDf(self, idx, sample):
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

            try:
                timestamp = [sample.reportedTimestamp for cuboid in sample.cuboids]
            except Exception as e:
                timestamp = [sample.timestamp for cuboid in sample.cuboids]
            track_id = [cuboid["uuid"] for cuboid in sample.cuboids]
            long_pos = [cuboid["c"]["x"] for cuboid in sample.cuboids]
            lat_pos = [cuboid["c"]["y"] for cuboid in sample.cuboids]
            long_vel = [cuboid["v"]["x"] for cuboid in sample.cuboids]
            lat_vel = [cuboid["v"]["y"] for cuboid in sample.cuboids]
            heading = [np.arctan2(2 * (cuboid["r"]["w"] * cuboid["r"]["z"] + cuboid["r"]["x"] * cuboid["r"]["y"]),
                                  1 - 2 * (cuboid["r"]["y"] * cuboid["r"]["y"] + cuboid["r"]["z"] * cuboid["r"]["z"]))
                       for
                       cuboid in sample.cuboids]  # Wikipedia formula
            length = [self.cubiodDimensions(cuboid["s"])['length'] for cuboid in sample.cuboids]
            width = [self.cubiodDimensions(cuboid["s"])['width'] for cuboid in sample.cuboids]
            height = [self.cubiodDimensions(cuboid["s"])['height'] for cuboid in sample.cuboids]
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

    def get_headers(self):
        """
        :return: returns headers
        """
        return self._headers

    def KPI_statistics(self, joined_df):
        group_list = []
        count_list = []
        long_pos_err = []
        long_pos_error_median_list = []
        lat_pos_error_median_list = []
        long_vel_error_median_list = []
        lat_vel_error_median_list = []
        heading_error_median_list = []
        length_error_median_list = []
        width_error_median_list = []

        long_pos_ref_error_median_list = []
        lat_pos_ref_error_median_list = []


        iqr_long_pos_error_list = []
        iqr_lat_pos_error_list = []
        iqr_long_pos_ref_error_list = []
        iqr_lat_pos_ref_error_list = []
        iqr_long_vel_error_list = []
        iqr_lat_vel_error_list = []
        iqr_heading_error_list = []
        iqr_length_error_list = []
        iqr_width_error_list = []

        long_pos_error_max_list = []
        lat_pos_error_max_list = []
        long_pos_error_ref_max_list = []
        lat_pos_error_ref_max_list = []
        long_vel_error_max_list = []
        lat_vel_error_max_list = []
        heading_error_max_list = []
        length_error_max_list = []
        width_error_max_list = []

        long_pos_error_limit_list = []
        lat_pos_error_limit_list = []
        long_pos_error_ref_limit_list = []
        lat_pos_error_ref_limit_list = []
        long_vel_error_limit_list = []
        lat_vel_error_limit_list = []
        heading_error_limit_list = []
        length_error_limit_list = []
        width_error_limit_list = []

        gt_ranges_values = list(joined_df['gt_range'].unique())

        for i in gt_ranges_values:
            # getting column from dataframe
            long_pos_error_temp = joined_df.groupby('gt_range').get_group(i).Longitudinal_position_error.abs()
            lat_pos_error_temp = joined_df.groupby('gt_range').get_group(i).Lateral_position_error.abs()
            long_pos_ref_error_temp = joined_df.groupby('gt_range').get_group(i).Longitudinal_position_ref_raw_error.abs()
            lat_pos_ref_error_temp = joined_df.groupby('gt_range').get_group(i).Lateral_position_ref_raw_error.abs()
            long_vel_error_temp = joined_df.groupby('gt_range').get_group(i).Longitudinal_velocity_error.abs()
            lat_vel_error_temp = joined_df.groupby('gt_range').get_group(i).Lateral_velocity_error.abs()
            heading_error_temp = joined_df.groupby('gt_range').get_group(i).vcsHeading_error.abs()
            length_error_temp = joined_df.groupby('gt_range').get_group(i).Target_Length_error.abs()
            width_error_temp = joined_df.groupby('gt_range').get_group(i).Target_Width_error.abs()

            # calculating mean value
            long_pos_error_mean = long_pos_error_temp.mean()
            lat_pos_error_mean = lat_pos_error_temp.mean()
            long_pos_ref_error_mean = long_pos_ref_error_temp.mean()
            lat_pos_ref_error_mean = lat_pos_ref_error_temp.mean()
            long_vel_error_mean = long_vel_error_temp.mean()
            lat_vel_error_mean = lat_vel_error_temp.mean()
            heading_error_mean = heading_error_temp.mean()
            length_error_mean = length_error_temp.mean()
            width_error_mean = width_error_temp.mean()

            # mean plus 3*std

            long_pos_error_limit = long_pos_error_mean + 3 * long_pos_error_temp.std()
            lat_pos_error_limit = lat_pos_error_mean + 3 * lat_pos_error_temp.std()
            long_pos_ref_error_limit = long_pos_ref_error_mean + 3 * long_pos_ref_error_temp.std()
            lat_pos_ref_error_limit = lat_pos_ref_error_mean + 3 * lat_pos_ref_error_temp.std()
            long_vel_error_limit = long_vel_error_mean + 3 * long_vel_error_temp.std()
            lat_vel_error_limit = lat_vel_error_mean + 3 * lat_vel_error_temp.std()
            heading_error_limit = heading_error_mean + 3 * heading_error_temp.std()
            length_error_limit = length_error_mean + 3 * length_error_temp.std()
            width_error_limit = width_error_mean + 3 * width_error_temp.std()

            # calculating median value
            long_pos_error_median = np.median(long_pos_error_temp)
            lat_pos_error_median = lat_pos_error_temp.median()
            long_pos_ref_error_median = np.median(long_pos_ref_error_temp)
            lat_pos_ref_error_median = lat_pos_ref_error_temp.median()
            long_vel_error_median = long_vel_error_temp.median()
            lat_vel_error_median = lat_vel_error_temp.median()
            heading_error_median = heading_error_temp.median()
            length_error_median = length_error_temp.median()
            width_error_median = width_error_temp.median()

            # calculating max value
            long_pos_error_max = long_pos_error_temp.max()
            lat_pos_error_max = lat_pos_error_temp.max()
            long_pos_ref_error_max = long_pos_ref_error_temp.max()
            lat_pos_ref_error_max = lat_pos_ref_error_temp.max()
            long_vel_error_max = long_vel_error_temp.max()
            lat_vel_error_max = lat_vel_error_temp.max()
            heading_error_max = heading_error_temp.max()
            width_error_max = width_error_temp.max()
            length_error_max = length_error_temp.max()

            # calculating IQR values
            # position
            long_pos_error_temp = np.abs(long_pos_error_temp)
            q3_long_error, q1_long_error = np.percentile(long_pos_error_temp, [75, 25])
            iqr_long_pos_error = q3_long_error - q1_long_error

            lat_pos_error_temp = np.abs(lat_pos_error_temp)
            q3_lat_error, q1_lat_error = np.percentile(lat_pos_error_temp, [75, 25])
            iqr_lat_pos_error = q3_lat_error - q1_lat_error

            long_pos_ref_error_temp = np.abs(long_pos_ref_error_temp)
            q3_long_error, q1_long_error = np.percentile(long_pos_ref_error_temp, [75, 25])
            iqr_long_pos_ref_error = q3_long_error - q1_long_error

            lat_pos_ref_error_temp = np.abs(lat_pos_ref_error_temp)
            q3_lat_error, q1_lat_error = np.percentile(lat_pos_ref_error_temp, [75, 25])
            iqr_lat_pos_ref_error = q3_lat_error - q1_lat_error

            # velocity
            long_vel_error_temp = np.abs(long_vel_error_temp)
            q3_long_vel_error, q1_long_vel_error = np.percentile(long_vel_error_temp, [75, 25])
            iqr_long_vel_error = q3_long_vel_error - q1_long_vel_error

            lat_vel_error_temp = np.abs(lat_vel_error_temp)
            q3_lat_vel_error, q1_lat_vel_error = np.percentile(lat_vel_error_temp, [75, 25])
            iqr_lat_vel_error = q3_lat_vel_error - q1_lat_vel_error

            # heading
            heading_error_temp = np.abs(heading_error_temp)
            q3_heading_error, q1_heading_error = np.percentile(heading_error_temp, [75, 25])
            iqr_heading_error = q3_heading_error - q1_heading_error

            # Length
            length_error_temp = np.abs(length_error_temp)
            q3_length_error, q1_length_error = np.percentile(length_error_temp, [75, 25])
            iqr_length_error = q3_length_error - q1_length_error
            # Width
            width_error_temp = np.abs(width_error_temp)
            q3_width_error, q1_width_error = np.percentile(width_error_temp, [75, 25])
            iqr_width_error = q3_width_error - q1_width_error

            # appending data
            # median
            long_pos_error_median_list.append(long_pos_error_median)
            lat_pos_error_median_list.append(lat_pos_error_median)
            long_pos_ref_error_median_list.append(long_pos_ref_error_median)
            lat_pos_ref_error_median_list.append(lat_pos_ref_error_median)
            long_vel_error_median_list.append(long_vel_error_median)
            lat_vel_error_median_list.append(lat_vel_error_median)
            heading_error_median_list.append(heading_error_median)
            length_error_median_list.append(length_error_median)
            width_error_median_list.append(width_error_median)

            # max value append
            long_pos_error_max_list.append(long_pos_error_max)
            lat_pos_error_max_list.append(lat_pos_error_max)
            long_pos_error_ref_max_list.append(long_pos_ref_error_max)
            lat_pos_error_ref_max_list.append(lat_pos_ref_error_max)
            long_vel_error_max_list.append(long_vel_error_max)
            lat_vel_error_max_list.append(lat_vel_error_max)
            heading_error_max_list.append(heading_error_max)
            length_error_max_list.append(length_error_max)
            width_error_max_list.append(width_error_max)

            # IQR appedning
            iqr_long_pos_error_list.append(iqr_long_pos_error)
            iqr_lat_pos_error_list.append(iqr_lat_pos_error)
            iqr_long_pos_ref_error_list.append(iqr_long_pos_ref_error)
            iqr_lat_pos_ref_error_list.append(iqr_lat_pos_ref_error)
            iqr_long_vel_error_list.append(iqr_long_vel_error)
            iqr_lat_vel_error_list.append(iqr_lat_vel_error)
            iqr_heading_error_list.append(iqr_heading_error)
            iqr_length_error_list.append(iqr_length_error)
            iqr_width_error_list.append(iqr_width_error)

            # 3-std limit
            long_pos_error_limit_list.append(long_pos_error_limit)
            lat_pos_error_limit_list.append(lat_pos_error_limit)
            long_pos_error_ref_limit_list.append(long_pos_ref_error_limit)
            lat_pos_error_ref_limit_list.append(lat_pos_ref_error_limit)
            long_vel_error_limit_list.append(long_vel_error_limit)
            lat_vel_error_limit_list.append(lat_vel_error_limit)
            heading_error_limit_list.append(heading_error_limit)
            length_error_limit_list.append(length_error_limit)
            width_error_limit_list.append(width_error_limit)

            # group
            cnt = len(joined_df.groupby('gt_range').get_group(i))
            count_list.append(cnt)
            group_list.append(i)

            long_pos_err.append(long_pos_error_temp)

        # position table
        kpi_pos_object_table = pd.DataFrame()
        kpi_pos_object_table['gt_range'] = group_list
        kpi_pos_object_table['Count'] = count_list
        kpi_pos_object_table['Median Longitudinal Error(m)'] = long_pos_error_median_list
        kpi_pos_object_table['IQR Longitudinal Error (m)'] = iqr_long_pos_error_list
        kpi_pos_object_table['Max Longitudinal Error (m)'] = long_pos_error_max_list
        kpi_pos_object_table['Three-Sigma-limit(upper) longitudinal error'] = long_pos_error_limit_list
        kpi_pos_object_table['Median Lateral Error (m)'] = lat_pos_error_median_list
        kpi_pos_object_table['IQR Lateral Error(m)'] = iqr_lat_pos_error_list
        kpi_pos_object_table['Max Lateral Error(m)'] = lat_pos_error_max_list
        kpi_pos_object_table['Three-Sigma-limit(upper) Lateral error'] = lat_pos_error_limit_list
        kpi_pos_object_table = self.sort_table(kpi_pos_object_table)

        # ref position table
        kpi_pos_ref_object_table = pd.DataFrame()
        kpi_pos_ref_object_table['gt_range'] = group_list
        kpi_pos_ref_object_table['Count'] = count_list
        kpi_pos_ref_object_table['Median Longitudinal Error(m)'] = long_pos_ref_error_median_list
        kpi_pos_ref_object_table['IQR Longitudinal Error (m)'] = iqr_long_pos_ref_error_list
        kpi_pos_ref_object_table['Max Longitudinal Error (m)'] = long_pos_error_ref_max_list
        kpi_pos_ref_object_table['Three-Sigma-limit(upper) longitudinal error'] = long_pos_error_ref_limit_list
        kpi_pos_ref_object_table['Median Lateral Error (m)'] = lat_pos_ref_error_median_list
        kpi_pos_ref_object_table['IQR Lateral Error(m)'] = iqr_lat_pos_ref_error_list
        kpi_pos_ref_object_table['Max Lateral Error(m)'] = lat_pos_error_ref_max_list
        kpi_pos_ref_object_table['Three-Sigma-limit(upper) Lateral error'] = lat_pos_error_ref_limit_list
        kpi_pos_ref_object_table = self.sort_table(kpi_pos_ref_object_table)


        # velocity table
        kpi_vel_object_table = pd.DataFrame()
        kpi_vel_object_table['gt_range'] = group_list
        kpi_vel_object_table['Count'] = count_list
        kpi_vel_object_table['Median Longitudinal velocity Error(m/s)'] = long_vel_error_median_list
        kpi_vel_object_table['IQR Longitudinal velocity Error (m/s)'] = iqr_long_vel_error_list
        kpi_vel_object_table['Max Longitudinal velocity Error (m/s)'] = long_vel_error_max_list
        kpi_vel_object_table['Three-Sigma-limit(upper) longitudinal velocity error'] = long_vel_error_limit_list
        kpi_vel_object_table['Median Lateral velocity Error (m/s)'] = lat_vel_error_median_list
        kpi_vel_object_table['IQR Lateral velocity Error (m/s)'] = iqr_lat_vel_error_list
        kpi_vel_object_table['Max Lateral velocity Error (m/s)'] = lat_vel_error_max_list
        kpi_vel_object_table['Three-Sigma-limit(upper) Lateral velocity error'] = lat_vel_error_limit_list
        kpi_vel_object_table = self.sort_table(kpi_vel_object_table)

        # Dimensions table
        kpi_dem_object_table = pd.DataFrame()
        kpi_dem_object_table['gt_range'] = group_list
        kpi_dem_object_table['Count'] = count_list
        kpi_dem_object_table['Median Length Error(m)'] = length_error_median_list
        kpi_dem_object_table['IQR Length Error (m)'] = iqr_length_error_list
        kpi_dem_object_table['Max Lenght Error (m)'] = length_error_max_list
        kpi_dem_object_table['Three-Sigma-limit(upper) length error'] = length_error_limit_list
        kpi_dem_object_table['Median Width Error (m)'] = width_error_median_list
        kpi_dem_object_table['IQR Width Error (m)'] = iqr_width_error_list
        kpi_dem_object_table['Max Width Error (m/s)'] = width_error_max_list
        kpi_dem_object_table['Three-Sigma-limit(upper) Width error'] = width_error_limit_list
        kpi_dem_object_table = self.sort_table(kpi_dem_object_table)

        # Heading table
        kpi_heading_object_table = pd.DataFrame()
        kpi_heading_object_table['gt_range'] = group_list
        kpi_heading_object_table['Count'] = count_list
        kpi_heading_object_table['Median heading Error (deg)'] = heading_error_median_list
        kpi_heading_object_table['IQR heading Error (deg)'] = iqr_heading_error_list
        kpi_heading_object_table['Max heading Error (deg)'] = heading_error_max_list
        kpi_heading_object_table['Three-Sigma-limit(upper) Heading error'] = heading_error_limit_list
        kpi_heading_object_table = self.sort_table(kpi_heading_object_table)

        return kpi_pos_object_table, kpi_vel_object_table, kpi_dem_object_table, kpi_heading_object_table, kpi_pos_ref_object_table

    def gt_range_enum_apply(self, i):
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

    def gt_range_zones(self, i):
        if i > 0:
            return "_front"
        else:
            return "_rear"

    def confusion_matrix(self, y_pred, y_actual):
        fp = np.sum((y_pred == 1) & (y_actual == 0))
        tp = np.sum((y_pred == 1) & (y_actual == 1))
        fn = np.sum((y_pred == 0) & (y_actual == 1))
        tn = np.sum((y_pred == 0) & (y_actual == 0))
        if fp == 0:
            fpr = 0.0
        else:
            fpr = fp / (fp + tp + fn)
        if tp == 0:
            tpr = 0.0
        else:
            tpr = tp / (tp + fn)
        return tpr, fpr

    def sort_table(self, df):
        from natsort import index_natsorted, order_by_index
        return df.reindex(index=order_by_index(df.index, index_natsorted(df['gt_range'], reverse=False)))

    def return_ref_long_lat_position(self, reference_position_GT, reference_position_track, reference_position_track_raw):
        return reference_position_GT[0], reference_position_GT[1], reference_position_track[0], reference_position_track[1], reference_position_track_raw[0], reference_position_track_raw[1]

    def return_ref_position(self, vertices_GT, ref_position_GT, vertices_track, ref_position_track):
        return vertices_GT[ref_position_GT], vertices_track[ref_position_GT], vertices_track[ref_position_track]

    def reference_position(self, df):
        df[['reference_positions_GT', 'reference_positions_track', 'reference_positions_track_raw']] = \
            df.apply(lambda row: self.return_ref_position(row['vertices_GT'], row['ref_position_GT'], row['vertices_JLBUX'], row['ref_position_JLBUX']), axis=1, result_type="expand")

        df[['ref_long_position_GT','ref_lat_position_GT', 'ref_long_position_JLBUX','ref_lat_position_JLBUX','ref_long_position_raw_JLBUX', 'ref_lat_position_raw_JLBUX']] = \
            df.apply(lambda row: self.return_ref_long_lat_position(row['reference_positions_GT'], row['reference_positions_track'], row['reference_positions_track_raw']), axis=1, result_type= "expand")

        return df

    def convert_string_to_numpy(self, array_string):
        array_string = array_string.replace('\n', ' ').replace('[', '').replace(']', '')
        numbers = array_string.split()
        array = np.array([float(num) for num in numbers]).reshape(-1, 2)
        return array



    def datapreparation_KPI(self, MatchedGT_df, MtachedJLBUX_df):
        MatchedGT_df.columns = [str(i) + '_GT' for i in list(MatchedGT_df.columns)]
        MtachedJLBUX_df.columns = [str(i) + '_JLBUX' for i in list(MtachedJLBUX_df.columns)]
        MatchedGT_df.index = [idx for idx in range(0, len(MatchedGT_df))]
        MtachedJLBUX_df.index = [idx for idx in range(0, len(MtachedJLBUX_df))]
        joined_df = pd.concat([MatchedGT_df, MtachedJLBUX_df], axis=1)
        joined_df.rename(columns={'Heading_GT': 'vcsHeading_GT'}, inplace=True)
        joined_df.rename(columns={'Heading_JLBUX': 'vcsHeading_JLBUX'}, inplace=True)
        joined_df['track_eulicdean'] = np.sqrt(np.array(joined_df.loc[:, 'Longitudinal_position_GT']) ** 2 + np.array(
            joined_df.loc[:, 'Lateral_position_GT']) ** 2)
        joined_df['gt_range'] = joined_df.iloc[:, -1].apply(self.gt_range_enum_apply)
        joined_df['range'] = joined_df.loc[:, 'Longitudinal_position_GT'].apply(self.gt_range_zones)
        joined_df['gt_range'] = joined_df['gt_range'] + joined_df['range']
        del joined_df['range']
        joined_df.rename(columns={'vertices_GT': 'vertices_GT_str'}, inplace=True)
        joined_df.rename(columns={'vertices_JLBUX': 'vertices_JLBUX_str'}, inplace=True)
        joined_df['vertices_GT'] = joined_df['vertices_GT_str'].apply(self.convert_string_to_numpy)
        joined_df['vertices_JLBUX'] = joined_df['vertices_JLBUX_str'].apply(self.convert_string_to_numpy)
        joined_df = self.reference_position(joined_df)
        joined_df['Longitudinal_position_error'] = joined_df['Longitudinal_position_GT'] - joined_df[
            'Longitudinal_position_JLBUX']
        joined_df['Lateral_position_error'] = joined_df['Lateral_position_GT'] - joined_df['Lateral_position_JLBUX']

        joined_df['Longitudinal_position_ref_error'] = joined_df['ref_long_position_GT'] - joined_df[
            'ref_long_position_JLBUX']
        joined_df['Lateral_position_ref_error'] = joined_df['ref_lat_position_GT'] - joined_df['ref_lat_position_JLBUX']

        joined_df['Longitudinal_position_ref_raw_error'] = joined_df['ref_long_position_GT'] - joined_df['ref_long_position_raw_JLBUX']
        joined_df['Lateral_position_ref_raw_error'] = joined_df['ref_lat_position_GT'] - joined_df['ref_lat_position_raw_JLBUX']

        joined_df['Longitudinal_velocity_error'] = joined_df['Longitudinal_velocity_GT'] - joined_df[
            'Longitudinal_velocity_JLBUX']
        joined_df['Lateral_velocity_error'] = joined_df['Lateral_velocity_GT'] - joined_df['Lateral_velocity_JLBUX']
        joined_df['vcsHeading_error'] = joined_df['vcsHeading_GT'] - joined_df['vcsHeading_JLBUX']
        joined_df['Target_Length_error'] = joined_df['Target_Length_GT'] - joined_df['Target_Length_JLBUX']
        joined_df['Target_Width_error'] = joined_df['Target_Width_GT'] - joined_df['Target_Width_JLBUX']


        return joined_df

    def assign_grid(self, df):
        import warnings
        warnings.filterwarnings("ignore")
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
        lat_array = np.array(df['Lateral_position_GT'])
        long_array = np.array(df['Longitudinal_position_GT'])
        for i in range(len(df)):
            try:
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

        return_df = df[none_array]


        return return_df

    def get_grid_stats(self, array):
        mean = array.mean()
        median = array.median()
        max = array.max()
        three_std = mean + 3 * array.std()
        return round(mean, 2), round(median, 2), round(max, 2), round(three_std, 2)

    def grid_output(self, joined_df, orphans, output_excel_path, run_mode):
        import shutil
        if "lin" in sys.platform:
            template_path = r"/mnt/usmidet/projects/STLA-THUNDER/8-Users/AlgoGroup/chaitanya_kondru/tracker_KPI_templete/OF_Zone_KPI.xlsx"
        elif "win" in sys.platform:
            template_path = r"C:\Users\pjp8lm\Desktop\Thunder\Trimble_Parser\grid_generation\OF_Zone_KPI.xlsx"

        # if long > 72:
        #     if lat < -8 or lat > 8:
        #         pass
        # if long > 249 or long < -73:
        #     if lat < -70 or lat > 70:
        #         pass

        # joined_grid_df = self.assign_grid(joined_df)
        # orphans_grid_df = self.assign_grid(orphans)
        joined_grid_df = self.assign_grid(joined_df)
        shutil.copy(template_path, output_excel_path)
        grid_excel_path = os.path.join(output_excel_path, "OF_Zone_KPI.xlsx")
        wb = openpyxl.load_workbook(grid_excel_path)

        # orphans_worksheet = wb['Orphans']
        # for grid_pos in orphans_grid_df['cell'].unique():
        #     if grid_pos[0] == 0 :
        #         continue
        #     no_of_orphans = len(orphans_grid_df[orphans_grid_df['cell'] == grid_pos])
        #     # print('///', grid_pos[0], no_of_orphans)
        #     orphans_worksheet[grid_pos[0]].value = no_of_orphans

        long_position_sheet = wb['Longitudinal_Position']
        lat_position_sheet = wb['Lateral_Position']
        long_velocity_sheet = wb['Longitudinal_Velocity']
        lat_velocity_sheet = wb['Lateral_Velocity']
        long_position_ref_sheet = wb['Long_ref_Position']
        lat_position_ref_sheet = wb['Lat_ref_Position']

        for grid_pos in joined_grid_df['cell'].unique():
            if grid_pos[0] == 0:
                continue
            grid_pos_df = joined_grid_df[joined_grid_df['cell'] == grid_pos]
            number_of_samples = len(grid_pos_df)
            long_pos_err_mean, long_pos_err_median, long_pos_err_max, long_pos_err_three_std = self.get_grid_stats(
                grid_pos_df.Longitudinal_position_error.abs())
            lat_pos_err_mean, lat_pos_err_median, lat_pos_err_max, lat_pos_err_three_std = self.get_grid_stats(
                grid_pos_df.Lateral_position_error.abs())
            long_vel_pos_err_mean, long_vel_err_median, long_vel_err_max, long_vel_err_three_std = self.get_grid_stats(
                grid_pos_df.Longitudinal_velocity_error.abs())
            lat_vel_pos_err_mean, lat_vel_err_median, lat_vel_err_max, lat_vel_err_three_std = self.get_grid_stats(
                grid_pos_df.Longitudinal_velocity_error.abs())

            long_pos_ref_err_mean, long_pos_ref_err_median, long_pos_ref_err_max, long_pos_ref_err_three_std = self.get_grid_stats(
                grid_pos_df.Longitudinal_position_ref_raw_error.abs())
            lat_pos_ref_err_mean, lat_pos_ref_err_median, lat_pos_ref_err_max, lat_pos_ref_err_three_std = self.get_grid_stats(
                grid_pos_df.Lateral_position_ref_raw_error.abs())

            long_position_sheet[grid_pos[0]].value = str(long_pos_err_mean) + '\n' + str(
                long_pos_err_median) + '\n' + str(long_pos_err_max) + '\n' + str(long_pos_err_three_std) + '\n' + str(
                number_of_samples)
            lat_position_sheet[grid_pos[0]].value = str(lat_pos_err_mean) + '\n' + str(lat_pos_err_median) + '\n' + str(
                lat_pos_err_max) + '\n' + str(lat_pos_err_three_std) + '\n' + str(number_of_samples)
            long_velocity_sheet[grid_pos[0]].value = str(long_vel_pos_err_mean) + '\n' + str(
                long_vel_err_median) + '\n' + str(long_vel_err_max) + '\n' + str(long_vel_err_three_std) + '\n' + str(
                number_of_samples)
            lat_velocity_sheet[grid_pos[0]].value = str(lat_vel_pos_err_mean) + '\n' + str(
                lat_vel_err_median) + '\n' + str(lat_vel_err_max) + '\n' + str(lat_vel_err_three_std) + '\n' + str(
                number_of_samples)

            long_position_ref_sheet[grid_pos[0]].value = str(long_pos_ref_err_mean) + '\n' + str(
                long_pos_ref_err_median) + '\n' + str(long_pos_ref_err_max) + '\n' + str(long_pos_ref_err_three_std) + '\n' + str(
                number_of_samples)
            lat_position_ref_sheet[grid_pos[0]].value = str(lat_pos_ref_err_mean) + '\n' + str(lat_pos_ref_err_median) + '\n' + str(
                lat_pos_ref_err_max) + '\n' + str(lat_pos_ref_err_three_std) + '\n' + str(number_of_samples)

        wb.save(grid_excel_path)
        if not os.path.exists(run_mode):
            os.makedirs(run_mode)
        else:
            shutil.rmtree(os.path.join(output_excel_path, run_mode))
            os.makedirs(run_mode)
        shutil.move(grid_excel_path, os.path.join(output_excel_path, run_mode))


    def runKPI(self, MatchedGT_df, MtachedJLBUX_df, orphans, output_excel_path, run_mode):
        print("Generating KPIs...")
        if "lin" in sys.platform:
            os.system('source /mnt/usmidet/projects/FCA-CADM/11-Development/8-DMA_Reports/Scripts/DMA_env/DMA_ENV/bin/activate')
        elif "win" in sys.platform:
            pass
        joined_df = self.datapreparation_KPI(MatchedGT_df, MtachedJLBUX_df)
        self.grid_output(joined_df, orphans, output_excel_path, run_mode)
        kpi_pos_object_table, kpi_vel_object_table, kpi_dem_object_table, kpi_heading_object_table, kpi_pos_ref_object_table = self.KPI_statistics(
            joined_df)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        count_orphans = len(orphans)
        count_matched_GT = len(MatchedGT_df)
        count_summary = pd.DataFrame({'Matched_count': count_matched_GT, 'count_orphans': count_orphans}, index=[0])
        os.chdir(output_excel_path)
        writer = pd.ExcelWriter('LIDAR_Thunder_KPI_' + str(run_mode) + '_' + str(timestr) + '.xlsx')
        # joined_df.to_excel(writer, 'All_data', index=False)
        kpi_pos_object_table.to_excel(writer, "position table", index=False)
        kpi_pos_ref_object_table.to_excel(writer, "Reference position table", index =False)
        kpi_vel_object_table.to_excel(writer, "velocity table", index=False)
        kpi_heading_object_table.to_excel(writer, 'Heading table', index=False)
        kpi_dem_object_table.to_excel(writer, 'Dimension table', index=False)
        count_summary.to_excel(writer, 'Matched_summary', index=False)
        # MatchedGT_df.to_excel(writer, 'Matched_GT_data', index=False)
        # MtachedJLBUX_df.to_excel(writer, 'Matched_JLBUX_data', index=False)
        # orphans.to_excel(writer, 'AutoGT_miss_matched_objects', index=False)
        writer.close()
        print("KPI Report is generated!")

    # Function to add KPI sheet in Excel output.
    def kpi_sheet_generation(self, output_excel_sheet):
        """
        This function will be called from write_excel_to_excel and take out excel as input and add KPI sheet to it.
        :param output_excel_sheet: Excel sheet generated after reducer.
        :return:
        """
        if "lin" in sys.platform:
            os.system('source /mnt/usmidet/projects/FCA-CADM/11-Development/8-DMA_Reports/Scripts/DMA_env/DMA_ENV/bin/activate')
        elif "win" in sys.platform:
            pass
        import warnings
        from natsort import index_natsorted, order_by_index


        output_excel_path, excel_file = os.path.split(output_excel_sheet)

        matched_agt_all_logs_OF = []
        matched_JLBUX_all_logs_OF = []

        matched_agt_all_logs_vision = []
        matched_JLBUX_all_logs_vision = []
        # output_excel_path = r'C:\Users\pjp8lm\Desktop\Thunder\Latest_of_logs_agt_tool\Golden_dataset\vision_30_log_results'
        os.chdir(output_excel_path)

        """
        autogt_matched_data_OF.to_excel(writer, "autogt_matched_data_OF", index=False)
        vehicle_matched_objects_OF.to_excel(writer, "vehicle_matched_objects_OF", index=False)
        autogt_matched_data_vision.to_excel(writer, "autogt_matched_data_vision", index=False)
        vehicle_matched_objects_vision.to_excel(writer, "vehicle_matched_objects_vision", index=False)
        """

        all_excels = [x for x in os.listdir() if x.endswith(".xlsx")]
        for i in all_excels:
            try:
                print('loading', i)
                gt_data_OF = pd.read_excel(i, sheet_name='autogt_matched_data_OF')
                matched_data_OF = pd.read_excel(i, sheet_name='vehicle_matched_objects_OF')
                gt_data_vision = pd.read_excel(i, sheet_name= 'autogt_matched_data_vision')
                matched_data_vision = pd.read_excel(i, sheet_name= 'vehicle_matched_objects_vision')
                run_mode_df = pd.read_excel(i, sheet_name= 'Run_mode')
                # orphans_data = pd.read_excel(i, sheet_name='orphans')
                matched_agt_all_logs_OF.append(gt_data_OF)
                matched_JLBUX_all_logs_OF.append(matched_data_OF)
                matched_agt_all_logs_vision.append(gt_data_vision)
                matched_JLBUX_all_logs_vision.append(matched_data_vision)
                run_mode = run_mode_df['object_type'][0]
                # orphans_all_logs.append(orphans_data)
            except Exception as e:
                continue
            # matched_agt_all_logs.append(gt_data)
            # matched_JLBUX_all_logs.append(matched_data)
            # orphans_all_logs.append(orphans_data)

        if run_mode == 'Object_fusion':
            gt_data_df_OF = pd.concat(matched_agt_all_logs_OF)
            JLBUX_data_df_OF = pd.concat(matched_JLBUX_all_logs_OF)
            if len(gt_data_df_OF) == 0:
                print('No matches')
                print("Report generated!")
            else:
                all_orphans_df = pd.DataFrame(columns=gt_data_df_OF.columns)
                all_orphans_df.rename(columns={'Lateral_position': 'Lateral_position_GT'}, inplace=True)
                all_orphans_df.rename(columns={'Longitudinal_position': 'Longitudinal_position_GT'}, inplace=True)
                self.runKPI(gt_data_df_OF, JLBUX_data_df_OF, all_orphans_df, output_excel_path, run_mode)
        elif run_mode == 'Vision':
            gt_data_df_vision = pd.concat(matched_JLBUX_all_logs_vision)
            JLBUX_data_df_vision = pd.concat(matched_JLBUX_all_logs_vision)

            if len(gt_data_df_vision) == 0:
                print('No matches')
                print("Report generated!")
            else:
                all_orphans_df = pd.DataFrame(columns=gt_data_df_vision.columns)
                all_orphans_df.rename(columns={'Lateral_position': 'Lateral_position_GT'}, inplace=True)
                all_orphans_df.rename(columns={'Longitudinal_position': 'Longitudinal_position_GT'}, inplace=True)
                self.runKPI(gt_data_df_vision, JLBUX_data_df_vision, all_orphans_df, output_excel_path, run_mode)


        elif run_mode == 'All':
            gt_data_df_OF = pd.concat(matched_agt_all_logs_OF)
            JLBUX_data_df_OF = pd.concat(matched_JLBUX_all_logs_OF)
            gt_data_df_vision = pd.concat(matched_JLBUX_all_logs_vision)
            JLBUX_data_df_vision = pd.concat(matched_JLBUX_all_logs_vision)

            if len(gt_data_df_vision) == 0:
                print('No matches')
                print("Report generated!")
            else:
                all_orphans_df = pd.DataFrame(columns=gt_data_df_vision.columns)
                all_orphans_df.rename(columns={'Lateral_position': 'Lateral_position_GT'}, inplace=True)
                all_orphans_df.rename(columns={'Longitudinal_position': 'Longitudinal_position_GT'}, inplace=True)
                self.runKPI(gt_data_df_vision, JLBUX_data_df_vision, all_orphans_df, output_excel_path, 'Vision')

            if len(gt_data_df_OF) == 0:
                print('No matches')
                print("Report generated!")
            else:
                all_orphans_df = pd.DataFrame(columns=gt_data_df_vision.columns)
                all_orphans_df.rename(columns={'Lateral_position': 'Lateral_position_GT'}, inplace=True)
                all_orphans_df.rename(columns={'Longitudinal_position': 'Longitudinal_position_GT'}, inplace=True)
                self.runKPI(gt_data_df_OF, JLBUX_data_df_OF, all_orphans_df, output_excel_path, 'Object_fusion')

    def rotation_position_correction(self, df_1):
        import math
        df_1['Longitudinal_position'] = df_1['Longitudinal_position'] - 3.853
        # Th1 = 1.7 * math.pi / 180
        # AA = np.array([[math.cos(Th1), math.sin(Th1) * -1], [math.sin(Th1), math.cos(Th1)]])  # computed once
        # for i in range(len(df_1)):
        #     BB = np.array([[df_1.loc[i, 'Lateral_position']], [df_1.loc[i, 'Longitudinal_position']]])
        #     res = np.dot(AA, BB)
        #     df_1.loc[i, 'Longitudinal_position'] = res[1][0]
        #     df_1.loc[i, 'Lateral_position'] = res[0][0]
        return df_1

    def get_pickle_file_path(self, file_name):
        try:
            ### get pickle file for the events logs
            import pandas as pd
            from sqlalchemy import create_engine
            try:
                # engine_ds_team_na = create_engine(
                #     'mysql+pymysql://aptiv_db_algo_team:DKwVd5Le@usmidet-db001.aptiv.com/ds_team_na')
                engine_ds_team_na = create_engine(
                    'mysql+pymysql://aptiv_db_algo_team:DKwVd5Le@10.192.229.101/ds_team_na')
            except Exception as e:
                print(e)
                exit(-1)

            table_name = 'AutoGTSubmissionInfo'
            # Read the table into a DataFrame
            AutoGT_table = pd.read_sql_table(table_name, con=engine_ds_team_na)
            log_name_path = AutoGT_table[['base_name', 'ATGT_pickle_path']]
            log_name_path_dict = log_name_path.set_index('base_name')['ATGT_pickle_path'].to_dict()

            if file_name in log_name_path_dict.keys():
                pickle_file_path = log_name_path_dict[file_name]
                return pickle_file_path
            else:
                print('pickle file not found for log', file_name)
        except Exception as e:
            print(e)

    def get_baselogname(self, logName):
        logName = logName.replace('_dma', '_bus')
        basename = logName.split('.')[0]
        basename_logname = "_".join([i for i in basename.split('_') if not i.startswith('r')])
        return basename_logname, len(basename_logname) == len(basename)

    def timesync_fus(self, data, log_type):
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
            if max(np.abs(fusion_measurement_age_at_tx_us)) > 1000000:
                fusion_measurement_age_at_tx_us = 200000
            else:
                pass
            fusCtime_us_corrected = fusCtime_us - fusion_measurement_age_at_tx_us * 1e-6
            fusion_ctime_utc = [self.to_datetime(i).isoformat() for i in fusCtime_us_corrected]
            fusion_ctime_corrected_string = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]

            return fusCtime_us_corrected, fusion_ctime_corrected_string
        except Exception as e:
            print(e)
            print('Time sync not done')
            fusion_ctime_utc = [self.to_datetime(i).isoformat() for i in fusCtime_us]
            fusion_ctime_corrected_string = [i.split("+")[0][:-3] + 'Z' for i in fusion_ctime_utc]
            return fusCtime_us, fusion_ctime_corrected_string


    def run(self, file_name, **kwargs):
        """
        This function is used for extracting events from input file.
        :param file_name: input file name
        :param kwargs: additional arguments required by below section
        :return:
        """

        from nexus_toolkit.interpolation.stream_interpolator import upload_interpolated_stream, interpolate_cuboids
        import pickle

        from natsort import index_natsorted, order_by_index
        from datetime import datetime, timezone

        # Below section can be modified.
        try:

            # extract KPI_mode

            # objects_type = kwargs['objects_type']
            objects_type = 'Object_fusion'

            # extract baselog name from RESIM logs
            logPath, logName = os.path.split(file_name)
            logName = logName.replace('_dma', '_bus')
            logName_temp = logName.split('.')[0]
            baselogname, type = self.get_baselogname(logName_temp)

            if type == True:
                log_type = 'Original'
            else:
                log_type = 'RESIM'
            print('Input log : ',log_type)
            data = loadmat(file_name)
            corrected_fusion_ctime, fusion_ctime_corrected_string = self.timesync_fus(data, log_type)

            if objects_type == 'Object_fusion':
                vehicledataRaw_df, fusion_ctime_str, fusion_ctime, fus_vision_index_ctime_str_dict = self.get_logdata(data, fusion_ctime_corrected_string)
            elif objects_type == 'Vision':
                _, _, _, fus_vision_index_ctime_str_dict = self.get_logdata(data, fusion_ctime_corrected_string)
                vehicle_vision_raw_df = self.get_vision_obj_data(data, fus_vision_index_ctime_str_dict)
            elif objects_type == 'All':
                vehicledataRaw_df, fusion_ctime_str, fusion_ctime, fus_vision_index_ctime_str_dict = self.get_logdata(data, fusion_ctime_corrected_string)
                vehicle_vision_raw_df = self.get_vision_obj_data(data, fus_vision_index_ctime_str_dict)
            else:
                print('No obejct type is selected is paased')
                sys.exit(1)

            print("Extracted vehicle data")

            picklefile_path = None
            if "win" in sys.platform:

                pickle_file_name = baselogname + '_agt_0_8_1' + '.pickle'
                picklefile_path = os.path.join(logPath, pickle_file_name)
                picklefile_path_linux = self.get_pickle_file_path(baselogname)

            elif "lin" in sys.platform:
                print('platform :', sys.platform)
                picklefile_path = self.get_pickle_file_path(baselogname)

            if os.path.exists(picklefile_path):
                print(' The file exists')
            else:
                print('The file does not exist')
            try:
                with open(picklefile_path, 'rb') as f:
                    pickle_data = pickle.load(f)
            except Exception as e:
                print('Unable to load the pickle file')
            print("Loaded pickle file")

            # data transformation for interpolating data
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
            print("converted cubiods to serial data")
            print(len(serial_cubiods), len(fusion_ctime))
            print("started interpolation")

            ## Time sync

            # Interpolating cuboids of nexus data to Fusion timestamp
            nexus_url, nexus_user, nexus_pwd = self.nexus_details()
            # interpolated_cuboids = interpolate_cuboids(serial_cubiods, fusion_ctime_str,
            #                                            pi=None,
            #                                            nexus_url=nexus_url,
            #                                            nexus_user=nexus_user,
            #                                            nexus_pw=nexus_pwd,
            #                                            timing_offset=0,
            #                                            timestamp_type='timestamp',
            #                                            perform_sweep_comp=True,
            #                                            sampling_rate=1)

            interpolated_cuboids = interpolate_cuboids(serial_cubiods, fusion_ctime_corrected_string,
                                                       pi=None,
                                                       nexus_url=nexus_url,
                                                       nexus_user=nexus_user,
                                                       nexus_pw=nexus_pwd,
                                                       timing_offset=0,
                                                       timestamp_type='timestamp',
                                                       perform_sweep_comp=True,
                                                       sampling_rate=1)
            print("Done interpolation")

            # Transforming nexus cuboids to auto ground truth dataframe
            lidarMountingCals = {
                'position': {'x': 1.181648716540159, 'y': -0.06774385394548615, 'z': -2.014510213476982},
                'rotation': {'w': -0.003799397572359602, 'x': 0.6929092643722031, 'y': 0.7210009912854143,
                             'z': -0.004459427172793808}}

            import copy

            interpolated_data_vcs = copy.deepcopy(interpolated_cuboids)
            for idx, sample in enumerate(interpolated_cuboids):
                interpolated_data_vcs[idx] = self.convertToVcs(sample, lidarMountingCals)

            list_df = []
            for i in range(len(interpolated_data_vcs)):
                sample = interpolated_data_vcs[i]
                idx = i
                temp_df = self.convertGTCuboidsToDf(idx, sample)
                # temp_df['cTime'] = fusion_ctime_str[i]
                list_df.append(temp_df)
            autoGT_df = pd.concat(list_df, axis=0, ignore_index=True)
            abcompare = ABCompare()
            autoGT_df.rename(columns={'vcsHeading': 'Heading'}, inplace=True)
            autoGT_df.rename(columns={'timestamp': 'cTime'}, inplace=True)

            autoGT_df = self.rotation_position_correction(autoGT_df)
            print("Transformed data for correlation")

            # abcompare = ABcompareEuclidean()
            # dfMatchedGT_df, dfMatchedlog_df, orphans_df = abcompare.correlation_ab(autoGT_df, vehicledataRaw_df)

            if objects_type == 'Object_fusion':
                dfMatchedlog_of_df, dfMatchedGT_of_df, orphans_veh_of_df , orphans_autogt_of_df = abcompare.run(autoGT_df, vehicledataRaw_df)

                dfMatchedGT_of_df.insert(0, 'log_path', picklefile_path)
                dfMatchedGT_of_df.insert(1, 'log_name', baselogname)
                dfMatchedlog_of_df.insert(0, 'log_path', logPath)
                dfMatchedlog_of_df.insert(1, 'log_name', logName)


                dfMatchedlog_vision_df, dfMatchedGT_vision_df, orphans_veh_vision_df, orphans_autogt_vision_df = \
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            elif objects_type == 'Vision':
                dfMatchedlog_vision_df,dfMatchedGT_vision_df, orphans_veh_vision_df, orphans_autogt_vision_df = abcompare.run(autoGT_df, vehicle_vision_raw_df)

                dfMatchedGT_vision_df.insert(0, 'log_path', picklefile_path)
                dfMatchedGT_vision_df.insert(1, 'log_name', baselogname)
                dfMatchedlog_vision_df.insert(0, 'log_path', logPath)
                dfMatchedlog_vision_df.insert(1, 'log_name', logName)

                dfMatchedlog_of_df, dfMatchedGT_of_df, orphans_veh_of_df, orphans_autogt_of_df = \
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),pd.DataFrame()

            elif objects_type == 'All':
                dfMatchedlog_of_df, dfMatchedGT_of_df, orphans_veh_of_df, orphans_autogt_of_df = abcompare.run(autoGT_df, vehicledataRaw_df)
                dfMatchedlog_vision_df, dfMatchedGT_vision_df, orphans_veh_vision_df, orphans_autogt_vision_df = abcompare.run(autoGT_df, vehicle_vision_raw_df)

                dfMatchedGT_of_df.insert(0, 'log_path', picklefile_path)
                dfMatchedGT_of_df.insert(1, 'log_name', baselogname)
                dfMatchedlog_of_df.insert(0, 'log_path', logPath)
                dfMatchedlog_of_df.insert(1, 'log_name', logName)

                dfMatchedGT_vision_df.insert(0, 'log_path', picklefile_path)
                dfMatchedGT_vision_df.insert(1, 'log_name', baselogname)
                dfMatchedlog_vision_df.insert(0, 'log_path', logPath)
                dfMatchedlog_vision_df.insert(1, 'log_name', logName)

            print("Done correlation")

            run_mode_df = pd.DataFrame([objects_type], columns=['object_type'])

            out = dict()
            # Sheet name needs to match with sheet_name in self._headers
            # Make sure header column count matches with output column count
            out['autogt_matched_data_OF'] = np.array(dfMatchedGT_of_df.values)
            out['vehicle_matched_objects_OF'] = np.array(dfMatchedlog_of_df.values)

            out['autogt_matched_data_vision'] = np.array(dfMatchedGT_vision_df.values)
            out['vehicle_matched_objects_vision'] = np.array(dfMatchedlog_vision_df.values)

            out['Run_mode'] = np.array(run_mode_df.values)
            # out['orphans'] = np.array(orphans_autogt_df.values)
            return out

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # print(str(tb.format_exc()))
            return str(exc_obj.args[0]) + " FOUND IN LINE: " + str(exc_tb.tb_lineno)


if __name__ == '__main__':
    kwargs = dict()

    # file_name = r'C:\Users\pjp8lm\Desktop\Thunder\Latest_of_logs_agt_tool\FP_analyzes\TNDR1_DRUK_20240530_155144_WDC5_dma_0026.mat'
    file_name= r'C:\Users\pjp8lm\Desktop\Thunder\ACC_AGT_Integration\kusthav_issue_log\TNDR1_ASHN_20240320_161241_WDC5_rFLR240008243199_r4SRR240011243198_rM05_rVs05060011_rA24000124329162_dma_0010.mat'
    kwargs['objects_type'] = 'Object_fusion'

    Test = Comparehpcnexusagtgrid()
    headers = Test.get_headers()
    cell = Test.run(file_name, **kwargs)
    # print(cell)
    try:
        autogt_matched_data_OF = pd.DataFrame(data =cell['autogt_matched_data_OF'],columns=headers['autogt_matched_data_OF'])
        vehicle_matched_objects_OF = pd.DataFrame(data =cell['vehicle_matched_objects_OF'], columns=headers['vehicle_matched_objects_OF'])
        # orphans = pd.DataFrame(data =cell['orphans'] ,columns=headers['orphans'])
    except Exception as e:
        autogt_matched_data_OF = pd.DataFrame(columns=headers['autogt_matched_data_OF'])
        vehicle_matched_objects_OF = pd.DataFrame(columns= headers['vehicle_matched_objects_OF'])

    try:
        autogt_matched_data_vision = pd.DataFrame(data=cell['autogt_matched_data_vision'], columns=headers['autogt_matched_data_vision'])
        vehicle_matched_objects_vision = pd.DataFrame(data=cell['vehicle_matched_objects_vision'],
                                               columns=headers['vehicle_matched_objects_vision'])
    except Exception as e:
        autogt_matched_data_vision = pd.DataFrame(columns=headers['autogt_matched_data_vision'])
        vehicle_matched_objects_vision= pd.DataFrame(columns=headers['vehicle_matched_objects_vision'])
    run_mode_df = pd.DataFrame(data= cell['Run_mode'], columns=headers['Run_mode'])

    path, name = os.path.split(file_name)
    current_apth = os.chdir(path)
    writer = pd.ExcelWriter('Thunder_tracker_KPI' + '.xlsx')
    autogt_matched_data_OF.to_excel(writer, "autogt_matched_data_OF", index=False)
    vehicle_matched_objects_OF.to_excel(writer, "vehicle_matched_objects_OF", index=False)
    autogt_matched_data_vision.to_excel(writer, "autogt_matched_data_vision", index=False)
    vehicle_matched_objects_vision.to_excel(writer, "vehicle_matched_objects_vision", index=False)
    run_mode_df.to_excel(writer, "Run_mode", index= False)

    # orphans.to_excel(writer, "orphans", index=False)
    writer.close()
    print("Report generated!")
    Test.kpi_sheet_generation(os.path.join(path, 'Thunder_tracker_KPI.xlsx'))