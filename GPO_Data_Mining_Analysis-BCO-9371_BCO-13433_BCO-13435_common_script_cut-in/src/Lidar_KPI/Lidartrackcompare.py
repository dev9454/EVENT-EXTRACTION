import pandas as pd
import numpy as np
import sys
from scipy.spatial import distance as dis
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

# Gloabl variables for threshold

POSITIONAL_THRESHOLD = 5
VELOCITY_THRESHOLD = 3
# LENGTH_THRESHOLD = 7
# WIDTH_THRESHOLD = 1.5
MAX_COST = 4

class ABCompare:
    """
    user class which compare objects from object fusion tracker and lidar cuboids
    returns the matched objects in dataframe format (matched TP, AutoGt FN, OF FP)
    """

    def get_vertices_and_side_centers(self, center, length, width, heading):
        c, s = np.cos(heading), np.sin(heading)
        dx, dy = length / 2, width / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated_corners = corners @ rotation_matrix.T
        vertices = center + rotated_corners
        side_centers = (vertices + np.roll(vertices, -1, axis=0)) / 2
        return vertices, side_centers

    def find_inf_rows_cols(self, arr):
        inf_rows = np.all(arr == np.inf, axis=1)
        inf_row_indices = np.where(inf_rows)[0]
        inf_cols = np.all(arr == np.inf, axis=0)
        inf_col_indices = np.where(inf_cols)[0]
        return inf_row_indices, inf_col_indices

    def element_with_min_distance(self, coords):
        distances = np.linalg.norm(coords, axis=1)  # Compute distances
        min_index = np.argmin(distances)  # Get index of minimum distance
        return coords[min_index], min_index  # Return the coordinate with minimum distance

    def get_ref_position_generic(self, long_position, lat_position, length, width, heading):
        positions = [long_position, lat_position]
        vertices, side_centers = self.get_vertices_and_side_centers(positions, length, width, heading)
        new_matrix = np.vstack((vertices, side_centers))
        result = np.empty((new_matrix.shape[0], new_matrix.shape[1]), dtype=new_matrix.dtype)
        result[::2] = new_matrix[:vertices.shape[0]]  # Fill even rows with arr1
        result[1::2] = new_matrix[vertices.shape[0]:]  # Fill odd rows with arr2
        new_matrix_1 = np.vstack((result, positions))
        """
        matrix:
        [vert-1] 0
        [sc-1] 1
        [vert-2] 2
        [sc-2] 3
        [ver-3] 4 
        [sc-3] 5
        [ver -4] 6
        [sc-4] 7
        [center] 8
        """
        if length < 2 and width < 2:  # returning the centroid as center of the object if the length and width are
            # less than 2
            return positions, 8, new_matrix_1
        ref_position, centriod_point = self.element_with_min_distance(new_matrix_1)
        return ref_position, centriod_point, new_matrix_1

    def return_ref_long_position(self, reference_position):
        return reference_position[0]

    def return_ref_lat_position(self, reference_position):
        return reference_position[1]

    def extract_data_tracker(self, tracker_objects):
        positions = np.array(tracker_objects[['Longitudinal_position', 'Lateral_position']])
        velocities = np.array(tracker_objects[['Longitudinal_velocity', 'Lateral_velocity']])
        lengths = np.array(tracker_objects['Target_Length'])
        widths = np.array(tracker_objects['Target_Width'])
        heading = np.array(tracker_objects['Heading'])
        return positions, velocities, lengths, widths, heading

    def compute_position_distance_cdist(self, pos1, pos2):
        matching_dist = dis.cdist(pos1, pos2)
        return matching_dist

    def compute_velocity_distance_cdist(self, vel1, vel2):
        matching_dist = dis.cdist(vel1, vel2, 'cityblock')
        return matching_dist

    def compute_legth_distance_cdist(self, len1, len2):
        len1 = len1.reshape(-1, 1)
        len2 = len2.reshape(-1, 1)
        inverse_len2 = 1/len2
        matching_dist = dis.cdist(len1, len2, 'cityblock')
        matching_dist_norm = np.multiply(matching_dist.T, inverse_len2)
        matching_dist_norm = matching_dist_norm.T
        return matching_dist_norm

    def compute_width_distance_cdist(self, wid1, wid2):
        wid1 = wid1.reshape(-1, 1)
        wid2 = wid2.reshape(-1, 1)
        inverse_wid2 = 1 / wid2
        matching_dist = dis.cdist(wid1, wid2, 'cityblock')
        matching_dist_norm = np.multiply(matching_dist.T, inverse_wid2)
        matching_dist_norm = matching_dist_norm.T
        return matching_dist_norm

    def create_similarity_distance_matrix(self, pos1, vel1, len1, wid1, pos2, vel2, len2, wid2):
        global POSITIONAL_THRESHOLD, VELOCITY_THRESHOLD, LENGTH_THRESHOLD, WIDTH_THRESHOLD

        position_distances = self.compute_position_distance_cdist(pos1, pos2)
        velocity_differences = self.compute_velocity_distance_cdist(vel1, vel2)
        length_differences = self.compute_legth_distance_cdist(len1, len2)
        width_differences = self.compute_width_distance_cdist(wid1, wid2)
        # similarity_matrix = np.divide(position_distances, POSITIONAL_THRESHOLD) + np.divide(velocity_differences, VELOCITY_THRESHOLD) + np.divide(
        #     length_differences, LENGTH_THRESHOLD) + np.divide(width_differences, WIDTH_THRESHOLD)
        similarity_matrix = np.divide(position_distances, POSITIONAL_THRESHOLD) + np.divide(velocity_differences, VELOCITY_THRESHOLD) + length_differences + width_differences
        return similarity_matrix

    def get_vertices(self, center, length, width, heading):
        c, s = np.cos(heading), np.sin(heading)
        dx, dy = length / 2, width / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated_corners = corners @ rotation_matrix.T
        return center + rotated_corners

    def polygon_intersection_area(self, poly1, poly2):
        polygon1 = Polygon(poly1)
        polygon2 = Polygon(poly2)
        intersection_area = polygon1.intersection(polygon2).area
        return intersection_area

    def polygon_area(self, vertices):
        polygon = Polygon(vertices)
        area = polygon.area
        return area

    def compute_ref_position(self, df):

        df[['reference_position','centriod_point','box_points']] = df.apply(
            lambda row: self.get_ref_position_generic(row['Longitudinal_position'], row['Lateral_position'],
                                                 row['Target_Length'], row['Target_Width'], row['Heading']), axis=1, result_type="expand")
        df['ref_long_position'] = df.apply(lambda row: self.return_ref_long_position(row['reference_position']), axis=1)
        df['ref_lat_position'] = df.apply(lambda row: self.return_ref_lat_position(row['reference_position']), axis=1)

        return df
    def compute_ref_position_new(self, df):
        df['potential_long_lat_positions'] = df.apply(lambda row: self.get_ref_position_matrix(row['Longitudinal_position'], row['Lateral_position'],
                                                 row['Target_Length'], row['Target_Width'], row['Heading']), axis=1)


    def calculate_iou(self, box1, box2):
        vertices1 = np.array([self.get_vertices(pos, l, w, h) for pos, l, w, h in
                              zip(box1['position'], box1['length'], box1['width'], box1['heading'])])
        vertices2 = np.array([self.get_vertices(pos, l, w, h) for pos, l, w, h in
                              zip(box2['position'], box2['length'], box2['width'], box2['heading'])])
        iou_matrix = np.zeros((len(box1['position']), len(box2['position'])))
        for i in range(len(box1['position'])):
            for j in range(len(box2['position'])):
                intersection = self.polygon_intersection_area(vertices1[i], vertices2[j])
                union = self.polygon_area(vertices1[i]) + self.polygon_area(vertices2[j]) - intersection
                iou_matrix[i, j] = intersection / union if union > 0 else 0
        return iou_matrix

    def run(self, autogt_df, vehicle_df):

        """
        main function run the comparison for vehicle_df and autoGT_df
        """
        global MAX_COST

        time_list = []
        # all_times = list(vehicle_df.cTime.unique())

        all_times = list(vehicle_df.corrected_ctime.unique())
        matched_vehicle_df_list = []
        matched_autoGT_df_list = []
        orphan_vehicle_df_list = []
        orphan_autoGT_df_list = []
        # all_times = ['2024-05-30T16:43:55.309Z']
        for i in all_times:

            try:
                # vehicle_time_df = vehicle_df.groupby('cTime').get_group(i)
                vehicle_time_df = vehicle_df.groupby('corrected_ctime').get_group(i)
                vehicle_time_df = vehicle_time_df.reset_index(drop=True)

            except Exception as e:
                print(e)
                print(f"vehicle dataframe not avail for {i}")
                continue
            try:
                autogt_time_df = autogt_df.groupby('cTime').get_group(i)
                autogt_time_df = autogt_time_df.reset_index(drop=True)

            except Exception as e:
                print(e)
                print(f"autoGT dataframe not avail for {i}")
                continue

            # compute ref_position for AutoGT and tracker

            # vehicle reference_position
            vehicle_time_df_temp = self.compute_ref_position(vehicle_time_df)

            # AutoGT reference position

            autogt_time_df_temp = self.compute_ref_position(autogt_time_df)

            # Extract positions and velocities
            tracker1_positions, tracker1_velocities, length_1, width_1, heading_1 = self.extract_data_tracker(
                vehicle_time_df_temp)
            tracker2_positions, tracker2_velocities, length_2, width_2, heading_2 = self.extract_data_tracker(
                autogt_time_df_temp)
            ref_position_1 = np.array(vehicle_time_df_temp[['ref_long_position', 'ref_lat_position']])
            ref_position_2 = np.array(autogt_time_df_temp[['ref_long_position', 'ref_lat_position']])

            # # Distance matrix

            dis_pos_matrix = self.create_similarity_distance_matrix(ref_position_1, tracker1_velocities, length_1, width_1,
                                                               ref_position_2, tracker2_velocities, length_2, width_2)

            ### IOU computation ####

            # box1 = {'position': tracker1_positions, 'length': length_1, 'width': width_1, 'heading': heading_1}
            # box2 = {'position': tracker2_positions, 'length': length_2, 'width': width_2, 'heading': heading_2}
            # iou_matrix = calculate_iou(box1,box2)
            # iou_cost = (1 - iou_matrix)
            # final_matrix = iou_cost + dis_pos_matrix

            final_matrix = dis_pos_matrix
            final_matrix[
                final_matrix > MAX_COST] = np.inf  # setting cost to 4 and assigning all the values to infinity to select orphans in tracker and AutoGT.
            orphan_tracker_objects, orphan_AutoGT_objects = self.find_inf_rows_cols(final_matrix)
            orphan_vehicle_temp_df = vehicle_time_df.iloc[list(orphan_tracker_objects), :]
            orphan_autogt_temp_df = autogt_time_df.iloc[list(orphan_AutoGT_objects), :]

            del orphan_autogt_temp_df['reference_position'], orphan_autogt_temp_df[
                'ref_long_position'], orphan_autogt_temp_df['ref_lat_position'], \
                orphan_vehicle_temp_df['reference_position'], orphan_vehicle_temp_df[
                'ref_long_position'], orphan_vehicle_temp_df['ref_lat_position']

            orphan_vehicle_df_list.append(orphan_vehicle_temp_df)
            orphan_autoGT_df_list.append(orphan_autogt_temp_df)

            potent_matched_tracker_objects = vehicle_time_df.drop(orphan_tracker_objects)
            potent_matched_autoGT_objects = autogt_time_df.drop(orphan_AutoGT_objects)
            potent_matched_tracker_objects = potent_matched_tracker_objects.reset_index(drop=True)
            potent_matched_autoGT_objects = potent_matched_autoGT_objects.reset_index(drop=True)
            if len(potent_matched_tracker_objects) == 0 or len(potent_matched_autoGT_objects) == 0:
                print(f" No matching objects in time frame{i}")
                continue
            else:
                # # matching
                # potent_matched_tracker_objects_temp = self.compute_ref_position(potent_matched_tracker_objects)
                # potent_matched_autoGT_objects_temp = self.compute_ref_position(potent_matched_autoGT_objects)
                potent_matched_tracker_objects_temp = potent_matched_tracker_objects
                potent_matched_autoGT_objects_temp = potent_matched_autoGT_objects
                tracker1_positions, tracker1_velocities, length_1, widths_1, heading_1 = self.extract_data_tracker(
                    potent_matched_tracker_objects_temp)
                tracker2_positions, tracker2_velocities, length_2, widths_2, heading_2 = self.extract_data_tracker(
                    potent_matched_autoGT_objects_temp)
                ref_position_1 = np.array(
                    potent_matched_tracker_objects_temp[['ref_long_position', 'ref_lat_position']])
                ref_position_2 = np.array(potent_matched_autoGT_objects_temp[['ref_long_position', 'ref_lat_position']])

                similarity_matrix = self.create_similarity_distance_matrix(ref_position_1, tracker1_velocities, length_1,
                                                                      widths_1, ref_position_2, tracker2_velocities,
                                                                      length_2, widths_2)
                del potent_matched_tracker_objects_temp['reference_position'], potent_matched_tracker_objects_temp[
                    'ref_long_position'], potent_matched_tracker_objects_temp['ref_lat_position'], \
                potent_matched_autoGT_objects_temp['reference_position'], potent_matched_autoGT_objects_temp[
                    'ref_long_position'], potent_matched_autoGT_objects_temp['ref_lat_position']
                box1 = {'position': tracker1_positions, 'length': length_1, 'width': widths_1, 'heading': heading_1}
                box2 = {'position': tracker2_positions, 'length': length_2, 'width': widths_2, 'heading': heading_2}
                iou_matrix = self.calculate_iou(box1, box2)
                iou_cost = (1 - iou_matrix)
                final_matrix = iou_cost + similarity_matrix + 1  # adding a penality of 1 to push objects in border not to match
                final_matrix[final_matrix > MAX_COST] = 1000
                rows_1, columns_1 = linear_sum_assignment(final_matrix)
                final_row = []
                final_column = []
                for i in range(len(rows_1)):
                    row = rows_1[i]
                    col = columns_1[i]
                    if final_matrix[row, col] == 1000:
                        # orph_vehicle_temp = list(potent_matched_tracker_objects_temp.iloc[row, :])
                        # orph_auto_temp = list(potent_matched_autoGT_objects_temp.iloc[col, :])
                        # orphan_vehicle_df_list.append(pd.DataFrame(orph_vehicle_temp, columns= potent_matched_tracker_objects_temp.columns.tolist()))
                        # orphan_autoGT_df_list.append(pd.DataFrame(orph_auto_temp, columns= potent_matched_autoGT_objects_temp.columns.tolist()))
                        orphan_vehicle_df_list.append(potent_matched_tracker_objects_temp.iloc[row, :])
                        orphan_autoGT_df_list.append(potent_matched_autoGT_objects_temp.iloc[col, :])
                    else:
                        final_row.append(row)
                        final_column.append(col)
                matched_vehicle_time_df = potent_matched_tracker_objects_temp.iloc[list(final_row), :]
                matched_autogt_time_df = potent_matched_autoGT_objects_temp.iloc[list(final_column), :]
                matched_vehicle_df_list.append(matched_vehicle_time_df)
                matched_autoGT_df_list.append(matched_autogt_time_df)
                time_list.append(i)
        if len(matched_vehicle_df_list) == 0:
            matched_vehicle_df = pd.DataFrame()
        else:
            matched_vehicle_df = pd.concat(matched_vehicle_df_list)
        if len(matched_autoGT_df_list) == 0:
            matched_autoGT_df = pd.DataFrame()
        else:
            matched_autoGT_df = pd.concat(matched_autoGT_df_list)
        if len(orphan_vehicle_df_list) == 0:
            orphan_vehicle_df = pd.DataFrame()
        else:
            orphan_vehicle_df = pd.concat(orphan_vehicle_df_list)
        if len(orphan_autoGT_df_list) == 0:
            orphan_autoGT_df = pd.DataFrame()
        else:
            orphan_autoGT_df = pd.concat(orphan_autoGT_df_list)
        return matched_vehicle_df, matched_autoGT_df, orphan_vehicle_df, orphan_autoGT_df

class ABcompareEuclidean:
    def correlation_ab(self, gt_data, log_data):
        # log_data = log_data[log_data['Longitudinal_position'] != 0] # just considering objects in front of the vehicle
        dfMatchedlog = pd.DataFrame()
        dfMatchedGT = pd.DataFrame()
        orphans = pd.DataFrame()
        # indexframes_persec = sum(vehicledata['Fusion_Index'] == vehicledata.Fusion_Index.iloc[1,])
        # Fusion_Indexes = np.unique(np.array(vehicledata.Fusion_Index))
        ctime = np.unique(log_data['corrected_ctime'])
        for timeframe in range(len(ctime)):  # len(Fusion_Indexes) ##testing

            ''' # Original code for comparing the CADM to RESIM objects
            #print(timeframe)
            # dataframe_1_part = a.iloc[ (timeframe * indexframes_persec):(indexframes_persec + timeframe * indexframes_persec)]
            # dataframe_2_part = b.iloc[(timeframe * indexframes_persec):(indexframes_persec + timeframe * indexframes_persec)]
            '''
            dataframe_1_part = gt_data[gt_data['cTime'] == ctime[timeframe]]
            dataframe_2_part = log_data[log_data['corrected_ctime'] == ctime[timeframe]]
            df1 = dataframe_1_part.loc[:, ['Longitudinal_position', 'Lateral_position']]
            df2 = dataframe_2_part.loc[:, ['Longitudinal_position', 'Lateral_position']]
            if not df1.empty and not df2.empty:
                matching_dist = dis.cdist(df1, df2,
                                          metric='euclidean')  # scipy.spatial.distance.cdist  returns 2D with ij refering eucd(df1[i],df2[j]) distance
                dfmatching = pd.DataFrame(matching_dist)  # 2D DataFrame - 96x2 size
                threshold_pos = 7  #####threshold for distance similarity######
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


if __name__ == '__main__':
    vehicle_df = pd.read_csv(r'C:\Users\pjp8lm\Desktop\Thunder\Latest_of_logs_agt_tool\FP_analyzes\vehicle_0026.csv')
    autogt_df = pd.read_csv(r'C:\Users\pjp8lm\Desktop\Thunder\Latest_of_logs_agt_tool\FP_analyzes\Autogt_0026.csv')
    test = ABCompare()
    matched_vehicle_df, matched_autoGT_df, orphan_vehicle_df, orphan_autoGT_df = test.run(autogt_df, vehicle_df)
    print('Here')