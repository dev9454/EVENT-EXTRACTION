import geopandas as gpd
import pandas as pd

from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file, \
    merge_gdf_information
from collections import defaultdict
from shapely import Point
from shapely.ops import linemerge
import numpy as np

''' author  : Padmesh (uqgqif)
    version : 1.1 '''


class StraightRoad:

    def __init__(self, output_path):
        self.output_file_name = "137_Feature_Straight_Road"
        self.output_directory = output_path / self.output_file_name

    def adjacency_list(self, roads_gdf):
        """
        Function to create dictionary with points as keys and list of indices as values
        :param roads_gdf: road_type GeoDataframe
        :return adj_list:  dict of list of indices as values and gps points as keys
        """
        adj_list = defaultdict(list)
        for idx, line in enumerate(roads_gdf.geometry):
            start, end = line.coords[0], line.coords[-1]
            adj_list[start].append(idx)
            adj_list[end].append(idx)
        return adj_list

    def calculate_slope(self, line):
        """
        Function to calculate slope of line
        :param line: linestring for calculating slope
        :return slope : slope of line
        """

        x1, y1, x2, y2 = line.coords[0] + line.coords[-1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        return angle

    def are_segments_aligned_and_connected(self, seg1, seg2, angle_tolerance=10 * np.pi / 180):
        """
        Function to check weather the segments are connected and have similar slope
        : param seg1: segment 1
        : param seg2: segment 2
        : param angle_tolerance: tolerance_angle (default is 10 degrees)
        """

        slope1 = abs(StraightRoad.calculate_slope(self, seg1))
        slope2 = abs(StraightRoad.calculate_slope(self, seg2))
        if abs(slope1 - slope2) > angle_tolerance:
            return False

        return seg1.coords[-1] == seg2.coords[0] or seg1.coords[0] == seg2.coords[-1] or seg1.coords[-1] == seg2.coords[
            -1] \
            or seg1.coords[0] == seg2.coords[0]

    def merge_connected_segments(self, roads_gdf):
        """"
        Main function to merge segments
        :param roads_gdf: road_type GeoDataframe
        :return merged_gdf: GeoDataframe with stright roads connected
        """

        adj_list = StraightRoad.adjacency_list(self, roads_gdf)
        visited = set()
        merged_lines = []
        for idx, line in roads_gdf.iterrows():
            if idx in visited:
                continue

            # Initialize a path starting with the current line
            current_line = line.geometry
            stack = [idx]
            visited.add(idx)

            while stack:
                segment_idx = stack.pop()
                segment = roads_gdf.geometry[segment_idx]
                visited.add(segment_idx)

                for point in [segment.coords[0], segment.coords[-1]]:
                    for neighbor_idx in adj_list[point]:
                        if neighbor_idx not in visited:
                            if StraightRoad.are_segments_aligned_and_connected(self, current_line, roads_gdf.geometry[neighbor_idx]):
                                temp_line = linemerge([current_line, roads_gdf.geometry[neighbor_idx]])
                                if Point(temp_line.coords[0]).distance(
                                        Point(temp_line.coords[-1])) > 0.75 * temp_line.length:
                                    current_line = temp_line
                                    stack.append(neighbor_idx)
                                    visited.add(neighbor_idx)

                            # Merge the current line with this neighbor segment
            merged_lines.append(current_line)
        # Create a new GeoDataFrame with the merged lines
        merged_gdf = gpd.GeoDataFrame(geometry=merged_lines, crs=roads_gdf.crs)
        merged_gdf['length'] = merged_gdf.to_crs(3857).length
        return merged_gdf[merged_gdf['length'] > 1000]

    def filter(self, road_type):
        """
        Filter function for tunnels scenario
        :param road_type: Road GeoDataframe
        :return: Filtered GeoDataframe with SLF=>0.99 and length > 1000m (0.621371 mi)
        """

        roads_single = road_type[(road_type['length'] >= 0.6213) & (road_type['SLF'] >= 0.99) &
                  (road_type['highway'].isin(['motorway', 'primary', 'secondary']))]
        roads_multiple = StraightRoad.merge_connected_segments(self, road_type[(road_type['SLF'] >= 0.99) &
                  (road_type['highway'].isin(['motorway', 'primary', 'secondary']))].reset_index(drop=True))
        filtered = gpd.GeoDataFrame(pd.concat([roads_single, roads_multiple]).reset_index(drop=True))
        return filtered


    def run(self, config):
        """
        Run function for tunnels scenarios, returns none, saves map, pptx and gpkg on specified location
        :param config: Config object
        :return: None
        """
        road_type = read_file(config.road_type['processed_file_path'])
        # road_type = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_road_type_processed.pkl").to_crs(4087) # local
        filtered_road_types = self.filter(road_type)
        if len(filtered_road_types) == 0:
            print("No events found")
        else:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            comments = ['highway_left', 'lanes', 'maxspeed', 'surface']

            create_map(filtered_road_types, self.output_directory / str(self.output_file_name + '.html'))
            save_file(filtered_road_types, self.output_directory / str(self.output_file_name + '.gpkg'))
            create_pptx(gpd.GeoDataFrame(filtered_road_types, geometry='geometry'),
                        self.output_directory / str(self.output_file_name + '.pptx'), comments)
