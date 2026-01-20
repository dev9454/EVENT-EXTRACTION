"""
@author: Filip Ulmaniec (gjl6gk)
last update: Jakub Grzesiak 10.10.2024

This is the main function for postprocessing, as an argument we provide the path to the pickle files. All calculations are performed.
As output we get new files with suffix "_processed" and "_merged_processed" and also json file in which the paths to processed files are saved.
If json files exists, then on the input function checks which have already been processed and which have not and processed only those that have
not yet been. If there is no json file, the function will process everything again.
"""
from shapely.geometry import LineString, Polygon, MultiPolygon
from extract_intersections import find_intersections_parallel
import argparse
import pickle
import json
import re
import os


class GpkgPostProcessing:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        path = folder_path.rstrip(os.sep)
        self.json_filename = os.path.basename(path) + "_config.json"
        self.json_path = os.path.join(folder_path, self.json_filename)

    def load_existing_json(self):
        """
        Input: None
        Output: empty dictionary

        The function checks if the json file exists, if so it loads it.
        """
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as json_file:
                    return json.load(json_file)
            except Exception as e:
                print(f"Error during loading JSON file: there is no file! {e}")
        return {}

    def load_pickle_files(self):
        """
        Output: pickle data, pickle file names, tunnel bridge data, tunel bridge file names

        The function loads all files from the specified folder path that are pickle and are not
        in the loaded json file. Then it divides into 2 lists where we have data from all pickle
        files and only for tunnels and bridges and 2 lists with the names of these files.
        """
        existing_data = self.load_existing_json()
        processed_files = set()

        if 'Road_types' in existing_data:
            for road_type_info in existing_data['Road_types']:
                if 'raw' in road_type_info:
                    processed_files.add(road_type_info['raw'])

        pickle_files = []
        file_names = []
        tunnel_bridge_files = []
        tunnel_bridge_file_names = []

        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.pkl') and '_processed' not in file_name and file_name not in processed_files and 'intersections' not in file_name:
                file_path = os.path.join(self.folder_path, file_name)
                try:
                    with open(file_path, 'rb') as file:
                        gdf = pickle.load(file)
                        pickle_files.append(gdf)
                        file_names.append(file_name)

                    if 'tunnel' in file_name.lower() or 'bridges' in file_name.lower():
                        tunnel_bridge_files.append(gdf)
                        tunnel_bridge_file_names.append(file_name)

                except Exception as e:
                    print(f'Error during loading the file: {file_name}: {e}')
        return pickle_files, file_names, tunnel_bridge_files, tunnel_bridge_file_names

    @staticmethod
    def line_to_polygon(geometry):
        if geometry is None:
            return geometry
        else:
            if geometry.is_ring and not isinstance(geometry, (Polygon, MultiPolygon)):
                return Polygon(geometry)
            else:
                return geometry

    @staticmethod
    def get_length(gdf):
        """
        Input: gdf data
        Output: gdf data with length calculated

        Function get_length gets gdf data as a parameter, it calculates the length of each object. As a return value we
        receive lengths.
        """
        return gdf.to_crs(3857).length * 0.000621371  # calculated length of each object and changed from meters -> mileage

    def get_area(self, gdf):
        """
        Input: gdf data
        Output: gdf data with area calculated

        Function get_area gets gdf data as a parameter, it calculates the area of closed object. As a return value we
        receive areas.
        """
        return gdf.to_crs(3857).geometry.apply(self.line_to_polygon).area

    @staticmethod
    def merge_objects(gdf):
        """
        Input: gdf data
        Output: gdf data with merged rows

        Function merge_objects gets gdf data as a parameter, it merges objects whose start and end coordinates are the same and type matches.
        As a return value we receive gdf data with reduced number of rows by those merged.
        """
        gdf['original_index'] = gdf.index

        filtered_gdf = gdf[(gdf.geometry.type == 'LineString')]

        filtered_gdf['start_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[0] if geom is not None else None)
        filtered_gdf['end_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[-1] if geom is not None else None)

        while True:
            matches = filtered_gdf.merge(
                filtered_gdf,
                left_on=['end_coords', 'type', 'highway'],
                right_on=['start_coords', 'type', 'highway'],
                suffixes=('', '_match')
            )

            if matches.empty:
                break

            def merge_geometries(row):
                geom1 = row['geometry']
                geom2 = row['geometry_match']
                if geom1 is None and geom2 is None:
                    return geom1 if geom1 is not None else geom2
                return LineString(list(geom1.coords) + list(geom2.coords)[1:])

            matches['new_geometry'] = matches.apply(merge_geometries, axis=1)

            filtered_gdf.loc[matches['original_index'], 'geometry'] = matches['new_geometry']

            filtered_gdf = filtered_gdf.drop(matches['original_index_match'])

            filtered_gdf['start_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[0] if geom is not None else None)
            filtered_gdf['end_coords'] = filtered_gdf.geometry.apply(lambda geom: geom.coords[-1] if geom is not None else None)

        return filtered_gdf.drop(columns=['start_coords', 'end_coords', 'original_index'])

    @staticmethod
    def calculate_slf(gdf):
        """
        Input: gdf data
        Output: gdf data with slf calculated

        Function  gets gpkg data as a parameter, it checks only LineString geometries and then calculates SLF parameter.
        As a return value we receive SLF calculation.
        """
        def calculate_row_slf(geometry):
            # Check if the geometry is a LineString
            if isinstance(geometry, LineString):
                # Calculate the Euclidean distance (straight line distance)
                start_point = geometry.coords[0]
                end_point = geometry.coords[-1]
                euclidean_distance = LineString([start_point, end_point]).length

                # Calculate the actual length of the way
                actual_length = geometry.length

                # Calculate the Straight Line Factor (SLF)
                slf = euclidean_distance / actual_length if actual_length > 0 else 0
                return slf
            else:
                return None
        return gdf['geometry'].apply(calculate_row_slf)

    @staticmethod
    def extract_region_and_road_type(file_name):
        """
        Input: file_name
        Output: region and road_type

        The function extracts from file name region and type of road.
        """
        file_name_spltted = re.match(r"(.+?)[_-](?:\d+|latest)[-_](.*)", file_name.split('.')[0])
        region = file_name_spltted.group(1) if file_name_spltted else 'unknown'
        road_type = file_name_spltted.group(2) if file_name_spltted else 'unknown'

        return region, road_type

    def save_to_json(self, file_names, pickle_all_data):
        """
        Input: file names and data
        Output: json file

        The function write to json region, road type, areas, raw file path, processed file path and merged processed file path.
        It checks if those are available, if not it stays False.
        """
        json_data = self.load_existing_json()
        for gdf, file_name in zip(pickle_all_data, file_names):

            bbox = gdf.total_bounds.tolist()
            region, road_type = self.extract_region_and_road_type(file_name)

            region_key = 'Region'
            if region_key not in json_data:
                json_data.update({region_key: region, 'Road_types': []})

            processed_file_name = f"{os.path.splitext(file_name)[0]}_processed.pkl"
            merged_processed_file_name = f"{os.path.splitext(file_name)[0]}_merged_processed.pkl"

            raw_exists = os.path.exists(os.path.join(self.folder_path, file_name))
            processed_exists = os.path.exists(os.path.join(self.folder_path, processed_file_name))
            merged_processed_exists = os.path.exists(os.path.join(self.folder_path, merged_processed_file_name))

            json_data['Road_types'].append({
                'road_type': road_type,
                'area': bbox,
                'raw': os.path.join(self.folder_path, file_name) if raw_exists else False,
                'processed': os.path.join(self.folder_path, processed_file_name) if processed_exists else False,
                'merged_processed': os.path.join(self.folder_path, merged_processed_file_name) if merged_processed_exists else False
            })

        json_path = os.path.join(self.folder_path, self.json_filename)
        try:
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
        except Exception as e:
            print(f'Error during saving to JSON file: {e}')

    def save_new_files(self, gdf_list, file_names, suffix):
        """
        Input: pickle files, file names, suffix for new files
        Output: new files

        The function saves new proceed files with the appropriate suffixes.
        """
        for gdf, file_name in zip(gdf_list, file_names):
            processed_file_name = f"{os.path.splitext(file_name)[0]}{suffix}.pkl"
            processed_file_path = os.path.join(self.folder_path, processed_file_name)
            try:
                with open(processed_file_path, 'wb') as file:
                    pickle.dump(gdf, file)
            except Exception as e:
                print(f'Error during saving processed file: {processed_file_name}: {e}')

    def run(self):
        """
         The function run is a main function in which all other functions mentioned earlier are called in turn.
         """
        pickle_all_data, file_names, pickle_tunnel_bridge_data, tunnel_bridge_file_names = self.load_pickle_files()
        road_type_files = dict()

        if not file_names and not tunnel_bridge_file_names:
            print("No new files to process.")
            return

        if file_names:
            processed_data = []
            total_files = len(file_names)
            for i, gdf in enumerate(pickle_all_data):
                current_file_name = file_names[i]
                percentage = ((i+1) / total_files) * 100
                print(f"File being processed ({i+1}/{total_files}): {current_file_name} [{percentage:.2f}%]")

                if 'administrative_area' in current_file_name:
                    gdf['area'] = self.get_area(gdf)
                else:
                    gdf['length'] = self.get_length(gdf)
                    gdf['area'] = self.get_area(gdf)
                    gdf['SLF'] = self.calculate_slf(gdf)
                processed_data.append(gdf)

                if 'road_type' in current_file_name:
                    road_type_files[current_file_name] = gdf

            self.save_new_files(processed_data, file_names, suffix='_processed')

        if tunnel_bridge_file_names:
            merged_data = []
            total_bridge_tunnel_files = len(tunnel_bridge_file_names)
            for i, gdf in enumerate(pickle_tunnel_bridge_data):
                current_file_name = tunnel_bridge_file_names[i]
                percentage = ((i + 1) / total_bridge_tunnel_files) * 100
                print(f"File being processed: ({i + 1}/{total_bridge_tunnel_files}): {current_file_name} [{percentage:.2f}%]")

                merged_gdf = self.merge_objects(gdf)
                merged_gdf['length'] = self.get_length(merged_gdf)
                merged_gdf['area'] = self.get_area(merged_gdf)
                merged_gdf['SLF'] = self.calculate_slf(merged_gdf)
                merged_data.append(merged_gdf)

            self.save_new_files(merged_data, tunnel_bridge_file_names, suffix='_merged_processed')

        # intersections
        if road_type_files:
            intersections = []
            intersections_file_names = []
            i = 0
            for file_name, gdf in road_type_files.items():
                percentage = ((i + 1) / len(road_type_files.keys())) * 100
                intersection_file_name = file_name.replace('road_type', 'intersections')
                print(f"Creating file: ({i + 1}/{len(road_type_files.keys())}): {intersection_file_name} [{percentage:.2f}%] (Be patient, this process may take up to several minutes)")
                intersection = find_intersections_parallel(gdf)
                intersections.append(intersection)
                intersections_file_names.append(intersection_file_name)

            self.save_new_files(intersections, intersections_file_names, suffix='_processed')
            pickle_all_data += intersections
            file_names += intersections_file_names

        self.save_to_json(file_names, pickle_all_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post processing GPKG files')
    parser.add_argument('folder_path', type=str, help='Path to the folder with GPKG files')

    args = parser.parse_args()

    folder_path = args.folder_path
    # folder_path = r"C:\\Users\\bjrlhg\\Desktop\\GEO_pickle\\CES"

    object = GpkgPostProcessing(folder_path)
    object.run()


