import geopandas as gpd
from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file, \
    merge_gdf_information
import pandas as pd

''' author  : Padmesh (uqgqif)
    version : 1.2 '''


class PolesGuardrails:

    def __init__(self, output_path):
        self.output_file_name_poles = "81_Poles"
        self.output_file_name_guard_rails = "82_86_Guard_Rails"
        self.output_directory_poles = output_path / self.output_file_name_poles
        self.output_directory_guard_rails = output_path / self.output_file_name_guard_rails

    @staticmethod
    def filter_pole(road_type, pole):
        """
        Filter function for poles; Access tag of road == ''
        :param road_type: Road GeoDataframe
        :param pole: Poles GeoDataframe
        :return: Filtered GeoDataframe with roads under 10m
        """

        road_type = road_type[road_type['access:lanes'] == '']
        return merge_gdf_information(pole, road_type, max_distance=10, gdf_a_name='pole', gdf_b_name='road')

    @staticmethod
    def filter_guardrail(road_type, barrier, tags):
        """
        Filter function for guard_rails; Access tag of road == '', barrier == 'guard_rail'
        :param road_type: Road GeoDataframe
        :param barrier: Barrier GeoDataframe
        :param tags: Tags to filter barrier for poles and guard_rails
        :return: Filtered GeoDataframe with tags and near roads under 10m
        """

        barrier = barrier[barrier['barrier'].isin(tags)]
        road_type = road_type[road_type['access:lanes'] == '']
        return merge_gdf_information(road_type, barrier, max_distance=10, gdf_a_name='road',
                                     gdf_b_name='object').dropna(subset=['el_id_object'])

    def run(self, config):
        """
        Run function for Poles and Guardrail scenarios, returns none, saves map, pptx and gpkg on specified directories
        :param config: Config object
        :return: None
        """
        road_type = read_file(config.road_type['processed_file_path'])
        #road_type = read_file(r"C:\Work\BCO-12581\Gpkg\CES2025\CES2025_07102024_road_type.pkl")  # local
        barrier = read_file(config.barrier['processed_file_path'])
        #barrier = read_file(r"C:\Work\BCO-12581\Gpkg\CES2025\CES2025_07102024_barrier.pkl")  # local
        poles = read_file(config.pole['processed_file_path'])
        #poles = read_file(r"C:\Work\BCO-12581\Gpkg\CES2025\CES2025_07102024_pole.pkl")  # local
        street_lamps = read_file(config.street_lamp['processed_file_path'])
        #street_lamps = read_file(r"C:\Work\BCO-12581\Gpkg\CES2025\CES2025_07102024_street_lamp.pkl")  # local
        filtered_poles = gpd.GeoDataFrame(pd.concat([self.filter_pole(road_type, pd.concat([poles, street_lamps])),
                                                     self.filter_guardrail(road_type, barrier,
                                                                           tags=['bollard', 'delineators']),
                                                     ]).reset_index(drop=True), geometry='geometry')
        filtered_guard_rails = self.filter_guardrail(road_type, barrier, tags=['guard_rail'])

        # Keep only one geometry dtype
        for col in filtered_guard_rails.columns:
            if col != filtered_guard_rails.geometry.name and filtered_guard_rails[col].dtype == 'geometry':
                filtered_guard_rails[col] = filtered_guard_rails[col].astype(str)

        for col in filtered_poles.columns:
            if col != filtered_poles.geometry.name and filtered_poles[col].dtype == 'geometry':
                filtered_poles[col] = filtered_poles[col].astype(str)

        comments = ['highway', 'lanes', 'maxspeed', 'surface']

        if len(filtered_guard_rails) == 0:
            print("No guard rails events found...")
        else:
            self.output_directory_guard_rails.mkdir(parents=True, exist_ok=True)

            create_map(filtered_guard_rails,
                       self.output_directory_guard_rails / str(self.output_file_name_guard_rails + '.html'))
            save_file(filtered_guard_rails,
                      self.output_directory_guard_rails / str(self.output_file_name_guard_rails + '.gpkg'))
            create_pptx(filtered_guard_rails,
                        self.output_directory_guard_rails / str(self.output_file_name_guard_rails + '.pptx'), comments)

        if len(filtered_poles) == 0:
            print("No poles events found...")
        else:
            self.output_directory_poles.mkdir(parents=True, exist_ok=True)

            create_map(filtered_poles, self.output_directory_poles / str(self.output_file_name_poles + '.html'))
            save_file(filtered_poles, self.output_directory_poles / str(self.output_file_name_poles + '.gpkg'))

            create_pptx(filtered_poles, self.output_directory_poles / str(self.output_file_name_poles + '.pptx'),
                        comments)

