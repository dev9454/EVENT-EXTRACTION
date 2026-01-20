import geopandas as gpd
from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file, \
    merge_gdf_information

''' author  : Padmesh (uqgqif)
    version : 1.1 '''


class RoadsNearSchools:

    def __init__(self, output_path):
        self.output_file_name = "6_Roads_Near_Schools"
        self.output_directory = output_path / self.output_file_name

    @staticmethod
    def filter(road_type, buildings):
        """
        Filter function for school scenario
        :param road_type:
        :param buildings: Buildings GeoDataframe
        :return: Filtered GeoDataframe with roads near schools under 100m
        """
        road_type = road_type[road_type['access:lanes'] == '']  # filter for access tag
        buildings = buildings[((buildings['amenity'] == 'school') |
                               (buildings['building'] == 'school'))].reset_index(
            drop=True)  # filter for school building
        return merge_gdf_information(road_type, buildings, max_distance=100, gdf_a_name='road',
                                     gdf_b_name='school').dropna(subset=['el_id_school'])

    def run(self, config):
        """
        Run function for school scenarios, returns none, saves map, pptx and gpkg on specified location
        :param config: Config object
        :return: None
        """
        buildings = read_file(config.buildings['processed_file_path'])
        # buildings = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_buildings_processed.pkl")  # local
        road_type = read_file(config.road_type['processed_file_path'])
        # road_type = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_road_type_processed.pkl")  # local
        filtered_school = self.filter(road_type, buildings)

        if len(filtered_school) == 0:
            print("No events found...")
        else:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(filtered_school, self.output_directory / str(self.output_file_name + '.html'))
            save_file(filtered_school.drop(columns=['geometry_school']),
                      self.output_directory / str(self.output_file_name + '.gpkg'))
            comments = ['highway', 'lanes', 'maxspeed', 'surface']
            create_pptx(gpd.GeoDataFrame(filtered_school, geometry='geometry'),
                        self.output_directory / str(self.output_file_name + '.pptx'), comments)