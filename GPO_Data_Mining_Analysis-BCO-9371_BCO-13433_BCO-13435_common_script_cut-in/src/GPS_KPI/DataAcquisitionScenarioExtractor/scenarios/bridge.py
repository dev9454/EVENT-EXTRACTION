import geopandas as gpd
from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file, \
    merge_gdf_information
import sys

''' author  : Padmesh (uqgqif)
    version : 1.0 '''


class Bridge:

    def __init__(self, output_path):
        self.output_file_name = "34_Bridge"
        self.output_directory = output_path / self.output_file_name

    @staticmethod
    def filter(bridges):
        """
        Filter function for tunnels scenario
        :param bridges: Tunnels GeoDataframe
        :return: Filtered GeoDataframe with tunnels of length > 50m
        """

        # filter for length > 20m and highway tag filter
        bridges_filtered = bridges[
            (bridges['length'] > 20) & bridges['highway'].isin(['motorway', 'trunk', 'trunk', 'primary',
                                                                'secondary', 'tertiary', 'unclassified', 'residential',
                                                                'living_street', 'service', 'road', 'track',
                                                                'motorway_link', 'trunk_link', 'primary_link',
                                                                'secondary_link', 'tertiary_link']) &
                       (bridges['access'] == '')]

        return bridges_filtered

    def run(self, config):
        """
        Run function for tunnels scenarios, returns none, saves map, pptx and gpkg on specified location
        :param config: Config object
        :return: None
        """
        bridges = read_file(config.bridges['processed_file_path'])
        # bridges = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_bridges_merged_processed.pkl")  # local
        filtered_bridges = self.filter(bridges)
        if len(filtered_bridges) == 0:
            print("No events found...")
        else:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            create_map(filtered_bridges, self.output_directory / str(self.output_file_name + '.html'))
            save_file(filtered_bridges,
                      self.output_directory / str(self.output_file_name + '.gpkg'))
            comments = ['highway', 'lanes', 'maxspeed', 'surface']
            create_pptx(gpd.GeoDataFrame(filtered_bridges, geometry='geometry'),
                        self.output_directory / str(self.output_file_name + '.pptx'), comments)
