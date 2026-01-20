import geopandas as gpd
from data_acquisition_utils import get_intersection, create_map, create_pptx, read_file, save_file, \
    merge_gdf_information

''' author  : Padmesh (uqgqif)
    version : 1.0 '''


class Tunnels:
    '''
    Class that extract tunnel sceanrios from OSM pkl files
    '''


    def __init__(self, output_path):
        self.output_file_name = "43_85_Tunnels"
        self.output_directory = output_path / self.output_file_name


    @staticmethod
    def filter(tunnels):
        """
        Filter function for tunnels scenario
        :param tunnels: Tunnels GeoDataframe
        :return: Filtered GeoDataframe with tunnels of length > 50m
        """

        tunnels = tunnels[(tunnels['length'] > 0.031068) & (tunnels['access:lanes'] == '')] # filter for length > 50m and
        # access tag empty
        return tunnels


    def run(self, config):
        """
        Run function for tunnels scenarios, returns none, saves map, pptx and gpkg on specified location
        :param config: Config object
        :return: None
        """
        tunnels = read_file(config.tunnel['processed_file_path'])
        # tunnels = read_file(r"C:\Work\BCO-12581\Gpkg\GCC\gcc-states-20241007_tunnel_processed.pkl")  # local
        filtered_tunnels = self.filter(tunnels)
        if len(filtered_tunnels) == 0:
            print("No events found")
        else:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            comments = ['highway', 'lanes', 'maxspeed', 'surface']
            create_map(filtered_tunnels, self.output_directory / str(self.output_file_name + '.html'))
            save_file(filtered_tunnels, self.output_directory / str(self.output_file_name + '.gpkg'))
            create_pptx(gpd.GeoDataFrame(filtered_tunnels, geometry='geometry'),
                self.output_directory / str(self.output_file_name + '.pptx'), comments)
