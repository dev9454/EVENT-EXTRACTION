import data_acquisition_utils
import geopandas as gpd


class Gantry:
    """
    Class that extracts gantry scenarios from OSM geopackage files.
    """

    def __init__(self, output_path):
        self.scenario = '42_Gantry'
        self.output_directory = output_path / self.scenario

    @staticmethod
    def filter(gantry: gpd.GeoDataFrame, road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter function for gantry scenario.
        Extracts gantry with matching road
        :param gantry: Gantry GeoDataframe
        :return: Filtered GeoDataframe with gantry merge with road

        """
        road_type = road_type[road_type['access'] == ''].reset_index().drop(columns='index')
        return data_acquisition_utils.merge_gdf_information(gantry, road_type, gdf_a_name='gantry',
                                                            gdf_b_name='road_type')

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Run function for gantry scenarios, returns none, saves map (.html), presentation (.pptx)
        and geopackage (.gpkg) files on specified location
        :param config: Config object
        :return: None

        """
        gantry = data_acquisition_utils.read_file(config.gantry['processed_file_path'])
        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])

        gdf_filtered = self.filter(gantry, road_type)

        if len(gdf_filtered) == 0:
            print("No events found...")
        else:
            comments = []
            self.output_directory.mkdir(parents=True, exist_ok=True)

            data_acquisition_utils.create_map(gdf_filtered, self.output_directory / '42_Gantry.html')
            data_acquisition_utils.create_pptx(gdf_filtered, self.output_directory / '42_Gantry.pptx', comments)
            data_acquisition_utils.save_file(gdf_filtered.drop(['geometry_road_type'], axis=1),
                                             self.output_directory / '42_Gantry.gpkg')
