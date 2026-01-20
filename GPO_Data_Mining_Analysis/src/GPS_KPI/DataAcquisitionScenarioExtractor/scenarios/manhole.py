import data_acquisition_utils
import geopandas as gpd


class Manhole:
    """
    Class that extracts manhole scenarios from OSM geopackage files.
    """

    def __init__(self, output_path):
        self.scenario = '35_Manhole'
        self.output_directory = output_path / self.scenario

    @staticmethod
    def filter(manhole: gpd.GeoDataFrame, road_type: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter function for manhole scenario.
        Extracts manhole with matching road.

        :param manhole: Gantry GeoDataframe
        :param road_type: Road_type GeoDataFrame
        :return: Filtered GeoDataframe with manholes merged with roads

        """
        road_type = road_type[road_type['access'] == '']
        manhole = data_acquisition_utils.merge_gdf_information(manhole, road_type, gdf_a_name='manhole',
                                                               gdf_b_name='road_type', max_distance=5)
        return manhole

    def run(self, config: data_acquisition_utils.Config) -> None:
        """
        Run function for manhole scenarios, returns none, saves map (.html), presentation (.pptx)
        and geopackage (.gpkg) files on specified location
        :param config: Config object
        :return: None
        """
        manhole = data_acquisition_utils.read_file(config.manhole['raw_file_path'])
        road_type = data_acquisition_utils.read_file(config.road_type['processed_file_path'])

        gdf_filtered = self.filter(manhole, road_type)

        comments = []
        self.output_directory.mkdir(parents=True, exist_ok=True)

        data_acquisition_utils.create_map(gdf_filtered, self.output_directory / '35_Manhole.html')
        data_acquisition_utils.create_pptx(gdf_filtered, self.output_directory / '35_Manhole.pptx', comments)
        data_acquisition_utils.save_file(gdf_filtered.drop(['geometry_road_type'], axis=1),
                                         self.output_directory / '35_Manhole.gpkg')
